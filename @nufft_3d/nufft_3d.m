classdef nufft_3d
    
    % Non-uniform 3D fourier transform (based on Fessler NUFFT).
    % Note: trajectory units are phase cycles/fov which means the
    % Nyquist distance is 1 unit (Fessler's om is in radians/fov).
    % If available, code uses gpu sparse or gpuSparse matrices.
    %
    % Inputs:
    %  om = trajectory centered at 0 (unit = cycles/fov) [3 npts]
    %  N = final matrix dimensions (must be an even number) [3 1]
    %  varargin = string/value pairs, e.g. ('radial',0)
    %
    % Output:
    %  obj = nufft object using sparse matrix coefficients
    %
    % Typical usage:
    %   N = 128; % matrix size
    %   om = trajectory from -64 to +64
    %   ksp = kspace data for trajectory 
    %   obj = nufft(om,N,'J',4,'radial',0);
    %   img = obj.iNUFT(ksp,10); % 10 iterations
    
    % key parameters
    properties (SetAccess = private)
        J(1,1) double         = 4  % kernel width (4)
        u(1,1) double         = 2  % oversampling factor (2)
        N(3,1) double = zeros(3,1) % final image dimensions
        alpha(1,1) double     = 0  % kaiser-bessel parameter (0=Beatty optimal)
        radial(1,1) logical   = 0  % radial kernel (1=yes 0=no)        
    end
    
    % behind the scenes parameters
    properties (SetAccess = private, Hidden = true)

        gpu(1,1) logical      = 1  % use gpu (gpuSparse) if present (1=yes 0=no)
        low(1,1) double       = 5  % lowpass filter: h = exp(-(-low:low).^2/low)
        K(3,1) double = zeros(3,1) % oversampled image dimensions   
        d(:,1)                     % density weighting vector           
        H(:,:)                     % sparse interpolation matrix
        HT(:,:)                    % transpose of H (faster if stored separately)
        U(:,:,:)                   % deapodization matrix
     
    end

    % experimental thing
    properties (SetAccess = private, Hidden = true)
        sd                         % the mean sample density in kspace
        dwsd                       % the mean density weighed sample density
        fnull                      % forward nullspace for pruno
        anull                      % adjunct nullspace for pruno   
    end
    
    methods
        
        %% constructor
        function obj = nufft_3d(om,N,varargin)

            if nargin==0
                return; % default constructor required by MATLAB
            end 
            if ~isnumeric(om) || ~isreal(om) || size(om,1)~=3 || isempty(om)
                error('om must be a real array with leading dimension of 3')
            else
                om = reshape(om,3,[]); % remove unnecessary shape
            end
            if ~exist('N','var') || isempty(N)
                N = 2 * ceil(max(abs(om),[],2)); % assume centered at [0 0 0]
                warning(sprintf('N argument not supplied - using [%i %i %i]',N))
            end

            % varargin handling - must be field/value pairs, e.g. ('J',5)
            for k = 1:2:numel(varargin)
                if k==numel(varargin) || ~ischar(varargin{k})
                    error('''varargin'' must be supplied in string/value pairs');
                end
                obj.(varargin{k}) = varargin{k+1};
            end
            
            %% check values, sizes, shapes, etc.
            if obj.J<1 || obj.J>7
                warning('value of J=%f is not recommended',obj.J);
            end
            if obj.u<=1 || obj.u>6 % doesn't work properly with u=1
                error('value of u=%f is not recommended',obj.u);
            end
            if obj.alpha==0
                % Beatty formula
                obj.alpha = pi*realsqrt(obj.J^2*(1-0.5/obj.u)^2-0.8);
                if obj.radial
                    % with radial, it seems better with a linear term in u
                    obj.alpha = pi*realsqrt(obj.J^2*(1-0.5/obj.u)^2-0.6*obj.u); 
                end
            end
            if obj.radial && (obj.J<3 || obj.u<1.5)
                warning('radial kernel not recommended with J<3 or u<1.5');
            end
            if ~isscalar(obj.low) || obj.low<0 || mod(obj.low,1)
                error('low must be a nonnegative integer');
            end
            if (numel(N)==1 || numel(N)==3) && isnumeric(N)
                obj.N = ones(3,1).*reshape(N,[],1); N = [];
            else
                error('N must be scalar or 3-vector of even integers');
            end
            if mod(obj.N(1),2) || mod(obj.N(2),2) % allow z=1 for 2D
                error('N must be an even integer');
            end
            if all(om(3,:)==0) % catch z=1 case for 2D
                obj.N(3) = 1;
            end
            
            % oversampled matrix size (must be even)
            obj.K = 2 * ceil(obj.N * obj.u / 2);
            if obj.N(3) == 1; obj.K(3) = 1; end
            
            % display trajectory limits
            disp(' Trajectory:     min        max        matrix')
            fprintf('   om(1):    %.3f       %.3f      %i\n',min(om(1,:)),max(om(1,:)),obj.N(1))
            fprintf('   om(2):    %.3f       %.3f      %i\n',min(om(2,:)),max(om(2,:)),obj.N(2))           
            fprintf('   om(3):    %.3f       %.3f      %i\n',min(om(3,:)),max(om(3,:)),obj.N(3))
            
            % scale trajectory units (need double to stay below flintmax in sparse indicies)
            kx = obj.u * double(om(1,:));
            ky = obj.u * double(om(2,:));
            kz = obj.u * double(om(3,:));

            % only keep points that are within bounds
            ok = kx >= -obj.K(1)/2 & kx <= obj.K(1)/2-1 & ky >= -obj.K(2)/2 & ky <= obj.K(2)/2-1;
            if obj.N(3)>1; ok = ok & kz >= -obj.K(3)/2 & kz <= obj.K(3)/2-1; end % allow 2d
            fprintf('  %i points (out of %i) are out of bounds.\n',sum(~ok),numel(ok))
            
            %% set up interpolation matrix
            
            % no. columns
            ncol = prod(obj.K);
            
            % no. rows
            nrow = numel(ok);
            
            % interpolation matrix: because H and H' have wildly
            % different performance, store them both and always
            % use the fastest operation. Cost = 2x memory.
            %
            % on CPU
            %  -transpose is faster to create due to sorted columns
            %  -transpose is faster to multiply (H'*y >> HT*y)
            % on GPU
            %  -non-transpose is faster to create due to sorted rows
            %  -non-tranpose is faster to multiply (HT*y >> H'*y)
            
            obj.H  = sparse(nrow,ncol);
            obj.HT = sparse(ncol,nrow);
            
            % push to gpu (try/catch fallback to cpu)
            if obj.gpu
                try
                    obj.H  = gpuArray(obj.H);
                    obj.HT = gpuArray(obj.HT);
                catch
                    obj.gpu = 0;
                    warning('GPU device error, fallback to CPU.');
                end
                if obj.gpu
                    try
                        obj.H  = gpuSparse(obj.H);
                        obj.HT = gpuSparse(obj.HT);
                    catch
                        warning('gpuSparse not found, fallback to gpuArray.');
                    end
                    kx = gpuArray(kx);
                    ky = gpuArray(ky);
                    kz = gpuArray(kz);
                end
            end
            
            %% calculate indices and convolution coefficients
            
            tic; fprintf(' Creating %s matrix H     ',class(obj.H)); count = 1;
            
            % overkill on the range to include endpoints
            range = -ceil(obj.J/2):ceil(obj.J/2);
            
            for ix = range
                for iy = range
                    for iz = range

                        % to allow 2d
                        if obj.N(3)==1 && iz~=0; continue; end
                        
                        % neighboring grid points (keep ix,iy,iz outside for consistent rounding)
                        x = round(kx) + ix;
                        y = round(ky) + iy;
                        z = round(kz) + iz;

                        % distance (squared) from the samples
                        dx2 = (x-kx).^2;
                        dy2 = (y-ky).^2;
                        dz2 = (z-kz).^2;
                        dist2 = dx2 + dy2 + dz2;
                        
                        % wrap negatives (kspace centered at 0)
                        x = mod(x,obj.K(1));
                        y = mod(y,obj.K(2));
                        z = mod(z,obj.K(3));
                        
                        % sparse matrix indices: use <= to include endpoints
                        if obj.radial
                            i = find(ok & 4*dist2 <= obj.J^2);
                            j = 1+x(i)+obj.K(1)*y(i)+obj.K(1)*obj.K(2)*z(i);
                            s = obj.kernel(dist2(i));
                        else
                            i = find(ok & 4*dx2 <= obj.J^2 & 4*dy2 <= obj.J^2 & 4*dz2 <= obj.J^2);
                            j = 1+x(i)+obj.K(1)*y(i)+obj.K(1)*obj.K(2)*z(i);
                            s = obj.kernel(dx2(i)).*obj.kernel(dy2(i)).*obj.kernel(dz2(i));
                        end

                        % accumulate sparse matrix
                        if obj.gpu
                            obj.H = obj.H + gpuSparse(i,j,s,nrow,ncol);
                        else
                            obj.HT = obj.HT + sparse(j,i,s,ncol,nrow);
                        end

                        % display progress
                        fprintf('\b\b\b\b%-2d%% ',floor(100*count/numel(range)^3)); count = count+1;
                    end
                end
            end
            fprintf('\b. '); toc

            % clear memory of large temporaries 
            clear i j s kx ky kz dist2 dx2 dy2 dz2 x y z

            %% final steps

            % transpose of H or HT           
            if obj.gpu
                try
                    obj.HT = full_ctranspose(obj.H);
                catch ME % out of memory?
                    obj.HT = obj.H'; % lazy transpose
                end
            else
                obj.H = obj.HT';
            end
            
            % check for boo boos
            if nnz(obj.H>1); error('invalid value(s) in H'); end
            
            % deapodization matrix
            obj.U = obj.deap();
            
            % density weighting
            [obj.d obj.sd obj.dwsd] = obj.density(ok);
            
            % we are going to do a lot of ffts of the same type so tune it
            fftw('planner','measure');           

            % display properties
            fprintf(' Created'); disp(obj);
            fprintf('\n')
            fprintf('\tH: [%ix%i] (nonzeros %i) %0.1fGbytes\n',size(obj.H),nnz(obj.H),3*nnz(obj.H)*4*(2-obj.gpu)/1e9);
            fprintf('\tU: [%ix%ix%i] min=%f max=%f\n',size(obj.U,1),size(obj.U,2),size(obj.U,3),min(obj.U(:)),max(obj.U(:)))
            fprintf('\td: [%ix%i] (zeros %i) min=%f max=%f\n',size(obj.d),nrow-nnz(obj.d),min(nonzeros(obj.d)),max(obj.d))
            fprintf('\n')
  
        end

    end
    
    %% utility functions - keep all the hacks in one place
    methods (Access = private, Hidden = true)
        
        % sparse matrix vector multiply
        function y = spmv(obj,k)
            if obj.gpu
                y = single(k);
                y = obj.H * y;
            else
                y = double(k);
                y = obj.HT' * y;
                y = full(y);
            end
        end
        
        % sparse transpose matrix vector multiply
        function y = spmv_t(obj,k)
            if obj.gpu
                y = single(k);
                y = obj.HT * y;
            else
                y = double(k);
                y = obj.H' * y;
                y = full(y);
            end
        end
        
        % 3d fft with pre-fft padding (cartesian kspace <- cartesian image)
        function k = fft3_pad(obj,x)
            for j = 1:3
                pad = size(x); pad(j) = (obj.K(j)-obj.N(j)) / 2;
                if j==1; x = cat(1,zeros(pad),x,zeros(pad)); end
                if j==2; x = cat(2,zeros(pad),x,zeros(pad)); end
                if j==3; x = cat(3,zeros(pad),x,zeros(pad)); end
                %if pad(j)>0
                %    % in the padded matrix, should the end-points should share the energy?
                %    if j==1; x(end-pad(j)+1,:,:,:) = x(end-pad(j)+1,:,:,:)/2; x(pad(j),:,:,:) = x(end-pad(j)+1,:,:,:); end
                %    if j==2; x(:,end-pad(j)+1,:,:) = x(:,end-pad(j)+1,:,:)/2; x(:,pad(j),:,:) = x(:,end-pad(j)+1,:,:); end
                %    if j==3; x(:,:,end-pad(j)+1,:) = x(:,:,end-pad(j)+1,:)/2; x(:,:,pad(j),:) = x(:,:,end-pad(j)+1,:); end
                %end
                x = fft(fftshift(x,j),[],j); 
            end
            k = x; % output is kspace
        end

        %% 3d ifft with post-ifft cropping (cartesian image <- cartesian kspace)
        function x = ifft3_crop(obj,k)
            for j = 1:3
                crop = (obj.K(j)-obj.N(j)) / 2;
                k = ifftshift(ifft(k,[],j),j); 
                if j==1; k = k(1+crop:end-crop,:,:,:); end
                if j==2; k = k(:,1+crop:end-crop,:,:); end
                if j==3; k = k(:,:,1+crop:end-crop,:); end
            end
            x = k * prod(obj.K) / prod(obj.N); % ifft scaling (1/N not 1/K)
        end
        
        %% image projection operator (image <- image)
        function y = iprojection(obj,x,damp,W)
            % y = A' * W * D * A * x + penalty on ||x||
            y = obj.fNUFT(x);
            y = (W.*obj.d).*y; % density weighting included
            y = obj.aNUFT(y);
            x = reshape(x,size(y));
            y = y + damp.^2.*x;
            if isempty(obj.fnull) || isempty(obj.anull)
                y = reshape(y,prod(obj.N),[]); % coils separate
            elseif size(x,4)~=size(obj.fnull,4)
                error('check number of coils (received %i expected %i)',size(x,4),size(obj.fnull,4));
            else
                % nullspace constraints
                r = sum(obj.fnull.*x,4);
                r = sum(obj.anull.*r,5);
                y = reshape(y+r,[],1); % coils coupled
            end
        end
        
        %% phase constrained projection operator (image <- image)
        function y = pprojection(obj,x,damp,lambda,W,P)
            % y = P' * A' * W * D * A * P * x + penalty on ||imag(x)||
            P = reshape(P,size(x));
            y = P.*x;
            y = obj.iprojection(y,damp,W);
            if ~isscalar(lambda)
                lambda = reshape(lambda,size(x,1),1);
            end
            y = conj(P).*y + i*lambda.^2.*imag(x);
        end
        
    end
    
end