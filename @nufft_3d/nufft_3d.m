classdef nufft_3d
    %obj = nufft_3d(om,N,varargin)
    % Non-uniform 3D fourier transform (based on Fessler NUFFT).
    % Note: trajectory units are phase cycles/fov so the Nyquist
    % distance is 1 unit (Fessler's om is in radians/fov).
    %
    % If available, code uses gpuSparse class (single precision)
    % or gpuArray sparse (double precision).
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
        N(1,3) double = zeros(1,3) % final image dimensions
        alpha(1,1) double     = 0  % kaiser-bessel parameter (0=Beatty optimal)
        radial(1,1) logical   = 1  % radial kernel (1=yes 0=no)        
    end
    
    % behind the scenes parameters
    properties (SetAccess = private)
        gpu(1,1) double     = 1    % use GPU (0=no 1=gpuSparse 2=gpuArray)
        K(1,3) double = zeros(1,3) % oversampled image dimensions
        H(:,:)                     % sparse interpolation matrix
        HT(:,:)                    % H transpose (stored separately on CPU)        
        U(:,:,:)                   % deapodization matrix
        d(:,1)                     % density weighting
        nnz(1,1) double     = 0    % number of sample points within bounds
    end

    % experimental parameters
    properties (SetAccess = private, Hidden = true)
        fnull                      % forward nullspace for pruno
        anull                      % adjunct nullspace for pruno
    end
    
    methods
        
        %% constructor
        function obj = nufft_3d(om,N,varargin)

            %% handle varargin
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
            if obj.u<1 || obj.u>6 % doesn't work properly with u=1
                error('value of u=%f is not recommended',obj.u);
            end
            if obj.alpha==0
                % Beatty formula
                obj.alpha = pi*sqrt(obj.J^2*(1-0.5/obj.u)^2-0.8);
                if obj.radial
                    % with radial, seems better with a linear term in u
                    obj.alpha = pi*sqrt(obj.J^2*(1-0.5/obj.u)^2-0.6*obj.u); 
                end
            end
            if obj.radial && (obj.J<3 || obj.u<1.5)
                warning('radial kernel not recommended with J<3 or u<1.5');
            end
            if (numel(N)==1 || numel(N)==3) && isnumeric(N)
                obj.N = ones(1,3).*reshape(N,[],1); N = [];
            else
                error('N must be scalar or 3-vector of even integers');
            end
            if all(obj.N==0) || mod(obj.N(1),2) || mod(obj.N(2),2) % allow z=1 for 2D
                error('N must be an even integer');
            end
            if all(om(3,:)==0) % catch z=1 case for 2D
                obj.N(3) = 1;
            end
            
            % oversampled matrix size (must be even and limited to ~1280^3)
            obj.K = 2 * ceil(obj.N * obj.u / 2);
            if obj.N(3)==1; obj.K(3) = 1; end
            if prod(obj.K)>=intmax('int32'); error('N or u too large'); end
            
            %% handle trajectory
            disp(' Trajectory:     min       max       matrix')
            fprintf('  om(1): %12.3f %9.3f %9i\n',min(om(1,:)),max(om(1,:)),obj.N(1))
            fprintf('  om(2): %12.3f %9.3f %9i\n',min(om(2,:)),max(om(2,:)),obj.N(2))           
            fprintf('  om(3): %12.3f %9.3f %9i\n',min(om(3,:)),max(om(3,:)),obj.N(3))
            
            % scale to oversampled matrix size (double to avoid flintmax('single') limit)
            om = obj.u * double(om);

            % only keep points that are within bounds
            ok = om(1,:) >= -obj.K(1)/2 & om(1,:) <= obj.K(1)/2-1;
            ok = om(2,:) >= -obj.K(2)/2 & om(2,:) <= obj.K(2)/2-1 & ok;
            ok = om(3,:) >= -obj.K(3)/2 & om(3,:) <= obj.K(3)/2-1 & ok | obj.N(3)==1;
            obj.nnz = nnz(ok);
            fprintf('  %i points (out of %i) are within bounds.\n',obj.nnz,numel(ok))
            
            %% set up interpolation matrix
            t = tic;
            
            % size
            nrow = numel(ok);            
            ncol = prod(obj.K);
            obj.H = sparse(nrow,ncol);
            
            % send to gpu (try/catch fallback to gpuArray, then cpu)
            if obj.gpu
                try
                    if obj.gpu==1; obj.H = gpuSparse(obj.H); end
                catch ME
                    warning('%s Trying gpuArray sparse.',ME.message);
                end
                try
                    om = gpuArray(om);
                    ok = gpuArray(ok);
                catch ME
                    obj.gpu = 0;
                    warning('%s Setting gpu=0.',ME.message);
                end
            end

            % overkill to include endpoints
            range = -ceil(obj.J/2):ceil(obj.J/2);
            
            % create sparse matrix - assemble in parts in 32-bit (lower memory requirement)
            for dx = range
                for dy = range
                    
                    I = int32([]); J = int32([]); S = single([]); 
                                    
                    for dz = range
                        
                        % allow for 2d case
                        if obj.N(3)==1 && dz~=0; continue; end
                        
                        % neighboring grid points (keep dx,dy,dz outside for consistent rounding)
                        x = round(om(1,:)) + dx;
                        y = round(om(2,:)) + dy;
                        z = round(om(3,:)) + dz;
                        
                        % distance (squared) from the samples
                        dx2 = (x-om(1,:)).^2;
                        dy2 = (y-om(2,:)).^2;
                        dz2 = (z-om(3,:)).^2;
                        
                        % wrap negatives (kspace centered at 0)
                        x = mod(x,obj.K(1));
                        y = mod(y,obj.K(2));
                        z = mod(z,obj.K(3));
                        
                        % sparse matrix indices: use <= to include endpoints for accuracy
                        if obj.radial
                            dist2 = dx2 + dy2 + dz2;
                            i = ok & 4*dist2 <= obj.J^2;
                            j = 1 + x(i) + obj.K(1)*y(i) + obj.K(1)*obj.K(2)*z(i);
                            s = obj.kernel(dist2(i));
                        else
                            i = ok & 4*dx2 <= obj.J^2 & 4*dy2 <= obj.J^2 & 4*dz2 <= obj.J^2;
                            j = 1 + x(i) + obj.K(1)*y(i) + obj.K(1)*obj.K(2)*z(i);
                            s = obj.kernel(dx2(i)).*obj.kernel(dy2(i)).*obj.kernel(dz2(i));
                        end

                        % store indices for sparse call
                        I = cat(2,I,find(i)); J = cat(2,J,j); S = cat(2,S,s);
                        
                        % clear temporaries
                        clearvars i j s x y z dx2 dy2 dz2 dist2
                        
                    end

                    if isa(obj.H,'gpuSparse')
                        obj.H = obj.H + gpuSparse(I,J,S,nrow,ncol);
                    else
                        S = double(S); % sparse only accepts double values
                        if verLessThan('matlab','9.8'); I = double(I); end
                        if verLessThan('matlab','9.8'); J = double(J); end
                        obj.H = obj.H + sparse(I,J,S,nrow,ncol);
                    end

                end
            end
            fprintf(' Created %s matrix. ',class(obj.H)); toc(t);
            
            % clear large temporaries
            clearvars -except om N varargin obj ok

            % store transpose matrix (CPU only where A*x and A'*x differ in speed) 
            if obj.gpu==0; obj.HT = obj.H'; end

            %% final steps

            % deapodization matrix
            obj.U = obj.deap();

            % density weighting (default or user supplied)
            if isempty(obj.d)
                obj.d = obj.density(ok);
            else
                if numel(obj.d)~=nrow
                    error('supplied density has wrong number of elements (should be %i)',nrow);
                end
                obj.d = cast(reshape(obj.d,[],1),'like',obj.U);
                obj.d(~ok) = 0; % exclude out of bounds points
                if mean(obj.d)~=1
                    warning('mean(density) should be 1 (%f)',mean(obj.d));
                end
            end
            
            % we are going to do a lot of ffts of the same type so tune it
            fftw('planner','measure');      

            % display properties
            fprintf(' Created'); disp(obj);
            fprintf('\n');
            fprintf('\tH: [%ix%i] (nonzeros %i) %0.1fGbytes\n',size(obj.H),nnz(obj.H),3*nnz(obj.H)*4*(2-isa(obj.H,'gpuSparse'))/1e9);
            fprintf('\tU: [%ix%ix%i] min=%f max=%f\n',size(obj.U,1),size(obj.U,2),size(obj.U,3),min(obj.U(:)),max(obj.U(:)));
            fprintf('\td: [%ix%i] (zeros %i) min=%f max=%f\n',size(obj.d),nnz(obj.d==0),min(nonzeros(obj.d)),max(obj.d));
            fprintf('\n');
  
        end

    end
    
    %% utility functions - keep all the hacks in one place
    methods (Access = private, Hidden = true)
        
        %% sparse matrix vector multiply
        function y = spmv(obj,k)
            if isa(obj.H,'gpuSparse')
                y = single(k);
            else
                y = double(k);
            end
            if ~isempty(obj.HT)
                y = obj.HT' * y;
            else
                y = obj.H * y;
            end
            y = full(y);            
        end
        
        %% sparse transpose matrix vector multiply
        function y = spmv_t(obj,k)
            if isa(obj.H,'gpuSparse')
                y = single(k);
            else
                y = double(k);
            end
            y = obj.H' * y;
            y = full(y);
        end
        
        %% 3d fft with pre-fft padding (cartesian kspace <- cartesian image)
        function x = fft3_pad(obj,x)
            for j = 1:3
                pad = size(x); pad(j) = (obj.K(j)-obj.N(j)) / 2;
                x = cat(j,zeros(pad,'like',x),x,zeros(pad,'like',x));
                x = fft(fftshift(x,j),[],j);
            end
        end

        %% 3d ifft with post-ifft cropping (cartesian image <- cartesian kspace)
        function k = ifft3_crop(obj,k)
            for j = 1:3
                mid = (obj.K(j)-obj.N(j)) / 2;
                k = ifftshift(ifft(k,[],j),j) * obj.K(j);
                if j==1; k = k(1+mid:end-mid,:,:,:); end
                if j==2; k = k(:,1+mid:end-mid,:,:); end
                if j==3; k = k(:,:,1+mid:end-mid,:); end
            end
        end
        
        %% image projection operator (image <- image)
        function y = iprojection(obj,x,damp,W)
            % y = A' * W * A * x + penalty on ||x||
            y = obj.fNUFT(x);
            y = W.*y;
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
            % y = P' * A' * W * A * P * x + penalty on ||imag(x)||
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