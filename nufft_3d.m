classdef nufft_3d

    properties (SetAccess = immutable)
        
        J      @ double scalar  = 4;  % kernel width (4)
        u      @ double scalar  = 2;  % oversampling factor (2)
        N      @ double = zeros(3,1); % final matrix dimensions       
        K      @ double = zeros(3,1); % oversampled matrix dimensions
        alpha  @ double scalar  = 0;  % kaiser-bessel parameter (0=use Fessler optimal)
        radial @ logical scalar = 1;  % radial kernel (1=yes 0=no)
        deap   @ logical scalar = 0;  % deapodization type (0=analytical 1=numerical)
        gpu    @ logical scalar = 1;  % use gpuSparse instead of sparse (1=yes 0=no)

        H;                            % sparse interpolation matrix
        HT;                           % transpose of H (faster if stored separately)
        U;                            % deapodization matrix
        d;                            % density weighting vector

    end

    methods

        %% Constructor
        function f = nufft_3d(om,N,J,u,alpha,radial,deap,gpu)
            
            % Non-uniform fourier transform (based on Fessler NUFFT).
            %
            % Inputs:
            %  om = trajectory centered at (0 0 0) units 1/fov [3 npts]
            %  N = final matrix dimensions (must be even number) [3 1]
            %  J, u, alpha, etc.: (optional) override default values
            %
            % Output:
            %  f = nufft object with precalculated coeffficients
            %

            if nargin==0
                return; % 0 arg option needed for class constructor
            end 
            if ~isnumeric(om) || isempty(om) || size(om,1)~=3
                error('om must be an array with leading dimension of 3')
            end
            if ~exist('N','var') || isempty(N) || ~isnumeric(N)
                N = 2 * ceil(max(max(abs(om),[],2),[],3));
                warning('N argument not valid. Using [%i %i %i].',N)
            end
            if exist('J','var') && ~isempty(J)
                f.J = J;
            end
            if exist('u','var') && ~isempty(u)
                f.u = u;
            end
            if exist('alpha','var') && ~isempty(alpha)
                f.alpha = alpha;
            else
                f.alpha = f.J * spline([1 1.5 2 3],[1.5 2.05 2.34 2.6],f.u);
            end
            if exist('radial','var') && ~isempty(radial)
                f.radial = radial;
            end
            if exist('deap','var') && ~isempty(deap)
                f.deap = deap;
            end
            if exist('gpu','var') && ~isempty(gpu)
                f.gpu = gpu;
            end

            %% check values, sizes, shapes, precision, etc.
            if f.J<1 || f.J>6
                error('value of J=%f is not recommended',f.J);
            end
            if f.u<1 || f.u>4
                error('value of u=%f is not recommended',f.u);
            end
            if f.radial && f.u~=2
                warning('radial kernel not tested with u=%f (try u=2).',f.u)
            end
            if numel(N)==1 || numel(N)==3
                f.N = ones(1,3).*reshape(N,1,[]); N = [];
            else
                error('N argument must be scalar or 3-vector.')
            end
            om = reshape(om,3,[]);
            
            % odd matrix sizes not tested - probably won't work
            if any(mod(f.N,2))
                error('matrix size must be even - not tested with odd values.')
            end
            
            % oversampled matrix size (must be even)
            f.K = 2 * ceil(f.N * f.u / 2);

            % display trajectory limits
            disp(' Trajectory:     min        max        matrix')
            fprintf('   om(1):    %.3f       %.3f      %i\n',min(om(1,:)),max(om(1,:)),f.N(1))
            fprintf('   om(2):    %.3f       %.3f      %i\n',min(om(2,:)),max(om(2,:)),f.N(2))           
            fprintf('   om(3):    %.3f       %.3f      %i\n',min(om(3,:)),max(om(3,:)),f.N(3))
            
            % convert trajectory to new units (double to avoid precision loss in H)
            kx = f.u * double(om(1,:));
            ky = f.u * double(om(2,:));
            kz = f.u * double(om(3,:));
           
            % only keep points that are within bounds
            ok = abs(kx<f.K(1)/2) & abs(ky<f.K(2)/2) & abs(kz<f.K(3)/2);
            fprintf('  %i points (out of %i) are out of bounds.\n',sum(~ok),numel(ok))


            %% set up indicies and convolution coefficients
            
            % no. columns
            ncol = prod(f.K);
            
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
            %
            f.H  = sparse(nrow,ncol);
            f.HT = sparse(ncol,nrow);
            f.U = zeros(f.N,'single'); % deapodization matrix
            
            % push to gpu if needed (try/catch fallback to cpu)
            if f.gpu
                try
                    f.H  = gpuSparse(f.H);
                    f.HT = gpuSparse(f.HT);
                    f.U = gpuArray(f.U);
                    kx = gpuArray(kx);
                    ky = gpuArray(ky);
                    kz = gpuArray(kz);
                    ok = gpuArray(ok);
                catch ME
                    f.gpu = 0;
                    warning('%s Setting gpu=0.',ME.message);
                end
            end
           
            tic; fprintf(' Creating sparse matrix H     ');

            for ix = 1:ceil(f.J)
                for iy = 1:ceil(f.J)
                    for iz = 1:ceil(f.J)
                        
                        % neighboring grid points: keep ix,iy,iz outside floor() to avoid problems
                        x = floor(kx-f.J/2) + ix;
                        y = floor(ky-f.J/2) + iy;
                        z = floor(kz-f.J/2) + iz;
                        
                        % Euclidian distance (squared) for the samples
                        dx2 = (x-kx).^2;
                        dy2 = (y-ky).^2;
                        dz2 = (z-kz).^2;
                        dist2 = dx2 + dy2 + dz2;
                        
                        % Euclidian distance (squared) for numerical deapodisation
                        ux2 = (ix-1).^2/4;
                        uy2 = (iy-1).^2/4;
                        uz2 = (iz-1).^2/4;
                        udist2 = ux2 + uy2 + uz2;
                        
                        % wrap out of bounds
                        x = mod(x,f.K(1));
                        y = mod(y,f.K(2));
                        z = mod(z,f.K(3));

                        % sparse matrix indices
                        if f.radial
                            % radial kernel
                            i = find(ok & dist2 < f.J.^2/4);
                            j = 1+x(i) + f.K(1)*y(i) + f.K(1)*f.K(2)*z(i);
                            s = f.convkernel(dist2(i));
                            % deapodization coefficients
                            if f.deap && udist2 <= f.J.^2/4
                                f.U(ix,iy,iz) = f.convkernel(udist2);
                            end
                        else
                            % separable kernel
                            i = find(ok & dx2 < f.J.^2/4 & dy2 < f.J.^2/4 & dz2 < f.J.^2/4);
                            j = 1+x(i) + f.K(1)*y(i) + f.K(1)*f.K(2)*z(i);
                            s = f.convkernel(dx2(i)).*f.convkernel(dy2(i)).*f.convkernel(dz2(i));
                            % deapodization coefficients
                            if f.deap && ux2 <= f.J.^2/4 && uy2 <= f.J.^2/4 && uz2 <= f.J.^2/4
                                f.U(ix,iy,iz) = f.convkernel(ux2).*f.convkernel(uy2).*f.convkernel(uz2);
                            end
                        end

                        % accumulate sparse matrix
                        if f.gpu
                            f.H = f.H + gpuSparse(i,j,s,nrow,ncol);
                        else
                            f.HT = f.HT + sparse(j,i,s,ncol,nrow);
                        end
                        
                        % display progress
                        fprintf('\b\b\b\b%-2d%% ',floor(100*sub2ind(ceil([f.J f.J f.J]),iz,iy,ix)/ceil(f.J).^3));
                    end
                end
            end
            toc

            % free memory for GPU
            clear dist2 dx2 dy2 dz2 kx ky kz x y z i j k s
            
            % un-transpose
            tic; fprintf(' Un-transposing sparse matrix. ');
            
            if f.gpu
                try
                    f.HT = full_ctranspose(f.H);
                catch ME % out of memory?
                    f.HT = f.H';
                    warning('Using lazy transpose. %s',ME.message);
                end
            else
                f.H = f.HT';
            end
            toc

            % deapodization matrix
            tic; fprintf(' Creating deapodization function. ');

            if f.deap
                
                % numerical deapodization (with 2x oversampling)
                for j = 1:3
                    f.U = ifft(f.U*f.K(j),2*f.K(j),j,'symmetric');
                    if j==1; f.U(1+N(1)/2:end-N(1)/2,:,:) = []; end
                    if j==2; f.U(:,1+N(2)/2:end-N(2)/2,:) = []; end
                    if j==3; f.U(:,:,1+N(3)/2:end-N(3)/2) = []; end
                end
                f.U = fftshift(f.U);

            else
                
                % analytical deapodization (Lewitt, J Opt Soc Am A 1990;7:1834)
                if false
                    % centered: do not use, requires centered fftshifts, no advantage in accuracy
                    x = ((1:f.N(1))-f.N(1)/2-0.5)./f.K(1);
                    y = ((1:f.N(2))-f.N(2)/2-0.5)./f.K(2);
                    z = ((1:f.N(3))-f.N(3)/2-0.5)./f.K(3);
                else
                    % not centered: gives almost the same deapodization matrix as numerical
                    x = ((1:f.N(1))-f.N(1)/2-1)./f.K(1);
                    y = ((1:f.N(2))-f.N(2)/2-1)./f.K(2);
                    z = ((1:f.N(3))-f.N(3)/2-1)./f.K(3);
                end
                [x y z] = ndgrid(x,y,z);

                if f.radial
                    % radial
                    a = f.J/2;
                    C = 4*pi*a.^3/f.bessi0(f.alpha);
                    R = realsqrt(x.^2 + y.^2 + z.^2);
                    
                    k = 2*pi*a*R < f.alpha;
                    sigma = realsqrt(f.alpha.^2 - (2*pi*a*R(k)).^2);
                    f.U(k) = C * (cosh(sigma)./sigma.^2 - sinh(sigma)./sigma.^3);
                    sigma = realsqrt((2*pi*a*R(~k)).^2 - f.alpha.^2);
                    f.U(~k) = C * (sin(sigma)./sigma.^3 - cos(sigma)./sigma.^2);
                else
                    % separable
                    a = f.J/2;
                    C = 2*a/f.bessi0(f.alpha);
                    
                    k = 2*pi*a*abs(x) < f.alpha;
                    sigma = realsqrt(f.alpha.^2 - (2*pi*a*x(k)).^2);
                    f.U(k) = C * (sinh(sigma)./sigma);
                    sigma = realsqrt((2*pi*a*x(~k)).^2 - f.alpha.^2);
                    f.U(~k) = C * (sin(sigma)./sigma);
                    
                    k = 2*pi*a*abs(y) < f.alpha;
                    sigma = realsqrt(f.alpha.^2 - (2*pi*a*y(k)).^2);
                    f.U(k) = C * (sinh(sigma)./sigma) .* f.U(k);
                    sigma = realsqrt((2*pi*a*y(~k)).^2 - f.alpha.^2);
                    f.U(~k) = C * (sin(sigma)./sigma) .* f.U(~k);
                    
                    k = 2*pi*a*abs(z) < f.alpha;
                    sigma = realsqrt(f.alpha.^2 - (2*pi*a*z(k)).^2);
                    f.U(k) = C * (sinh(sigma)./sigma) .* f.U(k);
                    sigma = realsqrt((2*pi*a*z(~k)).^2 - f.alpha.^2);
                    f.U(~k) = C * (sin(sigma)./sigma) .* f.U(~k);
                end
                
            end
            toc

            % turn into a deconvolution (catch div by zero)
            f.U = 1 ./ hypot(f.U, eps);
            if f.gpu; f.U = gpuArray(f.U); end

            % we are going to do a lot of ffts of the same size so tune it
            fftw('planner','measure');

            % calculate density weighting
            f.d = f.density(ok);

            % display properties
            fprintf(' Created'); disp(f);
            w = whos('f');
            fprintf('\n')
            fprintf('\t H: [%ix%i] (nonzeros %i) %0.1fMbytes\n',size(f.H),nnz(f.H),w.bytes/1e6);
            fprintf('\tHT: [%ix%i] (nonzeros %i) %0.1fMbytes\n',size(f.HT),nnz(f.HT),w.bytes/1e6);
            fprintf('\t U: [%ix%ix%i] min=%f max=%f\n',size(f.U),min(f.U(:)),max(f.U(:)))
            fprintf('\t d: [%ix%i] (zeros %i) min=%f max=%f\n',size(f.d),nnz(~f.d),min(f.d(~~f.d)),max(f.d))
            fprintf('\n')
  
        end
        
        %% utility functions
        
        % sparse matrix vector multiply (keep all the hacks in one place)
        function y = spmv(f,k)
            if f.gpu
                y = single(k);
                y = f.H * y;
            else
                y = double(k);
                y = f.HT' * y;
                y = full(y);
                y = single(y);
            end
        end
        
        % sparse transpose matrix vector multiply (keep all the hacks in one place)
        function y = spmv_t(f,k)
            if f.gpu
                y = single(k);
                y = f.HT * y;
            else
                y = double(k);
                y = f.H' * y;
                y = full(y);
                y = single(y);
            end
        end
        
        % 3d fft with pre-fft padding (cartesian kspace <- cartesian image)
        function k = fft3_pad(f,k)
            for j = 1:3
                pad = (f.K(j) - f.N(j)) / 2;
                if j==1; k = padarray(k,[pad 0 0]); end
                if j==2; k = padarray(k,[0 pad 0]); end
                if j==3; k = padarray(k,[0 0 pad]); end
                k = fft(fftshift(k,j),[],j);
            end
        end

        % 3d ifft with post-ifft cropping (cartesian image <- cartesian kspace)
        function x = ifft3_crop(f,x)
            for j = 1:3
                scale = f.K(j) / f.N(j); % undo ifft scaling and reapply with correct size
                crop = (f.K(j) - f.N(j)) / 2;
                x = ifftshift(ifft(x,[],j),j).*scale;
                if j==1; x = x(1+crop:end-crop,:,:); end
                if j==2; x = x(:,1+crop:end-crop,:); end
                if j==3; x = x(:,:,1+crop:end-crop); end
            end
        end

        % forward non-uniform FT (irregular kspace <- cartesian image)
        function k = fNUFT(f,x)
            % k = A * x
            k = reshape(x,f.N);
            k = k.*f.U;
            k = f.fft3_pad(k);
            k = reshape(k,[],1);
            k = f.spmv(k);
        end
        
        % adjoint non-uniform FT (cartesian image <- irregular kspace)
        function x = aNUFT(f,k)
            % x = A' * k
            x = reshape(k,[],1);
            x = f.spmv_t(x);
            x = reshape(x,f.K);
            x = f.ifft3_crop(x);
            x = x.*f.U;
        end
        
        % image projection operator (image <- image)
        function y = iprojection(f,x,damping,W)
            % y = A' * W * D * W * A * x
            y = f.fNUFT(x);
            y = (W.^2.*f.d).*y; % density weighting included
            y = f.aNUFT(y);
            y = reshape(y,size(x));
            if ~isscalar(damping)
                damping = reshape(damping,size(x));
            end
            y = y + damping.^2.*x;
        end

        % phase constrained projection operator (image <- image)
        function y = pprojection(f,x,damping,phase_constraint,W,P)
            % y = P' * A' * W * D * W * A * P * x + penalty on imag(x)
            P = reshape(P,size(x));
            y = P.*x;
            y = f.iprojection(y,damping,W);
            if ~isscalar(phase_constraint)
                phase_constraint = reshape(phase_constraint,size(x));
            end
            y = conj(P).*y + i.*phase_constraint.^2.*imag(x);
        end

        % replacement for matlab besseli function (from Numerical Recipes in C)
        function ans = bessi0(f,ax)
            ans = zeros(size(ax),'like',ax);
            
            % ax<3.75
            k=ax<3.75;
            y=ax(k)./3.75;
            y=y.^2;
            ans(k)=1.0+y.*(3.5156229+y.*(3.0899424+y.*(1.2067492+...
                   y.*(0.2659732+y.*(0.360768e-1+y.*0.45813e-2)))));
 
            % ax>=3.75
            k=~k;
            y=3.75./ax(k);
            ans(k)=(exp(ax(k))./realsqrt(ax(k))).*(0.39894228+y.*(0.1328592e-1+...
                   y.*(0.225319e-2+y.*(-0.157565e-2+y.*(0.916281e-2+y.*(-0.2057706e-1+...
                   y.*(0.2635537e-1+y.*(-0.1647633e-1+y.*0.392377e-2))))))));
        end
        
        % convolution kernel (no error checking, out of bounds will cause an error)
        function s = convkernel(f,dist2)
            s = f.bessi0(f.alpha*realsqrt(1-dist2/(f.J/2).^2)) / f.bessi0(f.alpha);
            %s = besseli(0,f.alpha*realsqrt(1-dist2/(f.J/2).^2)) / besseli(0,f.alpha);
        end
        
        % use with svds/eigs to calculate singular values of projection operator
        function y = svds_func(f,x,tflag)
            damping = 0; W = 1;
            if f.gpu; x = gpuArray(x); end
            y = f.iprojection(x,damping,W);
            if f.gpu; y = gather(y); end
        end
        
        %% density estimation
        function d = density(f,ok)

            % Pipe's method 
            maxiter = 10;
            fprintf(' Calculating density. '); tic

            % initial estimate (preserve zeros = out of bounds)
            d = reshape(ok,[],1);

            % iterative refinement
            for j = 1:maxiter
                q = f.spmv(f.spmv_t(d));
                d = d ./ hypot(q, eps); % prevent div by zero
            end
            
            % scale so regridding gives similar result to least squares: not working
            if false
                % s = max. sval of A'DA: should be 1 if d is scaled correctly
                opts = struct('issym',1,'isreal',0,'tol',1e-3);
                s = eigs(@f.svds_func, prod(f.N), 1, 'lm', opts);
            else
                % s = norm of D = diag(d): also not correct but fast
                s = max(d);
            end
            d = d ./ s;
            toc

        end
        
        %% inverse non-uniform FT (cartesian image <- irregular kspace)
        function im = iNUFT(f,raw,tol,maxiter,damping,phase_constraint,W)
            
            % raw = complex raw data [npts nc ne] or [nr ny nc ne]
            % maxiter = no. iterations (0=poorly scaled regridding 1=well-scaled regridding)
            % tol = tolerance (tol=0 allows early termination, only applies when maxiter>1)
            % damping = Tikhonov regularization term (only applies when maxiter>1)
            % phase_constraint = phase constraint term (only applies when maxiter>1)
            % W = weighting vector [scalar, ny or nr*ny] (only applies when maxiter>1)

            % no. data points
            nrow = size(f.H,1);
            
            % size checks
            if size(raw,1)==nrow
                nc = size(raw,2);
                nte = size(raw,3);
                fprintf('  %s received raw data: npts=%i nc=%i ne=%i.\n',mfilename,nrow,nc,nte);
            else
                nr = size(raw,1); % assume readout points
                ny = size(raw,2); % assume no. of spokes
                if nr*ny ~= nrow
                    error('raw data leading dimension(s) must be length %i (not %ix%i).',nrow,nr,ny)
                end
                nc = size(raw,3);
                nte = size(raw,4);
                fprintf('  %s received raw data: nr=%i ny=%i nc=%i ne=%i.\n',mfilename,nr,ny,nc,nte);
            end
            raw = reshape(raw,nrow,nc,nte);

            % optional argument checks
            if ~exist('maxiter','var') || isempty(maxiter)
                maxiter = 1;
            else
                validateattributes(maxiter,{'numeric'},{'scalar','finite','integer','nonnegative'},'','maxiter');
            end
            if ~exist('tol','var') || isempty(tol)
                tol = [];
            else
                validateattributes(tol,{'numeric'},{'scalar','finite','nonnegative'},'','tol');
            end
            if ~exist('damping','var') || isempty(damping)
                damping = 0;
            else
                validateattributes(damping,{'numeric'},{'scalar','finite','nonnegative'},'','damping');
            end
            if ~exist('phase_constraint','var') || isempty(phase_constraint)
                phase_constraint = 0;
            else
                validateattributes(phase_constraint,{'numeric'},{'scalar','finite','nonnegative'},'','phase_constraint');
            end
            if ~exist('W','var') || isempty(W)
                W = 1;
            else
                if numel(W)~=nrow
                    if ~exist('ny','var')
                        % guess - expect W to be vector of length ny
                        ny = numel(W);
                        nr = nrow/ny;
                    end
                    % this should catch most size mismatches
                    if mod(nr,1) || nr*ny~=nrow || isscalar(W)
                        error('W must be a vector of length ny or ny*nr.');
                    end
                    W = repmat(reshape(W,1,ny),nr,1);
                end
                W = reshape(W,nrow,1);
                if numel(unique(W))==1; W = W(1); end
                if ~any(W); error('W cannot be all zero.'); end
                validateattributes(W,{'numeric','gpuArray'},{'finite','nonnegative'},'','W');
            end

            % damping, weighting and phase_constraint require iterative recon
            if ~isempty(tol) && maxiter<=1
                error('tol is only active when maxiter>1.');
            end
            if damping~=0 && maxiter<=1
                error('damping is only active when maxiter>1.');
            end
            if ~isscalar(W) && maxiter<=1
                error('weighting is only active when maxiter>1.');
            end
            if phase_constraint~=0 && maxiter<=1
                error('phase constraint is only active when maxiter>1.');
            end
            fprintf('  maxiter=%i tol=%.1e damping=%.3f phase_constraint=%.3f weighted=%i\n',maxiter,tol,damping,phase_constraint,~isscalar(W))
            
 			% experimental method to inhibit noise amplification at edges of image
 			damping = damping * f.U / min(f.U(:));

            %  push to gpu if needed
            if f.gpu
                W = gpuArray(W);
                damping = gpuArray(damping);
                phase_constraint = gpuArray(phase_constraint);
            end

            % array for the final images
            im = zeros([size(f.U) nc nte],'single');

            % reconstruction. note: don't use parfor in these loops, it is REALLY slow
            tic
            for e = 1:nte
                for c = 1:nc

                    if maxiter==0 || phase_constraint
                        
                        % regridding x = (A'Db). hard to scale correctly, prefer pcg with 1 iteration
                        x = f.aNUFT(f.d.*raw(:,c,e));
                        
                    else

                        % least squares (A'W^2DA)(x) = (A'W^2Db)
                        b = f.aNUFT((W.^2.*f.d).*raw(:,c,e));
                        
                        % correct form for solver
                        b = reshape(b,[],1);
                        
                        [x,~,relres,iter] = pcgpc(@(x)f.iprojection(x,damping,W),b,tol,maxiter);
                        %fprintf('  pcg finished at iteration=%i with relres=%.3e\n',iter,relres);
                        
                    end
                   
                    % phase constrained: need to use pcgpc (real dot products) instead of pcg
                    if phase_constraint

					    % use non-constrained estimate for low-resolution phase
                        x = reshape(x,size(f.U));

                        % smooth in image space so voxel size is independent of osf
                        h = hamming(11);
                        P = fftshift(x); % shift center to mitigate edge effects
                        P = convn(P,reshape(h,numel(h),1,1),'same');
                        P = convn(P,reshape(h,1,numel(h),1),'same');
                        P = convn(P,reshape(h,1,1,numel(h)),'same');
                        P = ifftshift(P); % shift center back to origin
                        P = exp(i*angle(P));

                        % RHS vector
                        b = conj(P).*f.aNUFT((W.^2.*f.d).*raw(:,c,e));

                        % correct form for solver
                        P = reshape(P,[],1);
                        b = reshape(b,[],1);

                        % phase constrained (P'A'W^2DAP)(P'x) = (P'A'W^2Db) with penalty on imag(P'x)
                        % (REF: Bydder & Robson, Magnetic Resonance in Medicine 2005;53:1393)
                        [x,~,relres,iter] = pcgpc(@(x)f.pprojection(x,damping,phase_constraint,W,P),b,tol,maxiter);
                        %fprintf('  pcg finished at iteration=%i with relres=%.3e\n',iter,relres);
                        
                        % put back the low resolution phase
                        x = P.*x;

                    end
                    
                    % reshape into image format
                    im(:,:,:,c,e) = reshape(gather(x),size(f.U));

                end
            end
            fprintf('  %s returned %ix%ix%ix%i dataset. ',mfilename,size(im(:,:,:,1)),size(im(1,1,1,:),4)); toc

        end
        
    end
    
end

