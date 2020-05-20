%% inverse non-uniform FT (cartesian image <- irregular kspace)
function im = iNUFT(obj,raw,maxit,damp,lambda,pruno,W)
%im = iNUFT(obj,raw,maxit,damp,lambda,W)
%
% raw = complex raw kspace data [nr nc] or [nr ny nc]
% maxit [scalar] = no. iterations (use 0 or 1 for regridding)
% damp [scalar] = Tikhonov regularization (only when maxit>1)
% lambda [scalar] = phase constraint (only when maxit>1)
% pruno [scalar] = use pruno parallel imaging (1=yes 0=no)
% W [scalar, ny or nr*ny] = weighting (only when maxit>1)
%
%% argument checks

% expected no. data points
nrow = size(obj.H,1);

if size(raw,1)==nrow
    nc = size(raw,2);
    fprintf('  %s received raw data: nr=%i nc=%i\n',mfilename,nrow,nc);
else
    nr = size(raw,1); % assume readout points
    ny = size(raw,2); % assume radial spokes
    if nr*ny ~= nrow
        error('raw data leading dimension(s) must be length %i (not %ix%i)',nrow,nr,ny)
    end
    nc = size(raw,3);
    fprintf('  %s received raw data: nr=%i ny=%i nc=%i\n',mfilename,nr,ny,nc);
end
raw = reshape(raw,nrow,nc);

% optional argument checks
if ~exist('maxit','var') || isempty(maxit)
    maxit = 0; % default to regridding
else
    validateattributes(maxit,{'numeric'},{'scalar','finite','integer','nonnegative'},'','maxit');
end
if ~exist('damp','var') || isempty(damp)
    damp = 0;
else
    validateattributes(damp,{'numeric'},{'scalar','finite','nonnegative'},'','damp');
end
if ~exist('lambda','var') || isempty(lambda)
    lambda = 0;
else
    validateattributes(lambda,{'numeric'},{'scalar','finite','nonnegative'},'','lambda');
end
if ~exist('pruno','var') || isempty(pruno)
    pruno = 0;
else
    if nc==1; error('pruno option requires multiple coils'); end
    validateattributes(pruno,{'numeric'},{'scalar','finite','binary'},'','pruno');
end
if ~exist('W','var') || isscalar(W) || isempty(W)
    W = 1;
else
    if numel(W)~=nrow
        if ~exist('ny','var')
            % guess - expect W to be vector of length ny
            ny = numel(W);
            nr = nrow/ny;
        end
        % this should catch size mismatches
        if mod(nr,1) || nr*ny~=nrow
            error('W must be a vector of length ny or ny*nr');
        end
        W = repmat(reshape(W,1,ny),nr,1);
    end
    W = reshape(W,nrow,1);
    if numel(unique(W))==1; W = W(1); end
    if ~any(W); error('W cannot be all zero'); end
    validateattributes(W,{'numeric','gpuArray'},{'finite','nonnegative'},'','W');
end

% damp, weighting and lambda require iterative recon
if damp~=0 && maxit<=1
    warning('damp is only effective when maxit>1 - try 10');
end
if ~isscalar(W) && maxit<=1
    warning('weighting is only effective when maxit>1 - try 10');
end
if lambda~=0 && maxit<=1
    warning('phase constraint is only effective when maxit>1 - try 10');
end

%% finalize setup
fprintf('  maxit=%i damp=%.1e',maxit,damp);
fprintf(' lambda=%.1e weighted=%i\n',lambda,~isscalar(W));

% send to gpu if needed
if obj.gpu
    W = gpuArray(W);
    damp = gpuArray(damp);
    lambda = gpuArray(lambda);

    raw = single(raw);
    im = zeros(obj.N(1),obj.N(2),obj.N(3),nc,'single');
else
    raw = double(raw);
    im = zeros(obj.N(1),obj.N(2),obj.N(3),nc,'double');
end

%% iNUFT reconstruction: solve Ax=b
tic;

if maxit==0
    
    % regridding
    x = obj.aNUFT(obj.d.*raw);
    
else
    
    % rhs vector b = (A'WDb)
    b = obj.aNUFT((W.*obj.d).*raw);

    % correct shape for solver
    b = reshape(b,prod(obj.N),nc);
    
    % least squares (A'WDA)(x) = (A'WDb) + penalty on ||x||
    [x,~,relres,iter] = pcgpc(@(x)obj.iprojection(x,damp,W),b,[],maxit);
    
end

% pruno (slow!)
if pruno
    
    % use regridding solution for calibration
    x = reshape(x,obj.N(1),obj.N(2),obj.N(3),nc);
    x = fft(fft(fft(x,[],1),[],2),[],3); % kspace
    
    % make the nulling kernels
    obj.fnull = obj.pruno(x,'tol',0);
    obj.anull = conj(obj.fnull);

    % correct RHS for solver 
    b = obj.aNUFT((W.*obj.d).*raw);
    b = reshape(b,[],1);

    % in Cartesian the diagonal of iprojection is the mean of
    % the sample weighting squared (used as a preconditioner)
    D = dwsd + damp^2 + real(sum(obj.anull.*obj.fnull,5));
    M = @(x) x./reshape(D,size(x)); % diagonal preconditioner -
    
    % check: measure diagonal of iprojection (V. SLOW)
    if 1
        N = 200; % how many to test
        d = zeros(1,N);
        for j = 1:N
            tmp = zeros(size(D));
            tmp(j) = 1; tmp = obj.iprojection(tmp,damp,W);
            d(j) = real(tmp(j)); fprintf('%i/%i\n',j,N);
        end
        plot([d;D(1:N);d-D(1:N)]'); legend({'exact','estimate','diff'});
        keyboard
    end
    
    
    
    
    % least squares (A'WA)(x) = (A'Wb) + penalty on ||null*x||   
    iters = 100; % need about 100
    [x,~,~,~,resvec] = pcgpc(@(x)obj.iprojection(x,damp,W),b,[],iters,M);

end

% phase constrained
if lambda
    
    % smoothing kernel (in image space)
    h = exp(-(-obj.low:obj.low).^2/obj.low);
    
    % use regridding solution for phase
    P = reshape(x,obj.N(1),obj.N(2),obj.N(3),nc);
    P = circshift(P,fix(obj.N/2)); % mitigate edge effects (or use cconvn)
    P = convn(P,reshape(h,numel(h),1,1),'same');
    P = convn(P,reshape(h,1,numel(h),1),'same');
    P = convn(P,reshape(h,1,1,numel(h)),'same');
    P = circshift(P,-fix(obj.N/2)); % shift back again
    P = exp(i*angle(P));
    
    % rhs vector b = (P'A'WDb)
    b = conj(P).*obj.aNUFT((W.*obj.d).*raw);
    
    % correct shape for solver
    P = reshape(P,prod(obj.N),nc);
    b = reshape(b,prod(obj.N),nc);
    
    % phase constrained (P'A'WDAP)(P'x) = (P'A'WDb) + penalty on imag(P'x)
    % (REF: Bydder & Robson, Magnetic Resonance in Medicine 2005;53:1393)
    [x,~,relres,iter] = pcgpc(@(x)obj.pprojection(x,damp,lambda,W,P),b,[],maxit);
    
    % put back the low resolution phase
    x = P.*x;
    
end

% reshape into image format
im = reshape(gather(x),obj.N(1),obj.N(2),obj.N(3),nc);

fprintf('  %s returned %ix%ix%ix%i dataset. ',mfilename,obj.N(1),obj.N(2),obj.N(3),nc); toc;