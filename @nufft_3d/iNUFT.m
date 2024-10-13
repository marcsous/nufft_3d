%% inverse non-uniform FT (cartesian image <- irregular kspace)
function im = iNUFT(obj,raw,maxit,damp,W,constraint,lambda)
%im = iNUFT(obj,raw,maxit,damp,W,constraint,lambda)
%
% -raw: complex raw kspace data [nr nc] or [nr ny nc]
% -maxit [scalar]: no. iterations (0 or 1 for regridding)
% -damp [scalar]: Tikhonov L2 regularization parameter
% -W [scalar, ny or nr*ny]: weighting (maxit>1)
%
% Experimental options
% -constraint: 'phase-constraint'
%              'compressed-sensing'
%              'parallel-imaging-sake'
%              'parallel-imaging-pruno'
% -lambda [scalar]: regularization parameter (e.g. 0.5, 0.33, 0, 0.5)
%
% The constraints can often produce slightly better images but require
% tweaking of lambda. They are implemented as mutually exclusive options
% for evaluation but could be used simultaneously at great computational
% cost and probably marginal benefit.
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
raw = reshape(raw,nrow,nc,[]);
if size(raw,3)>1
    error('raw data size [%s] seems to have too many dimensions',num2str(size(raw)));
end

% optional argument checks
if ~exist('maxit','var') || isempty(maxit)
    maxit = 20; % default to iterative
else
    validateattributes(maxit,{'numeric'},{'scalar','finite','integer','nonnegative'},'','maxit');
end
if ~exist('damp','var') || isempty(damp)
    damp = 0;
else
    validateattributes(damp,{'numeric'},{'scalar','finite','nonnegative'},'','damp');
end
if ~exist('W','var') || isscalar(W) || isempty(W)
    W = obj.d;
elseif maxit<=1
    warning('weighting is only effective when maxit>1');
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
    W = reshape(W,nrow,1).*obj.d;
    validateattributes(W,{'numeric','gpuArray'},{'finite','nonnegative'},'','W');
end
if ~exist('constraint','var') || isempty(constraint)
    constraint = '';
elseif maxit<=1
    warning('phase constraint is only effective when maxit>1');
else
    switch constraint
        case 'phase-constraint';
        case 'compressed-sensing';
        case 'parallel-imaging-sake'; if nc==1; error('sake-low-rank requires multiple coils'); end    
        case 'parallel-imaging-pruno'; if nc==1; error('parallel-imaging requires multiple coils'); end            
        otherwise; error('unknown constraint');
    end
    if ~exist('lambda','var') || isempty(lambda)
        error('lambda must be supplied with %s',constraint);
    end
    validateattributes(lambda,{'numeric'},{'scalar','finite','nonnegative'},'','lambda');
end

%% finalize setup

% send to gpu if needed
if obj.gpu==0
    raw = double(gather(raw));
elseif obj.gpu==1
    W = gpuArray(single(W));
    raw = gpuArray(single(raw));
elseif obj.gpu==2
    W = gpuArray(double(W));
    raw = gpuArray(double(raw));   
end

fprintf('  maxit=%i damp=%.2e',maxit,damp);
if isempty(constraint)
    fprintf('\n')
else
    fprintf(' (%s lambda=%.3f)\n',constraint,lambda);
end

% fitting tolerance
tol = 1e-4;

%% iNUFT reconstruction: solve Ax=b
tic;

if maxit==0

    % classic regridding
    x = obj.aNUFT(W.*raw);

else

    % linear operator (A'WA)
    A = @(x)obj.iprojection(x,damp,W);

    % rhs vector b = (A'Wb)
    b = obj.aNUFT(W.*raw);

    % correct shape for solver
    b = reshape(b,prod(obj.N),nc);

    % solve (AWA)x = AWb with penalty on ||x||
    [x,~] = minres(A,b,tol,maxit);

end

%% experimental options

if ~isempty(constraint)
    
    % phase constrained least squares
    if isequal(constraint,'phase-constraint')
        
        % smooth phase (from previous solution)
        P = reshape(x,obj.N);
        P = convn(P,ones(3),'same');
        P = exp(i*angle(P));
        
        % linear operator (P'A'WAP)
        A = @(x)obj.pprojection(x,damp,lambda,W,P);
        
        % rhs vector b = (P'A'Wb)
        b = conj(P).*obj.aNUFT(W.*raw);
        
        % correct shape for solver
        P = reshape(P,prod(obj.N),nc);
        b = reshape(b,prod(obj.N),nc);
        
        % solve (P'A'WAP)(P'x) = (P'A'Wb) with penalty on ||imag(P'x)||
        [x,~] = pcgpc(A,b,tol,maxit);
        
        % put back the low resolution phase
        x = P.*x;
        
    end
    
    % compressed sensing (wavelet)
    if isequal(constraint,'compressed-sensing')
        
        % wavelet operator
        Q = HWT(obj.N);
        
        % rhs vector b = (A'Wb)
        b = obj.aNUFT(W.*raw);
        
        % correct shape for solver
        b = reshape(b,[],1);
        
        % linear operator (A'WA)
        A = @(x)reshape(obj.iprojection(x,damp,W),[],1);
        
        % solve (A'WA)x = (A'Wb) with penalty on ||Qx||_1
        %[x,~] = pcgL1(A,b,lambda,tol,maxit,Q);
        x = fistaN(A,b,lambda,tol,maxit,Q);
       
    end
    
    % parallel imaging (sake low rank)
    if isequal(constraint,'parallel-imaging-sake')
        
        % loraks is either on or off
        x = obj.sake(raw,'damp',damp,'W',W,'loraks',lambda>0);
        
    end
    
    % parallel imaging (pruno low rank)
    if isequal(constraint,'parallel-imaging-pruno')
        
        % use regridding solution for calibration
        x = reshape(x,obj.N(1),obj.N(2),obj.N(3),nc);
        x = fft(fft(fft(x,[],1),[],2),[],3); % kspace
        
        % make the nulling kernels
        obj.fnull = obj.pruno(x) * lambda;
        obj.anull = conj(obj.fnull);
        
        % rhs vector b = (A'Wb)
        b = obj.aNUFT(W.*raw);

        % correct shape for solver
        b = reshape(b,[],1);

        % least squares (A'WA)(x) = (A'Wb) with penalty on ||null*x||
        iters = 100; % need about 100
        [x,~] = minres(@(x)obj.iprojection(x,damp,W),b,tol,iters,M);
        
    end
    
end

%% reshape into image format
im = reshape(gather(x),obj.N(1),obj.N(2),obj.N(3),nc);

fprintf('  %s returned %ix%ix%ix%i dataset. ',mfilename,obj.N(1),obj.N(2),obj.N(3),nc); toc;
