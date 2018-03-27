function [x,flag,relres,iter,resvec] = pcgpc(A,b,tol,maxit,M1,M2,x0)
%
% Preconditioned conjugate gradient solver with modifications to
% support the use of a penalty on Im(x), aka phase constraint.
%
% Meant to be used with anonymous functions, A = @(x)myfunc(x),
% where myfunc(x) should return:
% - A*x           : minimize ||b-Ax||
% - A*x+λ*x       : minimize ||b-Ax|| with penalty on ||x||
% - A*x+i*λ*Im(x) : minimize ||b-Ax|| with penalty on ||Im(x)||
%
% References:
% - Partial Fourier Partially Parallel Imaging
%   Mark Bydder and Matthew D. Robson (MRM 2005;53:1393)
% - An Introduction to the CG Method Without the Agonizing Pain
%   Jonathan Richard Shewchuk (1994)
%
% Key modifications:
% - use the real part only of the dot products
% - clear variables to minimize memory usage (GPU)
% - make return args compatible with matlab's pcg
% - (experimental) set tol=0 to terminate early (Bazan, 2014)
%
% Usage: see Matlab's pcg function (except M2 is not supported)

% check arguments
if nargin<2; error('Not enough input arguments.'); end
if ~exist('tol') || isempty(tol); tol = 1e-6; end
if ~exist('maxit') || isempty(maxit); maxit = 20; end
if ~exist('M1') || isempty(M1); M1 = @(arg) arg; end
if exist('M2') && ~isempty(M2); error('M2 argument not supported'); end
if isnumeric(A); A = @(arg) A * arg; end
if isnumeric(M1); M1 = @(arg) M1 \ arg; end
if ~iscolumn(b); error('Right hand side must be a column vector.'); end
validateattributes(tol,{'numeric'},{'scalar','nonnegative','finite'},'','tol');
validateattributes(maxit,{'numeric'},{'scalar','nonnegative','integer'},'','maxit');

% initialize
t = tic;
iter = 1;
flag = 1;
imin = Inf;
if ~exist('x0') || isempty(x0)
    r = b;
    x = zeros(size(b),'like',b);
else
    if ~isequal(size(x0),size(b))
        error('x0 must be a column vector of length %i to match the problem size.',numel(b));
    end
    x = x0;
    r = b - A(x);
end
d = M1(r);
delta0 = norm(b);
delta_new = real(r'*d);
resvec(iter) = norm(r);
solvec(iter) = norm(x);

% main loop
while maxit>0
    
    iter = iter+1;
	clear q; q = A(d);
	alpha = delta_new./real(d'*q);
    
    % unsuccessful termination
    if ~isfinite(alpha); flag = 4; break; end
    
	x = x + alpha.*d;
    if mod(iter,50)==0
        r = b - A(x);
    else
        r = r - alpha.*q;
    end

    % residual and solution vectors
    resvec(iter) = norm(r);
    solvec(iter) = norm(x);

    % retain best solution
    if resvec(iter) < min(resvec(1:iter-1))
        xmin = x;
        imin = iter;
    end
    
    % successful termination
    if resvec(iter)<tol*delta0; flag = 0; break; end

    % early termination (Bazan et al, Linear Algebra Appl. 2014;21:316)
    psi(iter) = resvec(iter).*solvec(iter);
    if tol==0 && iter>3 && psi(iter) > psi(iter-1); flag = 5; break; end

    % unsuccessful termination
    if iter>maxit; flag = 1; break; end

    clear q; q = M1(r);
	delta_old = delta_new;
    delta_new = real(r'*q);
   
    % unsuccessful termination
    if delta_new<=0; flag = 4; break; end

    beta = delta_new./delta_old;
    d = q + beta.*d;

end

% return best solution (based on resvec)
if imin<iter
    flag = 3;
    x = xmin;
    iter = imin;
else
    iter = iter-1;
end

% return arguments
if nargout~=1
    
    % remeasure final residual
    resvec(end) = norm(b-A(x));
    
    % matlab compatiblity
    resvec = reshape(resvec,[],1);
    relres = resvec(end)./delta0;

    % display
    fprintf('%s terminated at iteration %i (flag %i): relres = %e. ',mfilename,iter,flag,relres); toc(t);

end

