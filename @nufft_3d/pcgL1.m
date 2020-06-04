function x = pcgL1(obj, A, b, lambda, varargin)
% x = pcgL1(A, b, lambda, varargin)
%
% Solves the following problem via ADMM:
%
%   minimize (1/2)*||Ax-b||_2^2 + λ*||x||_1,
%
% Meant to be used with anonymous functions, A = @(x)myfunc(x),
% where myfunc(x) should return:
%
% - A*x to minimize ||b-Ax||^2 + λ ||x||_1
%
% Derived from lasso_lsqr.m at
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%
%% default options

opts.tol = 1e-6;
opts.maxit = 100;
opts.rho = 1;
opts.alpha = 1.5;

% varargin handling (must be option/value pairs)
for k = 1:2:numel(varargin)
    if k==numel(varargin) || ~ischar(varargin{k})
        error('''varargin'' must be option/value pairs.');
    end
    if ~isfield(opts,varargin{k})
        warning('''%s'' is not a valid option.',varargin{k});
    end
    opts.(varargin{k}) = varargin{k+1};
end

%% check arguments

if nargin<2
    error('Not enough input arguments.');
end
if ~iscolumn(b)
    error('b argument must be a column vector');
else
    n = numel(b);
end

% not intended for matrix inputs but they are supported
if isnumeric(A)
    A = @(arg) A * arg;
end

% check A is square [n x n]
try
if ~isequal(size(A(b)),[n 1])
    error('A(x) did not return a vector of length %i',n);
end
catch
    error('A(x) failed when passed a vector of length %i',n);
end

%% ADMM solver

x = zeros(n,1,'like',b);
z = zeros(n,1,'like',b);
u = zeros(n,1,'like',b);

for k = 1:opts.maxit

    % x-update with pcg
    Ak = @(x)A(x) + opts.rho*x;
    bk = b + opts.rho*(z-u);
    [x flag relres iters resvec] = pcg(Ak,bk,[],[],[],[],x);

    if flag~=0
        warning('PCG problem (iter=%i flag=%i)',k,flag);
    end
    
    % z-update with relaxation
    zold = z;
    x_hat = opts.alpha*x + (1 - opts.alpha)*zold;
    z = shrinkage(x_hat + u, lambda/opts.rho);

    u = u + (x_hat - z);

    % check convergence
    if norm(x-z) < opts.tol
        break;
    end
    
end

x = z;


%% Helper function

function z = shrinkage(x, kappa)

z = sign(x) .* max( abs(x) - kappa, 0); % complex ok

