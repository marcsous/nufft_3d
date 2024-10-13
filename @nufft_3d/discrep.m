%% use discrepancy principle to estimate damping term
function damp = discrep(obj,data,stddev)

% Algorithm:
%  Chambolle A. An Algorithm for Total Variation Minimization and Applications.
%  Journal of Mathematical Imaging and Vision 20, 89–97 (2004)

%% argument checks
nrow = size(obj.H,1);

if size(data,1)==nrow
    nc = size(data,2);
else
    nr = size(data,1); % assume readout points
    ny = size(data,2); % assume radial spokes
    if nr*ny ~= nrow
        error('data leading dimension(s) must be length %i (not %ix%i)',nrow,nr,ny)
    end
    nc = size(data,3);
end
data = reshape(data,nrow,nc,[]);
if size(data,3)>1
    error('data size [%s] seems to have too many dimensions',num2str(size(data)));
end

% if stddev not supplied, make a valiant attempt...
if exist('stddev','var')
    validateattributes(stddev,{'numeric'},{'scalar','finite','nonnegative'},'','stddev');
else
    tmp = nonzeros(data);
    tmp = [real(tmp);imag(tmp)];
    stddev = median(abs(tmp-median(tmp))); clear tmp;
    warning('stddev not supplied. Guessing %.2e.',stddev);
end

%% the discrepancy (norm of noise vector)
discrepancy = stddev * sqrt(numel(data));

% density weighting
W = obj.d;

% rhs vector b = (A'Wb)
b = obj.aNUFT(W.*data);

% correct shape for solver
b = reshape(b,prod(obj.N),[]);

% rough initial estimates
x = [];
damp = 0.1; % norm of QDQ would be useful (~10-100)

fprintf('discrep: estimating damp = %.4e',damp); tic;

for iter = 1:20

    % linear operator (A'WA)
    A = @(x)obj.iprojection(x,damp,W);

    % solve (AWA)x = AWb
    [x,~,~,iters] = minres(A,b,1e-4,[],[],[],x);

    % update parameter
    resnorm = norm(obj.fNUFT(x)-data,'fro');
    damp = damp * (discrepancy / resnorm);

    fprintf('\b\b\b\b\b\b\b\b\b\b%.4e',damp);

end
fprintf('. '); toc