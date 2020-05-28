%% generate nullspace kernels
function K = pruno(obj,data,varargin)
%
% Returns nulling kernels for 2d/3d data with nc coils.
%
% Data must be [nx ny nz nc] in size even if nz=1 (2d).
%
% The goal is so to return K such that convn(data,K) is
% zero and can be used as a least squares penalty. But
% actually the convolutions are done as multiplies in
% image-space.
%
% Inputs:
%  -data [nx ny nz nc]: kspace data from nc coils
%  -varargin: pairs of options/values (e.g. 'width',5)
%
% Outputs:
%  -K [nx ny nz nc nk]: nulling kernels
%
% References:
% Zhang et al. Parallel reconstruction using null operations
% Magn Reson Med (2011) Nov;66(5):1241 doi:10.1002/mrm.22899

%% setup

% default options
opts.width = 5; % kernel width
opts.radial = 1; % radial kernel
opts.noise = []; % noise std, if available
opts.tol = 0.01; % threshold for singular value filter

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

%% initialize

% argument checks
[nx ny nz nc] = size(data);

if ndims(data)~=4 || ~isfloat(data)
    error('''data'' must be a 4d float array [%s]',num2str(size(data)))
end
if any([nx ny nc]==1) % allow z=1 for 2d
    warning('''data'' has a singleton dim [%i %i %i %i]',nx,ny,nz,nc);
end

% convolution kernel indicies
if nz==1
    [x y] = ndgrid(-fix(opts.width/2):fix(opts.width/2));
    z = zeros(size(x)); % still need the z coordinates
else
    [x y z] = ndgrid(-fix(opts.width/2):fix(opts.width/2));
end
if opts.radial
    k = sqrt(x.^2+y.^2+z.^2)<=opts.width/2;
else
    k = abs(x)<=opts.width/2 & abs(y)<=opts.width/2 & abs(z)<=opts.width/2;
end
nk = nnz(k);
opts.kernel.x = x(k);
opts.kernel.y = y(k);
opts.kernel.z = z(k);
opts.kernel.mask = k;

% dimensions of the data set
opts.dims = [nx ny nz nc nk];

% must be fully sampled
matrix_density = nnz(data) / numel(data);

if matrix_density < 0.99
    error('data must be fully sampled');
end

% display
fprintf('%s:\n',mfilename);
disp(rmfield(opts,'kernel'));

%% make the nulling kernels

% normal calibration matrix
AA = make_data_matrix(data,opts);

% row space and singular values (squared)
[V W] = svd(AA);
W = diag(W);

% estimate noise floor (sigma)
if isempty(opts.noise)
    hi = nnz(W > eps(numel(W)*W(1))); % skip true zeros
    for lo = 1:hi
        h = hist(W(lo:hi),sqrt(hi-lo));
        [~,k] = max(h);
        if k>1; break; end
    end
    sigma = sqrt(median(W(lo:hi)));
    opts.noise = sigma / sqrt(matrix_density*nx*ny);
    fprintf('Noise std estimate: %.2e\n',opts.noise);
end
sigma = opts.noise * sqrt(matrix_density*nx*ny);

% un-square the singular values
W = sqrt(gather(W));

% minimum variance filter
f = max(0,1-sigma^2./W.^2);
nv = nnz(f < 1-opts.tol);

% display
plot(W/W(1)); hold on; plot(f,'--'); hold off; xlim([0 numel(W)]);
line(xlim,[1 1]*sigma/W(1),ylim,'linestyle',':','color','black');
title(sprintf('tol=%.2f (%i nullspace vectors)',opts.tol,nv));
legend({'singular values','filter','noise floor'}); drawnow;

% make nullspace kernels: scale by (1-f)
V = reshape(V,nc,nk,[]);
V = permute(V,[2 1 3]);

K = zeros(numel(opts.kernel.mask),nc,nv,'like',V);
for v = nv:-1:1
    K(opts.kernel.mask,:,v) = (1-f(v)) * V(:,:,v);
end

% make shape compatible
if nz==1
    shape = [size(opts.kernel.mask) 1 nc nv];
else
   shape = [size(opts.kernel.mask)    nc nv];
end
K = reshape(K,shape);

% image space
K = ifft(K,obj.N(1),1) * obj.N(1);
K = ifft(K,obj.N(2),2) * obj.N(2);
K = ifft(K,obj.N(3),3) * obj.N(3);

%% make normal calibration matrix (low memory)
function AA = make_data_matrix(data,opts)

nx = size(data,1);
ny = size(data,2);
nz = size(data,3);
nc = size(data,4);
nk = opts.dims(5);

AA = zeros(nc,nk,nc,nk,'like',data);

for j = 1:nk

    x = opts.kernel.x(j);
    y = opts.kernel.y(j);
    z = opts.kernel.z(j);
    row = circshift(data,[x y z]); % rows of A.'
    
    for k = j:nk
        
        x = opts.kernel.x(k);
        y = opts.kernel.y(k);
        z = opts.kernel.z(k);
        col = circshift(data,[x y z]); % cols of A

        % matrix multiply A'*A
        AA(:,j,:,k) = reshape(row,[],nc)' * reshape(col,[],nc);
 
        % fill conjugate symmetric entries
        AA(:,k,:,j) = squeeze(AA(:,j,:,k))';
        
    end

end
AA = reshape(AA,nc*nk,nc*nk);
