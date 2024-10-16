function ksp = sake(obj,data,varargin)
% ksp = sake(data,varargin)
%
% 3D MRI reconstruction based on matrix completion.
% Low memory version does not form matrix but is slow.
%
% Non-Cartesian version.
%
% Singular value filtering is done based on opts.noise
% which is a key parameter that affects image quality.
%
% Inputs:
%  -data [npts nc]: kspace data points from nc coils
%  -varargin: pairs of options/values (e.g. 'radial',1)
%
% Outputs:
%  -ksp [nx ny nz nc]: 3D kspace data array from nc coils
%
% References:
%  -Haldar JP et al. LORAKS. IEEE Trans Med Imag 2014;33:668
%  -Shin PJ et al. SAKE. Magn Resonance Medicine 2014;72:959
%
%% setup

% default options
opts.width = 4; % kernel width
opts.radial = 1; % use radial kernel
opts.loraks = 0; % phase constraint (loraks)
opts.tol = 1e-6; % tolerance (fraction change in norm)
opts.maxit = 1e3; % maximum no. iterations
opts.noise = []; % noise std, if available
opts.center = []; % center of kspace, if available
opts.W = 1; % data weighting (same size as data)
opts.damp = 0; % tikhonov damping term
opts.sparsity = 0; % sparsity (0.1 = 10% zeros)

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

% sizes
nx = obj.N(1);
ny = obj.N(2);
nz = obj.N(3);
nc = size(data,2);

% argument checks
if size(data,1)~=size(obj.H,1) || nc<2 || ~isfloat(data)
    error('''data'' must be a float array [npts nc].')
end

% convolution kernel indicies
[x y z] = ndgrid(-fix(opts.width/2):fix(opts.width/2));
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
opts.dims = [nx ny nz nc nk 1];
if opts.loraks; opts.dims(6) = 2; end

% assume center of kspace is [0 0 0] (+matlab unit offset)
if isempty(opts.center)
    opts.center(1) = 1;
    opts.center(2) = 1;
    opts.center(3) = 1;
end

% indices for conjugate reflection about center
opts.flip.x = circshift(nx:-1:1,[0 2*opts.center(1)-1]);
opts.flip.y = circshift(ny:-1:1,[0 2*opts.center(2)-1]);
opts.flip.z = circshift(nz:-1:1,[0 2*opts.center(3)-1]);

% set up wavelet transform
if opts.sparsity
    Q = DWT([nx ny nz]);
end

% approximate density of k-space sampling
matrix_density = obj.sd;

% display
disp(rmfield(opts,{'flip','kernel'}));

%% Cadzow algorithm

ksp = zeros(nx,ny,nz,nc,'like',data);

for iter = 1:opts.maxit

    % data consistency
    tmp = ifft3(ksp);
    tmp = iNUFT(obj,obj.fNUFT(tmp)-data,10,opts.damp);
    tmp = fft3(tmp);
    ksp = ksp - tmp;
    
    % threshold in wavelet domain
    if opts.sparsity
        ksp = ifft3(ksp); % to image
        ksp = Q.thresh(ksp,opts.sparsity);
        ksp = fft3(ksp); % to kspace
    end
    
    % normal calibration matrix
    AA = make_data_matrix(ksp,opts);
    
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
    
    % unsquare singular values
    W = sqrt(gather(W));
    
    % minimum variance filter
    f = max(0,1-sigma.^2./W.^2); 
    F = V * diag(f) * V';
    
    % hankel structure (average along anti-diagonals)  
    ksp = undo_data_matrix(F,ksp,opts);

    % check convergence
    norms(1,iter) = norm(W,Inf); % L2 norm    
    norms(2,iter) = norm(W,1); % nuclear norm 
    norms(3,iter) = norm(W,2); % Frobenius norm
    if iter==1
        tol(iter) = opts.tol;
    else
        % tol = fractional change in L2 norm (or Frobenius?)
        tol(iter) = abs(norms(1,iter)-norms(1,iter-1))/norms(1,iter);
    end
    converged = tol(iter) < opts.tol;
    
    % display every few iterations
    if mod(iter,2)==1 || converged
        display(W,f,sigma,ksp,iter,tol,opts,norms);
    end

    % finish when nothing left to do
    if converged; break; end
 
end

if nargout==0; clear; end % avoid dumping to screen

%% make normal calibration matrix (low memory)
function AA = make_data_matrix(data,opts)

nx = size(data,1);
ny = size(data,2);
nz = size(data,3);
nc = size(data,4);
nk = opts.dims(5);

AA = zeros(nc,nk,nc,nk,'like',data);

if opts.loraks
    BB = zeros(nc,nk,nc,nk,'like',data);
end
  
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

        if opts.loraks
            col = conj(col(opts.flip.x,opts.flip.y,opts.flip.z,:));
            BB(:,j,:,k) = reshape(row,[],nc)' * reshape(col,[],nc);
            BB(:,k,:,j) = squeeze(BB(:,j,:,k)).';
        end
        
    end

end
AA = reshape(AA,nc*nk,nc*nk);

if opts.loraks
    BB = reshape(BB,nc*nk,nc*nk);
    AA = [AA BB;conj(BB) conj(AA)];
end

%% undo calibration matrix (low memory)
function ksp = undo_data_matrix(F,data,opts)

nx = opts.dims(1);
ny = opts.dims(2);
nz = opts.dims(3);
nc = opts.dims(4);
nk = opts.dims(5);

if ~opts.loraks
    F = reshape(F,nc,nk,nc,nk);
else
    F = reshape(F,nc,2*nk,nc,2*nk);
end

ksp = zeros(nx,ny,nz,nc,'like',data);

for j = 1:nk

    x = opts.kernel.x(j);
    y = opts.kernel.y(j);
    z = opts.kernel.z(j);
    colA = circshift(data,[x y z]); % cols of A

    if opts.loraks
        colZ = conj(colA(opts.flip.x,opts.flip.y,opts.flip.z,:)); % conj sym cols of A
    end
    
    for k = 1:nk
        
        chunkA = reshape(colA,[],nc) * squeeze(F(:,j,:,k));
        
        if opts.loraks
            chunkA = chunkA + reshape(colZ,[],nc) * squeeze(F(:,j+nk,:,k));
            chunkZ = reshape(colA,[],nc) * squeeze(F(:,j,:,k+nk));
            chunkZ = chunkZ + reshape(colZ,[],nc) * squeeze(F(:,j+nk,:,k+nk));
            chunkZ = reshape(chunkZ,nx,ny,nz,nc);
            chunkZ = conj(chunkZ(opts.flip.x,opts.flip.y,opts.flip.z,:));
            chunkA = reshape(chunkA,nx,ny,nz,nc) + chunkZ;
        else
            chunkA = reshape(chunkA,nx,ny,nz,nc);
        end

        % reorder and sum along rows      
        x = opts.kernel.x(k);
        y = opts.kernel.y(k);
        z = opts.kernel.z(k);
        ksp = ksp + circshift(chunkA,-[x y z]);
    
    end
    
end

% average
if ~opts.loraks
    ksp = ksp / nk;
else
    ksp = ksp / (2*nk);
end

%% show plots of various things
function display(W,f,sigma,ksp,iter,tol,opts,norms)

% plot singular values
subplot(1,4,1); plot(W/W(1)); title(sprintf('rank %i',nnz(f))); 
hold on; plot(f,'--'); hold off; xlim([0 numel(f)+1]);
line(xlim,gather([1 1]*sigma/W(1)),'linestyle',':','color','black');
legend({'singular vals.','sing. val. filter','noise floor'});

% show current kspace (center of kz)
subplot(1,4,2);
tmp = squeeze(log(sum(abs(ksp(:,:,opts.center(3),:)),4)));
imagesc(tmp); xlabel('kx'); ylabel('ky'); title('kspace');

% show current image (center of z)
subplot(1,4,3); slice = ceil(size(ksp,3)/2);
tmp = ifft(ksp,[],3); tmp = squeeze(tmp(:,:,slice,:));
imagesc(sum(abs(ifft2(tmp)),3)); xlabel('x'); ylabel('y');
title(sprintf('iter %i',iter));

% plot change in metrics
norms(1,:)=norms(1,1)./norms(1,:);
norms(2,:)=norms(2,:)./norms(2,1);
norms(3,:)=norms(3,1)./norms(3,:);
subplot(1,4,4);
ax = plotyy(1:iter,norms([1 3],:),1:iter,norms(2,:));
legend('||A||_F^{-1}','||A||_2^{-1}','||A||_*'); axis(ax,'tight');
xlim([0 iter+1]); xlabel('iters'); title(sprintf('tol %.2e',tol(end)));
drawnow;
