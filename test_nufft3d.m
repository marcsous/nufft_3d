% test nufft_3d.m (works much faster on GPU)
clear all

%% object - 3D Cartesian object
N = 128;
im0 = phantom3d(N); % 3d shepp logan phantom
im0(im0==1) = i; % add phase to make it realistic

%% generate koosh ball data
nInterleaves = 21; % Fibonacci number
nRadialSpokes = fix(5000 / nInterleaves) * nInterleaves;

% a (kind of) density adapted readout 
grad = [linspace(0,1,20) ones(1,20) linspace(1,0,N).^1.25];
traj = cumsum(grad); traj = traj * (N/2-1) / max(traj);

z = traj;
y = zeros(size(z));
x = zeros(size(z));

om = zeros(3,numel(traj),nRadialSpokes,'single');

for k = 1:nRadialSpokes
    
    % golden angle (http://blog.wolfram.com/2011/07/28/how-i-made-wine-glasses-from-sunflowers)
    dH = 1 - 2 * (k-1) / (nRadialSpokes-1);
    PolarAngle(k) = acos(dH);
    AzimuthalAngle(k) = max(k-1,0) * pi * (3 - sqrt(5));
    
    % rotation matrix
    RotationMatrix(1,1) = cos(AzimuthalAngle(k))*cos(PolarAngle(k));
    RotationMatrix(1,2) =-sin(AzimuthalAngle(k));
    RotationMatrix(1,3) = cos(AzimuthalAngle(k))*sin(PolarAngle(k));
    RotationMatrix(2,1) = sin(AzimuthalAngle(k))*cos(PolarAngle(k));
    RotationMatrix(2,2) = cos(AzimuthalAngle(k));
    RotationMatrix(2,3) = sin(AzimuthalAngle(k))*sin(PolarAngle(k));
    RotationMatrix(3,1) =-sin(PolarAngle(k));
    RotationMatrix(3,2) = 0.0;
    RotationMatrix(3,3) = cos(PolarAngle(k));
    
    % rotate the readout
    om(:,:,k) = RotationMatrix * [x; y; z];
    
end

% sort into interleaves (works for golden angle)
k = reshape(1:nRadialSpokes,nInterleaves,[])';
om = reshape(om(:,:,k),3,numel(traj),[],nInterleaves);

%% create nufft object
obj = nufft_3d(om,N);

%% generate data (forward transform)
data = obj.fNUFT(im0);
noise = complex(randn(size(data)),randn(size(data))) * 10;
data = data+noise;

%% reconstruction (inverse transform)
maxit = 10; % 0 or 1 for gridding, higher values for conjugate gradient
weight = []; % data weighting (optional)

damp = 1e-2; % L2 penalty on ||x||
im1 = obj.iNUFT(data,maxit,damp);

partial = 1e2; % L2 penalty on ||imag(x))||
im2 = obj.iNUFT(data,maxit,damp,weight,'phase-constraint',partial);

cs = 1e-2; % L1 penalty in wavelet domain
im3 = obj.iNUFT(data,maxit,damp,weight,'compressed-sensing',cs);

%% display
subplot(2,2,1); imagesc(abs(im0(:,:,N/2+1)),[0 0.5]); colorbar; title('original');
subplot(2,2,2); imagesc(abs(im1(:,:,N/2+1)),[0 0.5]); colorbar; title('least squares');
subplot(2,2,3); imagesc(abs(im2(:,:,N/2+1)),[0 0.5]); colorbar; title('phase constraint');
subplot(2,2,4); imagesc(abs(im3(:,:,N/2+1)),[0 0.5]); colorbar; title('compressed sensing');
