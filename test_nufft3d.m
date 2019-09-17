% test nufft_3d.m
clear all

%% object - 3D Cartesian object
N = 128;
im = phantom3d(N); % 3d shepp logan phantom
im(im==1) = i; % add phase to make it realistic

%% generate koosh ball data
nInterleaves = 21; % Fibonacci number
nRadialSpokes = fix(5000 / nInterleaves) * nInterleaves;

% a density adapted readout (kind of...)
grad = [linspace(0,1,20) ones(1,20) linspace(1,0,N).^1.25];
traj = cumsum(grad); traj = traj * (N/2-1) / max(traj);

z = traj;
y = zeros(size(z));
x = zeros(size(z));

om = zeros(3,numel(traj),nRadialSpokes,'single');

for k = 1:nRadialSpokes
    
    % golden angle (http://blog.wolfram.com/2011/07/28/how-i-made-wine-glasses-from-sunflowers)
    dH = 1 - 2 * (k-1) / (nRadialSpokes-1);
    AzimuthalAngle(k) = acos(dH);
    PolarAngle(k) = max(k-1,0) * pi * (3 - sqrt(5));
    
    % rotation matrix
    RotationMatrix(1,1) = cos(PolarAngle(k))*cos(AzimuthalAngle(k));
    RotationMatrix(1,2) =-sin(PolarAngle(k));
    RotationMatrix(1,3) = cos(PolarAngle(k))*sin(AzimuthalAngle(k));
    RotationMatrix(2,1) = sin(PolarAngle(k))*cos(AzimuthalAngle(k));
    RotationMatrix(2,2) = cos(PolarAngle(k));
    RotationMatrix(2,3) = sin(PolarAngle(k))*sin(AzimuthalAngle(k));
    RotationMatrix(3,1) =-sin(AzimuthalAngle(k));
    RotationMatrix(3,2) = 0.0;
    RotationMatrix(3,3) = cos(AzimuthalAngle(k));
    
    % rotate the readout
    om(:,:,k) = RotationMatrix * [x; y; z];
    
end

% sort into interleaves (works for golden angle)
k = reshape(1:nRadialSpokes,nInterleaves,[])';
om = reshape(om(:,:,k),3,numel(traj),[],nInterleaves);

%% create nufft object
obj = nufft_3d(om,N);

%% generate data (forward transform)
data = obj.fNUFT(im);
noise = complex(randn(size(data)),randn(size(data))) * 10;
data = data+noise;

%% reconstruction (inverse transform)
maxit = 10; % 0 or 1 for gridding, higher values for conjugate gradient
damp = 0; % Tikhonov penalty on ||x||
partial = 0; % Tikhobov penalty on ||imag(x))||
done = obj.iNUFT(data,maxit,damp,partial);

%% display
subplot(1,3,1);
plot3(squeeze(om(1,end,:,1))',squeeze(om(2,end,:,1))',squeeze(om(3,end,:,1))','.');
title('interleave 1'); grid on;
subplot(1,3,2); imagesc(abs(im(:,:,N/2)),[0 1]); colorbar; title('original');
subplot(1,3,3); imagesc(abs(done(:,:,N/2)),[0 1]); colorbar; title('radial');

