% test nufft_3d.m (works much faster on GPU)
clear all

%% object - 3d shepp logan phantom
N = 160;
im = phantom3d(N);
im(im==1) = i; % add phase

%% generate koosh ball data
if 1

    % radial density adapted readout
    nRadialSpokes = 5555;
    
    grad = [linspace(0,1,10) ones(1,10) linspace(1,0,N).^1.25];
    traj = cumsum(grad); traj = traj * (N/2-1) / max(traj);

    z = traj;
    y = zeros(size(z));
    x = zeros(size(z));

    om = zeros(3,numel(traj),nRadialSpokes,'single');

    for k = 1:nRadialSpokes

        % golden angle (http://blog.wolfram.com/2011/07/28/how-i-made-wine-glasses-from-sunflowers)
        dH = 1 - 2 * (k-1) / (nRadialSpokes-1);
        PolarAngle(k) = acos(dH);
        AzimuthalAngle(k) = (k-1) * pi * (3 - sqrt(5));

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
else

    % Cartesian
    im = im(:,:,81);
    [kx ky ] = ndgrid(-80:79,-80:79);
    om = [kx(:) ky(:) 0*ky(:)]';

end

%% create nufft object (gpu=0 CPU, gpu=1 gpuSparse, gpu=2 gpuArray)
obj = nufft_3d(om,N,'gpu',1);

%% generate data (forward transform)
data = obj.fNUFT(im);
randn('state',0);
noise = 2e-2 * complex(randn(size(data)),randn(size(data)));
data = data + noise;

%% reconstruction (inverse transform)
maxit = 20; % use 1 for gridding, higher values for conjugate gradient
weight = []; % data weighting (optional)
damp = obj.discrep(data,std(noise)); % estimate damping term

% regridding
im0 = obj.iNUFT(data,1);

% L2 penalty on ||x||
im1 = obj.iNUFT(data,maxit,damp);

% L2 penalty on ||imag(x))||
partial = 0.5; 
im2 = obj.iNUFT(data,maxit,damp,weight,'phase-constraint',partial);

% L1 penalty on ||Q(x)|| (Q=wavelet transform)
sparsity = 0.5; 
im3 = obj.iNUFT(data,maxit,damp,weight,'compressed-sensing',sparsity);

%% display
mid = floor(size(im,3)/2)+1;
subplot(2,2,1); imagesc(abs(im0(:,:,mid)),[0 0.5]); colorbar; title('regridding');
subplot(2,2,2); imagesc(abs(im1(:,:,mid)),[0 0.5]); colorbar; title('least squares');
subplot(2,2,3); imagesc(abs(im2(:,:,mid)),[0 0.5]); colorbar; title('phase constrained');
subplot(2,2,4); imagesc(abs(im3(:,:,mid)),[0 0.5]); colorbar; title('compressed sensing');
