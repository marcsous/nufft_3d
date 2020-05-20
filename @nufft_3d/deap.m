%% deapodization matrix 
function U = deap(obj)

%% convolution kernel (2x oversampling)

if obj.N(3)==1
    U = zeros(ceil([obj.J obj.J]+1));
else
    U = zeros(ceil([obj.J obj.J obj.J]+1));
end

for ix = 1:size(U,1)
    for iy = 1:size(U,2)
        for iz = 1:size(U,3)
            
            % distance (squared) in 1/2 increments
            dx2 = (ix-1)^2 / 4;
            dy2 = (iy-1)^2 / 4;
            dz2 = (iz-1)^2 / 4;
            dist2 = dx2 + dy2 + dz2;
            
            % kernel values - don't truncate for accuracy
            if obj.radial
                U(ix,iy,iz) = obj.kernel(dist2);
            else
                U(ix,iy,iz) = obj.kernel(dx2).*obj.kernel(dy2).*obj.kernel(dz2);
            end
            
        end
    end
end

%% inverse Fourier transform of kernel (remove 2x oversampling)

for j = 1:2+(obj.N(3)>1)
    U = ifft(U*obj.K(j),2*obj.K(j),j,'symmetric');
    if j==1; U(1+obj.N(1)/2:end-obj.N(1)/2,:,:) = []; end
    if j==2; U(:,1+obj.N(2)/2:end-obj.N(2)/2,:) = []; end
    if j==3; U(:,:,1+obj.N(3)/2:end-obj.N(3)/2) = []; end
end
U = fftshift(U);

%% analytical formule for testing

if 0
    
    % analytical deapodization (kaiser bessel)
    U = zeros(obj.N');
    
    % analytical deapodization (Lewitt, J Opt Soc Am A 1990;7:1834)
    if 0
        % centered: do not use, requires centered fftshifts, no advantage in accuracy
        x = ((1:obj.N(1))-obj.N(1)/2-0.5)./obj.K(1);
        y = ((1:obj.N(2))-obj.N(2)/2-0.5)./obj.K(2);
        z = ((1:obj.N(3))-obj.N(3)/2-0.5)./obj.K(3);
    else
        % not centered: gives essentially the same deapodization matrix as numerical
        x = ((1:obj.N(1))-obj.N(1)/2-1)./obj.K(1);
        y = ((1:obj.N(2))-obj.N(2)/2-1)./obj.K(2);
        z = ((1:obj.N(3))-obj.N(3)/2-1)./obj.K(3);
    end
    if obj.N(3)==1; z=0; end
    [x y z] = ndgrid(x,y,z);
    
    if obj.radial
        % radial
        a = obj.J/2;
        C = 4*pi*a.^3/besseli(0,obj.alpha);
        R = realsqrt(x.^2 + y.^2 + z.^2);
        
        k = 2*pi*a*R < obj.alpha;
        sigma = realsqrt(obj.alpha.^2 - (2*pi*a*R(k)).^2);
        U(k) = C * (cosh(sigma)./sigma.^2 - sinh(sigma)./sigma.^3);
        sigma = realsqrt((2*pi*a*R(~k)).^2 - obj.alpha.^2);
        U(~k) = C * (sin(sigma)./sigma.^3 - cos(sigma)./sigma.^2);
    else
        % separable
        a = obj.J/2;
        C = 2*a/besseli(0,obj.alpha);
        
        k = 2*pi*a*abs(x) < obj.alpha;
        sigma = realsqrt(obj.alpha.^2 - (2*pi*a*x(k)).^2);
        U(k) = C * (sinh(sigma)./sigma);
        sigma = realsqrt((2*pi*a*x(~k)).^2 - obj.alpha.^2);
        U(~k) = C * (sin(sigma)./sigma);

        k = 2*pi*a*abs(y) < obj.alpha;
        sigma = realsqrt(obj.alpha.^2 - (2*pi*a*y(k)).^2);
        U(k) = C * (sinh(sigma)./sigma) .* U(k);
        sigma = realsqrt((2*pi*a*y(~k)).^2 - obj.alpha.^2);
        U(~k) = C * (sin(sigma)./sigma) .* U(~k);
    
        if obj.N(3)>1
            k = 2*pi*a*abs(z) < obj.alpha;
            sigma = realsqrt(obj.alpha.^2 - (2*pi*a*z(k)).^2);
            U(k) = C * (sinh(sigma)./sigma) .* U(k);
            sigma = realsqrt((2*pi*a*z(~k)).^2 - obj.alpha.^2);
            U(~k) = C * (sin(sigma)./sigma) .* U(~k);
        end
    end
  
end

%% convert to a deconvolution

U = 1 ./ U;    

if obj.gpu
    U = gpuArray(single(U));
end
