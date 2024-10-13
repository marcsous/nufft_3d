%% deapodization matrix 
function U = deap(obj)

%% convolution kernel (with 2x oversampling)
for dx = 0:obj.J
    for dy = 0:obj.J
        for dz = 0:obj.J
            
            % allow for 2D case
            if obj.N(3)==1 && dz~=0; continue; end

            % distance in 1/2 increments, squared
            dx2 = (dx/2)^2;
            dy2 = (dy/2)^2;
            dz2 = (dz/2)^2;
            dist2 = dx2 + dy2 + dz2;
            
            % kernel values - don't truncate for accuracy
            if obj.radial
                U(1+dx,1+dy,1+dz) = obj.kernel(dist2);
            else
                U(1+dx,1+dy,1+dz) = obj.kernel(dx2).*obj.kernel(dy2).*obj.kernel(dz2);
            end
            
        end
    end
end

%% inverse Fourier transform of kernel (remove 2x oversampling)
for j = 1:ndims(U)
    U = ifft(U*sqrt(obj.K(j)),2*obj.K(j),j,'symmetric');
    if j==1; U(1+obj.N(1)/2:end-obj.N(1)/2,:,:) = []; end
    if j==2; U(:,1+obj.N(2)/2:end-obj.N(2)/2,:) = []; end
    if j==3; U(:,:,1+obj.N(3)/2:end-obj.N(3)/2) = []; end
end
U = fftshift(U);

%% analytical formulas for testing (3D only - ifft scaling not correct)
if 0
    
    % analytical deapodization (kaiser bessel)
    U = zeros(obj.N);
    
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
    
        k = 2*pi*a*abs(z) < obj.alpha;
        sigma = realsqrt(obj.alpha.^2 - (2*pi*a*z(k)).^2);
        U(k) = C * (sinh(sigma)./sigma) .* U(k);
        sigma = realsqrt((2*pi*a*z(~k)).^2 - obj.alpha.^2);
        U(~k) = C * (sin(sigma)./sigma) .* U(~k);

    end
  
end

%% convert to a deconvolution
U = 1 ./ U;    

if obj.gpu==1
    U = gpuArray(single(U));
elseif obj.gpu==2
    U = gpuArray(double(U));
end
