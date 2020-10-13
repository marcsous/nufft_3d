%% kaiser bessel kernel
function s = kernel(obj,dist2)

arg = obj.alpha*sqrt(1-dist2/(obj.J/2).^2);

% MATLAB's builtin doesn't accept gpuArray
%s = besseli(0,arg) / besseli(0,obj.alpha);

s = bessi0(arg) / bessi0(obj.alpha);

s = real(s); % remove trace imag component

%% substitute for matlab besseli function (Numerical Recipes in C)
function ans = bessi0(ax)

ans = zeros(size(ax),'like',ax);

% ax<3.75
k=abs(ax)<3.75;
y=ax(k)./3.75;
y=y.^2;
ans(k)=1.0+y.*(3.5156229+y.*(3.0899424+y.*(1.2067492+...
    y.*(0.2659732+y.*(0.360768e-1+y.*0.45813e-2)))));

% ax>=3.75
k=~k;
y=3.75./ax(k);
ans(k)=(exp(ax(k))./sqrt(ax(k))).*(0.39894228+y.*(0.1328592e-1+...
    y.*(0.225319e-2+y.*(-0.157565e-2+y.*(0.916281e-2+y.*(-0.2057706e-1+...
    y.*(0.2635537e-1+y.*(-0.1647633e-1+y.*0.392377e-2))))))));

% hack to get the same result as MATLAB besseli
k=k & real(ax)==0;
ans(k)=2*ans(k);
