%% adjoint non-uniform FT (cartesian image <- irregular kspace)
function x = aNUFT(obj,k)
% x = A' * k
x = reshape(k,size(obj.H,1),[]);
x = obj.spmv_t(x);
x = reshape(x,obj.K(1),obj.K(2),obj.K(3),[]);
x = obj.ifft3_crop(x);
x = x.*obj.U;
