%% forward non-uniform FT (irregular kspace <- cartesian image)
function k = fNUFT(obj,x)
% k = A * x
k = reshape(x,obj.N(1),obj.N(2),obj.N(3),[]);
k = k.*obj.U;
k = obj.fft3_pad(k);
k = reshape(k,size(obj.H,2),[]);
k = obj.spmv(k);
