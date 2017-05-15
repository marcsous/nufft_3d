MATLAB implementation of Non-Uniform Fast Fourier Transform in 3D based on Jeff Fessler's NUFFT package:

http://web.eecs.umich.edu/~fessler/code/index.html

The version here doesn't have as many options and so is maybe a little easier to understand.

* It uses the sparse matrix version only.
* It uses gpuSparse for single precision on GPU (if available).
* Radial kernel option (higher accuracy for the same number of convolution coefficients).
* As sparse matrix multiply and transpose multiply are vastly difference in performance, this code stores H and H' separately. Awaiting testing on CUDA8 to see if cusparseCcsrmv_mp can even up the difference.
