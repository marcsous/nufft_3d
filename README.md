MATLAB implementation of Non-Uniform Fast Fourier Transform in 3D based on Jeff Fessler's NUFFT package:

http://web.eecs.umich.edu/~fessler/code/index.html

The version here doesn't have as many options and so is maybe a little easier to understand.

* Uses the sparse matrix version only.
* Uses gpuSparse for single precision on GPU (if available).
* Radial kernel option ([higher accuracy](https://cds.ismrm.org/protected/16MPresentations/abstracts/1763.html) for the same number of convolution coefficients).
* Because sparse matrix multiply and transpose multiply are vastly different in performance, the code stores H and H' separately and uses the faster operation.
