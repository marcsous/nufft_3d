MATLAB implementation of Non-Uniform Fast Fourier Transform in 3D based on Jeff Fessler's NUFFT package:

http://web.eecs.umich.edu/~fessler/code/index.html

The version here doesn't have as many options and so is maybe a little easier to understand.

* To install: put the @nufft_3d folder in the path. See test_nufft_3d for an example. 
* Also need some files from [parallel](https://github.com/marcsous/parallel) on the MATLAB path (e.g. pcgpc, pcgL1, DWT)
* Uses the sparse matrix formulation only.
* Uses [gpuSparse](https://github.com/marcsous/gpuSparse) for single precision on GPU (if available).
* Radial kernel option ([higher accuracy](https://cds.ismrm.org/protected/16MPresentations/abstracts/1763.html) for the same number of convolution coefficients).
* Because sparse matrix multiply and transpose multiply are vastly different in performance, the code stores H and H' separately and uses the faster operation. <i>Since CUDA-11 no longer the case on GPU.</i>
