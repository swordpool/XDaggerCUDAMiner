# XDaggerCUDAMiner
A NEW miner for XDAG mining based on Nvidia CUDA Platform, forked from XdaggerGPUMiner which initial support OpenCL only.
The original code based on OpenCL has not been deleted. The CUDA version has only been modified on the original framework, making it possible to use CUDA directly for mining. So this branch can use OpenCL and CUDA to mine at the same time.

#Command line Pamarater

XDaggerGPUMinerCUDA.EXE -C -a kJiGT15WUp64QkfPQZfQhCTZzR6IO7jG -p xdagmine.com:13654 -opencl-platform 2 -opencl-devices 0 -nvidia-fix -cl-local-work 600 -worker A001

The original version uses OpenCL mining by setting the - G parameter on the command line. Updated, this version adds a new parameter - C to start CUDA mining.

For miner using,just downloading https://github.com/swordpool/XDaggerCUDAMiner/blob/master/XDAG-CUDA-OPENCL-Release.rar , that's all.
