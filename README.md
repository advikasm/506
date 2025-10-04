# Program 2: CUDA SpMV

## Project Overview
This project implements **Sparse Matrix–Vector multiplication (SpMV)** in the **Coordinate (COO)** format.  
Two implementations are compared:
1. A **sequential CPU baseline** (`spmv.c`)  
2. A **CUDA GPU version** (`spmv-cuda.cu`)  

Both versions are benchmarked on real sparse matrices from the SuiteSparse collection.

---

## COO Format
The COO format stores only the nonzero entries of a sparse matrix using three parallel arrays:
- `rows[i]`: row index of the i-th nonzero  
- `cols[i]`: column index of the i-th nonzero  
- `vals[i]`: value of the i-th nonzero  


## Implementation Details

### CPU (Baseline)
- Loads `.mtx` file and parses into COO arrays  
- Initializes dense vectors `x` (randomized) and `y` (zeroed)  
- Runs warm-up iteration, then dynamically sets number of benchmark iterations  
- Measures runtime, GFLOP/s, and GB/s  

### GPU (CUDA)
- Allocates GPU memory (`cudaMalloc`) for COO arrays and vectors  
- Transfers data to device and zero-initializes output  
- Kernel launch: one CUDA thread per nonzero element  
  ```cuda
  if (i < nnz) atomicAdd(&y[rows[i]], vals[i] * x[cols[i]]);
