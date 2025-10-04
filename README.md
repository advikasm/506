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
### CUDA Vs Base Implementation
- Same math, new schedule. We keep the exact arithmetic (y[row] += val * x[col]) but assign one CUDA thread per nonzero. That exposes massive parallelism versus the CPU’s single loop.
- Correctness under concurrency. Because many nonzeros share the same destination row, we use atomicAdd(&y[row], val*x[col]) to prevent lost updates—on CPU, the loop’s serial order avoids races; on GPU, atomics make concurrent accumulations correct.
- Launch configuration. We use a grid of (nnz + 255)/256 blocks with 256 threads each, which balances load and achieves good GPU occupancy.
- Device memory + transfers. We allocate device buffers for rows/cols/vals/x/y, copy rows/cols/vals/x from host to device, zero y on device, and copy results back to host if we want to compare.
- Timing symmetry. We mirror the CPU flow: a warm-up kernel timed with CUDA events, then compute num_iterations from the warm-up, then time the main kernel loop and report ms/iter, GFLOP/s, GB/s.
- The CUDA program (spmv-cuda.cu) adapts the same COO idea but offloads the computation to the GPU.

### Core Modifications
- Memory Allocation:
cudaMalloc used for all COO arrays and vectors.
- Data Movement:
Host–device copies with cudaMemcpy; result fetched back after kernel execution.
- Parallel Kernel:
Execution Setup:
- Threads per block: 256
Blocks: (nnz + 255)/256## Figures
- Atomic Operations:
Needed since multiple threads may update the same row in y.



<img width="1979" height="1180" alt="output (4)" src="https://github.com/user-attachments/assets/481dc157-3aa8-447c-88d4-a3d6509ceb2a" />


<img width="1979" height="1180" alt="output (5)" src="https://github.com/user-attachments/assets/fb4bb69f-21ca-4f70-8d85-b4fd6fd4b56b" />

### Performance Explanation
The performance results show a consistent and significant speedup for the CUDA version of SpMV compared to the sequential CPU version across all tested matrices. The reasons are:
- Massive parallelism.
In COO SpMV, each nonzero entry contributes one multiply-add. On CPU, these are executed one after another. On GPU, we launch thousands of threads so that each nonzero can be processed in parallel, leading to orders-of-magnitude reductions in runtime.
- Data-parallel suitability.
Sparse matrix–vector multiply is naturally data-parallel because each nonzero operation is independent except for accumulation into the same row. CUDA atomics handle this safely and efficiently in hardware, especially when row contention is low.
- Matrix size sensitivity.
For small matrices (D6-6, bfly, dictionary28), the absolute runtimes are very small for both CPU and GPU. Here, speedups are less dramatic in absolute time, but still measurable.
For large matrices (Ga3As3H12, pkustk14, roadNet-CA), the GPU parallelism really shows: speedups reach 8×–40×, as GPU threads can hide memory latency and keep many cores busy.
Memory bandwidth utilization.
SpMV is memory-bound: performance depends heavily on how fast values and indices can be read. GPUs offer much higher memory bandwidth than CPUs, which is why even with atomic operations, the CUDA kernel sustains far higher throughput (GB/s).
- Irregularity overhead.
The only limitation is row contention. For matrices with rows that have many nonzeros (e.g., roadNet-CA), atomics can serialize some updates, but even then the GPU is substantially faster than the CPU.







## Advantages
- Thousands of threads allow nearly all nonzeros to be processed in parallel.
- GPU memory bandwidth (~900 GB/s) far exceeds CPU bandwidth.
- SIMT model makes this fine-grained parallelism practical
