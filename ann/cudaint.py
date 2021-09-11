# cuda interface for the neural network
# module for linear algebraic calculations

import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
from pathlib import Path
import time

def matmul():
    X = 512
    Y = 512
    Z = 512
    a = np.random.randn(X, Y).astype(np.int32)
    b = np.random.randn(Y, Z).astype(np.int32)
    c = np.zeros((X, Z), dtype=np.int32)

    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)
    
    kernel_code = Path('cuda/matmul.cu').read_text()
    module = SourceModule(kernel_code)

    t1 = time.time()
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    block_size = 16
    grid = (X//block_size, Z//block_size)
    block = (block_size, block_size, 1)

    kernel = module.get_function("mulmat")
    kernel.prepare([np.intp, np.intp, np.intp, np.int32, np.int32])
    kernel.prepared_call(grid, block, a_gpu, b_gpu, c_gpu, 512, block_size, shared_size = 2 * block_size * block_size * 8)

    cuda.memcpy_dtoh(c, c_gpu)
    t2 = time.time()

    c_cpu = np.dot(a, b)
    t3 = time.time()

    print("gpu calculation time with PCI overhead: ", 1000 * (t2 - t1), " milliseconds")
    print("cpu calculation using numpy: ",  1000 * (t3 - t2), " milliseconds")

    return c, c_cpu
