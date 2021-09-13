/*
    kernel for matrix multiplication
    using matrix tiling algorithm
    currently does not support floats
*/

__global__ void mulmat(float *a, float *b, float *c, int n, int BLOCK_SIZE){
    extern __shared__ float s[];
    float *s_a = s;
    float *s_b = &s[BLOCK_SIZE * BLOCK_SIZE];

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    float tmp = 0.0f;
    int i, j;

    for(i = 0; i < n; i += BLOCK_SIZE){
        s_a[threadIdx.x * BLOCK_SIZE + threadIdx.y] = a[row * n + (threadIdx.y + i)];
        s_b[threadIdx.x * BLOCK_SIZE + threadIdx.y] = b[(threadIdx.x + i) * n + col];

        __syncthreads();

        for(j = 0; j < BLOCK_SIZE; j++){
            tmp = __fadd_rn(tmp, __fmul_rn(s_a[threadIdx.x * BLOCK_SIZE + j], s_b[j * BLOCK_SIZE + threadIdx.y]));
            // tmp += s_a[threadIdx.x * BLOCK_SIZE + j] * s_b[j * BLOCK_SIZE + threadIdx.y];
        }

        __syncthreads();
    }

    c[row * n + col] = (float)tmp;
}