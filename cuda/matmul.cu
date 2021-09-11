/*
    kernel for matrix multiplication
    using matrix tiling algorithm
    currently does not support floats
*/

__global__ void mulmat(int *a, int *b, int *c, int n, int BLOCK_SIZE){
    extern __shared__ int s[];
    int *s_a = s;
    int *s_b = &s[BLOCK_SIZE * BLOCK_SIZE];

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    int tmp = 0;
    int i, j;

    for(i = 0; i < n; i += BLOCK_SIZE){
        s_a[threadIdx.x * BLOCK_SIZE + threadIdx.y] = a[row * n + (threadIdx.y + i)];
        s_b[threadIdx.x * BLOCK_SIZE + threadIdx.y] = b[(threadIdx.x + i) * n + col];

        __syncthreads();

        for(j = 0; j < BLOCK_SIZE; j++){
            tmp += s_a[threadIdx.x * BLOCK_SIZE + j] * s_b[j * BLOCK_SIZE + threadIdx.y];
        }

        __syncthreads();
    }

    c[row * n + col] = tmp;
}