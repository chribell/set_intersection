#ifndef HELPERS_CUH
#define HELPERS_CUH

#pragma once

__forceinline__ __device__  int myMax(int a, int b)
{
    return (a < b) ? b : a;
}

__forceinline__ __device__  int myMin(int a, int b)
{
    return (a > b) ? b : a;
}

__forceinline__ __device__ unsigned int triangular_idx(unsigned int n, unsigned int i, unsigned int j) {
    return (n * (n - 1) / 2) - (n - i)*((n-i)-1)/2 + j - i - 1;
}

__forceinline__ __device__ unsigned int quadratic_idx(unsigned int n, unsigned int i, unsigned int j) {
    return i * n + j;
}

__device__ inline unsigned int serialIntersect(unsigned int* a, unsigned int aBegin, unsigned int aEnd,
                                                unsigned int* b, unsigned int bBegin, unsigned int bEnd, unsigned int vt)
{
    unsigned int count = 0;

    // vt parameter must be odd integer
    for (int i = 0; i < vt; i++)
    {
        bool p = false;
        if ( aBegin >= aEnd ) p = false; // a, out of bounds
        else if ( bBegin >= bEnd ) p = true; //b, out of bounds
        else {
            if (a[aBegin] < b[bBegin]) p = true;
            if (a[aBegin] == b[bBegin]) {
                count++;
            }
        }
        if(p) aBegin++;
        else bBegin++;

    }
    return count;
}

__forceinline__ __device__  int binarySearch(unsigned int* a, unsigned int aSize,
                                           unsigned int* b, unsigned int bSize, unsigned int diag)
{
    int begin = myMax(0, diag -  bSize);
    int end   = myMin(diag, aSize);

    while (begin < end) {
        int mid = (begin + end) / 2;

        if (a[mid] < b[diag - 1 - mid]) begin = mid + 1;
        else end = mid;
    }
    return begin;
}

// Taken from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define errorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Adapted from https://github.com/ogreen/MergePathGPU
template <bool selfJoin>
__global__ void findDiagonals(tile A, tile B, unsigned int numOfSets, unsigned int *sets,
                              const unsigned int *sizes, unsigned int *offsets,
                              unsigned int *globalDiagonals, unsigned int* counts) {

    for (unsigned int a = A.start; a < A.end; a++) {
        for (unsigned int b = (selfJoin ? a + 1 : B.start); b < B.end; b++) { // iterate every combination
            unsigned int offset = selfJoin ?
                    triangular_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets) :
                    quadratic_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets);
            unsigned int aSize = sizes[a];
            unsigned int bSize = sizes[b];
            unsigned int *aSet = sets + offsets[a];
            unsigned int *bSet = sets + offsets[b];
            unsigned int *diagonals = globalDiagonals + (2 * (gridDim.x + 1)) * offset;

            unsigned int combinedIndex =
                    (uint64_t) blockIdx.x * ((uint64_t) sizes[a] + (uint64_t) sizes[b]) / (uint64_t) gridDim.x;
            __shared__ int xTop, yTop, xBottom, yBottom, found;
            __shared__ unsigned int oneOrZero[32]; // array size must be equal to number of block threads
            __shared__ unsigned int increment; // use this as flag to ensure single increment, find a more elegant way

            increment = 0;

            unsigned int threadOffset = threadIdx.x - 16;

            xTop = myMin(combinedIndex, aSize);
            yTop = combinedIndex > aSize ? combinedIndex - aSize : 0;
            xBottom = yTop;
            yBottom = xTop;

            found = 0;

            // Search the diagonal
            while (!found) {
                // Update our coordinates within the 32-wide section of the diagonal
                int currentX = xTop - ((xTop - xBottom) >> 1) - threadOffset;
                int currentY = yTop + ((yBottom - yTop) >> 1) + threadOffset;

                // Are we a '1' or '0' with respect to A[x] <= B[x]
                if (currentX >= aSize || currentY < 0) {
                    oneOrZero[threadIdx.x] = 0;
                } else if (currentY >= bSize || currentX < 1) {
                    oneOrZero[threadIdx.x] = 1;
                } else {
                    oneOrZero[threadIdx.x] = (aSet[currentX - 1] <= bSet[currentY]) ? 1 : 0;
                    if (aSet[currentX - 1] == bSet[currentY] && increment == 0) { // count augmentation
                        atomicAdd(counts + offset,  1);
                        atomicAdd(&increment, 1);
                    }
                }

                __syncthreads();

                // If we find the meeting of the '1's and '0's, we found the
                // intersection of the path and diagonal
                if (threadIdx.x > 0 && (oneOrZero[threadIdx.x] != oneOrZero[threadIdx.x - 1]) && currentY >= 0 && currentX >= 0 ) {
                    found = 1;
                    diagonals[blockIdx.x] = currentX;
                    diagonals[blockIdx.x + gridDim.x + 1] = currentY;
                }

                __syncthreads();

                // Adjust the search window on the diagonal
                if (threadIdx.x == 16) {
                    if (oneOrZero[31] != 0) {
                        xBottom = currentX;
                        yBottom = currentY;
                    } else {
                        xTop = currentX;
                        yTop = currentY;
                    }
                }
                __syncthreads();
            }

            // Set the boundary diagonals (through 0,0 and A_length,B_length)
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                diagonals[0] = 0;
                diagonals[gridDim.x + 1] = 0;
                diagonals[gridDim.x] = aSize;
                diagonals[gridDim.x + gridDim.x + 1] = bSize;
            }

            oneOrZero[threadIdx.x] = 0;

            __syncthreads();

        }
    }
}

typedef unsigned long long word;

#define WORD_BITS (sizeof(word) * CHAR_BIT)
#define BITMAP_NWORDS(_n) (((_n) + WORD_BITS - 1) / WORD_BITS)

#endif //HELPERS_CUH
