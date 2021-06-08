#include <fmt/core.h>
#include "io.hpp"
#include "device_timer.cuh"
#include <climits>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "util.hpp"
#include "helpers.cuh"
#include <cxxopts.hpp>

__global__ void constructBitmap(word* bitmap, const unsigned int* elements, unsigned int n);

__global__ void singleInstanceIntersect(const word* bitmap, const unsigned int* elements,
                                        unsigned int n, unsigned int* counts);

__global__ void intersectPerPair(word* bitmaps, unsigned int numOfWords, unsigned int numOfSets,
        const unsigned int* elements, const unsigned int* sizes, const unsigned int* offsets, unsigned int* counts);


int main(int argc, char** argv) {
    try {

        fmt::print("{}\n", "Bitmap-based dynamic GPU set intersection");

        int multiprocessorCount;
        int maxThreadsPerBlock;

        cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        // arguments
        std::string input;
        std::string output;
        unsigned int blocks = multiprocessorCount * 16;
        unsigned int blockSize = maxThreadsPerBlock / 2;
        bool warpBased = true;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("input", "Input dataset path", cxxopts::value<std::string>(input))
                ("output", "Output result path", cxxopts::value<std::string>(output))
                ("blocks", "Number of blocks (default: " + std::to_string(blocks) + ")", cxxopts::value<unsigned int>(blocks))
                ("threads", "Threads per block (default: " + std::to_string(blockSize) + ")", cxxopts::value<unsigned int>(blockSize))
                ("warp", "Launch warp based kernel (default: false, runs block based)", cxxopts::value<bool>(warpBased))
                ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            fmt::print("{}\n", options.help());
            return 0;
        }

        if (!result.count("input")) {
            fmt::print("{}\n", "No input dataset given! Exiting...");
            return 1;
        }

        Dataset* d = readDataset(input);

        bool singleInstance = d->cardinality == 2;

        fmt::print("|A| = {}\n", d->sizes[0]);
        fmt::print("|B| = {}\n", d->sizes[1]);

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "└{9:─^{1}}┘\n", "Dataset characteristics", 51, 25,
                "Cardinality", d->cardinality,
                "Universe", d->universe,
                "Total elements", d->totalElements, ""
        );

        unsigned int bitmapWords = BITMAP_NWORDS(d->universe);
        size_t bitmapMemory = 0;

        if (singleInstance) {
            bitmapMemory = bitmapWords * sizeof(word);
        } else {
            bitmapMemory = bitmapWords * sizeof(word) * blocks;
        }

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "└{9:─^{1}}┘\n", "Bitmap Info", 51, 25,
                "Word size", WORD_BITS,
                "Bitmap words", bitmapWords,
                "Required Memory (MB)", bitmapMemory / 1000000, ""
        );

        cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        DeviceTimer deviceTimer;

        // allocate device memory space
        word* deviceBitmaps;
        unsigned int* deviceOffsets;
        unsigned int* deviceSizes;
        unsigned int* deviceElements;
        unsigned int* deviceCounts;

        EventPair *devMemAlloc = deviceTimer.add("Device memory allocation");
        errorCheck(cudaMalloc((void**)&deviceBitmaps, bitmapMemory))
        errorCheck(cudaMemset(deviceBitmaps, 0, bitmapMemory))
        errorCheck(cudaMalloc((void**)&deviceOffsets, sizeof(unsigned int) * d->cardinality))
        errorCheck(cudaMalloc((void**)&deviceSizes, sizeof(unsigned int) * d->cardinality))
        errorCheck(cudaMalloc((void**)&deviceElements, sizeof(unsigned int) * d->totalElements))
        if (singleInstance) {
            errorCheck(cudaMalloc((void**)&deviceCounts, sizeof(unsigned int) * blocks))
            errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * blocks))
        } else {
            errorCheck(cudaMalloc((void**)&deviceCounts, sizeof(unsigned int) * combination(d->cardinality, 2)))
            errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combination(d->cardinality, 2)))
        }
        DeviceTimer::finish(devMemAlloc);

        EventPair *dataTransfer = deviceTimer.add("Transfer to device");
        errorCheck(cudaMemcpy(deviceSizes, d->sizes, sizeof(unsigned int) * d->cardinality, cudaMemcpyHostToDevice))
        errorCheck(cudaMemcpy(deviceOffsets, deviceSizes, sizeof(unsigned int) * d->cardinality, cudaMemcpyDeviceToDevice))
        errorCheck(cudaMemcpy(deviceElements, d->elements, sizeof(unsigned int) * d->totalElements, cudaMemcpyHostToDevice))
        DeviceTimer::finish(dataTransfer);

        if (singleInstance) {
            EventPair* bitmapGen = deviceTimer.add("Generate bitmap");
            constructBitmap<<<blocks, blockSize>>>(deviceBitmaps, deviceElements, d->sizes[0]);
            DeviceTimer::finish(bitmapGen);
        } else {
            EventPair* setOffsets = deviceTimer.add("Compute set offsets");
            thrust::exclusive_scan(thrust::device, deviceOffsets, deviceOffsets + d->cardinality, deviceOffsets,
                                   0); // in-place scan
            DeviceTimer::finish(setOffsets);
        }

        EventPair *interBitmap = deviceTimer.add("Bitmap intersection");
        if (singleInstance) {
            singleInstanceIntersect<<<blocks, blockSize>>>
                    (deviceBitmaps, deviceElements + d->sizes[0], d->sizes[1], deviceCounts);
        } else {
            intersectPerPair<<<blocks, blockSize>>>
                    (deviceBitmaps, bitmapWords, d->cardinality, deviceElements, deviceSizes, deviceOffsets, deviceCounts);
        }

        DeviceTimer::finish(interBitmap);

        std::vector<unsigned int> counts(combination(d->cardinality, 2));

        if (singleInstance) {
            counts[0] = thrust::reduce(thrust::device, deviceCounts, deviceCounts + blocks);
        } else {
            EventPair *countTransfer = deviceTimer.add("Transfer result");
            errorCheck(cudaMemcpy(&counts[0], deviceCounts, sizeof(unsigned int) * combination(d->cardinality, 2),
                                  cudaMemcpyDeviceToHost));
            DeviceTimer::finish(countTransfer);
        }

        EventPair *freeDevMem = deviceTimer.add("Free device memory");
        errorCheck(cudaFree(deviceBitmaps))
        errorCheck(cudaFree(deviceOffsets))
        errorCheck(cudaFree(deviceSizes))
        errorCheck(cudaFree(deviceElements))
        errorCheck(cudaFree(deviceCounts))
        DeviceTimer::finish(freeDevMem);

        cudaDeviceSynchronize();

        deviceTimer.print();

        if (!output.empty()) {
            fmt::print("Writing result to file {}\n", output);
            writeResult(d->cardinality, counts, output);
            fmt::print("Finished\n");
        }
    } catch (const cxxopts::OptionException& e) {
        fmt::print("{}\n", e.what());
        return 1;
    }
    return 0;
}

__global__ void constructBitmap(word* bitmap, const unsigned int* elements, unsigned int n) {
    unsigned int elementsPerBlock = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);
    unsigned int start = elementsPerBlock * blockIdx.x;
    if (start >= n) return;
    unsigned int end = start + elementsPerBlock;
    if (end >= n) end = n;
    unsigned int size = end - start;

    elements += start;

    for (unsigned int i = threadIdx.x; i < size; i += blockDim.x) {
        unsigned int element = elements[i];
        atomicOr(bitmap + (element / WORD_BITS), 1ULL << (element % WORD_BITS));
    }
}

__global__ void singleInstanceIntersect(const word* bitmap, const unsigned int* elements, unsigned int n, unsigned int* counts) {
    unsigned int elementsPerBlock = n / gridDim.x + (n % gridDim.x == 0 ? 0 : 1);
    unsigned int start = elementsPerBlock * blockIdx.x;
    if (start >= n) return;
    unsigned int end = start + elementsPerBlock;
    if (end >= n) end = n;
    unsigned int size = end - start;

    elements += start;

    unsigned int count = 0;

    for (unsigned int i = threadIdx.x; i < size; i += blockDim.x) {
        unsigned int element = elements[i];
        if (bitmap[(element / WORD_BITS)] & (1ULL << (element % WORD_BITS))) {
            count++;
        }
    }

    atomicAdd(counts + blockIdx.x, count);
}



__global__ void intersectPerPair(word* bitmaps, unsigned int numOfWords, unsigned int numOfSets,
                                 const unsigned int* elements, const unsigned int* sizes, const unsigned int* offsets, unsigned int* counts)
{
    for (unsigned int a = blockIdx.x; a < numOfSets - 1; a += gridDim.x) {
        unsigned int aStart = offsets[a];
        unsigned int aEnd = offsets[a + 1];

        // first create bitmap of set a
        for (unsigned int j = aStart + threadIdx.x; j < aEnd; j += blockDim.x) {
            unsigned int element = elements[j]; // extract current element from set A
            atomicOr(bitmaps + (blockIdx.x * numOfWords) + (element / WORD_BITS), 1ULL << (element % WORD_BITS));
        }
        __syncthreads();

        // iterate combinations to calculate the intersections
        for (unsigned int b = a + 1; b < numOfSets; b++) {
            unsigned int bStart = offsets[b];
            unsigned int bEnd = bStart + sizes[b];
            for (unsigned int j = bStart + threadIdx.x; j < bEnd; j += blockDim.x) {
                unsigned int element = elements[j]; // extract current element from set B
                if (bitmaps[blockIdx.x * numOfWords + (element / WORD_BITS)] & (1ULL << (element % WORD_BITS))) { // check if bit is set
                    atomicAdd(counts + triangular_idx(numOfSets, a, b), 1);
                }
            }

        }

        __syncthreads();

        // reset bitmap to ensure correctness
        for (unsigned int j = aStart + threadIdx.x; j < aEnd; j += blockDim.x) {
            unsigned int element = elements[j];
            bitmaps[(blockIdx.x * numOfWords) + (element / WORD_BITS)] = 0;
        }

        __syncthreads();
    }
}


