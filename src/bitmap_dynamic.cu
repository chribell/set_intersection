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

template <bool selfJoin>
__global__ void intersectPerPair(tile A, tile B, word* bitmaps, unsigned int numOfWords, unsigned int numOfSets,
        const unsigned int* elements, const unsigned int* sizes, const unsigned int* offsets, unsigned int* counts);


int main(int argc, char** argv) {
    try {
        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{2: ^{1}}|\n"
                "└{0:─^{1}}┘\n", "", 51, "Bitmap-dynamic GPU set intersection"
        );


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
        unsigned int partition = 10000;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("input", "Input dataset path", cxxopts::value<std::string>(input))
                ("output", "Output result path", cxxopts::value<std::string>(output))
                ("blocks", "Number of blocks (default: " + std::to_string(blocks) + ")", cxxopts::value<unsigned int>(blocks))
                ("threads", "Threads per block (default: " + std::to_string(blockSize) + ")", cxxopts::value<unsigned int>(blockSize))
                ("warp", "Launch warp based kernel (default: false, runs block based)", cxxopts::value<bool>(warpBased))
                ("partition", "Number of sets to be processed per GPU invocation", cxxopts::value<unsigned int>(partition))
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

        std::vector<tile_pair> runs = findTilePairs(d->cardinality, partition);

        size_t combinations;
        if (runs.size() == 1) {
            combinations = combination(d->cardinality, 2);
        } else {
            combinations = partition * partition;
        }

        unsigned long long outputMemory = runs.size() * combinations * sizeof(unsigned int);

        unsigned int bitmapWords = BITMAP_NWORDS(d->universe);
        size_t bitmapMemory = 0;

        if (singleInstance) {
            bitmapMemory = bitmapWords * sizeof(word);
        } else {
            bitmapMemory = bitmapWords * sizeof(word) * blocks;
        }

        unsigned long long deviceMemory = (sizeof(unsigned int) * d->cardinality * 2)
                                          + (sizeof(unsigned int) * d->totalElements)
                                          + (sizeof(unsigned int) * combinations)
                                          + bitmapMemory;

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "│{9: ^{2}}|{10: ^{2}}│\n"
                "│{11: ^{2}}|{12: ^{2}}│\n"
                "│{13: ^{2}}|{14: ^{2}}│\n"
                "│{15: ^{2}}|{16: ^{2}}│\n"
                "│{17: ^{2}}|{18: ^{2}}│\n"
                "│{19: ^{2}}|{20: ^{2}}│\n"
                "│{21: ^{2}}|{22: ^{2}}│\n"
                "└{23:─^{1}}┘\n", "Launch Info", 51, 25,
                "Word size", WORD_BITS,
                "Bitmap words", bitmapWords,
                "Blocks", blocks,
                "Block Size", blockSize,
                "Warps per Block", blockSize / 32,
                "Level", (warpBased ? "Warp" : "Block"),
                "Partition", partition,
                "GPU invocations", runs.size(),
                "Required Memory (Output)", formatBytes(outputMemory),
                "Required Memory (GPU)", formatBytes(deviceMemory),
                ""
        );

        cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        std::vector<unsigned int> counts(runs.size() == 1 ? combinations : runs.size() * combinations);

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
            errorCheck(cudaMalloc((void**)&deviceCounts, sizeof(unsigned int) * combinations))
            errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combinations))
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
            EventPair *interBitmap = deviceTimer.add("Bitmap intersection");
            singleInstanceIntersect<<<blocks, blockSize>>>
                    (deviceBitmaps, deviceElements + d->sizes[0], d->sizes[1], deviceCounts);
            DeviceTimer::finish(interBitmap);
            counts[0] = thrust::reduce(thrust::device, deviceCounts, deviceCounts + blocks);
        } else {
            EventPair* setOffsets = deviceTimer.add("Compute set offsets");
            thrust::exclusive_scan(thrust::device, deviceOffsets, deviceOffsets + d->cardinality, deviceOffsets,
                                   0); // in-place scan
            DeviceTimer::finish(setOffsets);

            unsigned int iter = 0;
            for (auto& run : runs) {
                tile& A = run.first;
                tile& B = run.second;
                bool selfJoin = A.id == B.id;
                unsigned int numOfSets = selfJoin && runs.size() == 1 ? d->cardinality : partition;

                EventPair* interBitmap = deviceTimer.add("Bitmap intersection");
                if (selfJoin) {
                    intersectPerPair<true><<<blocks, blockSize>>>
                            (A, B, deviceBitmaps, bitmapWords, numOfSets, deviceElements, deviceSizes, deviceOffsets, deviceCounts);
                } else {
                    intersectPerPair<false><<<blocks, blockSize>>>
                            (A, B, deviceBitmaps, bitmapWords, numOfSets, deviceElements, deviceSizes, deviceOffsets, deviceCounts);
                }
                DeviceTimer::finish(interBitmap);

                EventPair* countTransfer = deviceTimer.add("Transfer result");
                errorCheck(cudaMemcpy(&counts[0] + (iter * combinations), deviceCounts, sizeof(unsigned int) * combinations,
                                      cudaMemcpyDeviceToHost))
                DeviceTimer::finish(countTransfer);

                EventPair* clearMemory = deviceTimer.add("Clear memory");
                errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combinations))
                DeviceTimer::finish(clearMemory);
                iter++;
            }
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
            if (runs.size() == 1) {
                writeResult(d->cardinality, counts, output);
            } else {
                writeResult(runs, partition, counts, output);
            }
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


template <bool selfJoin>
__global__ void intersectPerPair(tile A, tile B, word* bitmaps, unsigned int numOfWords, unsigned int numOfSets,
                                 const unsigned int* elements, const unsigned int* sizes, const unsigned int* offsets, unsigned int* counts)
{
    for (unsigned int a = blockIdx.x + A.start; a < A.end; a += gridDim.x) {
        unsigned int aStart = offsets[a];
        unsigned int aEnd = offsets[a + 1];

        // first create bitmap of set a
        for (unsigned int j = aStart + threadIdx.x; j < aEnd; j += blockDim.x) {
            unsigned int element = elements[j]; // extract current element from set A
            atomicOr(bitmaps + (blockIdx.x * numOfWords) + (element / WORD_BITS), 1ULL << (element % WORD_BITS));
        }
        __syncthreads();

        // iterate combinations to calculate the intersections
        for (unsigned int b = (selfJoin ? a + 1 : B.start); b < B.end; b++) {
            unsigned int bStart = offsets[b];
            unsigned int bEnd = bStart + sizes[b];
            for (unsigned int j = bStart + threadIdx.x; j < bEnd; j += blockDim.x) {
                unsigned int element = elements[j]; // extract current element from set B
                if (bitmaps[blockIdx.x * numOfWords + (element / WORD_BITS)] & (1ULL << (element % WORD_BITS))) { // check if bit is set
                    atomicAdd(counts + (selfJoin ?
                        triangular_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets) :
                        quadratic_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets)) , 1);
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


