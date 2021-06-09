#include <fmt/core.h>
#include <iterator>
#include "io.hpp"
#include "device_timer.cuh"
#include "util.hpp"
#include <cxxopts.hpp>
#include <thrust/scan.h>
#include "helpers.cuh"

// Adapted from https://github.com/concept-inversion/H-INDEX_Triangle_Counting
__device__
int linearSearch(unsigned int neighbor, const unsigned int* partition1, const unsigned int* binCounts, unsigned int bin,
        unsigned int BIN_OFFSET, unsigned int BIN_START, unsigned int BUCKETS)
{
    unsigned int len = binCounts[bin + BIN_OFFSET];
    unsigned int i = bin + BIN_START;
    unsigned int step = 0;
    while(step < len)
    {
        unsigned int test = partition1[i];
        if(test==neighbor)
        {
            return 1;
        }
        else
        {
            i+=BUCKETS;
        }
        step += 1;
    }
    return 0;
}

template <bool selfJoin>
__global__ void warpBasedHI(tile A, tile B, unsigned int numOfSets, const unsigned int* elements, const unsigned int* sizes,
        const unsigned int* offsets, unsigned int* counts, unsigned int* bins, unsigned int numOfBuckets, unsigned int bucketSize);

template <bool selfJoin, bool split>
__global__ void blockBasedHI(tile A, tile B, unsigned int numOfSets, const unsigned int* elements, const unsigned int* sizes,
        const unsigned int* offsets, unsigned int* counts, unsigned int* bins, unsigned int numOfBuckets, unsigned int bucketSize);

template <bool selfJoin>
__global__ void intersectPathHI(tile A, tile B, unsigned int numOfSets, const unsigned int* elements, const unsigned int* offsets,
        unsigned int* counts, unsigned int* bins, unsigned int numOfBuckets,
        unsigned int bucketSize, unsigned int* globalDiagonals);

int main(int argc, char** argv) {
    try {
        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{2: ^{1}}|\n"
                "└{0:─^{1}}┘\n", "", 51, "Hash-based GPU set intersection"
        );


        fmt::print("{}\n", "");

        int multiprocessorCount;
        int maxThreadsPerBlock;

        cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        // arguments
        std::string input;
        std::string output;
        unsigned int blocks = multiprocessorCount * 16;
        unsigned int blockSize = maxThreadsPerBlock / 2;
        unsigned int buckets = 512;
        unsigned int bucketSize = 1000;
        bool warpBased = false;
        bool path = false;
        unsigned int partition = 10000;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("input", "Input dataset path", cxxopts::value<std::string>(input))
                ("output", "Output result path", cxxopts::value<std::string>(output))
                ("blocks", "Number of blocks (default: " + std::to_string(blocks) + ")", cxxopts::value<unsigned int>(blocks))
                ("threads", "Threads per block (default: " + std::to_string(blockSize) + ")", cxxopts::value<unsigned int>(blockSize))
                ("buckets", "Number of buckets (default: 512)", cxxopts::value<unsigned int>(buckets))
                ("bucket-size", "Size of each bucket (default: 1000)", cxxopts::value<unsigned int>(bucketSize))
                ("warp", "Launch warp based kernel (default: false, runs block based)", cxxopts::value<bool>(warpBased))
                ("partition", "Number of sets to be processed per GPU invocation", cxxopts::value<unsigned int>(partition))
                ("path", "Adapt intersect path 1st level partitioning to distribute workload across thread blocks", cxxopts::value<bool>(path))
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


        unsigned int warpsPerBlock = blockSize / 32; // 32 is the warp size
        unsigned int binsMemory = blocks * warpsPerBlock * buckets * bucketSize * sizeof(unsigned int);

        std::vector<tile_pair> runs = findTilePairs(d->cardinality, partition);

        size_t combinations;
        if (runs.size() == 1) {
            combinations = combination(d->cardinality, 2);
        } else {
            combinations = partition * partition;
        }

        unsigned long long outputMemory = runs.size() * combinations * sizeof(unsigned int);

        unsigned long long deviceMemory = (sizeof(unsigned int) * d->cardinality * 2)
                                          + (sizeof(unsigned int) * d->totalElements)
                                          + (sizeof(unsigned int) * combinations)
                                          + binsMemory;

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
                "└{23:─^{1}}┘\n", "Launch info", 51, 25,
                "Blocks", blocks,
                "Block Size", blockSize,
                "Warps per Block", warpsPerBlock,
                "Level", (warpBased ? "Warp" : "Block"),
                "Buckets", buckets,
                "Bucket Size", bucketSize,
                "Partition", partition,
                "GPU invocations", runs.size(),
                "Required memory (Output)", formatBytes(outputMemory),
                "Required memory (GPU)", formatBytes(deviceMemory),
                ""
        );

        std::vector<unsigned int> counts(runs.size() == 1 ? combinations : runs.size() * combinations);

        cudaDeviceReset();

        unsigned int* deviceDiagonals;

        unsigned int* deviceOffsets;
        unsigned int* deviceSizes;
        unsigned int* deviceElements;
        unsigned int* deviceCounts;
        unsigned int* deviceBins;

        DeviceTimer deviceTimer;

        EventPair* devMemAlloc = deviceTimer.add("Device memory allocation");
        errorCheck(cudaMalloc((void**)&deviceOffsets, sizeof(unsigned int) * d->cardinality))
        errorCheck(cudaMalloc((void**)&deviceSizes, sizeof(unsigned int) * d->cardinality))
        errorCheck(cudaMalloc((void**)&deviceElements, sizeof(unsigned int) * d->totalElements))
        errorCheck(cudaMalloc((void**)&deviceCounts, sizeof(unsigned int) * combinations))
        errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combinations))
        errorCheck(cudaMalloc((void**)&deviceBins, binsMemory))
        errorCheck(cudaMemset(deviceBins, 0, binsMemory))
        if (path) {
            errorCheck(cudaMalloc((void**)&deviceDiagonals, sizeof(unsigned int) * 2 * (blocks + 1) * combinations))
            errorCheck(cudaMemset(deviceDiagonals, 0, sizeof(unsigned int) * 2 * (blocks + 1) * combinations))
        }
        DeviceTimer::finish(devMemAlloc);

        EventPair* dataTransfer = deviceTimer.add("Transfer to device");
        errorCheck(cudaMemcpy(deviceSizes, d->sizes, sizeof(unsigned int) * d->cardinality, cudaMemcpyHostToDevice))
        errorCheck(cudaMemcpy(deviceOffsets, deviceSizes, sizeof(unsigned int) * d->cardinality, cudaMemcpyDeviceToDevice))
        errorCheck(cudaMemcpy(deviceElements, d->elements, sizeof(unsigned int) * d->totalElements, cudaMemcpyHostToDevice))
        DeviceTimer::finish(dataTransfer);

        EventPair* setOffsets = deviceTimer.add("Compute set offsets");
        thrust::exclusive_scan(thrust::device, deviceOffsets, deviceOffsets + d->cardinality, deviceOffsets, 0); // in-place scan
        DeviceTimer::finish(setOffsets);

        unsigned int iter = 0;
        for (auto& run : runs) {
            tile& A = run.first;
            tile& B = run.second;
            bool selfJoin = A.id == B.id;
            unsigned int numOfSets = selfJoin && runs.size() == 1 ? d->cardinality : partition;

            if (path) {
                EventPair *findDiags = deviceTimer.add("Find diagonals");
                if (selfJoin) {
                    findDiagonals<true><<<blocks, 32>>>(A, B, numOfSets, deviceElements, deviceSizes, deviceOffsets,
                                                        deviceDiagonals, deviceCounts);
                } else {
                    findDiagonals<false><<<blocks, 32>>>(A, B, numOfSets, deviceElements, deviceSizes, deviceOffsets,
                                                         deviceDiagonals, deviceCounts);
                }
                DeviceTimer::finish(findDiags);
            }

            EventPair* hashInter = deviceTimer.add("Hash intersection");
            if (path) {
                if (selfJoin) {
                    intersectPathHI<true><<<blocks, blockSize, sizeof(unsigned int) * buckets>>>
                            (A, B, numOfSets, deviceElements, deviceOffsets, deviceCounts, deviceBins, buckets, bucketSize, deviceDiagonals);
                } else {
                    intersectPathHI<false><<<blocks, blockSize, sizeof(unsigned int) * buckets>>>
                            (A, B, numOfSets, deviceElements, deviceOffsets, deviceCounts, deviceBins, buckets, bucketSize, deviceDiagonals);
                }
            } else {
                if (warpBased) {
                    if (selfJoin) {
                        warpBasedHI<true><<<blocks, blockSize, sizeof(unsigned int) * buckets * warpsPerBlock>>>
                            (A, B, numOfSets, deviceElements, deviceSizes, deviceOffsets, deviceCounts, deviceBins, buckets, bucketSize);
                    } else {
                        warpBasedHI<false><<<blocks, blockSize, sizeof(unsigned int) * buckets * warpsPerBlock>>>
                            (A, B, numOfSets, deviceElements, deviceSizes, deviceOffsets, deviceCounts, deviceBins, buckets, bucketSize);
                    }
                } else {
                    if (selfJoin) {
                        blockBasedHI<true, false><<<blocks, blockSize, sizeof(unsigned int) * buckets>>>(A, B, numOfSets, deviceElements, deviceSizes,
                                                                                                   deviceOffsets, deviceCounts, deviceBins, buckets, bucketSize);
                    } else {
                        blockBasedHI<false, false><<<blocks, blockSize, sizeof(unsigned int) * buckets>>>(A, B, numOfSets, deviceElements, deviceSizes,
                                                                                                   deviceOffsets, deviceCounts, deviceBins, buckets, bucketSize);
                    }
                }
            }
            DeviceTimer::finish(hashInter);

            EventPair *countTransfer = deviceTimer.add("Transfer result");
            errorCheck(cudaMemcpy(&counts[0] + (iter * combinations), deviceCounts, sizeof(unsigned int) * combinations,
                                  cudaMemcpyDeviceToHost))
            DeviceTimer::finish(countTransfer);

            EventPair* clearMemory = deviceTimer.add("Clear memory");
            if (path) {
                errorCheck(cudaMemset(deviceDiagonals, 0, sizeof(unsigned int) * 2 * (blocks + 1) * combinations))
            }
            errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combinations))
            errorCheck(cudaMemset(deviceBins, 0, binsMemory))
            DeviceTimer::finish(clearMemory);

            iter++;
        }

        EventPair* freeDevMem = deviceTimer.add("Free device memory");
        errorCheck(cudaFree(deviceBins))
        errorCheck(cudaFree(deviceOffsets))
        errorCheck(cudaFree(deviceSizes))
        errorCheck(cudaFree(deviceElements))
        errorCheck(cudaFree(deviceCounts))
        if (path) {
            errorCheck(cudaFree(deviceDiagonals))
        }
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

template <bool selfJoin>
__global__ void warpBasedHI(tile A, tile B, unsigned int numOfSets, const unsigned int* elements, const unsigned int* sizes,
                            const unsigned int* offsets, unsigned int* counts, unsigned int* bins, unsigned int numOfBuckets, unsigned int bucketSize) {
    extern unsigned int __shared__ s[]; // [256*4]
    unsigned int* binCounts = s;

    unsigned int warpSize = 32;
    unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warpsPerBlock = blockDim.x / warpSize;
    unsigned int globalWarpId = globalThreadId / warpSize;
    unsigned int warpId = threadIdx.x / warpSize; // local warp id
    unsigned int binSize = numOfBuckets * bucketSize;
    unsigned int binStart = globalWarpId * binSize;
    unsigned int binOffset = warpId * numOfBuckets;

    // Sets must be sorted in ascending order, thus set a will always be smaller than set b in order to minimize collisions
    // and avoid redundant work
    for (unsigned int a = globalWarpId + A.start; a < A.end; a += gridDim.x * warpsPerBlock) {
        unsigned int aStart = offsets[a];
        unsigned int aEnd = offsets[a + 1];

        __syncwarp();

        // ensure bit counts are empty
        for (unsigned int i = threadIdx.x % warpSize + binOffset; i < binOffset + numOfBuckets; i += warpSize) {
            binCounts[i] = 0;
        }

        __syncwarp();

        // Hash shorter set
        for (unsigned int i = threadIdx.x % warpSize + aStart; i < aEnd; i += warpSize) {
            unsigned int element = elements[i];
            unsigned int bin = element % numOfBuckets;
            unsigned int index = atomicAdd(&binCounts[bin + binOffset], 1);
            bins[index * numOfBuckets + bin + binStart] = element;
        }

        __syncwarp();

        for (unsigned int b = (selfJoin ? a + 1 : B.start); b < B.end; b++) {

            unsigned int bStart = offsets[b];
            unsigned int bEnd = bStart + sizes[b];

            unsigned int count = 0;

            // probe larger set
            for (unsigned int i = threadIdx.x % warpSize + bStart; i < bEnd; i += warpSize) {
                unsigned int element = elements[i];
                unsigned int bin = element % numOfBuckets;
                count += linearSearch(element, bins, binCounts, bin, binOffset, binStart, numOfBuckets);
            }

            if (count > 0) {
                atomicAdd(counts + (selfJoin ?
                                    triangular_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets) :
                                    quadratic_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets)) , count);
            }

            __syncwarp();
        }

    }
}

template <bool selfJoin, bool split>
__global__ void blockBasedHI(tile A, tile B, unsigned int numOfSets, const unsigned int* elements, const unsigned int* sizes,
                             const unsigned int* offsets, unsigned int* counts, unsigned int* bins, unsigned int numOfBuckets, unsigned int bucketSize)
{
    extern unsigned int __shared__ s[];
    unsigned int* binCounts = s;

    unsigned int binSize = numOfBuckets * bucketSize;
    unsigned int binStart = blockIdx.x * binSize;

    // Sets must be sorted in ascending order, thus set a will always be smaller than set b in order to minimize collisions
    // and avoid redundant work
    for (unsigned int a = (split ? 0 : blockIdx.x) + A.start; a < A.end; a += (split ? 1 : gridDim.x)) {

        unsigned int aStart = offsets[a];
        unsigned int aEnd = offsets[a + 1];

        // ensure bit counts are empty
        for (unsigned int i = threadIdx.x; i < numOfBuckets; i += blockDim.x) {
            binCounts[i] = 0;
        }

        __syncthreads();

        // Hash shorter set
        for (unsigned int i = threadIdx.x + aStart; i < aEnd; i += blockDim.x) {
            unsigned int element = elements[i];
            unsigned int bin = element % numOfBuckets;
            unsigned int index = atomicAdd(&binCounts[bin], 1);
            bins[index * numOfBuckets + bin + binStart] = element;
        }

        __syncthreads();

        for (unsigned int b = (split ? blockIdx.x : 1) + (selfJoin ? a : B.start); b < B.end; b += (split ? gridDim.x : 1)) {

            unsigned int bStart = offsets[b];
            unsigned int bEnd = bStart + sizes[b];

            unsigned int count = 0;

            // probe larger set
            for (unsigned int i = threadIdx.x + bStart; i < bEnd; i += blockDim.x) {
                unsigned int element = elements[i];
                unsigned int bin = element % numOfBuckets;
                count += linearSearch(element, bins, binCounts, bin, 0, binStart, numOfBuckets);
            }

            if (count > 0) {
                atomicAdd(counts + (selfJoin ?
                                    triangular_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets) :
                                    quadratic_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets)) , count);
            }
            __syncthreads();
        }
    }

}

template <bool selfJoin>
__global__ void intersectPathHI(tile A, tile B, unsigned int numOfSets, const unsigned int* elements, const unsigned int* offsets,
                                unsigned int* counts, unsigned int* bins, unsigned int numOfBuckets,
                                unsigned int bucketSize, unsigned int* globalDiagonals)
{
    extern unsigned int __shared__ s[];
    unsigned int* binCounts = s;

    unsigned int binSize = numOfBuckets * bucketSize;
    unsigned int binStart = blockIdx.x * binSize;

    for (unsigned int a = 0; a < numOfSets - 1; a++) {
        for (unsigned int b = a + 1; b < numOfSets; b++) { // iterate every combination
            unsigned int *diagonals = globalDiagonals + (2 * (gridDim.x + 1)) * triangular_idx(numOfSets, a, b);

            unsigned int aOffset = offsets[a];
            unsigned int bOffset = offsets[b];

            unsigned int aStart = diagonals[blockIdx.x];
            unsigned int aEnd = diagonals[blockIdx.x + 1];

            unsigned int bStart = diagonals[(gridDim.x + 1) + blockIdx.x];
            unsigned int bEnd = diagonals[(gridDim.x + 1) + blockIdx.x + 1];

            // ensure bit counts are empty
            for (unsigned int i = threadIdx.x; i < numOfBuckets; i += blockDim.x) {
                binCounts[i] = 0;
            }

            __syncthreads();

            // Hash shorter set
            for (unsigned int i = threadIdx.x + aStart; i < aEnd; i += blockDim.x) {
                unsigned int element = (elements + aOffset)[i];
                unsigned int bin = element % numOfBuckets;
                unsigned int index = atomicAdd(&binCounts[bin], 1);
                bins[index * numOfBuckets + bin + binStart] = element;
            }

            __syncthreads();

            unsigned int count = 0;

            // probe larger set
            for (unsigned int i = threadIdx.x + bStart; i < bEnd; i += blockDim.x) {
                unsigned int element = (elements + bOffset)[i];
                unsigned int bin = element % numOfBuckets;
                count += linearSearch(element, bins, binCounts, bin, 0, binStart, numOfBuckets);
            }

            if (count > 0) {
                atomicAdd(counts + (selfJoin ?
                                    triangular_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets) :
                                    quadratic_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets)) , count);
            }
            __syncthreads();

        }
    }
}
