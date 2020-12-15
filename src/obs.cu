#include <fmt/core.h>
#include "io.hpp"
#include <cxxopts.hpp>
#include "helpers.cuh"
#include "device_timer.cuh"
#include <thrust/scan.h>
#include "util.hpp"

__global__ void warpBasedOBS(unsigned int numOfSets,
                                           const unsigned int* elements,
                                           const unsigned int* sizes,
                                           const unsigned int* offsets,
                                           unsigned int* counts);

__global__ void blockBasedOBS(unsigned int numOfSets,
                                            const unsigned int* elements,
                                            const unsigned int* sizes,
                                            const unsigned int* offsets,
                                            unsigned int* counts);

__global__ void intersectPathOBS(unsigned int numOfSets,
                                            const unsigned int* elements,
                                            const unsigned int* sizes,
                                            const unsigned int* offsets,
                                            unsigned int* counts,
                                            unsigned int* globalDiagonals);

int main(int argc, char** argv) {
    try {
        fmt::print("{}\n", "OBS GPU set intersection");

        int multiprocessorCount;
        int maxThreadsPerBlock;

        cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        // arguments
        std::string input;
        std::string output;
        unsigned int blocks = multiprocessorCount * 16;
        unsigned int blockSize = maxThreadsPerBlock / 2;
        bool warpBased = false;
        bool partition = false;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("input", "Input dataset path", cxxopts::value<std::string>(input))
                ("output", "Output result path", cxxopts::value<std::string>(output))
                ("blocks", "Number of blocks (default: " + std::to_string(blocks) + ")", cxxopts::value<unsigned int>(blocks))
                ("threads", "Threads per block (default: " + std::to_string(blockSize) + ")", cxxopts::value<unsigned int>(blockSize))
                ("warp", "Launch warp based kernel (default: false, runs block based)", cxxopts::value<bool>(warpBased))
                ("path", "Adapt intersect path 1st level partitioning to distribute workload across thread blocks (default: false)", cxxopts::value<bool>(partition))
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

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "│{9: ^{2}}|{10: ^{2}}│\n"
                "└{11:─^{1}}┘\n", "Launch info", 51, 25,
                "Blocks", blocks,
                "Block Size", blockSize,
                "Warps per Block", warpsPerBlock,
                "Level", (warpBased ? "Warp" : "Block"), ""
        );

        unsigned int* deviceOffsets;
        unsigned int* deviceSizes;
        unsigned int* deviceElements;
        unsigned int* deviceCounts;
        unsigned int* deviceDiagonals;

        DeviceTimer deviceTimer;

        EventPair* devMemAlloc = deviceTimer.add("Device memory allocation");
        errorCheck(cudaMalloc((void**)&deviceOffsets, sizeof(unsigned int) * d->cardinality))
        errorCheck(cudaMalloc((void**)&deviceSizes, sizeof(unsigned int) * d->cardinality))
        errorCheck(cudaMalloc((void**)&deviceElements, sizeof(unsigned int) * d->totalElements))
        errorCheck(cudaMalloc((void**)&deviceCounts, sizeof(unsigned int) * combination(d->cardinality, 2)))
        errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combination(d->cardinality, 2)))
        if (partition) {
            errorCheck(cudaMalloc((void**)&deviceDiagonals, sizeof(unsigned int) * 2 * (blocks + 1) * (combination(d->cardinality, 2))))
            errorCheck(cudaMemset(deviceDiagonals, 0, sizeof(unsigned int) * 2 * (blocks + 1) * (combination(d->cardinality, 2))))
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

        if (partition) {
            EventPair *findDiags = deviceTimer.add("Find diagonals");
            findDiagonals<<<blocks, 32>>>(d->cardinality, deviceElements, deviceSizes, deviceOffsets, deviceDiagonals, deviceCounts);
            DeviceTimer::finish(findDiags);
        }

        EventPair* hashInter = deviceTimer.add("Binary search intersection");
        if (partition) {
            intersectPathOBS<<<blocks, blockSize, sizeof(unsigned int) * blockSize>>>(d->cardinality, deviceElements, deviceSizes,
                                                                                                 deviceOffsets, deviceCounts, deviceDiagonals);
        } else {
            if (warpBased) {
                warpBasedOBS<<<blocks, blockSize, sizeof(unsigned int) * blockSize>>>(d->cardinality, deviceElements, deviceSizes,
                                                                                                    deviceOffsets, deviceCounts);
            } else {
                blockBasedOBS<<<blocks, blockSize, sizeof(unsigned int) * blockSize>>>(d->cardinality, deviceElements, deviceSizes,
                                                                                                     deviceOffsets, deviceCounts);
            }
        }
        DeviceTimer::finish(hashInter);


        std::vector<unsigned int> counts(combination(d->cardinality, 2));

        EventPair* countTransfer = deviceTimer.add("Transfer result");
        errorCheck(cudaMemcpy(&counts[0], deviceCounts, sizeof(unsigned int) * combination(d->cardinality, 2), cudaMemcpyDeviceToHost))
        DeviceTimer::finish(countTransfer);

        EventPair* freeDevMem = deviceTimer.add("Free device memory");
        errorCheck(cudaFree(deviceOffsets))
        errorCheck(cudaFree(deviceSizes))
        errorCheck(cudaFree(deviceElements))
        errorCheck(cudaFree(deviceCounts))
        if (partition) {
            errorCheck(cudaFree(deviceDiagonals))
        }
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


__global__ void warpBasedOBS(unsigned int numOfSets,
                                           const unsigned int* elements,
                                           const unsigned int* sizes,
                                           const unsigned int* offsets,
                                           unsigned int* counts) {
    extern unsigned int __shared__ s[];
    unsigned int* cache = s;
    unsigned int warpSize = 32;

    unsigned int globalWarpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    unsigned int localWarpId = threadIdx.x / warpSize;
    unsigned int threadOffset = threadIdx.x % warpSize;
    for (unsigned int a = globalWarpId; a < numOfSets - 1; a += (blockDim.x * gridDim.x / warpSize)) {
        unsigned int aSize = sizes[a];
        unsigned int aStart = offsets[a];
        unsigned int aEnd = offsets[a + 1];

        // cache first levels of the binary tree of the larger set
        cache[localWarpId * warpSize + threadOffset] = elements[aStart + (threadOffset * aSize / warpSize)];
        __syncthreads();

        for (unsigned int b = a + 1; b < numOfSets; b++) {

            unsigned int bStart = offsets[b];
            unsigned int bEnd = bStart + sizes[b];

            unsigned int count = 0;

            // search smaller set
            for (unsigned int i = threadOffset + bStart; i < bEnd; i += warpSize) {
                unsigned int x = elements[i];
                unsigned int y;

                // phase 1: cache
                int bottom = 0;
                int top = warpSize;
                int mid;
                while (top > bottom + 1) {
                    mid = (top + bottom) / 2;
                    y = cache[localWarpId * warpSize + mid];
                    if (x == y) {
                        count++;
                        bottom = top + warpSize;
                    }
                    if (x < y) {
                        top = mid;
                    }
                    if (x > y) {
                        bottom = mid;
                    }
                }

                //phase 2
                bottom = bottom * aSize / warpSize;
                top = top * aSize / warpSize - 1;
                while (top >= bottom) {
                    mid = (top + bottom) / 2;
                    y = elements[aStart + mid];
                    if (x == y) {
                        count++;
                    }
                    if (x <= y) {
                        top = mid - 1;
                    }
                    if (x >= y) {
                        bottom = mid + 1;
                    }
                }
            }
            atomicAdd(counts + pos(numOfSets, a, b), count);
            __syncthreads();
        }

    }

}

__global__ void blockBasedOBS(unsigned int numOfSets,
                                      const unsigned int* elements,
                                      const unsigned int* sizes,
                                      const unsigned int* offsets,
                                      unsigned int* counts) {
    extern unsigned int __shared__ s[];
    unsigned int* cache = s;

    for (unsigned int a = blockIdx.x; a < numOfSets - 1; a += gridDim.x) {
        unsigned int aSize = sizes[a];
        unsigned int aStart = offsets[a];
        unsigned int aEnd = offsets[a + 1];

        // cache first levels of the binary tree of the larger set
        cache[threadIdx.x] = elements[aStart + (threadIdx.x * aSize / blockDim.x)];
        __syncthreads();

        for (unsigned int b = a + 1; b < numOfSets; b++) {

            unsigned int bStart = offsets[b];
            unsigned int bEnd = bStart + sizes[b];

            unsigned int count = 0;

            // search smaller set
            for (unsigned int i = threadIdx.x + bStart; i < bEnd; i += blockDim.x) {
                unsigned int x = elements[i];
                unsigned int y;

                // phase 1: cache
                int bottom = 0;
                int top = blockDim.x;
                int mid;
                while (top > bottom + 1) {
                    mid = (top + bottom) / 2;
                    y = cache[mid];
                    if (x == y) {
                        count++;
                        bottom = top + blockDim.x;
                    }
                    if (x < y) {
                        top = mid;
                    }
                    if (x > y) {
                        bottom = mid;
                    }
                }

                //phase 2
                bottom = bottom * aSize / blockDim.x;
                top = top * aSize / blockDim.x - 1;
                while (top >= bottom) {
                    mid = (top + bottom) / 2;
                    y = elements[aStart + mid];
                    if (x == y) {
                        count++;
                    }
                    if (x <= y) {
                        top = mid - 1;
                    }
                    if (x >= y) {
                        bottom = mid + 1;
                    }
                }
            }
            atomicAdd(counts + pos(numOfSets, a, b), count);
            __syncthreads();
        }

    }

}


__global__ void intersectPathOBS(unsigned int numOfSets,
                                          const unsigned int* elements,
                                          const unsigned int* sizes,
                                          const unsigned int* offsets,
                                          unsigned int* counts,
                                          unsigned int* globalDiagonals) {
    extern unsigned int __shared__ s[];
    unsigned int* cache = s;

    for (unsigned int a = 0; a < numOfSets; a++) {
        for (unsigned int b = a + 1; b < numOfSets; b++) {
            unsigned int *diagonals = globalDiagonals + (2 * (gridDim.x + 1)) * pos(numOfSets, a, b);

            unsigned int aOffset = offsets[a];
            unsigned int bOffset = offsets[b];

            unsigned int aStart = diagonals[blockIdx.x];
            unsigned int aEnd = diagonals[blockIdx.x + 1];

            unsigned int bStart = diagonals[(gridDim.x + 1) + blockIdx.x];
            unsigned int bEnd = diagonals[(gridDim.x + 1) + blockIdx.x + 1];

            unsigned int aSize = aEnd - aStart;
            unsigned int bSize = bEnd - bStart;

            // cache first levels of the binary tree of the larger set
            cache[threadIdx.x] = (elements + aOffset)[aStart + (threadIdx.x * aSize / blockDim.x)];
            __syncthreads();

            unsigned int count = 0;

            // search smaller set
            for (unsigned int i = threadIdx.x + bStart; i < bEnd; i += blockDim.x) {
                unsigned int x = (elements + bOffset)[i];
                unsigned int y;

                // phase 1: cache
                int bottom = 0;
                int top = blockDim.x;
                int mid;
                while (top > bottom + 1) {
                    mid = (top + bottom) / 2;
                    y = cache[mid];
                    if (x == y) {
                        count++;
                        bottom = top + blockDim.x;
                    }
                    if (x < y) {
                        top = mid;
                    }
                    if (x > y) {
                        bottom = mid;
                    }
                }

                //phase 2
                bottom = bottom * aSize / blockDim.x;
                top = top * aSize / blockDim.x - 1;
                while (top >= bottom) {
                    mid = (top + bottom) / 2;
                    y = elements[aStart + mid];
                    if (x == y) {
                        count++;
                    }
                    if (x <= y) {
                        top = mid - 1;
                    }
                    if (x >= y) {
                        bottom = mid + 1;
                    }
                }
            }
            atomicAdd(counts + pos(numOfSets, a, b), count);
            __syncthreads();

        }
    }
}