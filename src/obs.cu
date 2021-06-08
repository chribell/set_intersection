#include <fmt/core.h>
#include "io.hpp"
#include <cxxopts.hpp>
#include "helpers.cuh"
#include "device_timer.cuh"
#include <thrust/scan.h>
#include "util.hpp"
#include <fmt/ranges.h>

template <bool selfJoin>
__global__ void warpBasedOBS(tile A,
                             tile B,
                             unsigned int numOfSets,
                             const unsigned int* elements,
                             const unsigned int* sizes,
                             const unsigned int* offsets,
                             unsigned int* counts);

template <bool selfJoin>
__global__ void blockBasedOBS(tile A,
                              tile B,
                              unsigned int numOfSets,
                              const unsigned int* elements,
                              const unsigned int* sizes,
                              const unsigned int* offsets,
                              unsigned int* counts);

template <bool selfJoin>
__global__ void intersectPathOBS(tile A,
                                 tile B,
                                 unsigned int numOfSets,
                                 const unsigned int* elements,
                                 const unsigned int* sizes,
                                 const unsigned int* offsets,
                                 unsigned int* counts,
                                 unsigned int* globalDiagonals);

int main(int argc, char** argv) {
    try {
        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{2: ^{1}}|\n"
                "└{0:─^{1}}┘\n", "", 51, "OBS GPU set intersection"
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
        bool warpBased = false;
        bool path = false;
        unsigned int partition = 10000;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("input", "Input dataset path", cxxopts::value<std::string>(input))
                ("output", "Output result path", cxxopts::value<std::string>(output))
                ("blocks", "Number of blocks (default: " + std::to_string(blocks) + ")", cxxopts::value<unsigned int>(blocks))
                ("threads", "Threads per block (default: " + std::to_string(blockSize) + ")", cxxopts::value<unsigned int>(blockSize))
                ("warp", "Launch warp based kernel (default: false, runs block based)", cxxopts::value<bool>(warpBased))
                ("partition", "Number of sets to be processed per GPU invocation", cxxopts::value<unsigned int>(partition))
                ("path", "Adapt intersect path 1st level partitioning to distribute workload across thread blocks (default: false)", cxxopts::value<bool>(path))
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

        std::vector<tile_pair> runs = findTilePairs(d->cardinality, partition);

        size_t combinations;
        if (d->cardinality >= partition) {
            combinations = combination(d->cardinality, 2);
        } else {
            combinations = partition * partition;
        }

        unsigned long long outputMemory = (runs.size() == 1 ?
                                           combination(d->cardinality, 2) * sizeof(unsigned int) :
                                           runs.size() * partition * sizeof(unsigned int)
        );

        unsigned long long deviceMemory = (sizeof(unsigned int) * d->cardinality * 2)
                                          + (sizeof(unsigned int) * d->totalElements)
                                          + (sizeof(unsigned int) * combinations);

        if (path) {
            deviceMemory += (sizeof(unsigned int) * 2 * (blocks + 1) * combinations);
        }

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
                "└{19:─^{1}}┘\n", "Launch info", 51, 25,
                "Blocks", blocks,
                "Block Size", blockSize,
                "Warps per Block", warpsPerBlock,
                "Level", (warpBased ? "Warp" : "Block"),
                "Partition", partition,
                "GPU invocations", runs.size(),
                "Required memory (Output)", formatBytes(outputMemory),
                "Required memory (GPU)", formatBytes(deviceMemory),
                ""
        );

        std::vector<unsigned int> counts(runs.size() == 1 ? combinations : runs.size() * combinations);

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
        errorCheck(cudaMalloc((void**)&deviceCounts, sizeof(unsigned int) * combinations))
        errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combinations))
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
                EventPair* findDiags = deviceTimer.add("Find diagonals");
                if (selfJoin) {
                    findDiagonals<true><<<blocks, 32>>>(A, B, numOfSets, deviceElements, deviceSizes, deviceOffsets,
                                                        deviceDiagonals, deviceCounts);
                } else {
                    findDiagonals<false><<<blocks, 32>>>(A, B, numOfSets, deviceElements, deviceSizes, deviceOffsets,
                                                        deviceDiagonals, deviceCounts);
                }
                DeviceTimer::finish(findDiags);
            }

            EventPair* obsInter = deviceTimer.add("OBS intersection");
            if (path) {
                if (selfJoin) {
                    intersectPathOBS<true><<<blocks, blockSize, sizeof(unsigned int) * blockSize>>>(A, B, numOfSets,
                                                                                                    deviceElements, deviceSizes,
                                                                                                    deviceOffsets, deviceCounts,
                                                                                                    deviceDiagonals);
                } else {
                    intersectPathOBS<false><<<blocks, blockSize, sizeof(unsigned int) * blockSize>>>(A, B, numOfSets,
                                                                                                    deviceElements, deviceSizes,
                                                                                                    deviceOffsets, deviceCounts,
                                                                                                    deviceDiagonals);
                }

            } else {
                if (warpBased) {
                    if (selfJoin) {
                        warpBasedOBS<true><<<blocks, blockSize, sizeof(unsigned int) * blockSize>>>(A, B, numOfSets,
                                                                                              deviceElements, deviceSizes,
                                                                                              deviceOffsets, deviceCounts);
                    } else {
                        warpBasedOBS<false><<<blocks, blockSize, sizeof(unsigned int) * blockSize>>>(A, B, numOfSets,
                                                                                              deviceElements, deviceSizes,
                                                                                              deviceOffsets, deviceCounts);
                    }
                } else {
                    if (selfJoin) {
                        blockBasedOBS<true><<<blocks, blockSize, sizeof(unsigned int) * blockSize>>>(A, B, numOfSets,
                                                                                                     deviceElements, deviceSizes,
                                                                                                     deviceOffsets, deviceCounts);
                    } else {
                        blockBasedOBS<false><<<blocks, blockSize, sizeof(unsigned int) * blockSize>>>(A, B, numOfSets,
                                                                                                     deviceElements, deviceSizes,
                                                                                                     deviceOffsets, deviceCounts);
                    }
                }
            }
            DeviceTimer::finish(obsInter);

            EventPair *countTransfer = deviceTimer.add("Transfer result");
            errorCheck(cudaMemcpy(&counts[0] + (iter * combinations), deviceCounts, sizeof(unsigned int) * combinations,
                                  cudaMemcpyDeviceToHost))
            DeviceTimer::finish(countTransfer);

            EventPair* clearMemory = deviceTimer.add("Clear memory");
            if (path) {
                errorCheck(cudaMemset(deviceDiagonals, 0, sizeof(unsigned int) * 2 * (blocks + 1) * combinations))
            }
            errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combinations))
            DeviceTimer::finish(clearMemory);
            iter++;
        }

        EventPair* freeDevMem = deviceTimer.add("Free device memory");
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
            writeResult(d->cardinality, counts, output);
            fmt::print("Finished\n");
        }
    } catch (const cxxopts::OptionException& e) {
        fmt::print("{}\n", e.what());
        return 1;
    }
    return 0;
}

template <bool selfJoin>
__global__ void warpBasedOBS(tile A,
                             tile B,
                             unsigned int numOfSets,
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
    for (unsigned int a = globalWarpId + A.start; a < (selfJoin ? numOfSets - 1 : A.end); a += (blockDim.x * gridDim.x / warpSize)) {
        unsigned int aSize = sizes[a];
        unsigned int aStart = offsets[a];
        unsigned int aEnd = offsets[a + 1];

        // cache first levels of the binary tree of the larger set
        cache[localWarpId * warpSize + threadOffset] = elements[aStart + (threadOffset * aSize / warpSize)];
        __syncthreads();

        for (unsigned int b = (selfJoin ? a + 1 : B.start); b < (selfJoin ? numOfSets : B.end); b++) {

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
            if (selfJoin) {
                atomicAdd(counts + triangular_idx(numOfSets, a, b), count);
            } else {
                atomicAdd(counts + quadratic_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets), count);
            }
            __syncthreads();
        }

    }

}
template <bool selfJoin>
__global__ void blockBasedOBS(tile A,
                              tile B,
                              unsigned int numOfSets,
                              const unsigned int* elements,
                              const unsigned int* sizes,
                              const unsigned int* offsets,
                              unsigned int* counts) {
    extern unsigned int __shared__ s[];
    unsigned int* cache = s;

    for (unsigned int a = blockIdx.x + A.start; a < (selfJoin ? numOfSets - 1 : A.end); a += gridDim.x) {
        unsigned int aSize = sizes[a];
        unsigned int aStart = offsets[a];
        unsigned int aEnd = offsets[a + 1];

        // cache first levels of the binary tree of the larger set
        cache[threadIdx.x] = elements[aStart + (threadIdx.x * aSize / blockDim.x)];
        __syncthreads();

        for (unsigned int b = (selfJoin ? a + 1 : B.start); b < (selfJoin ? numOfSets : B.end); b++) {

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
            if (selfJoin) {
                atomicAdd(counts + triangular_idx(numOfSets, a, b), count);
            } else {
                atomicAdd(counts + quadratic_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets), count);
            }
            __syncthreads();
        }

    }

}

template <bool selfJoin>
__global__ void intersectPathOBS(tile A,
                                 tile B,
                                 unsigned int numOfSets,
                                 const unsigned int* elements,
                                 const unsigned int* sizes,
                                 const unsigned int* offsets,
                                 unsigned int* counts,
                                 unsigned int* globalDiagonals) {
    extern unsigned int __shared__ s[];
    unsigned int* cache = s;

    for (unsigned int a = A.start; a < (selfJoin ? numOfSets : B.start); a++) {
        for (unsigned int b = (selfJoin ? a + 1 : B.start); b < (selfJoin ? numOfSets : B.end); b++) {
            unsigned int offset =
                    (selfJoin ?
                     triangular_idx(numOfSets, a, b) :
                     quadratic_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets));

            unsigned int *diagonals = globalDiagonals + (2 * (gridDim.x + 1)) * offset;

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
            atomicAdd(counts + offset, count);
            __syncthreads();
        }
    }
}