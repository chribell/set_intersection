#include <fmt/core.h>
#include "io.hpp"
#include "device_timer.cuh"
#include "util.hpp"
#include "helpers.cuh"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cxxopts.hpp>

template <bool selfJoin>
__global__ void intersectPath(tile A, tile B, unsigned int numOfSets, unsigned int* sets, unsigned int* offsets,
                              unsigned int* globalDiagonals, unsigned int* counts);

int main(int argc, char **argv) {
    try {
        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{2: ^{1}}|\n"
                "└{0:─^{1}}┘\n", "", 51, "Intersect-Path GPU set intersection"
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
        unsigned int partition = 10000;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("input", "Input dataset path", cxxopts::value<std::string>(input))
                ("output", "Output result path", cxxopts::value<std::string>(output))
                ("blocks", "Number of blocks (default: " + std::to_string(blocks) + ")", cxxopts::value<unsigned int>(blocks))
                ("threads", "Threads per block (default: " + std::to_string(blockSize) + ")", cxxopts::value<unsigned int>(blockSize))
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

        Dataset *d = readDataset(input);

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

        unsigned long long deviceMemory = (sizeof(unsigned int) * 2 * (blocks + 1) * combinations)
                + (sizeof(unsigned int) * d->cardinality * 2)
                + (sizeof(unsigned int) * d->totalElements)
                + (sizeof(unsigned int) * combinations);

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
                "Warps per Block", blockSize / 32,
                "Level", "Block",
                "Partition", partition,
                "GPU invocations", runs.size(),
                "Required memory (Output)", formatBytes(outputMemory),
                "Required memory (GPU)", formatBytes(deviceMemory),
                ""
        );

        std::vector<unsigned int> counts(runs.size() == 1 ? combinations : runs.size() * combinations);

        DeviceTimer deviceTimer;

        // allocate device memory space
        unsigned int* deviceOffsets;
        unsigned int* deviceSizes;
        unsigned int* deviceElements;
        unsigned int* deviceCounts;
        unsigned int* deviceDiagonals;

        EventPair *devMemAlloc = deviceTimer.add("Device memory allocation");
        errorCheck(cudaMalloc((void**)&deviceOffsets, sizeof(unsigned int) * d->cardinality))
        errorCheck(cudaMalloc((void**)&deviceSizes, sizeof(unsigned int) * d->cardinality))
        errorCheck(cudaMalloc((void**)&deviceElements, sizeof(unsigned int) * d->totalElements))
        errorCheck(cudaMalloc((void**)&deviceCounts, sizeof(unsigned int) * combinations + 1))
        errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combinations + 1))
        errorCheck(cudaMalloc((void**)&deviceDiagonals, sizeof(unsigned int) * 2 * (blocks + 1) * combinations))
        errorCheck(cudaMemset(deviceDiagonals, 0, sizeof(unsigned int) * 2 * (blocks + 1) * combinations))
        DeviceTimer::finish(devMemAlloc);

        EventPair *dataTransfer = deviceTimer.add("Transfer to device");
        errorCheck(cudaMemcpy(deviceSizes, d->sizes, sizeof(unsigned int) * d->cardinality, cudaMemcpyHostToDevice))
        errorCheck(cudaMemcpy(deviceOffsets, deviceSizes, sizeof(unsigned int) * d->cardinality, cudaMemcpyDeviceToDevice))
        errorCheck(cudaMemcpy(deviceElements, d->elements, sizeof(unsigned int) * d->totalElements, cudaMemcpyHostToDevice))
        DeviceTimer::finish(dataTransfer);

        EventPair *setOffsets = deviceTimer.add("Compute set offsets");
        thrust::exclusive_scan(thrust::device, deviceOffsets, deviceOffsets + d->cardinality, deviceOffsets,
                               0); // in-place scan
        DeviceTimer::finish(setOffsets);

        unsigned int iter = 0;
        for (auto& run : runs) {
            tile &A = run.first;
            tile &B = run.second;
            bool selfJoin = A.id == B.id;
            unsigned int numOfSets = selfJoin && runs.size() == 1 ? d->cardinality : partition;

            EventPair *findDiags = deviceTimer.add("Find diagonals");
            if (selfJoin) {
                findDiagonals<true><<<blocks, 32>>>(A, B, numOfSets, deviceElements, deviceSizes,
                                              deviceOffsets, deviceDiagonals, deviceCounts);
            } else {
                findDiagonals<false><<<blocks, 32>>>(A, B, numOfSets, deviceElements, deviceSizes,
                                                    deviceOffsets, deviceDiagonals, deviceCounts);
            }
            DeviceTimer::finish(findDiags);

            EventPair *computeIntersections = deviceTimer.add("Intersect path");
            if (selfJoin) {
                intersectPath<true><<<blocks, blockSize, blockSize * sizeof(unsigned int)>>>
                        (A, B, numOfSets, deviceElements, deviceOffsets, deviceDiagonals, deviceCounts);
            } else {
                intersectPath<false><<<blocks, blockSize, blockSize * sizeof(unsigned int)>>>
                        (A, B, numOfSets, deviceElements, deviceOffsets, deviceDiagonals, deviceCounts);
            }
            DeviceTimer::finish(computeIntersections);

            EventPair *countTransfer = deviceTimer.add("Transfer result");
            errorCheck(cudaMemcpy(&counts[0] + (iter * combinations), deviceCounts, sizeof(unsigned int) * combinations,
                                  cudaMemcpyDeviceToHost))
            DeviceTimer::finish(countTransfer);

            EventPair* clearMemory = deviceTimer.add("Clear memory");
            errorCheck(cudaMemset(deviceDiagonals, 0, sizeof(unsigned int) * 2 * (blocks + 1) * combinations))
            errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combinations))
            DeviceTimer::finish(clearMemory);
            iter++;
        }

        EventPair *freeDevMem = deviceTimer.add("Free device memory");
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

template <bool selfJoin>
__global__ void intersectPath(tile A, tile B, unsigned int numOfSets, unsigned int* sets, unsigned int* offsets,
                              unsigned int* globalDiagonals, unsigned int* counts) {

    for (unsigned int a = A.start; a < A.end; a++) {
        for (unsigned int b = (selfJoin ? a + 1 : B.start); b < B.end; b++) { // iterate every combination
            unsigned int offset = (selfJoin ?
                                   triangular_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets) :
                                   quadratic_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets));
            unsigned int* aSet = sets + offsets[a];
            unsigned int* bSet = sets + offsets[b];
            unsigned int* diagonals = globalDiagonals + (2 * (gridDim.x + 1)) * offset;

            unsigned int aStart = diagonals[blockIdx.x];
            unsigned int aEnd = diagonals[blockIdx.x + 1];

            unsigned int bStart = diagonals[(gridDim.x + 1) + blockIdx.x];
            unsigned int bEnd = diagonals[(gridDim.x + 1) + blockIdx.x + 1];

            unsigned int aSize = aEnd - aStart;
            unsigned int bSize = bEnd - bStart;

            unsigned int vt = ((aSize + bSize) / blockDim.x) + 1;

            // local diagonal
            unsigned int diag = threadIdx.x * vt;

            int mp = binarySearch(aSet + aStart, aSize, bSet + bStart, bSize, diag);

            unsigned int intersection = serialIntersect(aSet + aStart, mp,
                                                         aSize,
                                                         bSet + bStart,
                                                         diag - mp,
                                                         bSize,
                                                         vt);
            atomicAdd(counts + offset, intersection);
        }
    }
}