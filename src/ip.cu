#include <fmt/core.h>
#include "io.hpp"
#include "device_timer.cuh"
#include "util.hpp"
#include "helpers.cuh"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cxxopts.hpp>

__global__ void intersectPath(unsigned int numOfSets, unsigned int* sets, unsigned int* offsets,
                              unsigned int* globalDiagonals, unsigned int* counts);

int main(int argc, char **argv) {
    try {
        fmt::print("{}\n", "Intersect-Path GPU set intersection");

        int multiprocessorCount;
        int maxThreadsPerBlock;

        cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        // arguments
        std::string input;
        std::string output;
        unsigned int blocks = multiprocessorCount * 16;
        unsigned int blockSize = maxThreadsPerBlock / 2;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("input", "Input dataset path", cxxopts::value<std::string>(input))
                ("output", "Output result path", cxxopts::value<std::string>(output))
                ("blocks", "Number of blocks (default: " + std::to_string(blocks) + ")", cxxopts::value<unsigned int>(blocks))
                ("threads", "Threads per block (default: " + std::to_string(blockSize) + ")", cxxopts::value<unsigned int>(blockSize))
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
        errorCheck(cudaMalloc((void**)&deviceCounts, sizeof(unsigned int) * combination(d->cardinality, 2) + 1))
        errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combination(d->cardinality, 2) + 1))
        errorCheck(cudaMalloc((void**)&deviceDiagonals, sizeof(unsigned int) * 2 * (blocks + 1) * (combination(d->cardinality, 2))))
        errorCheck(cudaMemset(deviceDiagonals, 0, sizeof(unsigned int) * 2 * (blocks + 1) * (combination(d->cardinality, 2))))
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

        EventPair *findDiags = deviceTimer.add("Find diagonals");
        findDiagonals<<<blocks, 32>>>(d->cardinality, deviceElements, deviceSizes, deviceOffsets, deviceDiagonals, deviceCounts);
        DeviceTimer::finish(findDiags);

        EventPair *computeIntersections = deviceTimer.add("Intersect path");
        intersectPath<<<blocks, blockSize, blockSize * sizeof(unsigned int)>>>
                (d->cardinality, deviceElements, deviceOffsets, deviceDiagonals, deviceCounts);
        DeviceTimer::finish(computeIntersections);

        std::vector<unsigned int> counts(combination(d->cardinality, 2));

        EventPair *countTransfer = deviceTimer.add("Transfer result");
        errorCheck(cudaMemcpy(&counts[0], deviceCounts, sizeof(unsigned int) * combination(d->cardinality, 2), cudaMemcpyDeviceToHost))
        DeviceTimer::finish(countTransfer);

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
            writeResult(d->cardinality, counts, output);
            fmt::print("Finished\n");
        }

    } catch (const cxxopts::OptionException& e) {
        fmt::print("{}\n", e.what());
        return 1;
    }
    return 0;
}

__global__ void intersectPath(unsigned int numOfSets, unsigned int* sets, unsigned int* offsets,
                              unsigned int* globalDiagonals, unsigned int* counts) {

    for (unsigned int a = 0; a < numOfSets - 1; a++) {
        for (unsigned int b = a + 1; b < numOfSets; b++) { // iterate every combination
            unsigned int* aSet = sets + offsets[a];
            unsigned int* bSet = sets + offsets[b];
            unsigned int* diagonals = globalDiagonals + (2 * (gridDim.x + 1)) * triangular_idx(numOfSets, a, b);

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
            atomicAdd(counts + triangular_idx(numOfSets, a, b), intersection);
        }
    }
}