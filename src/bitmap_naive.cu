#include <fmt/core.h>
#include <cxxopts.hpp>
#include "io.hpp"
#include "device_timer.cuh"
#include <climits>
#include <thrust/scan.h>
#include "util.hpp"
#include "helpers.cuh"


__global__ void generateBitmaps(word* bitmaps, unsigned int numOfWords, unsigned int numOfSets,
                                const unsigned int* sizes, const unsigned int* offsets, const unsigned int* elements);

__global__ void intersectPerPair(const word* words, unsigned int numOfSets, unsigned int numOfWords, unsigned int* counts);


int main(int argc, char** argv) {
    try {

        fmt::print("{}\n", "Bitmap-based naive GPU set intersection");

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

        unsigned int bitmapWords = BITMAP_NWORDS(d->universe);
        size_t bitmapMemory = bitmapWords * sizeof(word) * d->cardinality;

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


        DeviceTimer deviceTimer;

        // allocate device memory space
        word* deviceBitmaps;
        unsigned int* deviceOffsets;
        unsigned int* deviceSizes;
        unsigned int* deviceElements;
        unsigned int* deviceCounts;

        EventPair* devMemAlloc = deviceTimer.add("Device memory allocation");
        errorCheck(cudaMalloc((void**)&deviceBitmaps, bitmapMemory))
        errorCheck(cudaMemset(deviceBitmaps, 0, bitmapMemory))
        errorCheck(cudaMalloc((void**)&deviceOffsets, sizeof(unsigned int) * d->cardinality))
        errorCheck(cudaMalloc((void**)&deviceSizes, sizeof(unsigned int) * d->cardinality))
        errorCheck(cudaMalloc((void**)&deviceElements, sizeof(unsigned int) * d->totalElements))
        errorCheck(cudaMalloc((void**)&deviceCounts, sizeof(unsigned int) * combination(d->cardinality, 2)))
        errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combination(d->cardinality, 2)))
        DeviceTimer::finish(devMemAlloc);

        EventPair* dataTransfer = deviceTimer.add("Transfer to device");
        errorCheck(cudaMemcpy(deviceSizes, d->sizes, sizeof(unsigned int) * d->cardinality, cudaMemcpyHostToDevice))
        errorCheck(cudaMemcpy(deviceOffsets, deviceSizes, sizeof(unsigned int) * d->cardinality, cudaMemcpyDeviceToDevice))
        errorCheck(cudaMemcpy(deviceElements, d->elements, sizeof(unsigned int) * d->totalElements, cudaMemcpyHostToDevice))
        DeviceTimer::finish(dataTransfer);

        EventPair* setOffsets = deviceTimer.add("Compute set offsets");
        thrust::exclusive_scan(thrust::device, deviceOffsets, deviceOffsets + d->cardinality, deviceOffsets, 0); // in-place scan
        DeviceTimer::finish(setOffsets);

        EventPair* genBitmaps = deviceTimer.add("Generate set bitmaps");
        generateBitmaps<<<blocks, blockSize>>>
            (deviceBitmaps, bitmapWords, d->cardinality, deviceSizes, deviceOffsets, deviceElements);
        DeviceTimer::finish(genBitmaps);

        EventPair* interBitmap = deviceTimer.add("Bitmap intersection");
        intersectPerPair<<<blocks, blockSize>>>
            (deviceBitmaps, bitmapWords, d->cardinality, deviceCounts);
        DeviceTimer::finish(interBitmap);

        std::vector<unsigned int> counts(combination(d->cardinality, 2));

        EventPair* countTransfer = deviceTimer.add("Transfer result");
        errorCheck(cudaMemcpy(&counts[0], deviceCounts, sizeof(unsigned int) * combination(d->cardinality, 2), cudaMemcpyDeviceToHost))
        DeviceTimer::finish(countTransfer);

        EventPair* freeDevMem = deviceTimer.add("Free device memory");
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

__global__ void generateBitmaps(word* bitmaps, unsigned int numOfWords, unsigned int numOfSets,
                                const unsigned int* sizes, const unsigned int* offsets, const unsigned int* elements)
{
    unsigned int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int set = 0; set < numOfSets; set++) { // iterate all sets

        for (unsigned int i = globalID; i < sizes[set]; i += blockDim.x * gridDim.x) {
            unsigned int element = elements[offsets[set] + i];
            atomicOr(bitmaps + (set * numOfWords) + (element / WORD_BITS), 1ULL << (element % WORD_BITS));
        }
    }
}

__global__ void intersectPerPair(const word* words, unsigned int numOfWords, unsigned int numOfSets, unsigned int* counts)
{
    unsigned int globalID = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int a = 0; a < numOfSets - 1; a++) {

        for (unsigned int b = a + 1; b < numOfSets; b++) {
            unsigned intersection = 0;
            for (unsigned int i = globalID; i < numOfWords; i += blockDim.x * gridDim.x) {
                intersection += __popcll( words[a * numOfWords + i] & words[b * numOfWords + i] );
            }
            atomicAdd(counts + triangular_idx(numOfSets, a, b), intersection);
        }
    }
}