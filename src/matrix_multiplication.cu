#include <fmt/core.h>
#include "io.hpp"
#include <cxxopts.hpp>
#include "helpers.cuh"
#include "device_timer.cuh"
#include <thrust/scan.h>
#include "util.hpp"
#include "host_timer.hpp"
#include <cublas_v2.h>

void populateBinaryMatrix(float* input, Dataset* dataset, unsigned int start, unsigned int end);
void populateBinaryTransposeMatrix(float* input, Dataset* dataset, unsigned int partition, unsigned int start, unsigned int end);

int main(int argc, char** argv) {
    try {
        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{2: ^{1}}|\n"
                "└{0:─^{1}}┘\n", "", 51, "GPU Matrix multiplication set intersection"
        );

        int multiprocessorCount;
        int maxThreadsPerBlock;

        cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        size_t freeDeviceMemory, totalDeviceMemory;

        cudaMemGetInfo(&freeDeviceMemory, &totalDeviceMemory);

        // arguments
        std::string input;
        std::string output;
        unsigned int partition = 10000;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("input", "Input dataset path", cxxopts::value<std::string>(input))
                ("output", "Output result path", cxxopts::value<std::string>(output))
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

        d->universe++;
        partition = std::min(d->cardinality, partition);


        d->offsets = new unsigned int[d->cardinality];
        // calculate offsets
        thrust::exclusive_scan(thrust::host, d->sizes, d->sizes + d->cardinality, d->offsets, 0);

        std::vector<tile> tiles = splitToTiles(d->cardinality, partition);
        std::vector<tile_pair> runs = findTilePairs(d->cardinality, partition);

        size_t combinations = partition * partition;

        unsigned long long outputMemory = runs.size() * combinations * sizeof(float);

        unsigned long long deviceMemory = (sizeof(float) * d->universe * partition * 2)
                                          + (sizeof(float) * combinations);

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "│{9: ^{2}}|{10: ^{2}}│\n"
                "└{11:─^{1}}┘\n", "Launch info", 51, 25,
                "Partition", partition,
                "GPU invocations", runs.size(),
                "Required memory (Output)", formatBytes(outputMemory),
                "Required memory (GPU)", formatBytes(deviceMemory),
                ""
        );

        if (deviceMemory > freeDeviceMemory) {
            fmt::print("Error not enough GPU memory ({})!\nExiting...", formatBytes(freeDeviceMemory));
            return 1;
        }

        HostTimer hostTimer;
        DeviceTimer deviceTimer;

        Interval* hostMemAlloc = hostTimer.add("Allocate memory");
        // errorCheck(cudaMallocHost((void**)&hostCounts, runs.size() * combinations * sizeof(float))))
        std::vector<float> counts(runs.size() * combinations);
        float* hostInput = new float[d->universe * partition];
        float* hostInvInput = new float[d->universe * partition];
        HostTimer::finish(hostMemAlloc);

        Interval* clearMem = hostTimer.add("Clear memory");
        memset(hostInput, 0, d->universe * partition * sizeof(float));
        memset(hostInvInput, 0, d->universe * partition * sizeof(float));
        HostTimer::finish(clearMem);

        float* devInput;
        float* devInvInput;
        float* devOutput;

        EventPair* devMemAlloc = deviceTimer.add("Device memory allocation");
        errorCheck(cudaMalloc((void**) &devInput, d->universe * partition * sizeof(float)))
        errorCheck(cudaMalloc((void**) &devInvInput, d->universe * partition * sizeof(float)))
        errorCheck(cudaMalloc((void**) &devOutput, combinations * sizeof(float)))
        DeviceTimer::finish(devMemAlloc);

        cublasHandle_t handle;
        cublasCreate_v2(&handle);

        float alpha = 1.0;
        float beta = 0.0;

        unsigned int iter = 0;
        for (unsigned int i = 0; i < tiles.size(); ++i) {
            tile& A = tiles[i];

            Interval* binaryTileA = hostTimer.add("Create binary matrix");
            populateBinaryMatrix(hostInput, d, A.start, A.end);
            HostTimer::finish(binaryTileA);

            EventPair* tileATransfer = deviceTimer.add("Transfer to device");
            errorCheck(cudaMemcpy(devInput, hostInput, d->universe * partition * sizeof(float), cudaMemcpyHostToDevice))
            DeviceTimer::finish(tileATransfer);

            for (unsigned int j = i; j < tiles.size(); ++j) {
                tile& B = tiles[j];
                Interval* binaryTileB = hostTimer.add("Create binary matrix");
                populateBinaryTransposeMatrix(hostInvInput, d, partition, B.start, B.end);
                HostTimer::finish(binaryTileB);

                EventPair* tileBTransfer = deviceTimer.add("Transfer to device");
                errorCheck(cudaMemcpy(devInvInput, hostInvInput, d->universe * partition * sizeof(float), cudaMemcpyHostToDevice))
                DeviceTimer::finish(tileBTransfer);

                EventPair* matrixMultiplication = deviceTimer.add("Matrix multiplication");
                auto status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          B.length, A.length, d->universe,
                                          &alpha, devInvInput, B.length,
                                          devInput, d->universe,
                                          &beta, devOutput, B.length);
                DeviceTimer::finish(matrixMultiplication);

                EventPair *countTransfer = deviceTimer.add("Transfer result");
                errorCheck(cudaMemcpy(&counts[0] + (iter * combinations), devOutput, sizeof(float) * combinations,
                                      cudaMemcpyDeviceToHost))
                DeviceTimer::finish(countTransfer);

                clearMem = hostTimer.add("Clear memory");
                // clear binary matrix to ensure correctness
                memset(hostInvInput, 0, d->universe * partition * sizeof(float));
                HostTimer::finish(clearMem);
                // transfer result back to host
                iter++;
            }
            clearMem = hostTimer.add("Clear memory");
            // clear binary matrix to ensure correctness
            memset(hostInput, 0, d->universe * partition * sizeof(float));
            HostTimer::finish(clearMem);
        }

        EventPair* freeDevMem = deviceTimer.add("Free device memory");
        errorCheck(cudaFree(devInput))
        errorCheck(cudaFree(devInvInput))
        errorCheck(cudaFree(devOutput))
        cublasDestroy_v2(handle);
        DeviceTimer::finish(freeDevMem);

        cudaDeviceSynchronize();

        hostTimer.print();
        deviceTimer.print();

        if (!output.empty()) {
            fmt::print("Writing result to file {}\n", output);
            writeResult<float, true>(runs, partition, counts, output);
            fmt::print("Finished\n");
        }
    } catch (const cxxopts::OptionException& e) {
        fmt::print("{}\n", e.what());
        return 1;
    }
    return 0;
}


void populateBinaryMatrix(float* input, Dataset* d, unsigned int start, unsigned int end)
{
    unsigned int idx = 0;
    for (unsigned int i = start; i < end; ++i) {
        for (size_t j = d->offsets[i]; j < d->offsets[i] + d->sizes[i]; ++j) {
            input[idx * d->universe + d->elements[j]] = 1.0f;
        }
        idx++;
    }
}

void populateBinaryTransposeMatrix(float* input, Dataset* d, unsigned int partition, unsigned int start, unsigned int end)
{
    unsigned int idx = 0;
    for (unsigned int i = start; i < end; ++i) {
        for (size_t j = d->offsets[i]; j < d->offsets[i] + d->sizes[i]; ++j) {
            input[d->elements[j] * partition + idx] = 1.0f;
        }
        idx++;
    }
}