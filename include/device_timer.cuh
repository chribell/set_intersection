
#ifndef DEVICE_TIMER_CUH
#define DEVICE_TIMER_CUH

#pragma once

#include <string>
#include <algorithm>
#include <vector>
#include <fmt/core.h>

struct EventPair {
    std::string name;
    cudaEvent_t start;
    cudaEvent_t end;
    cudaStream_t stream;
    EventPair(std::string const& name, cudaStream_t const& stream) : name(name), stream(stream)  {}
};

struct DeviceTimer {
    std::vector<EventPair*> pairs;
    EventPair* add(std::string const& name, cudaStream_t const& stream = 0) {
        auto pair = new EventPair(name, stream);
        cudaEventCreate(&(pair->start));
        cudaEventCreate(&(pair->end));

        cudaEventRecord(pair->start, stream);

        pairs.push_back(pair);
        return pair;
    }
    static void finish(EventPair* pair) {
        cudaEventRecord(pair->end, pair->stream);
    }
    float sum(std::string const &name) const {
        float total = 0.0;
        auto it = pairs.begin();
        for(; it != pairs.end(); ++it) {
            if ((*it)->name == name) {
                float millis = 0.0;
                cudaEventElapsedTime(&millis, (*it)->start, (*it)->end);
                total += millis;
            }
        }
        return total;
    }
    float total() const {
        float total = 0.0;
        auto it = pairs.begin();
        for(; it != pairs.end(); ++it) {
            float millis = 0.0;
            cudaEventElapsedTime(&millis, (*it)->start, (*it)->end);
            total += millis;
        }
        return total;
    }
    void print() {
        fmt::print("┌{0:─^{1}}┐\n", "Device Timings (in ms)", 51);
        std::vector<std::string> distinctNames;
        for(auto& pair : pairs) {
            if (std::find(distinctNames.begin(), distinctNames.end(), pair->name) == distinctNames.end())
                distinctNames.push_back(pair->name);
        }
        for(auto& name : distinctNames) {
            fmt::print("│{1: ^{0}}|{2: ^{0}}│\n", 25, name, sum(name));
        }
        fmt::print("└{1:─^{0}}┘\n", 51, "");
        fmt::print("│{1: ^{0}}|{2: ^{0}}│\n", 25, "Total", total());
        fmt::print("└{1:─^{0}}┘\n", 51, "");
    }
    ~DeviceTimer() {
        for (auto& pair : pairs) {
            cudaEventDestroy(pair->start);
            cudaEventDestroy(pair->end);
            delete pair;
        }
    };
};

#endif // DEVICE_TIMER_CUH
