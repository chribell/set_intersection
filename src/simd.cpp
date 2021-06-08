#include <fmt/core.h>
#include "io.hpp"
#include "util.hpp"
#include "host_timer.hpp"
#include <algorithm>
#include <cxxopts.hpp>
#include "codecfactory.h"
#include "intersection.h"
#include <chrono>
#include <string>
#include <fstream>

int main(int argc, char** argv) {
    try {
        fmt::print("{}\n", "SIMD CPU set intersection");

        std::string input;
        std::string output;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("input", "Input dataset path", cxxopts::value<std::string>(input))
                ("output", "Output result path", cxxopts::value<std::string>(output))
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

        SIMDCompressionLib::intersectionfunction inter = SIMDCompressionLib::IntersectionFactory::getFromName("simd"); // using SIMD intersection

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

        HostTimer hostTimer;

        std::vector<std::vector<unsigned int>> sets = datasetToCollection(d);

        std::vector<unsigned int> counts(combination(sets.size(), 2));

        Interval* setInter = hostTimer.add("std::set intersection");
        for (unsigned int a = 0; a < d->cardinality - 1; a++) {
            for (unsigned int b = a + 1; b < d->cardinality; b++) {
                std::vector<unsigned int> v(std::min(sets[a].size(), sets[b].size()));
                size_t intersize = inter(sets[a].data(), sets[a].size(), sets[b].data(), sets[b].size(), v.data());
                v.resize(intersize);
                v.shrink_to_fit();
                counts[triangular_index(d->cardinality, a, b)] = v.size();
            }
        }
        HostTimer::finish(setInter);

        hostTimer.print();

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