#include <fmt/core.h>
#include <cxxopts.hpp>
#include "io.hpp"
#include "host_timer.hpp"


unsigned int serialIntersect(std::vector<unsigned int>& a, std::vector<unsigned int>& b);


int main(int argc, char** argv) {
    try {
        fmt::print("{}\n", "CPU serial intersection");

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
                counts[index(d->cardinality, a, b)] = serialIntersect(sets[a], sets[b]);
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

unsigned int serialIntersect(std::vector<unsigned int>& a, std::vector<unsigned int>& b) {
    unsigned int aSize = a.size();
    unsigned int bSize = b.size();
    unsigned int aIter = 0;
    unsigned int bIter = 0;

    unsigned int count = 0;

    while(aIter < aSize && bIter < bSize) {
        if(a[aIter] == b[bIter]) {
            ++aIter;
            ++bIter;
            ++count;
        } else if (a[aIter] < b[bIter]) {
            ++aIter;
        } else {
            ++bIter;
        }
    }
    return count;
}

