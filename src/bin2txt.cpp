#include <vector>
#include <string>
#include <sstream>
#include <fmt/core.h>
#include "io.hpp"

void writeTextDataset(std::vector<std::vector<unsigned int>>& dataset, std::string& path);


int main(int argc, char** argv) {

    std::string path = std::string(argv[1]);

    Dataset* d = readDataset(path);
    std::vector<std::vector<unsigned int>> dataset = datasetToCollection(d);

    std::string filename = argv[2];
    std::string outname = filename + ".txt";
    fmt::print("Writing dataset to {}\n", outname);
    writeTextDataset(dataset, outname);

    return 0;
}

void writeTextDataset(std::vector<std::vector<unsigned int>>& dataset, std::string& path) {

    std::ofstream outfile;
    outfile.open(path.c_str());

    for (auto& set : dataset) {
        std::copy(set.cbegin(), set.cend(),
                  std::ostream_iterator<int>(outfile, " "));
        outfile << std::endl;
    }

    outfile.close();
}