#include <vector>
#include <string>
#include <sstream>
#include <fmt/core.h>
#include <algorithm>
#include "io.hpp"

std::vector<unsigned int> split(const std::string& s, char delimiter);
void readTextDataset(std::string& path, unsigned long& totalElements, unsigned int& universe, std::vector<std::vector<unsigned int>>& dataset);


int main(int argc, char** argv) {

    unsigned long totalElements = 0;
    unsigned int universe = 0;
    std::vector<std::vector<unsigned int>> dataset;

    std::string path = std::string(argv[1]);

    readTextDataset(path, totalElements, universe, dataset);

    std::string filename = argv[2];
    std::string outname = filename + "_asc.bin";
    fmt::print("Writing dataset to {}\n", outname);
    writeDataset(dataset.size(), universe, totalElements, dataset, outname);

    fmt::print("Sorting sets in descending order\n");

    std::sort(dataset.begin(), dataset.end(), [](const std::vector<unsigned int>& a, const std::vector<unsigned int>& b) {
        return a.size() > b.size();
    });


    outname = filename + "_desc.bin";
    fmt::print("Writing dataset to {}\n", outname);
    writeDataset(dataset.size(), universe, totalElements, dataset, outname);

    return 0;
}

std::vector<unsigned int> split(const std::string& s, char delimiter) {
    std::stringstream ss(s);
    std::string item;
    std::vector<unsigned int> elements;
    while (std::getline(ss, item, delimiter)) {
        elements.push_back(std::stoi(item));
    }
    return elements;
}

void readTextDataset(std::string& path, unsigned long& totalElements,
        unsigned int& universe, std::vector<std::vector<unsigned int>>& dataset) {

    std::ifstream infile;
    std::string line;
    infile.open(path.c_str());

    while (!infile.eof()) {
        std::getline(infile, line);
        if (line.empty()) continue;

        std::vector<unsigned int> set = split(line, ' ');

        if (universe < set[set.size() - 1]) {
            universe = set[set.size() - 1];
        }

        dataset.push_back(set);
        totalElements += set.size();
    }

    infile.close();
}