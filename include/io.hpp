#ifndef IO_HPP
#define IO_HPP

#pragma once

#include <fstream>
#include <string>
#include <fmt/ostream.h>
#include "util.hpp"

struct Dataset {
    unsigned int cardinality; // k
    unsigned int universe;
    unsigned long totalElements;
    unsigned int* sizes;
    unsigned int* elements;

    ~Dataset() {
        delete[] sizes;
        delete[] elements;
    }
};

Dataset* readDataset(std::string& path)
{
    Dataset* d = new Dataset;

    std::ifstream infile;
    infile.open(path, std::ios::binary | std::ios::in);
    infile.read((char*)&(d->cardinality), sizeof(d->cardinality));
    infile.read((char*)&(d->universe), sizeof(d->universe));
    infile.read((char*)&(d->totalElements), sizeof(d->totalElements));

    d->sizes = new unsigned int[d->cardinality];

    for (unsigned int i = 0; i < d->cardinality; ++i) {
        unsigned int tmp;
        infile.read((char*)&tmp, sizeof(tmp));
        d->sizes[i] = tmp;
    }

    d->elements = new unsigned int[d->totalElements];

    for (unsigned long i = 0; i < d->totalElements; ++i) {
        unsigned int tmp;
        infile.read((char*)&tmp, sizeof(tmp));
        d->elements[i] = tmp;
    }

    infile.close();

    return d;
}

std::vector<std::vector<unsigned int>> datasetToCollection(Dataset* d) {
    std::vector<std::vector<unsigned int>>  sets(d->cardinality);
    unsigned long offset = 0;
    // initialize sets
    for (unsigned int i = 0; i < d->cardinality; ++i) {
        sets[i].reserve(d->sizes[i]);
        for (unsigned int j = 0; j < d->sizes[i]; ++j) {
            sets[i].push_back(d->elements[offset + j]);
        }
        offset += d->sizes[i];
    }
    return sets;
}

void writeDataset(unsigned int k, unsigned int universe, unsigned long totalElements,
                  std::vector<std::vector<unsigned int>>& dataset, std::string& path)
{
    std::ofstream outfile;
    outfile.open(path, std::ios::binary | std::ios::out);
    outfile.write((char*)&k, sizeof(k));
    outfile.write((char*)&universe, sizeof(universe));
    outfile.write((char*)&totalElements, sizeof(totalElements));
    // write set lengths
    for (const auto& set : dataset) {
        unsigned int tmp = set.size();
        outfile.write((char*)&tmp, sizeof(tmp));
    }
    for (const auto& set : dataset) {
        for (unsigned int tmp : set) {
            outfile.write((char*)&tmp, sizeof(tmp));
        }
    }
    outfile.close();
}

void writeResult(unsigned int k, std::vector<unsigned int> counts, std::string& output) {
    std::ofstream file;
    file.open(output.c_str());

    for (unsigned int a = 0; a < k; a++) {
        for (unsigned int b = a + 1; b < k; b++) {
            fmt::print(file, "({},{}): {}\n", a + 1, b + 1, counts[index(k, a, b)]);
        }
    }

    file.close();
}

#endif //IO_HPP
