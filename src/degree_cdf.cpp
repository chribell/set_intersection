#include <vector>
#include <string>
#include <sstream>
#include <fmt/core.h>
#include "io.hpp"


typedef std::vector<unsigned int> uint_vector;
typedef std::vector<uint_vector> inverted_index;
typedef std::pair<unsigned int, unsigned int> degree_pair;
typedef std::pair<unsigned long long, unsigned long long> cdf_pair;

template <typename T>
void writePairs(std::vector<T>& pairs, std::string& path);

int main(int argc, char** argv) {

    std::string path = std::string(argv[1]);

    Dataset* d = readDataset(path);
    std::vector<uint_vector> dataset = datasetToCollection(d);

    inverted_index index(d->universe++);
    std::vector<degree_pair> degreeSet;
    std::vector<cdf_pair> setCDF;
    std::vector<degree_pair> degreeElement;
    std::vector<cdf_pair> elementCDF;
    unsigned int id = 0;
    for (auto& s : dataset) {
        for (auto& el : s) {
            index[el].push_back(id);
        }
        id++;
        degreeSet.push_back(std::make_pair(s.size(), id));
    }

    unsigned int invListSize = 0;
    for (auto& s : dataset) {
        for (auto& el : s) {
            invListSize += index[el].size();
        }
        setCDF.push_back(std::make_pair(s.size(), invListSize));
        invListSize = 0;
    }

    unsigned int element = 0;
    for (auto& invList : index) {
        elementCDF.push_back(std::make_pair(invList.size(), invList.size())); // this only for the self join scenario
        degreeElement.push_back(std::make_pair(invList.size(), element++));
    }

    sort(degreeSet.begin(), degreeSet.end(), [] (const degree_pair& a, degree_pair& b) {
        return a.first < b.first;
    });
    sort(degreeElement.begin(), degreeElement.end(), [] (const degree_pair& a, degree_pair& b) {
        return a.first < b.first;
    });

    sort(setCDF.begin(), setCDF.end(), [] (const cdf_pair& a, const cdf_pair& b) {
        return a.first < b.first;
    });
    sort(elementCDF.begin(), elementCDF.end(), [] (const cdf_pair& a, const cdf_pair& b) {
        return a.first < b.first;
    });

    for (int i = 0 ; i < elementCDF.size(); i++) {
        if (i == 0)
            elementCDF.at(i).second = elementCDF.at(i).second * elementCDF.at(i).second;
        if (i > 0)
            elementCDF.at(i).second = elementCDF.at(i - 1).second + elementCDF.at(i).second * elementCDF.at(i).second;
    }

    for (int i = 0 ; i < setCDF.size(); i++) {
        if (i > 0)
            setCDF.at(i).second += setCDF.at(i - 1).second;
    }

    std::string filename = argv[2];
    std::string outname = filename + "_degree_set.txt";

    writePairs(degreeSet, outname);
    outname = filename + "_degree_element.txt";
    writePairs(degreeElement, outname);

    outname = filename + "_set_cdf.txt";
    writePairs(setCDF, outname);
    outname = filename + "_element_cdf.txt";
    writePairs(setCDF, outname);

    fmt::print("Finished\n");
    return 0;
}

template <typename T>
void writePairs(std::vector<T>& pairs, std::string& path) {

    std::ofstream outfile;
    outfile.open(path.c_str());

    for (auto& pair : pairs) {
        fmt::print(outfile, "{} {}\n", pair.first, pair.second);
    }

    outfile.close();
}