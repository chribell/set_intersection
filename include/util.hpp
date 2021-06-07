#ifndef UTIL_HPP
#define UTIL_HPP

#pragma once

#include <string>
#include <sstream>
#include <iomanip>

unsigned long long combination(unsigned int n, unsigned int k)
{
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;

    unsigned long long result = n;
    for( unsigned long long i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;
}

unsigned int index(unsigned int n, unsigned i, unsigned j) {
    return (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
}

std::string formatBytes(size_t bytes)
{
    size_t gb = 1073741824;
    size_t mb = 1048576;
    size_t kb = 1024;

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3);

    if (bytes >= gb) {
        stream << ((double) bytes / (double) gb) << " GB";
    } else if (bytes >= mb) {
        stream << ((double) bytes / (double) mb) << " MB";
    } else if (bytes >= kb) {
        stream << ((double) bytes / (double) kb) << " KB";
    } else {
        stream << bytes << " bytes";
    }

    return stream.str();
}

#endif //UTIL_HPP
