#ifndef UTIL_HPP
#define UTIL_HPP

unsigned long combination(unsigned int n, unsigned int k)
{
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;

    unsigned int result = n;
    for( int i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;
}

unsigned int index(unsigned int n, unsigned i, unsigned j) {
    return (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
}

#endif //UTIL_HPP
