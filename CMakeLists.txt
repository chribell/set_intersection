cmake_minimum_required(VERSION 3.18)
project(set_intersection_eval CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(SM_ARCH 61 CACHE STRING "CUDA SM architecture e.g. 50, 52, 61")

include(FetchContent)

# CXX options library
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        v2.2.1
)
FetchContent_MakeAvailable(cxxopts)
# Adds cxxopts

# Formatting library
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        7.1.2
)
FetchContent_MakeAvailable(fmt)
# Adds fmt::fmt

# Zipf header library
FetchContent_Declare(
        zipf
        GIT_REPOSITORY https://github.com/llersch/cpp_random_distributions.git
        GIT_TAG        "6ae38ecf5dbeb1150b80ecf051cb43cb88ba697a"
)
FetchContent_MakeAvailable(zipf)

# Progress bar library
FetchContent_Declare(
        progressbar
        GIT_REPOSITORY https://github.com/prakhar1989/progress-cpp.git
        GIT_TAG        "7bfba0d22d19c41323aa35541618b6ebec9d737c"
)
FetchContent_MakeAvailable(progressbar)

# Progress bar library
FetchContent_Declare(
        simdcompressionlib
        GIT_REPOSITORY https://github.com/lemire/SIMDCompressionAndIntersection.git
        GIT_TAG        "b666a60c8fca18227d6532fb2d3b4d4dbc466cc9"
)
FetchContent_MakeAvailable(simdcompressionlib) #SIMDCompressionLib

find_package(OpenMP)

# Generate dataset binary
add_executable(generate_dataset src/generate_dataset.cpp)
target_include_directories(generate_dataset PRIVATE include ${zipf_SOURCE_DIR} ${progressbar_SOURCE_DIR}/include/progresscpp/)
target_link_libraries(generate_dataset PRIVATE cxxopts fmt::fmt)

# Text to bin binary
add_executable(txt2bin src/txt2bin.cpp)
target_include_directories(txt2bin PRIVATE include)
target_link_libraries(txt2bin PRIVATE fmt::fmt)

# Bin to text binary
add_executable(bin2txt src/bin2txt.cpp)
target_include_directories(bin2txt PRIVATE include)
target_link_libraries(bin2txt PRIVATE fmt::fmt)

# Plain vector intersection binary
add_executable(stdintersection src/std_set_intersection.cpp)
target_link_libraries(stdintersection PRIVATE OpenMP::OpenMP_CXX fmt::fmt cxxopts)
target_include_directories(stdintersection PRIVATE include)

# Boost bitset binary
add_executable(boost_bitset src/boost_bitset.cpp)
target_link_libraries(boost_bitset PRIVATE OpenMP::OpenMP_CXX fmt::fmt cxxopts)
target_include_directories(boost_bitset PRIVATE include)
target_compile_options(boost_bitset PRIVATE -fpic -mavx -std=c++11  -Weffc++ -pedantic -O3 -Wall -fopenmp -Wextra -Wsign-compare  -Wwrite-strings -Wpointer-arith -Winit-self  -Wno-sign-conversion)

# Serial intersect binary
add_executable(serial_intersect src/serial_intersect.cpp)
target_link_libraries(serial_intersect PRIVATE OpenMP::OpenMP_CXX fmt::fmt cxxopts)
target_include_directories(serial_intersect PRIVATE include)

# SIMD CPU intersect binary
add_executable(simd src/simd.cpp ${simdcompressionlib_SOURCE_DIR}/src/intersection.cpp)
target_link_libraries(simd PRIVATE OpenMP::OpenMP_CXX fmt::fmt cxxopts)
target_include_directories(simd PRIVATE include ${simdcompressionlib_SOURCE_DIR}/include)
target_compile_options(simd PRIVATE -fpic -mavx -std=c++11  -Weffc++ -pedantic -O3 -Wall -fopenmp -Wextra -Wsign-compare  -Wwrite-strings -Wpointer-arith -Winit-self  -Wno-sign-conversion)

# Bitmap-based naive binary
add_executable(bitmap_naive src/bitmap_naive.cu)
target_link_libraries(bitmap_naive PRIVATE fmt::fmt cxxopts)
target_include_directories(bitmap_naive PRIVATE include)
set_target_properties(
        bitmap_naive
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})

# Intersect-Path binary
add_executable(ip src/ip.cu)
target_link_libraries(ip PRIVATE fmt::fmt cxxopts)
target_include_directories(ip PRIVATE include)
set_target_properties(
        ip
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(ip PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)

# Bitmap-based dynamic binary
add_executable(bitmap_dynamic src/bitmap_dynamic.cu)
target_link_libraries(bitmap_dynamic PRIVATE fmt::fmt cxxopts)
target_include_directories(bitmap_dynamic PRIVATE include)
set_target_properties(
        bitmap_dynamic
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(bitmap_dynamic PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

# Hash-based intersection binary
add_executable(hi src/hi.cu)
target_link_libraries(hi PRIVATE cxxopts fmt::fmt)
target_include_directories(hi PRIVATE include)
set_target_properties(
        hi
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(hi PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

# Optimized binary search binary
add_executable(obs src/obs.cu)
target_link_libraries(obs PRIVATE cxxopts fmt::fmt)
target_include_directories(obs PRIVATE include)
set_target_properties(
        obs
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(obs PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)

# Optimized binary search binary
add_executable(mm src/matrix_multiplication.cu)
target_link_libraries(mm PRIVATE cxxopts fmt::fmt -lcublas)
target_include_directories(mm PRIVATE include)
set_target_properties(
        mm
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${SM_ARCH})
target_compile_options(mm PRIVATE -m64 --expt-extended-lambda --expt-relaxed-constexpr)



