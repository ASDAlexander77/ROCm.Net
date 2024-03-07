#include "utils.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <string>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

// SAXPY kernel stored as a string
static constexpr auto saxpy_kernel{
    // MSVC 19.16 does not properly handle R-strings in its preprocessor when they are on a separate line,
    // if the /E flag is passed (as NVCC does).
    "#include \"test_header.h\"\n"
    "#include \"test_header1.h\"\n"
    R"(
extern "C"
__global__ void saxpy_kernel(const real a, const realptr d_x, realptr d_y, const unsigned int size)
{
    const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx < size)
    {
        d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
    }
}
)"};

int main()
{

    return 0;
}
