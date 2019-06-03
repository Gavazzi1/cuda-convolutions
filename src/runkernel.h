#pragma once

#include <cuda.h>

void naivekernel(float* d_in, int height, int width, int channels,
                 float* filter, int radius,
                 float* d_out,
                 dim3 h_gridDim, dim3 h_blockDim);
