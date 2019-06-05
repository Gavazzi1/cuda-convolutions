#include <cuda.h>
#include <stdio.h>

__global__ void naive_kernel(float* d_in, int height, int width, 
                             float* filter, int radius, 
                             float* d_out) {
    // Get global position in grid
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // actual location within image data
    unsigned int loc = (y * width) + x;
    
    // sum of all element-wise multiplications
    float sum = 0;
    int filter_size = radius*2 + 1;

    // only perform convolution on pixels within radius
    // Global memory use and O(N^2) loop in kernel kill performance
    if (x >= radius && y >= radius && x < (width - radius) && y < (height - radius)) {
#pragma unroll
        for (int i = -radius; i <= radius; ++i) {
#pragma unroll
            for (int j = -radius; j <= radius; ++j) {
                // x, y, and global location adjusted for filter radius
                int img_x   = x + i;
                int img_y   = y + j;
                int img_loc = (img_y * width) + img_x;

                // filter location based just on x and y
                int filt_x     = i + radius;
                int filt_y     = j + radius;
                int filter_loc = filt_y * filter_size + filt_x;

                // add element-wise product to accumulator
                sum += d_in[img_loc] * filter[filter_loc];
            }
        }

        // add pixel value to output
        d_out[loc] = sum;
    }
}

void naivekernel(float* d_in, int height, int width,
                 float* filter, int radius,
                 float* d_out,
                 dim3 h_gridDim, dim3 h_blockDim) {
    naive_kernel<<<h_gridDim, h_blockDim>>>(d_in, height, width,
                                            filter, radius,
                                            d_out);
}
