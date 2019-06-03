#include <cuda.h>
#include <stdio.h>

__global__ void naive_kernel(float* d_in, int height, int width, int channels, 
                             float* filter, int radius, 
                             float* d_out) {
    // Get global position in grid
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = threadIdx.z;

    // actual location within image data
    // since image data is interleaved RGB values, offset like you would a 2D
    // image, multiply that by the number of channels (3) and add the z value
    // representing whether the pixel is R, G, or B
    unsigned int loc = channels * (y * width + x) + z;
    
    // sum of all element-wise multiplications
    float sum = 0;
    int filter_size = radius*2 + 1;

    // only perform convolution on pixels within radius
    // Global memory use and O(N^2) loop in kernel kill performance
    if (x >= radius && y >= radius && x < (width - radius) && y < (height - radius)) {
        int img_z = z;
#pragma unroll
        for (int i = -radius; i <= radius; ++i) {
#pragma unroll
            for (int j = -radius; j <= radius; ++j) {
                // x, y, and global location adjusted for filter radius
                int img_x   = x + i;
                int img_y   = y + j;
                int img_loc = channels * (img_y * width + img_x) + img_z;

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

#if 0
        if ((d_in[loc] - 0.0) > 0.001) {
            printf("x=%d, y=%d, z=%d, loc=%d, d_in=%f, d_out=%f\n", 
                   x, y, z, loc, d_in[loc], d_out[loc]);
        }
#endif
    }
}

void naivekernel(float* d_in, int height, int width, int channels,
                 float* filter, int radius,
                 float* d_out,
                 dim3 h_gridDim, dim3 h_blockDim) {
    naive_kernel<<<h_gridDim, h_blockDim>>>(d_in, height, width, channels,
                                            filter, radius,
                                            d_out);
}
