#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "errchk.h"
#include "imgutils.h"

#define FILTER_SIZE 3
#define RADIUS ((FILTER_SIZE - 1) / 2)
#define BLOCK_SIZE 16
#define CHANNELS 3

#define DEBUG 0

__global__ void naive_kernel(float* d_in, int height, int width, float* filter, float* d_out) {
    // Get global position in grid
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = threadIdx.z;

    // actual location within image data
    // since image data is interleaved RGB values, offset like you would a 2D
    // image, multiply that by the number of channels (3) and add the z value
    // representing whether the pixel is R, G, or B
    unsigned int loc = CHANNELS * (y * width + x) + z;

    // sum of all element-wise multiplications
    float sum = 0;

    // only perform convolution on pixels within radius
    // Global memory use and O(N^2) loop in kernel kill performance
    if (x >= RADIUS && y >= RADIUS && x < (width - RADIUS) && y < (height - RADIUS)) {
        int img_z = z;
        for (int i = -RADIUS; i <= RADIUS; ++i) {
            for (int j = -RADIUS; j <= RADIUS; ++j) {
                // x, y, and global location adjusted for filter radius
                int img_x   = x + i;
                int img_y   = y + j;
                int img_loc = CHANNELS * (img_y * width + img_x) + img_z;

                // filter location based just on x and y
                int filt_x     = i + RADIUS;
                int filt_y     = j + RADIUS;
                int filter_loc = filt_y * FILTER_SIZE + filt_x;

                // add element-wise product to accumulator
                sum += d_in[img_loc] * filter[filter_loc];
            }
        }

        // add pixel value to output
        d_out[loc] = sum;

#if DEBUG
        if ((d_in[loc] - 0.0) > 0.001) {
            printf("x=%d, y=%d, z=%d, loc=%d, d_in=%f, d_out=%f\n", 
                   x, y, z, loc, d_in[loc], d_out[loc]);
        }
#endif
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./naive_conv <image>\n");
        return 0;
    }

    // read in image
    cv::Mat h_in     = read_image(argv[1]);
    int     height   = h_in.rows;
    int     width    = h_in.cols;
    int     channels = h_in.channels();

#if DEBUG
    printf("width=%d, height=%d, channels=%d, FILTER_SIZE=%d\n", 
           width, height, channels, FILTER_SIZE);
#endif

    // Declare image and filter variables for host and device
    float *h_filter, *h_out, *d_in, *d_filter, *d_out;

    // size to allocate for image and filter variables
    unsigned int img_size         = width * height * CHANNELS * sizeof(float);
    unsigned int full_filter_size = FILTER_SIZE * FILTER_SIZE * sizeof(float);

#if DEBUG
    printf("img_size=%u, full_filter_size=%u\n", img_size, full_filter_size);
#endif

    // Allocate host data
    h_filter = (float*)malloc(full_filter_size);
    h_out    = (float*)malloc(img_size);

    // Initialize filter template
    // clang-format off
    const float filt_template[FILTER_SIZE][FILTER_SIZE] = {
        {0,  0, 0},
        {0,  1, 0},
        {0,  0, 0}
    };
    // clang-format on

    // copy filter template to actual filter (maybe redundant)
    for (int row = 0; row < FILTER_SIZE; ++row) {
        for (int col = 0; col < FILTER_SIZE; ++col) {
            int idx       = row * FILTER_SIZE + col;
            h_filter[idx] = filt_template[row][col];
        }
    }

    // Allocate device data
    cudaMalloc((void**)&d_in, img_size);
    cudaMalloc((void**)&d_filter, full_filter_size);
    cudaMalloc((void**)&d_out, img_size);

    // Copy host memory to device
    cudaMemcpy(d_in, h_in.data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, full_filter_size, cudaMemcpyHostToDevice);

    // Let grid size be based on block size
    // Have just enough blocks to cover whole image
    // The -1 is to cover the case where image dimensions are multiples of
    // BLOCKS_SIZE
    int gridXSize = 1 + ((width - 1) / BLOCK_SIZE);
    int gridYSize = 1 + ((height - 1) / BLOCK_SIZE);
#if DEBUG
    printf("gridXSize=%d, gridYSize=%d, BLOCK_SIZE=%d\n", 
           gridXSize, gridYSize, BLOCK_SIZE);
#endif
    dim3 h_gridDim(gridXSize, gridYSize);
    dim3 h_blockDim(BLOCK_SIZE, BLOCK_SIZE, CHANNELS);

    // Run on GPU 0
    cudaSetDevice(0);

    // Timing stuff
    cudaEvent_t start, stop;
    float       time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel call
    cudaEventRecord(start, 0);

    naive_kernel<<<h_gridDim, h_blockDim>>>(d_in, height, width, d_filter, d_out);
    cudaDeviceSynchronize();

    // Get time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel time = %.2f ms\n", time);

    // Copy result back to host
    gpuErrchk(cudaMemcpy(h_out, d_out, img_size, cudaMemcpyDeviceToHost));

    // write image to file for displaying
    save_image("output.png", h_out, height, width);

    // Free device data
    cudaFree(d_in);
    cudaFree(d_filter);
    cudaFree(d_out);

    // Free host data
    free(h_filter);
    free(h_out);
}
