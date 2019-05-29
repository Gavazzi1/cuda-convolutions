#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <helper_cuda.h>
#include <helper_timer.h>

#include "imgutils.h"

#define RADIUS 1
#define FILTER_SIZE ((RADIUS * 2) + 1)
#define BLOCK_SIZE 16
#define ITERATIONS 128

#define PRINT 1
#define RANDOM 0

__global__ void kernel(float* d_in, int height, int width, int channels, float* filter, float* d_out) {
    // shared memory block - accessible to all threads in a block
    // for each pixel in a block, you need access to all pixels in the block
    // plus one radius of pixels on all sides
    __shared__ float sh_data[BLOCK_SIZE + 2*RADIUS][BLOCK_SIZE + 2*RADIUS];

    // Get global position in grid
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = threadIdx.z;

    // actual location within image data
    // since image data is interleaved RGB values, offset like you would a 2D
    // image, multiply that by the number of channels (3) and add the z value
    // representing whether the pixel is R, G, or B
    unsigned int loc = channels * (y * width + x) + z;

    // Copy into shared memory
    sh_data[threadIdx.x][threadIdx.y] = (x-RADIUS < 0 || y-RADIUS < 0) ?
                                        0 :
                                        d_in[];

    // sum of all element-wise multiplications
    float sum = 0;

    // only perform convolution on pixels within radius
    // Global memory use and O(N^2) loop in kernel kill performance
    if (x >= RADIUS && y >= RADIUS && x < (width - RADIUS) && y < (height - RADIUS)) {
        int img_z = z;
#pragma unroll
        for (int i = -RADIUS; i <= RADIUS; ++i) {
#pragma unroll
            for (int j = -RADIUS; j <= RADIUS; ++j) {
                // x, y, and global location adjusted for filter radius
                int img_x   = x + i;
                int img_y   = y + j;
                int img_loc = channels * (img_y * width + img_x) + img_z;

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

#if 0
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
    cv::Mat h_in     = read_image_bw(argv[1]);
    int     height   = h_in.rows;
    int     width    = h_in.cols;
    int     channels = h_in.channels();

#if PRINT
    printf("width=%d, height=%d, channels=%d, FILTER_SIZE=%d\n", 
           width, height, channels, FILTER_SIZE);
#endif

    // Declare image and filter variables for host and device
    float *h_filter, *h_out, *d_in, *d_filter, *d_out;

    // size to allocate for image and filter variables
    unsigned int img_size         = width * height * channels * sizeof(float);
    unsigned int full_filter_size = FILTER_SIZE * FILTER_SIZE * sizeof(float);

#if PRINT
    printf("img_size=%u, full_filter_size=%u\n", img_size, full_filter_size);
#endif

    // Allocate host data
    h_filter = (float*)malloc(full_filter_size);
    h_out    = (float*)malloc(img_size);

    // copy filter template to actual filter (maybe redundant)
#if RANDOM
    srand(200);
#else
    // Initialize filter template
    // clang-format off
    const float filt_template[FILTER_SIZE][FILTER_SIZE] = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };
    // clang-format on
#endif

    for (int row = 0; row < FILTER_SIZE; ++row) {
        for (int col = 0; col < FILTER_SIZE; ++col) {
            int idx = row * FILTER_SIZE + col;
#if RANDOM
            h_filter[idx] = (float)(rand() % 16);
#else
            h_filter[idx] = filt_template[row][col];
#endif
        }
    }

    // Allocate device data
    checkCudaErrors(cudaMalloc((void**)&d_in, img_size));
    checkCudaErrors(cudaMalloc((void**)&d_filter, full_filter_size));
    checkCudaErrors(cudaMalloc((void**)&d_out, img_size));

    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_in, h_in.data, img_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, full_filter_size, cudaMemcpyHostToDevice));

    // Let grid size be based on block size
    // Have just enough blocks to cover whole image
    // The -1 is to cover the case where image dimensions are multiples of
    // BLOCKS_SIZE
    int gridXSize = 1 + ((width - 1) / BLOCK_SIZE);
    int gridYSize = 1 + ((height - 1) / BLOCK_SIZE);
#if PRINT
    printf("gridXSize=%d, gridYSize=%d, BLOCK_SIZE=%d\n", 
           gridXSize, gridYSize, BLOCK_SIZE);
#endif
    dim3 h_gridDim(gridXSize, gridYSize);
    dim3 h_blockDim(BLOCK_SIZE, BLOCK_SIZE, channels);

    // Run on GPU 0
    cudaSetDevice(0);

    // Timing stuff
    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    // Kernel call
    for (int i = -1; i < ITERATIONS; ++i) {

        if (i == 0) {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }

        kernel<<<h_gridDim, h_blockDim>>>(d_in, height, width, channels, d_filter, d_out);
    }

    // Get time
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double time = sdkGetTimerValue(&hTimer) / (double)ITERATIONS;
    printf("Kernel time = %.5f ms\n", time);

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_out, d_out, img_size, cudaMemcpyDeviceToHost));

    // write image to file for displaying
    save_image_bw("output.png", h_out, height, width);

    // Free device data
    cudaFree(d_in);
    cudaFree(d_filter);
    cudaFree(d_out);

    // Free host data
    free(h_filter);
    free(h_out);
}
