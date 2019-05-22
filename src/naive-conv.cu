#include <cuda.h>
#include <time.h>
#include <cstdlib>
#include <iostream>

#define FILTER_SIZE 3
#define RADIUS ((FILTER_SIZE - 1) / 2)
#define BLOCK_SIZE 16

#define DEBUG 0

__global__ void naive_kernel(float* d_in, int height, int width, float* filter, float* d_out) {
    // Get global position in image
    unsigned int x   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y   = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int loc = y * blockDim.x + x;

    // sum of all element-wise multiplications
    float sum = 0;

    // only perform convolution on pixels within radius
    // Global memory use and O(N^2) loop in kernel kill performance
    if (x >= RADIUS && y >= RADIUS && x < (width - RADIUS) && y <= (height - RADIUS)) {
#if DEBUG
        printf("x=%d, y=%d, loc=%d, value=%f\n", x, y, loc, d_in[loc]);
#endif
        for (int i = -RADIUS; i <= RADIUS; ++i) {
            for (int j = -RADIUS; j <= RADIUS; ++j) {
                // x, y, and global location adjusted for filter radius
                int img_x   = x + i;
                int img_y   = y + j;
                int img_loc = y_new * width * x_new;

                // filter location based just on x and y
                int filter_loc = j * filter_size + i;

                // add element-wise product to accumulator
                sum += d_in[loc_new] * filter[filter_loc];
            }
        }
    }

    // add pixel value to output
    d_out[loc] = sum;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./naive_conv <image>\n");
        return 0;
    }
    
    // read in image
    cv::Mat h_in = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    h_in.convertTo(h_in, CV_32FC3);
    cv::normalize(h_in, h_in, 0, 1, cv::NORM_MINMAX);

    int height = image.rows;
    int width = image.cols;

    printf("WIDTH=%d, HEIGHT=%d, FILTER_SIZE=%d\n", width, height, FILTER_SIZE);

    // Declare image and filter variables for host and device
    float *h_filter, *h_out, *d_in, *d_filter, *d_out;

    // size to allocate for image and filter variables
    unsigned int img_size         = width * height * sizeof(float);
    unsigned int full_filter_size = FILTER_SIZE * FILTER_SIZE * sizeof(float);

    // Allocate host data
    h_filter = new float[FILTER_SIZE * FILTER_SIZE];
    h_out    = new float[width * height];

    // Initialize filter template
    // clang-format off
    const float template[FILTER_SIZE][FILTER_SIZE] = {
        {1, 0, 1},
        {0, 1, 0},
        {1, 0, 1}
    };
    // clang-format on

    // copy filter template to actual filter (maybe redundant)
    for (int r = 0; r < FILTER_SIZE; ++r) {
        for (int c = 0; c < FILTER_SIZE; ++c) {
            int idx       = r * FILTER_SIZE + c;
            h_filter[idx] = template[r][c];
        }
    }

    // Allocate device data
    cudaMalloc((void**)&d_in, img_size);
    cudaMalloc((void**)&d_filter, full_filter_size);
    cudaMalloc((void**)&d_out, img_size);

    // Copy host memory to device
    cudaMemcpy(d_in, h_in.ptr<float>(0), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, cudaMemcpyHostToDevice);

    // Let grid size be based on block size
    // Have just enough blocks to cover whole image
    // The -1 is to cover the case where image dimensions are multiples of
    // BLOCKS_SIZE
    int  gridXSize = 1 + ((WIDTH - 1) / BLOCK_SIZE);
    int  gridYSize = 1 + ((HEIGHT - 1) / BLOCK_SIZE);
    dim3 h_gridDim(gridXSize, gridYSize);
    dim3 h_blockDim(BLOCK_SIZE, BLOCK_SIZE);

    // Run on GPU 0
    cudaSetDevice(0);

    // Timing stuff
    cudaDevent_t start, stop;
    float        time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel call
    cudaEventRecord(start, 0);

    naive_kernel<<<h_gridDim, h_blockDim>>>(d_in, d_filter, d_out);

    // Get time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel time = %.2f ms\n", time);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, cudaMemcpyDeviceToHost);

    // TODO: write image to file for displaying

    // Free device data
    cudaFree(d_in);
    cudaFree(d_filter);
    cudaFree(d_out);

    // Free host data
    delete(h_filter);
    delete(h_out);
}
