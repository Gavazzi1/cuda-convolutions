#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <helper_cuda.h>
#include <helper_timer.h>

#include "imgutils.h"

#define checkCUDNN(expression)                             \
{                                                          \
    cudnnStatus_t status = (expression);                   \
    if (status != CUDNN_STATUS_SUCCESS) {                  \
        std::cerr << "Error on line " << __LINE__ << ": "  \
        << cudnnGetErrorString(status) << std::endl;       \
        std::exit(EXIT_FAILURE);                           \
    }                                                      \
}

#define ITERATIONS 16
#define KERNEL_SIZE 5
#define KERNEL_RADIUS ((KERNEL_SIZE - 1) / 2)

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: conv <image> [gpu=0]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
    std::cerr << "GPU: " << gpu_id << std::endl;

    cv::Mat image   = read_image_bw(argv[1]);
    int im_channels = image.channels();

    cudaSetDevice(gpu_id);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                /*format=*/CUDNN_TENSOR_NHWC,
                /*dataType=*/CUDNN_DATA_FLOAT,
                /*batch_size=*/1,
                /*channels=*/im_channels,
                /*image_height=*/image.rows,
                /*image_width=*/image.cols));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                /*dataType=*/CUDNN_DATA_FLOAT,
                /*format=*/CUDNN_TENSOR_NCHW,
                /*out_channels=*/im_channels,
                /*in_channels=*/im_channels,
                /*kernel_height=*/KERNEL_SIZE,
                /*kernel_width=*/KERNEL_SIZE));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                /*pad_height=*/KERNEL_RADIUS,
                /*pad_width=*/KERNEL_RADIUS,
                /*vertical_stride=*/1,
                /*horizontal_stride=*/1,
                /*dilation_height=*/1,
                /*dilation_width=*/1,
                /*mode=*/CUDNN_CROSS_CORRELATION,
                /*computeType=*/CUDNN_DATA_FLOAT));

    int batch_size{0}, channels{0}, height{0}, width{0};
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                input_descriptor,
                kernel_descriptor,
                &batch_size,
                &channels,
                &height,
                &width));

    std::cerr << "Output Image: " << height << " x " << width << " x " << channels
        << std::endl;

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                /*format=*/CUDNN_TENSOR_NHWC,
                /*dataType=*/CUDNN_DATA_FLOAT,
                /*batch_size=*/1,
                /*channels=*/channels,
                /*image_height=*/image.rows,
                /*image_width=*/image.cols));

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn,
                input_descriptor,
                kernel_descriptor,
                convolution_descriptor,
                output_descriptor,
                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                /*memoryLimitInBytes=*/0,
                &convolution_algorithm));

    size_t workspace_bytes{0};
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                input_descriptor,
                kernel_descriptor,
                convolution_descriptor,
                output_descriptor,
                convolution_algorithm,
                &workspace_bytes));
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
        << std::endl;
    assert(workspace_bytes > 0);

    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    int image_bytes = batch_size * channels * height * width * sizeof(float);

    float* d_input{nullptr};
    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

    float* d_output{nullptr};
    cudaMalloc(&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);

    // clang-format off
    //const float kernel_template[KERNEL_SIZE][KERNEL_SIZE] = {
    //    {1, 1, 1},
    //    {1, -8, 1},
    //    {1, 1, 1}
    //};
    // clang-format on

    float h_kernel[1][channels][KERNEL_SIZE][KERNEL_SIZE];
    for (int channel = 0; channel < channels; ++channel) {
        for (int row = 0; row < KERNEL_SIZE; ++row) {
            for (int column = 0; column < KERNEL_SIZE; ++column) {
                h_kernel[0][channel][row][column] = (float)(rand() % 16);
            }
        }
    }

    float* d_kernel{nullptr};
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;

    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    std::cerr << "Running convolutions..." << std::endl;

    for (int i = -1; i < ITERATIONS; ++i) {
        if (i == 0) {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }

        checkCUDNN(cudnnConvolutionForward(cudnn,
                    &alpha,
                    input_descriptor,
                    d_input,
                    kernel_descriptor,
                    d_kernel,
                    convolution_descriptor,
                    convolution_algorithm,
                    d_workspace,
                    workspace_bytes,
                    &beta,
                    output_descriptor,
                    d_output));
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double time = sdkGetTimerValue(&hTimer) / (double)ITERATIONS;
    printf("Kernel time = %.5f ms\n", time);

    float* h_output = new float[image_bytes];
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

    save_image_bw("cudnn-out.png", h_output, height, width);

    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
}
