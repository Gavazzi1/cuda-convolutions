#ifndef FLEX_IMG_UTILS_H_
#define FLEX_IMG_UTILS_H_

#include <opencv2/opencv.hpp>

cv::Mat read_image(const char* filename) {
    cv::Mat h_in = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    h_in.convertTo(h_in, CV_32FC3);
    cv::normalize(h_in, h_in, 0, 1, cv::NORM_MINMAX);
    return h_in;
}

cv::Mat read_image_bw(const char* filename) {
    cv::Mat h_in = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    h_in.convertTo(h_in, CV_32FC1);
    cv::normalize(h_in, h_in, 0, 1, cv::NORM_MINMAX);
    return h_in;
}

void save_image(const char* filename, float* buffer, int height, int width) {
    cv::Mat output_image(height, width, CV_32FC3, buffer);
    cv::threshold(output_image, output_image, 0, 0, cv::THRESH_TOZERO);
    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8UC3);
    cv::imwrite(filename, output_image);
}

void save_image_bw(const char* filename, float* buffer, int height, int width) {
    cv::Mat output_image(height, width, CV_32FC1, buffer);
    cv::threshold(output_image, output_image, 0, 0, cv::THRESH_TOZERO);
    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8UC1);
    cv::imwrite(filename, output_image);
}

#endif
