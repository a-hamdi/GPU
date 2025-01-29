/*
nvcc cmpFHd_real_image.cu -o cmpFHd_real_image `pkg-config --cflags --libs opencv4`

and don't forget to put the image file in the same directory.
*/



#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define FHD_THREADS_PER_BLOCK 256
#define PI 3.14159265358979323846
#define CHUNK_SIZE 256

using namespace cv;
using namespace std;

__constant__ float kx_c[CHUNK_SIZE], ky_c[CHUNK_SIZE], kz_c[CHUNK_SIZE];

__global__ void cmpFHd(float* rPhi, float* iPhi, float* phiMag,
                       float* x, float* y, float* z,
                       float* rMu, float* iMu, int M) {
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;
    
    float xn_r = x[n]; 
    float yn_r = y[n]; 
    float zn_r = z[n];

    float rFhDn_r = rPhi[n]; 
    float iFhDn_r = iPhi[n];

    for (int m = 0; m < M; m++) {
        float expFhD = 2 * PI * (kx_c[m] * xn_r + ky_c[m] * yn_r + kz_c[m] * zn_r);
        
        float cArg = __cosf(expFhD);
        float sArg = __sinf(expFhD);

        rFhDn_r += rMu[m] * cArg - iMu[m] * sArg;
        iFhDn_r += iMu[m] * cArg + rMu[m] * sArg;
    }

    rPhi[n] = rFhDn_r;
    iPhi[n] = iFhDn_r;
}

int main() {
    // Load an image using OpenCV
    Mat image = imread("lena_gray.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Could not open the image!" << endl;
        return -1;
    }

    // Normalize image to range [0,1]
    image.convertTo(image, CV_32F, 1.0 / 255);

    int N = image.rows * image.cols; // Number of pixels
    int M = 256; // Number of frequency components
  
    float *x, *y, *z, *rMu, *iMu, *rPhi, *iPhi, *phiMag;
    
    // Allocate CUDA memory
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&z, N * sizeof(float));
    cudaMallocManaged(&rMu, M * sizeof(float));
    cudaMallocManaged(&iMu, M * sizeof(float));
    cudaMallocManaged(&rPhi, N * sizeof(float));
    cudaMallocManaged(&iPhi, N * sizeof(float));
    cudaMallocManaged(&phiMag, N * sizeof(float));

    // Initialize x, y coordinates from image pixels
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int idx = i * image.cols + j;
            x[idx] = (float)j / image.cols;  // Normalize to [0,1]
            y[idx] = (float)i / image.rows;  // Normalize to [0,1]
            z[idx] = image.at<float>(i, j);  // Use intensity as "z"
            rPhi[idx] = z[idx];              // Initial real part
            iPhi[idx] = 0.0f;                // Initial imaginary part
        }
    }

    // Randomly initialize rMu and iMu
    for (int i = 0; i < M; i++) {
        rMu[i] = static_cast<float>(rand()) / RAND_MAX;
        iMu[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Copy chunks of kx, ky, kz to constant memory
    for (int i = 0; i < M / CHUNK_SIZE; i++) {
        cudaMemcpyToSymbol(kx_c, &x[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaMemcpyToSymbol(ky_c, &y[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaMemcpyToSymbol(kz_c, &z[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));

        // Launch CUDA kernel
        cmpFHd<<<N / FHD_THREADS_PER_BLOCK, FHD_THREADS_PER_BLOCK>>>(rPhi, iPhi, phiMag, x, y, z, rMu, iMu, CHUNK_SIZE);
        cudaDeviceSynchronize();
    }

    // Convert results back to image format
    Mat outputImage(image.rows, image.cols, CV_32F);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int idx = i * image.cols + j;
            outputImage.at<float>(i, j) = sqrt(rPhi[idx] * rPhi[idx] + iPhi[idx] * iPhi[idx]);
        }
    }

    // Normalize and save output image
    normalize(outputImage, outputImage, 0, 255, NORM_MINMAX);
    outputImage.convertTo(outputImage, CV_8U);
    imwrite("output.jpg", outputImage);

    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(rMu);
    cudaFree(iMu);
    cudaFree(rPhi);
    cudaFree(iPhi);
    cudaFree(phiMag);

    cout << "Processed image saved as output.jpg" << endl;
    return 0;
}