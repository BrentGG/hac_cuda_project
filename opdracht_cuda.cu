#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define POOLSTRIDE 2

struct Pixel
{
    unsigned char r, g, b, a;
};

__global__ void convoluteGPU(unsigned char* input, unsigned char* output, int width, int height, float* kernel)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height - 2 && col < width - 2) {
        int sum[4] = {0, 0, 0, 0};
        int opacity = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Pixel* p = (Pixel*)&input[((row + 1) + (i - 1)) * width * 4 + 4 * ((col + 1) + (j - 1))];
                sum[0] += p->r * kernel[i * 3 + j];
                sum[1] += p->g * kernel[i * 3 + j];
                sum[2] += p->b * kernel[i * 3 + j];
                sum[3] += p->a * kernel[i * 3 + j];
                if (i == 1 || j == 1)
                    opacity = p->a;
            }
        }
        Pixel* ptrPixel = (Pixel*)&output[row * width * 4 + 4 * col];
        for (int i = 0; i < 3; ++i) {
            if (sum[i] < 0)
                sum[i] = 0;
            else if (sum[i] > 255)
                sum[i] = 255;
        }
        ptrPixel->r = sum[0];
        ptrPixel->g = sum[1];
        ptrPixel->b = sum[2];
        ptrPixel->a = opacity;
    }
}

__global__ void pool(unsigned char* input, unsigned char* outputMaxPool, unsigned char* outputMinPool, unsigned char* outputAvgPool, int width, int height, int poolStride) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    int poolWidth = (int)(width / poolStride);
    int poolHeight = (int)(height / poolStride);

    if (row < poolHeight && col < poolWidth) {
        Pixel* p = (Pixel*)&input[row * poolStride * width * 4 + 4 * col * poolStride];
        int max[3] = {p->r, p->g, p->b};
        int min[3] = {p->r, p->g, p->b};
        float avg[3] = {0, 0, 0};
        for (int i = row * poolStride; i < (row * poolStride) + poolStride; ++i) {
            for (int j = col * poolStride; j < (col * poolStride) + poolStride; ++j) {
                Pixel* p = (Pixel*)&input[i * width * 4 + 4 * j];
                int values[3] = {p->r, p->g, p->b};
                for (int v = 0; v < 3; ++v) {
                    if (values[v] > max[v])
                        max[v] = values[v];
                    if (values[v] < min[v])
                        min[v] = values[v];
                    avg[v] += values[v];
                }
                for (int v = 0; v < 3; ++v)
                    avg[v] /= poolStride * poolStride;
                Pixel* q = (Pixel*)&outputMaxPool[row * poolWidth * 4 + 4 * col];
                q->r = max[0];
                q->g = max[1];
                q->b = max[2];
                q->a = 255;
                Pixel* r = (Pixel*)&outputMinPool[row * poolWidth * 4 + 4 * col];
                r->r = min[0];
                r->g = min[1];
                r->b = min[2];
                r->a = 255;
                Pixel* s = (Pixel*)&outputAvgPool[row * poolWidth * 4 + 4 * col];
                s->r = round(avg[0]);
                s->g = round(avg[1]);
                s->b = round(avg[2]);
                s->a = 255;
            }
        }
    }
}

int main(int argc, char** argv)
{
    // Convolution kernel options
    float gaussianBlur[3][3] = {
        {0.0625*1, 0.0625*2, 0.0625*1},
        {0.0625*2, 0.0625*4, 0.0625*2},
        {0.0625*1, 0.0625*2, 0.0625*1}
    };
    float edgeDetection[3][3] = {
        {-1, -1, -1},
        {-1, 8, -1},
        {-1, -1, -1}
    };
    float example[3][3] = {
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    };
    float *gaussianBlurGPU = nullptr;
    cudaMalloc(&gaussianBlurGPU, 3 * 3 * sizeof(float));
    cudaMemcpy(gaussianBlurGPU, gaussianBlur, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    float *edgeDetectionGPU = nullptr;
    cudaMalloc(&edgeDetectionGPU, 3 * 3 * sizeof(float));
    cudaMemcpy(edgeDetectionGPU, edgeDetection, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
    float *exampleGPU = nullptr;
    cudaMalloc(&exampleGPU, 3 * 3 * sizeof(float));
    cudaMemcpy(exampleGPU, example, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Execute all the operations for every image
    // Open image
    int width, height, componentCount;
    printf("Loading png file...\n");
    unsigned char* inputData = stbi_load(argv[1], &width, &height, &componentCount, 4);
    if (!inputData)
    {
        printf("Failed to open image\r\n");
        return -1;
    }
    printf(" DONE\n" );

    // Copy data to GPU
    printf("Copy data to GPU...\n");
    unsigned char* inputDataGPU = nullptr;
    cudaMalloc(&inputDataGPU, width * height * 4);
    cudaMemcpy(inputDataGPU, inputData, width * height * 4, cudaMemcpyHostToDevice);
    printf(" DONE\n" );

    // Convolution on GPU
    unsigned char* outputConvolutionGPU = nullptr;
    cudaMalloc(&outputConvolutionGPU, (width - 2) * (height - 2) * 4);
    dim3 blockSize(32, 32);
    dim3 convGridSize(width / blockSize.x, height / blockSize.y);
    printf("Applying convolution...\n");
    convoluteGPU<<<convGridSize, blockSize>>>(inputDataGPU, outputConvolutionGPU, width, height, edgeDetectionGPU);
    cudaDeviceSynchronize();
    printf(" DONE\n" );

    // Pooling on GPU
    int poolWidth = (int)(width / POOLSTRIDE);
    int poolHeight = (int)(height / POOLSTRIDE);
    unsigned char* outputMaxPoolGPU = nullptr;
    cudaMalloc(&outputMaxPoolGPU, poolWidth * poolHeight * 4);
    unsigned char* outputMinPoolGPU = nullptr;
    cudaMalloc(&outputMinPoolGPU, poolWidth * poolHeight * 4);
    unsigned char* outputAvgPoolGPU = nullptr;
    cudaMalloc(&outputAvgPoolGPU, poolWidth * poolHeight * 4);
    dim3 poolGridSize(poolWidth / blockSize.x + 1, poolHeight / blockSize.y + 1);
    printf("Pooling...\n");
    pool<<<poolGridSize, blockSize>>>(inputDataGPU, outputMaxPoolGPU, outputMinPoolGPU, outputAvgPoolGPU, width, height, POOLSTRIDE);
    cudaDeviceSynchronize();
    printf(" DONE\n" );

    // Copy data from the GPU
    printf("Copy data from GPU...\n");
    unsigned char* outputConvolution = (unsigned char*) malloc((width - 2) * (height - 2) * 4);
    cudaMemcpy(outputConvolution, outputConvolutionGPU, (width - 2) * (height - 2) * 4, cudaMemcpyDeviceToHost);
    unsigned char* outputMaxPool = (unsigned char*) malloc(poolWidth * poolHeight * 4);
    cudaMemcpy(outputMaxPool, outputMaxPoolGPU, poolWidth * poolHeight * 4, cudaMemcpyDeviceToHost);
    unsigned char* outputMinPool = (unsigned char*) malloc(poolWidth * poolHeight * 4);
    cudaMemcpy(outputMinPool, outputMinPoolGPU, poolWidth * poolHeight * 4, cudaMemcpyDeviceToHost);
    unsigned char* outputAvgPool = (unsigned char*) malloc(poolWidth * poolHeight * 4);
    cudaMemcpy(outputAvgPool, outputAvgPoolGPU, poolWidth * poolHeight * 4, cudaMemcpyDeviceToHost);
    printf(" DONE\n");

    // Write images back to disk
    printf("Writing output pngs to disk...\n");
    stbi_write_png("convolutionGPU.png", width - 2, height - 2, 4, outputConvolution, 4 * width);
    stbi_write_png("maxpoolGPU.png", poolWidth, poolHeight, 4, outputMaxPool, 4 * poolWidth);
    stbi_write_png("minpoolGPU.png", poolWidth, poolHeight, 4, outputMinPool, 4 * poolWidth);
    stbi_write_png("avgpoolGPU.png", poolWidth, poolHeight, 4, outputAvgPool, 4 * poolWidth);
    printf(" DONE\n");

    // Free memory
    cudaFree(inputDataGPU);
    cudaFree(outputConvolutionGPU);
    cudaFree(outputMaxPoolGPU);
    cudaFree(outputMinPoolGPU);
    cudaFree(outputAvgPoolGPU);
    stbi_image_free(inputData);
    stbi_image_free(outputConvolution);
    stbi_image_free(outputMaxPool);
    stbi_image_free(outputMinPool);
    stbi_image_free(outputAvgPool);

    cudaFree(gaussianBlurGPU);
    cudaFree(edgeDetectionGPU);
    cudaFree(exampleGPU);
}
