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

__global__ void convoluteGPU(unsigned char* input, unsigned char* output, int width, int height, float kernel[3][3])
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height - 2 && col < width - 2) {
        int sum[4] = {0, 0, 0, 0};
        int opacity = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Pixel* p = (Pixel*)&input[((row + 1) + (i - 1)) * width * 4 + 4 * ((col + 1) + (j - 1))];
                sum[0] += p->r * kernel[i][j];
                sum[1] += p->g * kernel[i][j];
                sum[2] += p->b * kernel[i][j];
                sum[3] += p->a * kernel[i][j];
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

int main(int argc, char** argv)
{
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

    // Check argument count
    if (argc < 2)
    {
        printf("Not enough arguments.");
        return -1;
    }

    // Open image
    int width, height, componentCount;
    printf("Loading png file...\r\n");
    unsigned char* inputData = stbi_load(argv[1], &width, &height, &componentCount, 4);
    if (!inputData)
    {
        printf("Failed to open image\r\n");
        return -1;
    }
    printf(" DONE \r\n" );

    /* --- */
    /* GPU */
    /* --- */

    // Copy data to the gpu
    printf("Copy data to GPU...\r\n");
    unsigned char* inputDataGPU = nullptr;
    cudaMalloc(&inputDataGPU, width * height * 4);
    cudaMemcpy(inputDataGPU, inputData, width * height * 4, cudaMemcpyHostToDevice);
    printf(" DONE \r\n");

    // Process image on GPU
    unsigned char* outputConvolutionGPU = nullptr;
    cudaMalloc(&outputConvolutionGPU, (width - 2) * (height - 2) * 4);
    printf("Running CUDA Kernel...\r\n");
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    convoluteGPU<<<gridSize, blockSize>>>(inputDataGPU, outputConvolutionGPU, width, height, gaussianBlur);
    cudaDeviceSynchronize();
    printf(" DONE \r\n" );

    // Copy data from the gpu
    printf("Copy data from GPU...\r\n");
    unsigned char* outputConvolution = nullptr;
    cudaMemcpy(outputConvolution, outputConvolutionGPU, (width - 2) * (height - 2) * 4, cudaMemcpyDeviceToHost);
    printf(" DONE \r\n");

    // Write images back to disk
    printf("Writing pngs to disk...\r\n");
    stbi_write_png("convolutionGPU.png", width - 2, height - 2, 4, outputConvolution, 4 * (width - 2));
    printf(" DONE\r\n");

    // Free memory
    cudaFree(inputDataGPU);
    cudaFree(outputConvolutionGPU);
    stbi_image_free(inputData);
    stbi_image_free(outputConvolution);
}
