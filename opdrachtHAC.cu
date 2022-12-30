#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

struct Pixel
{
    unsigned char r, g, b, a;
};

void convoluteCPU(unsigned char* input, unsigned char* output, int width, int height, float kernel[3][3])
{
    for (int row = 0; row < height - 2; row++) {
        for (int col = 0; col < width - 2; col++) {
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
}

__global__ void convoluteGPU(unsigned char* imageRGBA)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    Pixel* ptrPixel = (Pixel*)&imageRGBA[idy * gridDim.x*blockDim.x * 4 + 4 * idx];
    unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
    ptrPixel->r = pixelValue;
    ptrPixel->g = pixelValue;
    ptrPixel->b = pixelValue;
    ptrPixel->a = 255;
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

    // Process image on cpu
    unsigned char* outputData = (unsigned char*) malloc(sizeof(unsigned char) * (width - 2) * (height - 2) * 4);
    printf("Applying convolution...\r\n");
    convoluteCPU(inputData, outputData, width, height, edgeDetection);
    printf(" DONE \r\n");

    // Copy data to the gpu
    /*printf("Copy data to GPU...\r\n");
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, inputData, width * height * 4, cudaMemcpyHostToDevice);
    printf(" DONE \r\n");*/

    // Process image on gpu
    /*printf("Running CUDA Kernel...\r\n");
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    //convoluteGPU<<<gridSize, blockSize>>>(ptrImageDataGpu);
    printf(" DONE \r\n" );*/

    // Copy data from the gpu
    /*printf("Copy data from GPU...\r\n");
    cudaMemcpy(inputData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    printf(" DONE \r\n");*/

    // Build output filename
    const char * fileNameOut = "out.png";

    // Write image back to disk
    printf("Writing png to disk...\r\n");
    stbi_write_png(fileNameOut, width, height, 4, outputData, 4 * width);
    printf(" DONE\r\n");

    // Free memory
    //cudaFree(ptrImageDataGpu);
    stbi_image_free(inputData);
}
