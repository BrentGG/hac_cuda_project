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

void pool(unsigned char* input, unsigned char* outputMaxPool, unsigned char* outputMinPool, unsigned char* outputAvgPool, int width, int height, int poolStride) {
    int poolWidth = (int)(width / POOLSTRIDE);
    int poolHeight = (int)(height / POOLSTRIDE);
    int row = 0;
    for (int i = 0; i < height; i += poolStride) {
        int col = 0;
        for (int j = 0; j < width; j += poolStride) {
            Pixel* p = (Pixel*)&input[i * width * 4 + 4 * j];
            int max[3] = {p->r, p->g, p->b};
            int min[3] = {p->r, p->g, p->b};
            float avg[3] = {0, 0, 0};
            for (int k = i; k < i + poolStride; ++k) {
                for (int l = j; l < j + poolStride; ++l) {
                    Pixel* q = (Pixel*)&input[k * width * 4 + 4 * l];
                    int values[3] = {q->r, q->g, q->b};
                    for (int v = 0; v < 3; ++v) {
                        if (values[v] > max[v])
                            max[v] = values[v];
                        if (values[v] < min[v])
                            min[v] = values[v];
                        avg[v] += values[v];
                    }
                }
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
            ++col;
        }
        ++row;
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

    // Convolution on CPU
    unsigned char* outputConvolution = (unsigned char*) malloc(sizeof(unsigned char) * (width - 2) * (height - 2) * 4);
    printf("Applying convolution...\r\n");
    convoluteCPU(inputData, outputConvolution, width, height, edgeDetection);
    printf(" DONE \r\n");

    // Pooling on GPU
    int poolWidth = (int)(width / POOLSTRIDE);
    int poolHeight = (int)(height / POOLSTRIDE);
    printf("%d %d\n", width, height);
    printf("%d %d\n", poolWidth, poolHeight);
    unsigned char* outputMaxPool = (unsigned char*) malloc(sizeof(unsigned char) * poolWidth * poolHeight * 4);
    unsigned char* outputMinPool = (unsigned char*) malloc(sizeof(unsigned char) * poolWidth * poolHeight * 4);
    unsigned char* outputAvgPool = (unsigned char*) malloc(sizeof(unsigned char) * poolWidth * poolHeight * 4);
    printf("Pooling...\r\n");
    pool(inputData, outputMaxPool, outputMinPool, outputAvgPool, width, height, POOLSTRIDE);
    printf(" DONE \r\n");

    // Write images back to disk
    printf("Writing pngs to disk...\r\n");
    stbi_write_png("convolution.png", (width - 2), (height - 2), 4, outputConvolution, 4 * (width - 2));
    stbi_write_png("maxpool.png", poolWidth, poolHeight, 4, outputMaxPool, 4 * poolWidth);
    stbi_write_png("minpool.png", poolWidth, poolHeight, 4, outputMinPool, 4 * poolWidth);
    stbi_write_png("avgpool.png", poolWidth, poolHeight, 4, outputAvgPool, 4 * poolWidth);
    printf(" DONE\r\n");

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

    // Free memory
    //cudaFree(ptrImageDataGpu);
    stbi_image_free(inputData);
    stbi_image_free(outputConvolution);
    stbi_image_free(outputMaxPool);
    stbi_image_free(outputMinPool);
    stbi_image_free(outputAvgPool);
}
