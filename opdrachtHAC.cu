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

void ConvertImageToGrayCpu(unsigned char* imageRGBA, int width, int height)
{
  int i=0;
  long int avgPixel;
  int maxPixel;
  int minPixel;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);


            //wip ideals deel 2 min/max/avg pooling

            //avg pooling RED
            avgPixel=avgPixel+ptrPixel->r
            i++;
            avgPixel= avgPixel/i; //doing end of convertion
            //max pooling RED
            if (ptrPixel->r > maxPixel)
            {
              maxPixel=ptrPixel->r;
            }
            //min pooling RED
            if (ptrPixel->r < minPixel)
            {
              minPixel=ptrPixel->r;
            }


            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
        }
    }
    avgPixel= avgPixel/i;
    printf("avg pooling for red %li\n", avgPixel );
    printf("max pooling for red %li\n", maxPixel );
    printf("min pooling for red %li\n", minPixel );

}


__global__ void ConvertImageToGrayGpu(unsigned char* imageRGBA)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * blockDim.y + threadIdx.y;

   Pixel* ptrPixel = (Pixel*)&imageRGBA[idy * gridDim.x*blockDim.x * 4 + 4 * idx];
            unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;


  // TODO
}

int main(int argc, char** argv)
{
    // Check argument count
    if (argc < 2)
    {
        printf("Usage: im2gray <filename>\r\n");
        return -1;
    }

    // Open image
    int width, height, componentCount;
    printf("Loading png file...\r\n");
    unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, 4);
    if (!imageData)
    {
        printf("Failed to open Image\r\n");
        return -1;
    }
    printf(" DONE \r\n" );


    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        printf("Width and/or Height is not dividable by 32!\r\n");
        return -1;
    }


    // Process image on cpu
    printf("Processing image...\r\n");
    ConvertImageToGrayCpu(imageData, width, height);
    printf(" DONE \r\n");

    // Copy data to the gpu
    printf("Copy data to GPU...\r\n");
    unsigned char* ptrImageDataGpu = nullptr;
    cudaMalloc(&ptrImageDataGpu, width * height * 4);
    cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice);
    printf(" DONE \r\n");

    // Process image on gpu
    printf("Running CUDA Kernel...\r\n");
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    //ConvertImageToGrayGpu<<<gridSize, blockSize>>>(ptrImageDataGpu);
    printf(" DONE \r\n" );

    // Copy data from the gpu
    printf("Copy data from GPU...\r\n");
    cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost);
    printf(" DONE \r\n");

    // Build output filename
    const char * fileNameOut = "gray.png";

    // Write image back to disk
    printf("Writing png to disk...\r\n");
    stbi_write_png(fileNameOut, width, height, 4, imageData, 4 * width);
    printf("DONE\r\n");

    // Free memory
    cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);
}
