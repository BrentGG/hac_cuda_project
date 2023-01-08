#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curl/curl.h>
#include <string.h>
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

static size_t write_data(void *ptr, size_t size, size_t nmemb, void *stream)
{
  size_t written = fwrite(ptr, size, nmemb, (FILE *)stream);
  return written;
}

int main(int argc, char** argv)
{
    // Image URLs and names
    int imageAmount = 10;
    char urls[imageAmount][1024] = {
        "https://images.nintendolife.com/835d44208f0a6/mario-movie.large.jpg", // mario
        "https://img.buzzfeed.com/buzzfeed-static/static/2022-03/16/0/enhanced/c1de3db394fb/original-1460-1647389279-4.png", // the rock
        "https://images.squarespace-cdn.com/content/v1/5b788d28697a98e17a6d4c7a/b83f0eab-7dd6-4e9b-83a1-13139ac2a03b/rickroll+cropped.png", // rick astley
        "https://a-z-animals.com/media/2021/02/Kinkajou-header.jpg", //kinkajou
        "https://static.wikia.nocookie.net/marveldatabase/images/6/64/Incredible_Hulk_Vol_2_75_Textless.jpg/revision/latest/scale-to-width-down/300?cb=20050830175533", //hulk
        "https://laughingsquid.com/wp-content/uploads/bert-20110421-082506.jpg", //bert
        "https://car-anwb.akamaized.net/aas-afbeeldingen/117222g.jpg?imwidth=760&imheight=500", //ford
        "https://i.pinimg.com/originals/41/0b/2e/410b2eb9e59520a7ad4de7aa4fc9f722.jpg", //micky minnie mouse
        "https://pbs.twimg.com/media/CDN2AnCWYAIquzo.jpg", //canadian poop
        "https://media.istockphoto.com/id/1153678999/nl/vector/de-titel-van-het-eind-handschrift-op-rode-ronde-bacground-oude-film-einde-scherm-vector.jpg?s=612x612&w=0&k=20&c=OmFVqkf5TgXxizS0pWq5lEgbNQTNSny5W-BPmBahr1I=", //the end
        //extra URL links 
        /*
        "https://images0.persgroep.net/rcs/MP9RjFYsAOAE1ArHRLyorkHlKXU/diocontent/203486167/_fitwidth/1240?appId=93a17a8fd81db0de025c8abd1cca1279&quality=0.9",  //biden
        "https://cdn.shopify.com/s/files/1/0351/9630/5545/products/Good-Smile-Company-Pocket-Maquette-Demon-Slayer-Kimetsu-no-Yaiba-01-Single-Box-Random-7_bef2ee96-861e-41f5-9afb-b5650961e626_1200x.jpg?v=1631869901", //demon slayer
        */
    };
    char names[imageAmount][20] = {
        "image1.png",
        "image2.png",
        "image3.png",
        "image4.png",
        "image5.png",
        "image6.png",
        "image7.png",
        "image8.png",
        "image9.png",
        "image10.png"
    };

    // Get all the images from their URL
    CURL *curl_handle;
    char *imageFileName = (char*) malloc(sizeof(char) * 20);
    FILE *imageFile;
    curl_global_init(CURL_GLOBAL_ALL);
    curl_handle = curl_easy_init();
    //curl_easy_setopt(curl_handle, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curl_handle, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, write_data);
    for (int i = 0; i < imageAmount; ++i) {
        printf("Looking for image %d...\n", i + 1);
        curl_easy_setopt(curl_handle, CURLOPT_URL, urls[i]);
        imageFile = fopen(names[i], "wb");
        if(imageFile) {
            curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, imageFile);
            curl_easy_perform(curl_handle);
            fclose(imageFile);
        }
        else {
            printf(" FAILED\n");
            return 1;
        }
        printf(" DONE\n");
    }
    printf("\n");
    curl_easy_cleanup(curl_handle);
    curl_global_cleanup();
    
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
    for(int f = 0; f < imageAmount; f++) {
        // Open image
        int width, height, componentCount;
        printf("Loading png file %d...\n", f + 1);
        unsigned char* inputData = stbi_load(names[f], &width, &height, &componentCount, 4);
        if (!inputData) {
            printf(" FAILED\n");
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
        char buffer[255];
        memset(buffer, '\0', sizeof(buffer));
        strncpy(buffer, names[f], strlen(names[f]) - 4);
        strcat(buffer, "_convolution.png");
        stbi_write_png(buffer, width - 2, height - 2, 4, outputConvolution, 4 * width);
        memset(buffer, '\0', sizeof(buffer));
        strncpy(buffer, names[f], strlen(names[f]) - 4);
        strcat(buffer, "_maxpool.png");
        stbi_write_png(buffer, poolWidth, poolHeight, 4, outputMaxPool, 4 * poolWidth);
        memset(buffer, '\0', sizeof(buffer));
        strncpy(buffer, names[f], strlen(names[f]) - 4);
        strcat(buffer, "_minpool.png");
        stbi_write_png(buffer, poolWidth, poolHeight, 4, outputMinPool, 4 * poolWidth);
        memset(buffer, '\0', sizeof(buffer));
        strncpy(buffer, names[f], strlen(names[f]) - 4);
        strcat(buffer, "_avgpool.png");
        stbi_write_png(buffer, poolWidth, poolHeight, 4, outputAvgPool, 4 * poolWidth);
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

        printf("\n");
    }

    cudaFree(gaussianBlurGPU);
    cudaFree(edgeDetectionGPU);
    cudaFree(exampleGPU);
}
