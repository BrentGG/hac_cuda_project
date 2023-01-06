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

static size_t write_data(void *ptr, size_t size, size_t nmemb, void *stream)
{
  size_t written = fwrite(ptr, size, nmemb, (FILE *)stream);
  return written;
}

int main(int argc, char** argv)
{
    // Image URLs and names
    int imageAmount = 3;
    char urls[imageAmount][1024] = {
        "https://images.nintendolife.com/835d44208f0a6/mario-movie.large.jpg", // mario
        "https://img.buzzfeed.com/buzzfeed-static/static/2022-03/16/0/enhanced/c1de3db394fb/original-1460-1647389279-4.png", // the rock
        "https://images.squarespace-cdn.com/content/v1/5b788d28697a98e17a6d4c7a/b83f0eab-7dd6-4e9b-83a1-13139ac2a03b/rickroll+cropped.png", // rick astley
    };
    char names[imageAmount][20] = {
        "image1.png",
        "image2.png",
        "image3.png"
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

        // Convolution on CPU
        unsigned char* outputConvolution = (unsigned char*) malloc(sizeof(unsigned char) * (width - 2) * (height - 2) * 4);
        printf("Applying convolution...\n");
        convoluteCPU(inputData, outputConvolution, width, height, edgeDetection);
        printf(" DONE\n");

        // Pooling on CPU
        int poolWidth = (int)(width / POOLSTRIDE);
        int poolHeight = (int)(height / POOLSTRIDE);
        unsigned char* outputMaxPool = (unsigned char*) malloc(sizeof(unsigned char) * poolWidth * poolHeight * 4);
        unsigned char* outputMinPool = (unsigned char*) malloc(sizeof(unsigned char) * poolWidth * poolHeight * 4);
        unsigned char* outputAvgPool = (unsigned char*) malloc(sizeof(unsigned char) * poolWidth * poolHeight * 4);
        printf("Pooling...\n");
        pool(inputData, outputMaxPool, outputMinPool, outputAvgPool, width, height, POOLSTRIDE);
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
        stbi_image_free(inputData);
        stbi_image_free(outputConvolution);
        stbi_image_free(outputMaxPool);
        stbi_image_free(outputMinPool);
        stbi_image_free(outputAvgPool);
    }
}
