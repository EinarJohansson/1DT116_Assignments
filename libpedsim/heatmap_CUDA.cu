#include <iostream>
#include "ped_model.h"
#include <stdio.h>

#define CHECK_CUDA_ERROR(call)                                              \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void initHeatmap(int **d_heatmap, int *d_hm, size_t size)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    d_heatmap[i] = d_hm + size * i;
}

__global__ void fadeHeatmap(int *d_heatmap)
{
    size_t i = threadIdx.x + blockIdx.x * SIZE;
    d_heatmap[i] = (int)round(d_heatmap[i] * 0.80);
}

__global__ void agentCount(int *d_heatmap, int *d_agents_desired_x, int *d_agents_desired_y)
{
    size_t i = threadIdx.x;

    int x = d_agents_desired_x[i];
    int y = d_agents_desired_y[i];

    
    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
    {
        return;
    }

    size_t idx = x + y * SIZE;

    // intensify heat for better color results
    atomicAdd(&d_heatmap[idx], 40);
}

__global__ void scaleData(int *d_heatmap, int *d_scaled_heatmap) {
    // Calculate thread and block indices
    size_t x = threadIdx.x; // Column index (0 to SIZE-1)
    size_t y = blockIdx.x;  // Row index (0 to SIZE-1)

    // Read the value from the original heatmap
    int value = d_heatmap[y * SIZE + x];

    // Scale the value to the larger heatmap
    for (int cellY = 0; cellY < CELLSIZE; cellY++) {
        for (int cellX = 0; cellX < CELLSIZE; cellX++) {
            // Calculate the index in the scaled heatmap
            size_t scaledY = y * CELLSIZE + cellY;
            size_t scaledX = x * CELLSIZE + cellX;
            size_t scaledIndex = scaledY * (SIZE * CELLSIZE) + scaledX;

            // Write the value to the scaled heatmap
            d_scaled_heatmap[scaledIndex] = value;
        }
    }
}

__global__ void blur(int *d_scaled_heatmap, int *d_blurred_heatmap) {
    #define WEIGHTSUM 273
        const int w[5][5] = {
            {1, 4, 7, 4, 1},
            {4, 16, 26, 16, 4},
            {7, 26, 41, 26, 7},
            {4, 16, 26, 16, 4},
            {1, 4, 7, 4, 1}};
    
        size_t x = threadIdx.x; // 0 to SIZE-1
        size_t y = blockIdx.x;  // 0 to SIZE-1
    
        for (int cellY = 0; cellY < CELLSIZE; cellY++) {
            for (int cellX = 0; cellX < CELLSIZE; cellX++) {
                size_t i = y * CELLSIZE + cellY;
                size_t j = x * CELLSIZE + cellX;
    
                // Boundary checks
                if (i < 2) {
                    i = 2;
                } else if (i >= SCALED_SIZE - 2) {
                    i = SCALED_SIZE - 3;
                }
                if (j < 2) {
                    j = 2;
                } else if (j >= SCALED_SIZE - 2) {
                    j = SCALED_SIZE - 3;
                }
    
                int sum = 0;
                for (int k = -2; k <= 2; k++) {
                    for (int l = -2; l <= 2; l++) {
                        int weight = w[2 + k][2 + l];
                        int idx = (i + k) * SCALED_SIZE + (j + l);
                        sum += weight * d_scaled_heatmap[idx];
                    }
                }
    
                int value = sum / WEIGHTSUM;
                int idx = i * SCALED_SIZE + j;
                d_blurred_heatmap[idx] = 0x00FF0000 | (value << 24);
                if (idx == 1337) {
                    printf("d_blurred_heatmap[%d]=%d\n", idx, d_blurred_heatmap[idx]);
                }
            }
        }
    }
    
void Ped::Model::setupHeatmapCUDA() 
{
    // All kernel launches are asynchronous
    // • Control returns to CPU before kernel finishes
    // • Kernel executes after all previous CUDA calls have completed
    // cudaMemcpy() is synchronous


    // cudaDeviceSynchronize()
    // • Blocks host until all issued CUDA calls are complete
    this->setupHeatmapSeq();

    size_t agentSize = agents.size();

    CHECK_CUDA_ERROR(cudaMalloc(&d_heatmap, heatmapPointerSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_scaled_heatmap, scaledHeatmapPointerSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blurred_heatmap, scaledHeatmapPointerSize));

    CHECK_CUDA_ERROR(cudaMalloc(&d_agents_desired_x, agentSize * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_agents_desired_y, agentSize * sizeof(int)));
    
    int *h_agents_desired_x = (int*)malloc(agentSize * sizeof(int));
    int *h_agents_desired_y = (int*)malloc(agentSize * sizeof(int));

    for (size_t i = 0; i < agentSize; i++)
    {
        h_agents_desired_x[i] = agents[i]->getDesiredX();
        h_agents_desired_y[i] = agents[i]->getDesiredY();
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_agents_desired_x, h_agents_desired_x, agentSize * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_agents_desired_y, h_agents_desired_y, agentSize * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMalloc(&d_hm, heatmapSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_shm, scaledHeatmapSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bhm, scaledHeatmapSize));

    initHeatmap<<<1, SIZE>>>(d_heatmap, d_hm, SIZE);
    initHeatmap<<<CELLSIZE, SIZE>>>(d_scaled_heatmap, d_shm, SCALED_SIZE);
    initHeatmap<<<CELLSIZE, SIZE>>>(d_blurred_heatmap, d_bhm, SCALED_SIZE);

    cudaDeviceSynchronize();
}

void Ped::Model::updateHeatmapCUDA()
{
    size_t agentSize = agents.size();

    fadeHeatmap<<<SIZE, SIZE>>>(d_hm);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    agentCount<<<1, agentSize>>>(d_hm, d_agents_desired_x, d_agents_desired_y);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    scaleData<<<SIZE, SIZE>>>(d_hm, d_shm);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    blur<<<SIZE, SIZE>>>(d_shm, d_bhm);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(hm, d_hm, heatmapSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(shm, d_shm, scaledHeatmapSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(bhm, d_bhm, scaledHeatmapSize, cudaMemcpyDeviceToHost));
}