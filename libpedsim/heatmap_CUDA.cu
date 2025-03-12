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

            size_t scaledIndex = scaledY * (SCALED_SIZE) + scaledX;

            // Write the value to the scaled heatmap
            d_scaled_heatmap[scaledIndex] = value;
        }
    }
}

__global__ void blur(int *d_scaled_heatmap, int *d_blurred_heatmap) {   
    const int w[5][5] = {
        { 1, 4, 7, 4, 1 },
        { 4, 16, 26, 16, 4 },
        { 7, 26, 41, 26, 7 },
        { 4, 16, 26, 16, 4 },
        { 1, 4, 7, 4, 1 }
    };

    // Calculate thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    // Determine the size of the shared memory area
    #define SHARED_SIZE (16 + 4) // 16x16 block + 2 halo on each side
    __shared__ int sharedMem[SHARED_SIZE][SHARED_SIZE];

    // Global index
    int col = bx + tx;
    int row = by + ty;

    // Load main data into shared memory if within bounds
    if (row < SCALED_SIZE && col < SCALED_SIZE) {
        sharedMem[ty + 2][tx + 2] = d_scaled_heatmap[row * SCALED_SIZE + col];
    }

    // Load top and bottom halo
    if (ty < 2) {
        // Top halo (row - 2)
        int halo_row = row - 2;
        if (halo_row >= 0 && col < SCALED_SIZE) {
            sharedMem[ty][tx + 2] = d_scaled_heatmap[halo_row * SCALED_SIZE + col];
        }
        // Bottom halo (row + 2)
        halo_row = row + 2;
        if (halo_row < SCALED_SIZE && col < SCALED_SIZE) {
            sharedMem[ty + blockDim.y + 2][tx + 2] = d_scaled_heatmap[halo_row * SCALED_SIZE + col];
        }
    }

    // Load left and right halo
    if (tx < 2) {
        // Left halo (col - 2)
        int halo_col = col - 2;
        if (halo_col >= 0 && row < SCALED_SIZE) {
            sharedMem[ty + 2][tx] = d_scaled_heatmap[row * SCALED_SIZE + halo_col];
        }
        // Right halo (col + 2)
        halo_col = col + 2;
        if (halo_col < SCALED_SIZE && row < SCALED_SIZE) {
            sharedMem[ty + 2][tx + blockDim.x + 2] = d_scaled_heatmap[row * SCALED_SIZE + halo_col];
        }
    }

    // Synchronize to ensure all data is loaded
    __syncthreads();

    // Apply Gaussian blur if within valid image region (excluding borders)
    if (row >= 2 && row < SCALED_SIZE - 2 && col >= 2 && col < SCALED_SIZE - 2) {
        int sum = 0;
        for (int k = -2; k <= 2; ++k) {
            for (int l = -2; l <= 2; ++l) {
                sum += w[k + 2][l + 2] * sharedMem[ty + k + 2][tx + l + 2];
            }
        }
        #define WEIGHTSUM 273
        int value = sum / WEIGHTSUM;
        d_blurred_heatmap[row * SCALED_SIZE + col] = 0x00FF0000 | (value << 24);
    }
}


__global__ void initHeatmap(int **d_heatmap, int *d_hm, int size)
{
    size_t i = threadIdx.x + blockIdx.x * SIZE;
    d_heatmap[i] = d_hm + size*i;
}

void Ped::Model::setupHeatmapCUDA() 
{
    cudaEvent_t start_create, stop_create;
    cudaEventCreate(&start_create);
    cudaEventCreate(&stop_create);
    
    int **d_heatmap, **d_scaled_heatmap, **d_blurred_heatmap;

    heatmap = (int**)malloc(heatmapPointerSize);
	scaled_heatmap = (int**)malloc(scaledHeatmapPointerSize);
	blurred_heatmap = (int**)malloc(scaledHeatmapPointerSize);

    hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	shm = (int*)malloc(scaledHeatmapSize);
	bhm = (int*)malloc(scaledHeatmapSize);

    CHECK_CUDA_ERROR(cudaMalloc(&d_heatmap, heatmapSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_scaled_heatmap, scaledHeatmapSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blurred_heatmap, scaledHeatmapSize));

    CHECK_CUDA_ERROR(cudaMalloc(&d_hm, heatmapSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_shm, scaledHeatmapSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bhm, scaledHeatmapSize));

    agentSize = agents.size();
    h_agents_desired_x = (int*)malloc(agentSize * sizeof(int));
    h_agents_desired_y = (int*)malloc(agentSize * sizeof(int));

	CHECK_CUDA_ERROR(cudaMalloc(&d_agents_desired_x, agentSize * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_agents_desired_y, agentSize * sizeof(int)));

    cudaEventRecord(start_create);
    // init heatmap
    initHeatmap<<<1, SIZE>>>(d_heatmap, hm, SIZE);
    initHeatmap<<<CELLSIZE, SIZE>>>(d_scaled_heatmap, shm, SCALED_SIZE);
    initHeatmap<<<CELLSIZE, SIZE>>>(d_blurred_heatmap, bhm, SCALED_SIZE);
    cudaEventRecord(stop_create);

    CHECK_CUDA_ERROR(cudaMemcpy(heatmap, d_heatmap, heatmapPointerSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(scaled_heatmap, d_scaled_heatmap, scaledHeatmapPointerSize, cudaMemcpyDeviceToHost)); 
    CHECK_CUDA_ERROR(cudaMemcpy(blurred_heatmap, d_blurred_heatmap, scaledHeatmapPointerSize, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_heatmap));
    CHECK_CUDA_ERROR(cudaFree(d_scaled_heatmap));
    CHECK_CUDA_ERROR(cudaFree(d_blurred_heatmap));
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_create, stop_create);
    // printf("Elapsed time between creation: %f\n", milliseconds);
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void Ped::Model::updateHeatmapCUDA()
{
    //cudaEvent_t start_scale, stop_scale, start_blur,stop_blur, start_total, stop_total;
/*     cudaEventCreate(&start_scale);
    cudaEventCreate(&stop_scale);
    cudaEventCreate(&start_blur);
    cudaEventCreate(&stop_blur); */
    //cudaEventCreate(&start_total);
    //cudaEventCreate(&stop_total);

    float time;

    //cudaEventRecord(start_total);

    for (size_t i = 0; i < agentSize; i++)
    {
        h_agents_desired_x[i] = agents[i]->getDesiredX();
        h_agents_desired_y[i] = agents[i]->getDesiredY();
    }

    dim3 blockDims(16, 16);  // Optimal for shared memory usage
    dim3 gridDims(
        (SCALED_SIZE + blockDims.x - 1) / blockDims.x, // ca 321
        (SCALED_SIZE + blockDims.y - 1) / blockDims.y // ca 321
    );

	cudaMemcpyAsync(d_agents_desired_x, h_agents_desired_x, agentSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_agents_desired_y, h_agents_desired_y, agentSize * sizeof(int), cudaMemcpyHostToDevice);

    //cudaEventRecord(start);
    fadeHeatmap<<<SIZE, SIZE>>>(d_hm);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // cudaEventRecord(stop);

    // cudaEventRecord(start_scale);
    agentCount<<<1, agents.size()>>>(d_hm, d_agents_desired_x, d_agents_desired_y);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    scaleData<<<SIZE, SIZE>>>(d_hm, d_shm);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    /*cudaEventRecord(stop_scale);
    cudaEventSynchronize(stop_scale);
	cudaEventElapsedTime(&time, start_scale, stop_scale);
	cout << "Scale time: " << time << "\n";
 */


    //cudaEventRecord(start_blur);
    blur<<<gridDims, blockDims>>>(d_shm, d_bhm);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    //cudaEventRecord(stop_blur);

    
    
    // cudaEventSynchronize(stop_blur);
    // cudaEventElapsedTime(&time, start_blur, stop_blur);
    
	// cout << "Blur time: " << time << "\n";

   //  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    

    /*cudaEventRecord(stop_total);
    cudaEventElapsedTime(&time, start_total, stop_total);
    cout << "total GPU time: " << time << "\n";*/
}

void Ped::Model::cuda_fin()
{
    cudaMemcpyAsync(bhm, d_bhm, scaledHeatmapSize, cudaMemcpyDeviceToHost);
}