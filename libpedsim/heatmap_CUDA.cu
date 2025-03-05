#include <iostream>
#include "ped_model.h"

__global__ void initHeatMap(int **heatmap, int* hm, int size)
{
    int idx = blockIdx.x * SIZE + threadIdx.x;
    heatmap[idx] = hm + size * idx;
    // heatmap[idx] = &hm[size * idx];
}

__global__ void heatmapFade(int **heatmap)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	heatmap[blockDim.x * y][x] = (int)round(heatmap[blockDim.x * y][x] * 0.80);
}


__global__ void agentCount(int **heatmap, Ped::Tagent **agents)
{
	int i = threadIdx.x;
	Ped::Tagent* agent = agents[i];
	
	int x = agent->desiredPositionX;
	int y = agent->desiredPositionY;
	
	if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
	{
		return;
	}
	
	// intensify heat for better color results
	// TODO: Datarace? fix med atomic
	if (heatmap[y][x] > 215) {
		heatmap[y][x] = 255;
	}
	else {
		heatmap[y][x] += 40;
		// atomicAdd(&heatmap[y][x], 40);
	}
}

__global__ void scaleData(int **heatmap, int **scaled_heatmap)
{
	// TODO: Indexering fel just nu
	int x = threadIdx.x;
	int y = blockIdx.x;
	int value = heatmap[y][x];
	
	for (int cellY = 0; cellY < CELLSIZE; cellY++)
	{
		for (int cellX = 0; cellX < CELLSIZE; cellX++)
		{
			// TODO: fix DATA RACE
			scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
		}
	}
}

void Ped::Model::setupHeatmapCUDA()
{
    int **d_heatmap, **d_scaled_heatmap, **d_blurred_heatmap;

    size_t heatmapPointerSize = SIZE * sizeof(int*);
    size_t scaledHeatmapPointerSize = SCALED_SIZE * sizeof(int*);
    size_t scaledHeatmapSize = SCALED_SIZE*SCALED_SIZE*sizeof(int);

    // TODO: use cudamallochost
    heatmap = (int**)malloc(heatmapPointerSize);
	scaled_heatmap = (int**)malloc(scaledHeatmapPointerSize);
	blurred_heatmap = (int**)malloc(scaledHeatmapPointerSize);

    int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	int *shm = (int*)malloc(scaledHeatmapSize);
	int *bhm = (int*)malloc(scaledHeatmapSize);

    // 1.cudaMalloc 
    cudaMalloc(&d_heatmap, heatmapPointerSize);
    cudaMalloc(&d_scaled_heatmap, scaledHeatmapPointerSize);
    cudaMalloc(&d_blurred_heatmap, scaledHeatmapPointerSize);

    // 2.cudaMemcpyHostToDevice 
    cudaMemcpy(d_heatmap, heatmap, heatmapPointerSize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_scaled_heatmap, scaled_heatmap, scaledHeatmapPointerSize,cudaMemcpyHostToDevice);
	cudaMemcpy(d_blurred_heatmap, blurred_heatmap, scaledHeatmapPointerSize,cudaMemcpyHostToDevice);
    
    // 3.kernel <<<numBlocks,numThreads>>>() 
    initHeatMap<<<1, SIZE>>>(d_heatmap, hm, SIZE);
	initHeatMap<<<CELLSIZE, SIZE>>>(d_scaled_heatmap, shm, SCALED_SIZE);
	initHeatMap<<<CELLSIZE, SIZE>>>(d_blurred_heatmap, bhm, SCALED_SIZE);
    
    // 4.cudaMemcpyDeviceToHost 
    cudaMemcpy(heatmap, d_heatmap, heatmapPointerSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(scaled_heatmap, d_scaled_heatmap, scaledHeatmapPointerSize,cudaMemcpyDeviceToHost);
	cudaMemcpy(blurred_heatmap, d_blurred_heatmap, scaledHeatmapPointerSize,cudaMemcpyDeviceToHost);

    // 5.cudaFree 
    /*
    cudaFree(d_heatmap);
    cudaFree(d_scaled_heatmap);
    cudaFree(d_blurred_heatmap);
    */
}

void Ped::Model::updateHeatmapCUDA()
{
    this->updateHeatmapSeq();
}

