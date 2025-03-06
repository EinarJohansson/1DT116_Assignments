#include <iostream>
#include "ped_model.h"

#define CHECK_CUDA_ERROR(call)                                              \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void initHeatMap(int **heatmap, int* hm, int size)
{
    int idx = blockIdx.x * SIZE + threadIdx.x;
	// TODO: Datarace?
	// atomicAdd(&heatmap[idx], (hm + size * idx));
    heatmap[idx] = hm + size * idx;

	// heatmap[idx] pekar på början av varje rad i heatmap
}

__global__ void heatmapFade(int **heatmap)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
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

    /*size_t heatmapPointerSize = SIZE * sizeof(int*);
    size_t scaledHeatmapPointerSize = SCALED_SIZE * sizeof(int*);
    size_t scaledHeatmapSize = SCALED_SIZE*SCALED_SIZE*sizeof(int);
	size_t heatmapSize = SIZE*SIZE*sizeof(int);*/

    // TODO: use cudamallochost
    heatmap = (int**)malloc(heatmapPointerSize);
	scaled_heatmap = (int**)malloc(scaledHeatmapPointerSize);
	blurred_heatmap = (int**)malloc(scaledHeatmapPointerSize);

    hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	shm = (int*)malloc(scaledHeatmapSize);
	bhm = (int*)malloc(scaledHeatmapSize);

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
	/*size_t heatmapPointerSize = SIZE * sizeof(int*);
	size_t scaledHeatmapPointerSize = SCALED_SIZE * sizeof(int*);
	size_t scaledHeatmapSize = SCALED_SIZE*SCALED_SIZE*sizeof(int);
	size_t heatmapSize = SIZE*SIZE*sizeof(int);*/

	int *d_hm,*d_bhm,*d_shm;
	int **d_heatmap, **d_scaled_heatmap, **d_blurred_heatmap;
	
	cudaMalloc(&d_hm, heatmapSize);
	cudaMalloc(&d_shm, scaledHeatmapSize);
	cudaMalloc(&d_bhm, scaledHeatmapSize);

	cudaMalloc(&d_heatmap, heatmapPointerSize);
    cudaMalloc(&d_scaled_heatmap, scaledHeatmapPointerSize);
    cudaMalloc(&d_blurred_heatmap, scaledHeatmapPointerSize);

	cudaMemcpy(d_hm, hm, heatmapSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_shm, shm, scaledHeatmapSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bhm, bhm, scaledHeatmapSize, cudaMemcpyHostToDevice);

	int **h_devicePointers = (int**)malloc(heatmapPointerSize);
	int **hb_devicePointers = (int**)malloc(scaledHeatmapPointerSize);
	int **hs_devicePointers = (int**)malloc(scaledHeatmapPointerSize);
	for(int i =0; i<SIZE;i++)
	{
		h_devicePointers[i]= d_hm + i*SIZE;
	}
	for(int i =0; i<SCALED_SIZE;i++)
	{
		hs_devicePointers[i]= d_shm + i*SCALED_SIZE;
	}
	for(int i =0; i<SCALED_SIZE;i++)
	{
		hb_devicePointers[i]= d_bhm + i*SCALED_SIZE;
	}
	
	cudaMemcpy(d_heatmap, h_devicePointers, heatmapPointerSize,cudaMemcpyHostToDevice);
	cudaMemcpy(d_scaled_heatmap, hs_devicePointers, heatmapPointerSize,cudaMemcpyHostToDevice);
	cudaMemcpy(d_blurred_heatmap, hb_devicePointers, heatmapPointerSize,cudaMemcpyHostToDevice);

	heatmapFade<<<SIZE, SIZE>>>(d_heatmap);

	free(hb_devicePointers);
	free(h_devicePointers);
	free(hs_devicePointers);

	Ped::Tagent **d_agents;
	cudaMalloc(&d_agents, sizeof(Ped::Tagent)*agents.size());
	Ped::Tagent** h_deviceAgents =(Ped::Tagent**)malloc(sizeof(Ped::Tagent)*agents.size());

	for (size_t i = 0; i < agents.size(); i++) {
		Ped:: Tagent* d_agent;
		cudaMalloc(&d_agent, sizeof(Ped::Tagent *));  // Allocate memory for each agent
		cudaMemcpy(d_agent, agents[i], sizeof(Ped::Tagent *), cudaMemcpyHostToDevice);
		h_deviceAgents[i] = d_agent;  // Store device pointer in host array
	}

	// Copy the host array of pointers to device memory
	cudaMemcpy(d_agents, h_deviceAgents, agents.size() * sizeof(Ped::Tagent*), cudaMemcpyHostToDevice);

	// delete[] h_deviceAgents;  // Clean up temporary host pointer array	
	agentCount<<<SIZE, SIZE>>>(d_heatmap, d_agents);

	scaleData<<<SIZE, SIZE>>>(d_heatmap, d_scaled_heatmap);

	free(h_deviceAgents);

	cudaMemcpy(heatmap, d_heatmap, heatmapPointerSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(scaled_heatmap, d_scaled_heatmap, scaledHeatmapPointerSize,cudaMemcpyDeviceToHost);
	cudaMemcpy(blurred_heatmap, d_blurred_heatmap, scaledHeatmapPointerSize,cudaMemcpyDeviceToHost);
	
		// Weights for blur filter
		const int w[5][5] = {
			{ 1, 4, 7, 4, 1 },
			{ 4, 16, 26, 16, 4 },
			{ 7, 26, 41, 26, 7 },
			{ 4, 16, 26, 16, 4 },
			{ 1, 4, 7, 4, 1 }
		};
	
		#define WEIGHTSUM 273
		// Apply gaussian blurfilter		       
		for (int i = 2; i < SCALED_SIZE - 2; i++)
		{
			for (int j = 2; j < SCALED_SIZE - 2; j++)
			{
				int sum = 0;
				for (int k = -2; k < 3; k++)
				{
					for (int l = -2; l < 3; l++)
					{
						sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
					}
				}
				int value = sum / WEIGHTSUM;
				blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
			}
		}
}

