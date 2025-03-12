//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>
#include <stdint.h>

#include "ped_agent.h"

namespace Ped{
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ };

	class Model
	{
	public:

		// Sets everything up
		void setup(
			std::vector<Ped::Tagent*> agentsInScenario, 
			std::vector<Twaypoint*> destinationsInScenario, 
			IMPLEMENTATION implementation);
		void split(std::vector<Ped::Tagent *> &temp_agents);
        void thread_func(const std::vector<Ped::Tagent *> &agents, int id);
        void omp_run(std::vector<Ped::Tagent *> &agents);
		void sort(Ped::Tagent *agent, int xPrev);
        // Coordinates a time step in the scenario: move all agents by one step (if applicable).
        void tick();

		// Returns the agents of this scenario
		const std::vector<Tagent*>& getAgents() const { return agents; };

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

	private:

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		// The agents in this scenario
		std::vector<Tagent*> agents;
		std::vector<Tagent*> temp;
		std::vector<Tagent*> agentsQ1;
		std::vector<Tagent*> agentsQ2;
		std::vector<Tagent*> agentsQ3;
		std::vector<Tagent*> agentsQ4;

        uint32_t* agents_x; 
		uint32_t* agents_y;

		float* dest_x; 
		float* dest_y;
		float* dest_r;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		size_t heatmapPointerSize = SIZE * sizeof(int*);
		size_t scaledHeatmapPointerSize = SCALED_SIZE * sizeof(int*);
		size_t scaledHeatmapSize = SCALED_SIZE*SCALED_SIZE*sizeof(int);
		size_t heatmapSize = SIZE*SIZE*sizeof(int);

		// The heatmap representing the density of agents
		int **heatmap, **scaled_heatmap, **blurred_heatmap;
		int *d_hm, *d_shm, *d_bhm;
		int *hm, *shm, *bhm;
		int *d_agents_desired_x, *d_agents_desired_y;
		int *h_agents_desired_x, *h_agents_desired_y;
		
		size_t agentSize;
		// SEQ
		void setupHeatmapSeq();
		void updateHeatmapSeq();

		// CUDA
		void setupHeatmapCUDA();
		void updateHeatmapCUDA();
		void cuda_fin();
	};
}
#endif
