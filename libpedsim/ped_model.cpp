//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <chrono>
using namespace std::chrono;

#ifdef __arm__
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#include <emmintrin.h>
#else
#error "Unsupported Architecture"
#endif

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>
#include <cmath>
#define CORES 4
#define REGIONS 4
#define MAX_X 160
#define MAX_Y 120

omp_lock_t Q1Q2lock;
omp_lock_t Q2Q3lock;
omp_lock_t Q3Q4lock;

void Ped::Model::setup(
	std::vector<Ped::Tagent*> agentsInScenario,
	std::vector<Twaypoint*> destinationsInScenario, 
	IMPLEMENTATION implementation)
{
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	cuda_test();
#else
    std::cout << "Not compiled for CUDA" << std::endl;
#endif

    agents = std::vector<Ped::Tagent *>(agentsInScenario.begin(), agentsInScenario.end());
    temp = std::vector<Ped::Tagent *>();
    agentsQ1 = std::vector<Ped::Tagent *>();
    agentsQ2 = std::vector<Ped::Tagent *>();
    agentsQ3 = std::vector<Ped::Tagent *>();
    agentsQ4 = std::vector<Ped::Tagent *>();


    omp_init_lock(&Q1Q2lock);
    omp_init_lock(&Q2Q3lock);
    omp_init_lock(&Q3Q4lock);

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	int size = (1 + (agents.size() / 4))*4;

	agents_x = (uint32_t *) _mm_malloc(size* sizeof(uint32_t), 16);
	agents_y = (uint32_t *) _mm_malloc(size* sizeof(uint32_t), 16);

	dest_x = (float *) _mm_malloc(size * sizeof(float), 16);
	dest_y = (float *) _mm_malloc(size * sizeof(float), 16);
	dest_r = (float *) _mm_malloc(size * sizeof(float), 16);

	// Initialize values of coordinates
	for (int i = 0; i < agents.size(); i++) {
		agents_x[i] = agents[i]->getX();
		agents_y[i] = agents[i]->getY();
		dest_x[i] = agents[i]->getDestX();
		dest_y[i] = agents[i]->getDestY();
		dest_r[i] = agents[i]->getDestR();
	}
    split(agents);
	// Set up heatmap (relevant for Assignment 4)
    if (implementation == SEQ)
    {
        setupHeatmapSeq();
    }
    else
    {
        setupHeatmapCUDA();
    }
}
// Gå igenom alla agenter och lägg till agenterna i respektive kvadrant.
void Ped::Model::split(std::vector<Ped::Tagent *> &temp_agents)
{
    for (Ped::Tagent *agent : temp_agents)
    {
        const int xPos = agent->getX();
        
        if (xPos<(MAX_X/REGIONS)){
            // Quadrant 1 åt vänster
            agentsQ1.push_back(agent);
        }
        else if(xPos<2*(MAX_X/REGIONS)){
            // Quadrant 1 åt vänster
            agentsQ2.push_back(agent);
        } 
        else if(xPos<3*(MAX_X/REGIONS)){
            // Quadrant 1 åt vänster
            agentsQ3.push_back(agent);
        }
        else {
            agentsQ4.push_back(agent);
        }
    }
}
void Ped::Model::thread_func(const std::vector<Ped::Tagent *> &agents, int id)
{
    size_t agentsPerThread = std::ceil(agents.size() / CORES);

    size_t start = agentsPerThread * id;
    size_t end = (id + 1 == CORES) ? agents.size() : start + agentsPerThread;

    for (int i = start; i < end; i++)
    {
        agents[i]->computeNextDesiredPosition();

        // agents[i]->setX(agents[i]->getDesiredX());
        // agents[i]->setY(agents[i]->getDesiredY());

        // 4. Assignment 3:
        move(agents[i]);
    }
}

// Removes an agent from its old quadrant.
void Ped::Model::sort(Ped::Tagent *agent, int xPrev)
{
   int xNew = agent->getX();

   if (xPrev < 40 && xNew >= 40) {
        temp.push_back(agent);
        agentsQ1.erase(std::remove_if(agentsQ1.begin(), agentsQ1.end(),
        [&](Ped::Tagent *a) { return a == agent; }),agentsQ1.end());
   }
   else if (xPrev >= 40 && xPrev < 80 && (xNew < 40 || xNew >= 80)) {
        temp.push_back(agent);
        agentsQ2.erase(std::remove_if(agentsQ2.begin(), agentsQ2.end(),
        [&](Ped::Tagent *a) { return a == agent; }),agentsQ2.end());
    }
     
    else if (xPrev >= 80 && xPrev < 120 && (xNew < 80 || xNew >= 120)) {
        temp.push_back(agent);
        agentsQ3.erase(std::remove_if(agentsQ3.begin(), agentsQ3.end(),
        [&](Ped::Tagent *a) { return a == agent; }),agentsQ3.end());
    } 
   else if (xPrev >= 120 && xNew < 120) {
        temp.push_back(agent);
        agentsQ4.erase(std::remove_if(agentsQ4.begin(), agentsQ4.end(),
        [&](Ped::Tagent *a) { return a == agent; }),agentsQ4.end());
    }
}

// Gå igenom alla agenter och flytta dem till nästa position.
void Ped::Model::omp_run(std::vector<Ped::Tagent *> &agents)
{
    for (Ped::Tagent *agent : agents)
    {
        // 2. Calculate its next desired position
        agent->computeNextDesiredPosition();
        int xPos = agent->getX();

        int region_size = MAX_X / REGIONS;
        int q1_border = region_size;
        int q2_border = 2* region_size;
        int q3_border = 3* region_size;
        int q4_border = 4* region_size;

        if ((xPos < q1_border-2 || xPos > q3_border+1) || ((xPos%region_size) > 1) && ((xPos % region_size) < q1_border-2)) // Not in any boarder region
        {
            move(agent);
        }
        else if (xPos < q1_border+2 && xPos > q1_border-3) //REGION 1,2 LOCK
        {
            omp_set_lock(&Q1Q2lock);
            move(agent);
            sort(agent, xPos);
            omp_unset_lock(&Q1Q2lock);
        }
        else if (xPos < q2_border+2 && xPos > q2_border-3) // REGION 2,3 LOCK
        {
            omp_set_lock(&Q2Q3lock);
            move(agent);
            sort(agent, xPos);
            omp_unset_lock(&Q2Q3lock);
        }
        else  // REGION 3,4 LOCK
        {
            omp_set_lock(&Q3Q4lock);
            move(agent);
            sort(agent, xPos);
            omp_unset_lock(&Q3Q4lock);
        }
    }
}

void Ped::Model::tick()
{
    switch (implementation)
    {
    case SEQ:
    {
        // 1. Retrieve each agent.
        for (size_t i = 0; i < agents.size(); i++)
        {
            // 2. Calculate its next desired position
            agents[i]->computeNextDesiredPosition();

            // 3. Set its position to the calculated desired one
            // agents[i]->setX(agents[i]->getDesiredX());
            // agents[i]->setY(agents[i]->getDesiredY());

            // 4. Assignment 3:
            move(agents[i]);
        }
        updateHeatmapSeq();
        break;
    }
    case PTHREAD: {
      std::vector<std::thread> threads;

      for(int i =0;i<CORES;i++)
      {
        threads.push_back(std::thread(&Ped::Model::thread_func, this, std::cref(agents), i));
      }
      
      for (size_t i = 0; i < CORES; i++)
      {
        threads[i].join();
      }
      break;
    }
    case OMP:
    {
        std::vector<Ped::Tagent *> *quadrants[REGIONS] = {&agentsQ1, &agentsQ2, &agentsQ3, &agentsQ4};
        int thread_id;
        auto start = high_resolution_clock::now();
        // updateHeatmapCUDA();
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        
        // cout << "Time taken by function: "
        //    << duration.count() << " microseconds" << endl;
        
        omp_set_num_threads(CORES);
        #pragma omp parallel private(thread_id)
        {
            thread_id = omp_get_thread_num();
            // Flytta alla agenter inom en kvadrant.
            omp_run(*quadrants[thread_id]);
        }
        // Tilldela agenter till kvadrant vektorerna.
        split(temp);
        temp.clear();
        updateHeatmapSeq();
        //cuda_fin();
        break;
    }
	case VECTOR: {
		for (size_t i = 0; i < agents.size(); i+=4)
	  	{
			//////////////// getNextDestination() ////////////////
			__m128i x = _mm_load_si128((__m128i *) &agents_x[i]);
			__m128i y = _mm_load_si128((__m128i *) &agents_y[i]);

			__m128 destX = _mm_load_ps(&dest_x[i]);
			__m128 destY = _mm_load_ps(&dest_y[i]);
			__m128 destR = _mm_load_ps(&dest_r[i]);

			__m128 diffX = _mm_sub_ps(destX, _mm_cvtepi32_ps(x));
			__m128 diffY = _mm_sub_ps(destY, _mm_cvtepi32_ps(y));
			__m128 len = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));

			__m128 agentReach = _mm_cmplt_ps(len, destR);
			int mask = _mm_movemask_ps(agentReach);

			for (int j = 0; j < 4; j++) {
				if (mask & 1) {
					if (i+j < agents.size()) {
						// Destination är uppdaterad
						agents.at(i+j)->updateWaypoints();
						dest_x[i+j] = agents.at(i+j)->getDestX();
						dest_y[i+j] = agents.at(i+j)->getDestY();
						dest_r[i+j] = agents.at(i+j)->getDestR();
					}
				}
				mask >>= 1;
			}

			__m128i desiredPositionX = _mm_cvtps_epi32(_mm_add_ps(_mm_cvtepi32_ps(x), _mm_div_ps(diffX, len)));
			__m128i desiredPositionY = _mm_cvtps_epi32(_mm_add_ps(_mm_cvtepi32_ps(y), _mm_div_ps(diffY, len))); 

			// Update agent x and y vectors with desired position
			_mm_store_si128((__m128i *) &agents_x[i], desiredPositionX);
			_mm_store_si128((__m128i *) &agents_y[i], desiredPositionY);
     	}

	
		for (size_t j = 0; j < agents.size(); j++)
		{
			agents[j]->setX(agents_x[j]);
			agents[j]->setY(agents_y[j]);
		}
		break;
	}
    default:
      cout << "undefined implementation\n";
  }
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}
