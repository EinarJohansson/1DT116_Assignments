//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2015
//

#ifndef _parsescenario_h_
#define _parsescenario_h_

#include "ped_agent.h"
#include "ped_waypoint.h"
#include <tinyxml2.h>
#include <map>
#include <vector>
#include <set>
#include <string>
#include <cstdlib>

using namespace std;
using namespace tinyxml2;

class ParseScenario
{
public:
	ParseScenario() {}
	ParseScenario(std::string filename, bool verbose = false);
	~ParseScenario() {}

	// returns the collection of agents defined by this scenario
	vector<Ped::Tagent*> getAgents() const;
	uint32_t* getX() const;
	uint32_t* getY() const;

	// contains all defined waypoints
	vector<Ped::Twaypoint*> getWaypoints();

private:
	XMLDocument doc;

	// final collection of all created agents
	vector<Ped::Tagent*> agents;
	uint32_t* agents_x;
	uint32_t* agents_y;

	// temporary collection of agents used to
	// keep track of all agents that are generated
	// within the current opened agents xml tag
	vector<Ped::Tagent*> tempAgents;


	// contains all defined waypoints
	map<string, Ped::Twaypoint*> waypoints;
};

#endif
