//
//  Created by Harm van Seijen on 2015-05-20.
//  Copyright (c) 2015 Harm van Seijen. All rights reserved.
//

#include "Mountain_Car_Task.h"



Mountain_Car_Task::Mountain_Car_Task(Domain_Settings& dsettings)
{
    num_actions = 3;
    num_tilings = dsettings.num_tilings;
    num_x_tiles = dsettings.num_x_tiles;
    num_v_tiles = dsettings.num_v_tiles;
    gamma = dsettings.gamma;
    num_active_features = num_tilings;
    num_total_features = num_tilings*num_x_tiles*num_v_tiles;
    x_range[0] = -1.2;
    x_range[1] = 0.5;
    v_range[0] = -0.07;
    v_range[1] = 0.07;
    tiling_x_offset = new double[num_tilings];
    tiling_v_offset = new double[num_tilings];
    for (int t = 0; t < num_tilings; t++)
    {
        tiling_x_offset[t] = 0;
        tiling_v_offset[t] = 0;
    }
}

Mountain_Car_Task::~Mountain_Car_Task()
{
    delete[] tiling_x_offset;
    delete[] tiling_v_offset;
}



State Mountain_Car_Task::getNextState(State s, int a)
{
    // implement dynamics. set s.x and s.v
    State s2;
    s2.terminal = false;
    s2.v = s.v + 0.001*(a-1)-0.0025*cos(3*s.x);
    if (s2.v < v_range[0])
    {
        s2.v = v_range[0];
    }
    else if (s2.v > v_range[1])
    {
        s2.v = v_range[1];
    }
    
    s2.x = s.x + s2.v;
    if (s2.x <= x_range[0])
    {
        s2.x = x_range[0];
        s2.v = 0;
    }
    else if (s2.x >= x_range[1])
    {
        s2.x = x_range[1];
        s2.v = 0;
        s2.terminal = true;
    }
    
    return s2;
}

void Mountain_Car_Task::getActiveStateFeatures(State s, int active_features[])
{
    double x_size = (x_range[1]-x_range[0])/double(num_x_tiles-1);
    double v_size = (v_range[1]-v_range[0])/double(num_v_tiles-1);
    
    for(int t = 0; t < num_active_features; ++t)
    {
        double x = s.x + tiling_x_offset[t];
        double v = s.v + tiling_v_offset[t];
        
        int fx = int(floor((x - x_range[0])/x_size));
        fx = fmin(fx,num_x_tiles); // catch border case
        int fv = int(floor((v - v_range[0])/v_size));
        fv = fmin(fv,num_v_tiles); // catch border case
        
        int ft = fx + (num_x_tiles)*fv + t*(num_x_tiles*num_v_tiles);
        assert(ft >= 0);
        assert(ft < num_total_features);
        active_features[t] = ft;
    }
    
    return;
}

void Mountain_Car_Task::initialize()
{
    // This function can be used to initialize the domain, for example
    // by randomizing the offset of the tilings.
    // For our experiment, we used fixed tile positions. The reason is that
    // because we only use 3 tilings, randomization can cause huge variances.
    // with some representations being very bad (f.e., if all three tilings
    // have similar offset).
    
    double x_tile_size = (x_range[1]-x_range[0])/double(num_x_tiles);
    double v_tile_size = (v_range[1]-v_range[0])/double(num_v_tiles);
    
//    for (int t = 0; t < num_tilings; ++t)
//    {
//        tiling_x_offset[t] = random_double(0,x_tile_size);
//        tiling_v_offset[t] = random_double(0,v_tile_size);
//    }
    
    tiling_x_offset[0] = 0;
    tiling_v_offset[0] = 0;
    tiling_x_offset[1] = 1.5/double(num_tilings)*x_tile_size;
    tiling_v_offset[1] = 1.5/double(num_tilings)*v_tile_size;
    tiling_x_offset[2] = 2/double(num_tilings)*x_tile_size;
    tiling_v_offset[2] = 2/double(num_tilings)*v_tile_size;
    return;
}


void Mountain_Car_Task::reset_state()
{
    current_state.x = random_double(x_range[0],x_range[1]);
    current_state.v = 0;
    current_state.terminal = false;
    return;
    
}

void Mountain_Car_Task::get_state_features(double &gamma_, int state_features[])
{
    getActiveStateFeatures(current_state, state_features);
    if (current_state.terminal)
    {
        gamma_ = 0.0;
    }
    else
    {
        gamma_ = gamma;
    }
    return;
    
}


void Mountain_Car_Task::take_action(int action, double &reward, double &gamma_, int next_active_features[])
{
    assert(action >= 0);
    assert(action < num_actions);
    current_state = getNextState(current_state, action);
    
    
    reward = -1;
    if (current_state.terminal)
    {
        gamma_ = 0;
    }
    else
    {
        gamma_ = gamma;
    }
    getActiveStateFeatures(current_state, next_active_features);
    
    return; 
}


void Mountain_Car_Task::get_domain_size(int &n_total_features, int &n_active_features, int &n_actions)
{
    n_total_features= num_total_features;
    n_active_features = num_active_features;
    n_actions = num_actions;
}
