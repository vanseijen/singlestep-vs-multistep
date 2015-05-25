//
//  Created by Harm van Seijen on 2015-05-20.
//  Copyright (c) 2015 Harm van Seijen. All rights reserved.
//

#ifndef __MountainCar__Mountain_Car_Task__
#define __MountainCar__Mountain_Car_Task__

#include <cmath>
#include <cassert>
#include <cstdlib>
#include "ControlDomain.h"
#include "Utilities.h"
using namespace std;


struct Domain_Settings {
    int num_tilings;
    int num_x_tiles;
    int num_v_tiles;
    double gamma;
};


struct State {
    double x;
    double v;
    bool terminal;
    
};

class Mountain_Car_Task: public ControlDomain
{
        State current_state;
    
        double gamma;
        int num_actions;
        int num_tilings;
        int num_x_tiles;
        int num_v_tiles;
        int num_active_features;
        int num_total_features;
        double x_range[2];
        double v_range[2];
        double *tiling_x_offset;
        double *tiling_v_offset;
        void getActiveStateFeatures(State s, int F[]);
        State getNextState(State s, int a);
    public:
        Mountain_Car_Task(Domain_Settings&);
        ~Mountain_Car_Task();
        void get_domain_size(int &n_total_features, int &n_active_features, int &n_actions);
        void initialize();
        void reset_state();
        void get_state_features(double &gamma_, int state_features[]);
        void take_action(int action, double &reward, double &gamma, int next_active_features[]);
};

#endif /* defined(__MountainCar__Mountain_Car_Task__) */
