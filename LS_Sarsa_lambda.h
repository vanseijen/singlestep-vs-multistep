//
//  Created by Harm van Seijen on 2015-05-20.
//  Copyright (c) 2015 Harm van Seijen. All rights reserved.
//

#ifndef __MountainCar__LS_Sarsa_lambda__
#define __MountainCar__LS_Sarsa_lambda__

#include <cmath>
#include "ControlDomain.h"
#include "Utilities.h"


struct Agent_Settings {
    double alpha;
    double lambda;
    int k;
    double epsilon;
    double theta_init;
    bool replay_like;
    double beta;
    double d_init;
    double A_init;
};

class LS_Sarsa_lambda
{
    ControlDomain* domain;
    int num_state_features;
    int num_actions;
    int num_active_features;
    int total_features;

    double alpha;
    double lambda;
    int k;
    double epsilon;
    double theta_init;
    bool replay_like;
    double beta;
    double d_init;
    double A_init;

    double *theta;
    double *e_trace;
    double *d;
    double *A;
    
    int select_action(int state_features[]);
    void get_action_features(int state_features[], int action, int action_features[]);

public:
    LS_Sarsa_lambda(ControlDomain*, Agent_Settings&);
    ~LS_Sarsa_lambda();
    void initialize();
    double run_episode(int &num_steps, int max_steps);
};


#endif /* defined(__MountainCar__LS_Sarsa_lambda__) */
