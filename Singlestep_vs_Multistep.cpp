//
//  Created by Harm van Seijen on 2015-05-20.
//  Copyright (c) 2015 Harm van Seijen. All rights reserved.
//

#include <cmath>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

#include "LS_Sarsa_lambda.h"
#include "Mountain_Car_Task.h"


int main ()
{
    srand((unsigned int)time(0));
    
    // Domain Settings ///////////////////////////////////////////
    Domain_Settings dsettings;
    dsettings.num_tilings = 3;
    dsettings.num_x_tiles = 3;
    dsettings.num_v_tiles = 3;
    dsettings.gamma = 1;
    
    // Agent Settings ///////////////////////////////////////////
    Agent_Settings asettings;
    asettings.alpha = 0.01/double(dsettings.num_tilings);
    // asettings.lambda = 0;  // for lambda value: see below
    asettings.k = 1;
    asettings.epsilon = 0.01;
    asettings.theta_init = 0;
    asettings.replay_like = true; // if true, beta := alpha, d_init:= theta_init/alpha, and A_init:= I/alpha
    
    // Experiment Settings ///////////////////////////////////////////
    const int num_experiments = 2;
    const int num_episodes = 2000;
    const int num_runs = 100;
    const int max_steps_per_episode = 10000;
    const char directory[] = "./";
    const char filename[] = "singlestep_vs_multistep.txt";

    double avg_steps_per_episode[num_experiments][num_episodes];
    for (int i = 0; i < num_experiments; i++)
    {
        for (int j = 0; j < num_episodes; j++)
        {
            avg_steps_per_episode[i][j] = 0;
        }
    }

    
    for (int exp = 0; exp < num_experiments; exp++)
    {
        if (exp == 0)
        {
            asettings.lambda = 0;
        }
        else
        {
            asettings.lambda = 0.95;
        }
        
        Mountain_Car_Task domain(dsettings);
        LS_Sarsa_lambda agent(&domain, asettings);
        
        clock_t start_time,end_time;
        start_time = clock();
        for (int run = 0; run < num_runs; run++)
        {
            cout << "exp: " << exp << ", run: " << run << endl;
            double steps_per_episode[num_episodes];
            agent.initialize();
            for (int episode = 0; episode < num_episodes; episode++)
            {
                int num_steps;
                agent.run_episode(num_steps, max_steps_per_episode);
                steps_per_episode[episode] = num_steps;
            }
            
            double avg_factor = 1/double(run+1);
            for (int ep = 0; ep < num_episodes; ++ep)
            {
                avg_steps_per_episode[exp][ep] *= 1 - avg_factor;
                avg_steps_per_episode[exp][ep] += avg_factor * steps_per_episode[ep];
            }
        }
        end_time = clock();
        double dtime = double(end_time-start_time)/double(CLOCKS_PER_SEC);
        cout << "calc time : " << dtime << " seconds " << endl;
    }
    
    for (int exp = 0; exp < num_experiments; exp++)
    {
        double avg_steps = 0;
        for (int i = 0; i < num_episodes; ++i)
        {
            avg_steps += avg_steps_per_episode[exp][i];
        }
        avg_steps = avg_steps/double(num_episodes);
        cout << "Average number of steps: " << avg_steps << endl;
    }
    
    // print results to file
    char file_dir_name[100];
    FILE * fp;
    sprintf(file_dir_name,"%s%s",directory,filename);
    fp = fopen(file_dir_name,"w");
    if (fp == NULL) assert(false);
    for (int ep = 0; ep < num_episodes; ep++)
    {
        fprintf(fp,"%1.6f  %1.6f\n",avg_steps_per_episode[0][ep],avg_steps_per_episode[1][ep]);
    }
    fclose(fp);
    
}