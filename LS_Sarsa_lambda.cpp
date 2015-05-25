//
//  Created by Harm van Seijen on 2015-05-20.
//  Copyright (c) 2015 Harm van Seijen. All rights reserved.
//

#include "LS_Sarsa_lambda.h"

LS_Sarsa_lambda::LS_Sarsa_lambda(ControlDomain* domain_, Agent_Settings& settings)
{
    domain = domain_;
    domain->get_domain_size(num_state_features,num_active_features,num_actions);
    total_features = num_state_features*num_actions;
    
    alpha = settings.alpha;
    lambda = settings.lambda;
    k = settings.k;
    epsilon = settings.epsilon;
    theta_init = settings.theta_init;
    replay_like = settings.replay_like;
    if (!replay_like)
    {
        beta = settings.beta;
        d_init = settings.d_init;
        A_init = settings.A_init;
        
    }
    
    theta = new double[total_features];
    e_trace = new double[total_features];
    d = new double[total_features];
    A = new double[total_features*total_features];
}


LS_Sarsa_lambda::~LS_Sarsa_lambda()
{
    delete[] theta;
    delete[] e_trace;
    delete[] d;
    delete[] A;
}


double LS_Sarsa_lambda::run_episode(int &num_steps, int max_steps)
{
    int *state_features = new int[num_active_features];
    int *action_features = new int[num_active_features];
    int *next_action_features = new int[num_active_features];
    double gamma, reward;
    
    double* dummy = new double[total_features]; // allocate memory for helper vector
    
    for (int i = 0; i < total_features; i++)
    {
        e_trace[i] = 0;
    }
    
    domain->reset_state();
    domain->get_state_features(gamma, state_features);
    
    int action = select_action(state_features);
    get_action_features(state_features, action, action_features);
    
    num_steps = 0;
    double G = 0;  // return from initial state to end of episode (or until max_steps have been reached)
    double total_discount = 1;
    while ((gamma != 0) && num_steps < max_steps)
    {
        
        domain->take_action(action, reward, gamma, state_features);
        int next_action = select_action(state_features);
        get_action_features(state_features, next_action, next_action_features);
        
        num_steps++;
        G += total_discount*reward;
        total_discount *= gamma;

        
        // update trace ///////////////////////////////////////////////
        double e_phi = 0;
        for (int i = 0; i < num_active_features; i++)
        {
            int f = action_features[i];
            e_phi +=  e_trace[f];
        }
        
        for (int i = 0; i < num_active_features; i++)
        {
            int f = action_features[i];
            e_trace[f] +=  1 - beta*e_phi;
        }

        // update A /////////////////////////////////////////////////
        // A = (I - beta*phi*phi^T)A + e(phi - gamma*phi')^T

        // A -= beta*phi*phi^T*A
        double *phiA = dummy;
        for (int col = 0; col < total_features; col++)
        {
            phiA[col] = 0;
            for (int i =0; i < num_active_features; i++)
            {
                int row = action_features[i];
                phiA[col] += A[row*total_features + col];
            }
        }
        
        for (int i =0; i < num_active_features; i++)
        {
            int row = action_features[i];
            for (int col =0; col < total_features; col++)
            {
                A[row*total_features + col] -= beta*phiA[col];
            }
        }
        
        // A += e*phi^T
        for (int i =0; i < num_active_features; i++)
        {
            int col = action_features[i];
            for (int row =0; row < total_features; row++)
            {
                A[row*total_features + col] += e_trace[row];
            }
        }
        // A -= gamma*e*(phi')^T
        for (int i =0; i < num_active_features; i++)
        {
            int col = next_action_features[i];
            for (int row =0; row < total_features; row++)
            {
                A[row*total_features + col] -= gamma*e_trace[row];
            }
        }
 
        
        
         // update d ///////////////////////////////////////////////
        double d_phi = 0;
        for (int i = 0; i < num_active_features; i++)
        {
            int f = action_features[i];
            d_phi +=  d[f];
        }
        for (int i = 0; i < num_active_features; i++)
        {
            int f = action_features[i];
            d[f] -= beta*d_phi;
        }
        
        for (int i = 0; i < total_features; i++)
        {
            d[i] += e_trace[i]*reward;
        }
        
        // decay trace ///////////////////////////////////////////////
        for (int i = 0; i < total_features; i++)
        {
            e_trace[i] *= gamma*lambda;
        }
        
        
        // Update theta ////////////////////////////////
        for (int i = 0; i < k; ++i)
        {
            double *Aphi = dummy;
            for (int row = 0; row < total_features; ++row)
            {
                Aphi[row] = 0;
                for (int col = 0; col < total_features; ++col)
                {
                    Aphi[row] += A[row*total_features + col]*theta[col];
                }
            }
            
            for (int row = 0; row < total_features; ++row)
            {
                theta[row] += alpha*(d[row] - Aphi[row]);
            }
        }
        
        for (int i = 0; i < num_active_features; i++)
        {
            action_features[i] =  next_action_features[i];
        }
        action = next_action;
        
    }
    
    delete[] state_features;
    delete[] action_features;
    delete[] next_action_features;
    delete[] dummy;
    return G;
}


int LS_Sarsa_lambda::select_action(int state_features[])
{
    if (num_actions == 1)
    {
        return 0;
    }
    
    if (epsilon == 1.0)
    {
        return random_int(0, num_actions-1);
    }
    
    // determine Q-values
    double *Q = new double[num_actions];
    for (int a = 0; a < num_actions; a++)
    {
        Q[a] = 0;
        for (int j = 0; j < num_active_features; j++)
        {
            int f = state_features[j] + a*num_state_features;
            Q[a] += theta[f];
        }
    }

    // determine Qmax & no_max of the array Q[s]
    double Qmax = Q[0];
    int num_max = 1;
    for (int i=1; i<num_actions; i++)
    {
        if (Q[i] > Qmax)
        {
            Qmax = Q[i];
            num_max = 1;
        }
        else if (Q[i] == Qmax)
        {
            ++num_max;
        }
    }
    
    // simultaneously compute selection probability for each action and select action
    double rnd = random_double(0.0, 1.0);
    double cumulative_prob = 0.0; // prob_cs is the cumulative probablity
    int action = num_actions - 1;
    for (int a = 0; a < num_actions - 1; a++)
    {
        double prob = epsilon / double(num_actions);
        if (Q[a] == Qmax)
        {
            prob += (1-epsilon)/double(num_max);
        }
        cumulative_prob += prob;
        
        if (rnd < cumulative_prob)
        {
            action = a;
            break;
        }
    }
    
    delete[] Q;
    return action;
}


void LS_Sarsa_lambda::initialize()
{
    domain->initialize();
    
    if (replay_like)
    {
        beta = alpha;
        d_init = theta_init/double(alpha);
        A_init = 0;
    }
    
    for (int i = 0; i < total_features; i++)
    {
        theta[i] = theta_init;
        e_trace[i] = 0;
        d[i] = d_init;
        for (int j = 0; j < total_features; j++)
        {
            A[i*total_features + j] = A_init;
        }
        if (replay_like)
        {
            A[i*total_features + i] = 1/double(alpha);
        }
    }
    
}

void LS_Sarsa_lambda::get_action_features(int state_features[], int action, int action_features[])
{
    for (int i = 0; i < num_active_features; i++)
    {
        action_features[i] = state_features[i] + action*num_state_features;
    }
}