//
//  Created by Harm van Seijen on 2015-05-24.
//  Copyright (c) 2015 Harm van Seijen. All rights reserved.
//

#ifndef MountainCar_ControlDomain_h
#define MountainCar_ControlDomain_h

class ControlDomain
{
    public:
        virtual void get_domain_size(int &num_total_features, int &num_active_features, int &num_actions) = 0;
        virtual void initialize() = 0;
        virtual void reset_state() = 0;
        virtual void get_state_features(double &gamma, int state_features[]) = 0;
        virtual void take_action(int action, double &reward, double &gamma, int next_active_features[]) = 0;
};

#endif
