//
//  Created by Harm van Seijen on 2015-05-24.
//  Copyright (c) 2015 Harm van Seijen. All rights reserved.
//

#include "Utilities.h"


double random_double(double lo, double hi)
{
    return lo + (hi-lo) * double(rand()) / double(RAND_MAX);
}

int random_int(int lo, int hi)
{
    return lo + rand() % (hi - lo + 1);
}