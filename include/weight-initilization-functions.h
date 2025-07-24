#ifndef WEIGHT_INITILIZATION_FUNCTIONS_H
#define WEIGHT_INITILIZATION_FUNCTIONS_H

#include <math.h>
#include <stdlib.h>

static double get_weight(double maximum, double minimum)
{
     int scale = 1000000;
     int sdifference = (int)((maximum - minimum) * scale);
     return minimum + (1.0 * (rand() % (int)sdifference) / scale);
}

static double get_uniform_distribution(double ifan, double ofan)
{
     ofan = ifan;
     double minimum = -1 / sqrt(ifan);
     double maximum = 1 / sqrt(ofan);
     return get_weight(maximum, minimum);
}

static double get_xavier_normal_distribution(double ifan, double ofan)
{
     double minimum = 0;
     double maximum = sqrt(2 / (ifan + ofan));
     return get_weight(maximum, minimum);
}

static double get_xavier_uniform_distribution(double ifan, double ofan)
{
     double minimum = -sqrt(6) / sqrt(ifan + ofan);
     double maximum = sqrt(6) / sqrt(ifan + ofan);
     return get_weight(maximum, minimum);
}

#endif
