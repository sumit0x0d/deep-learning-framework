#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <math.h>

static double get_sigmoid(double data)
{
     return 1.0 / (1 + exp(-data));
}

static double get_derivative_sigmoid(double data)
{
     return get_sigmoid(data) * (1 - get_sigmoid(data));
}

static double get_relu(double data)
{
     if (data < 0) {
          return 0;
     }
     return data;
}

static double get_derivative_relu(double data)
{
     if (data < 0) {
          return 0;
     }
     return 1;
}

static double get_tanh(double data)
{
     return (2 / (1 + exp(-data * 2))) - 1;
}

#endif
