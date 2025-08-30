#include <weight-initilization-functions.h>

#include <math.h>
#include <stdlib.h>

static double _GetWeight(double maximum, double minimum);

double NeuralNetwork_GetUniformDistribution(double fIn, double fOut)
{
    fOut = fIn;
    double minimum = -1 / sqrt(fIn);
    double maximum = 1 / sqrt(fOut);
    return _GetWeight(maximum, minimum);
}

double NeuralNetwork_GetXavierNormalDistribution(double fIn, double fOut)
{
    double minimum = 0;
    double maximum = sqrt(2 / (fIn + fOut));
    return _GetWeight(maximum, minimum);
}

double NeuralNetwork_GetXavierUniformDistribution(double fIn, double fOut)
{
    double minimum = -sqrt(6) / sqrt(fIn + fOut);
    double maximum = sqrt(6) / sqrt(fIn + fOut);
    return _GetWeight(maximum, minimum);
}

static double _GetWeight(double maximum, double minimum)
{
    int scale = 1000000;
    int sdifference = (int)((maximum - minimum) * scale);
    return minimum + (1.0 * (rand() % (int)sdifference) / scale);
}