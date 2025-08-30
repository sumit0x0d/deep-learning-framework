#include <math.h>

#include <activation-functions.h>

double NeuralNetwork_GetSigmoid(double data)
{
     return 1.0 / (1 + exp(-data));
}

double NeuralNetwork_GetDerivativeSigmoid(double data)
{
     return get_sigmoid(data) * (1 - get_sigmoid(data));
}

double NeuralNetwork_GetRelu(double data)
{
     if (data < 0) {
          return 0;
     }
     return data;
}

double NeuralNetwork_GetDerivativeRelu(double data)
{
     if (data < 0) {
          return 0;
     }
     return 1;
}

double NeuralNetwork_GetTanh(double data)
{
     return (2 / (1 + exp(-data * 2))) - 1;
}