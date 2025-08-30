#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

double NeuralNetwork_GetSigmoid(double data);
double NeuralNetwork_GetDerivativeSigmoid(double data);
double NeuralNetwork_GetRelu(double data);
double NeuralNetwork_GetDerivativeRelu(double data);
double NeuralNetwork_GetTanh(double data);

#endif