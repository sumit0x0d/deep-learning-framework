#ifndef WEIGHT_INITILIZATION_FUNCTIONS_H
#define WEIGHT_INITILIZATION_FUNCTIONS_H

double NeuralNetwork_GetUniformDistribution(double fIn, double fOut);
double NeuralNetwork_GetXavierNormalDistribution(double fIn, double fOut);
double NeuralNetwork_GetXavierUniformDistribution(double fIn, double fOut);

#endif