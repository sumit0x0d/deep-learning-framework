#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>

#include <matrix.h>

typedef struct neural_network NeuralNetwork;
typedef Matrix NeuralNetworkLayer;
typedef double (*NeuralNetworkActivation)(double weight);
typedef double (*NeuralNetworkWeightInitilization)(double fIn, double fOut);

typedef struct neural_network_create_info {
     size_t input_neuron_count;
     size_t output_neuron_count;
     size_t hidden_layer_count;
     size_t *hidden_neuron_count;
     double learning_rate;
     NeuralNetworkWeightInitilization weight_initilization;
} NeuralNetworkCreateInfo;

NeuralNetwork *NeuralNetwork_Create(const NeuralNetworkCreateInfo *nncInfo);
void NeuralNetwork_Destroy(NeuralNetwork *nNetwork);
void NeuralNetwork_TrainOnCsv(NeuralNetwork *nNetwork, Matrix *lInput, NeuralNetworkLayer *lOutput,
                              NeuralNetworkActivation nnaFunction);
void NeuralNetwork_Predict(NeuralNetwork *nNetwork, NeuralNetworkLayer *lInput);
void NeuralNetwork_Save(NeuralNetwork *nNetwork, const char *dName);
NeuralNetwork *NeuralNetwork_Load(const char *dName);
void NeuralNetwork_Print(NeuralNetwork *nNetwork);

#endif
