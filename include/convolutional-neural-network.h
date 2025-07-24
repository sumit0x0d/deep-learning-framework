#ifndef CONVOLUTIONAL_NEURAL_NETWORK_H
#define CONVOLUTIONAL_NEURAL_NETWORK_H

#include <stdlib.h>
#include <matrix.h>

typedef struct convolutional_neural_network ConvolutionalNeuralNetwork;
typedef double (*ConvolutionalNeuralNetworkActivation)(double weight);
typedef double (*ConvolutionalNeuralNetworkWeightInitilization)(double fIn, double fOut);

typedef struct convolutional_neural_network_create_info {
     size_t input_neuron_count;
     size_t output_neuron_count;
     size_t hidden_layer_count;
     size_t *hidden_neuron_count;
     double learning_rate;
     ConvolutionalNeuralNetworkWeightInitilization weight_initilization;
} ConvolutionalNeuralNetworkCreateInfo;

ConvolutionalNeuralNetwork *ConvolutionalNeuralNetwork_Create(const ConvolutionalNeuralNetworkCreateInfo *cnncInfo);
void ConvolutionalNeuralNetwork_Destroy(ConvolutionalNeuralNetwork *cnNetwork);
void ConvolutionalNeuralNetwork_TrainOnCsv(ConvolutionalNeuralNetwork *cnNetwork, Matrix *iLayer, Matrix *oLayer,
                                           ConvolutionalNeuralNetworkActivation cnnActivation);
void ConvolutionalNeuralNetwork_Predict(ConvolutionalNeuralNetwork *cnNetwork, Matrix *iLayer);
void ConvolutionalNeuralNetwork_Save(ConvolutionalNeuralNetwork *cnNetwork, const char *dName);
ConvolutionalNeuralNetwork *ConvolutionalNeuralNetwork_Load(const char *dName);
void ConvolutionalNeuralNetwork_Print(ConvolutionalNeuralNetwork *cnNetwork);

#endif
