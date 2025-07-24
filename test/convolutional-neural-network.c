#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <convolutional-neural-network.h>
// #include "activation-functions.h"
#include <weight-initilization-functions.h>

int main(void)
{
     size_t hnCount[] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
     struct convolutional_neural_network_create_info cnncInfo = {
          .input_neuron_count = 10,
          .output_neuron_count = 10,
          .hidden_layer_count = 10,
          .hidden_neuron_count = hnCount,
          .learning_rate = 10.0,
          .weight_initilization = get_uniform_distribution
     };
     ConvolutionalNeuralNetwork *cnNetwork = ConvolutionalNeuralNetwork_Create(&cnncInfo);
     // convolutional_neural_network_train(neural_network, neural_network->hidden_layer[0],
     // neural_network->hiddenLayer[neural_network->hiddenLayerCount], get_relu);
     // convolutional_neural_network_save(neural_network);
     ConvolutionalNeuralNetwork_Print(cnNetwork);
}
