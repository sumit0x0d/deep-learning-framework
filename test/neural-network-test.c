#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <neural-network.h>
// #include "activation-functions.h"
#include <weight-initilization-functions.h>

int main(void)
{
    size_t hnCount[] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
    NeuralNetworkCreateInfo nncInfo = {
        .input_neuron_count = 10,
        .output_neuron_count = 10,
        .hidden_layer_count = 10,
        .hidden_neuron_count = hnCount,
        .learning_rate = 10.0,
        .weight_initilization = NeuralNetwork_GetUniformDistribution
    };
    NeuralNetwork *cnNetwork = NeuralNetwork_Create(&nncInfo);
    // neural_network_train(neural_network, neural_network->hidden_layer[0],
    // neural_network->hiddenLayer[neural_network->hiddenLayerCount], get_relu);
    // neural_network_save(neural_network);
    NeuralNetwork_Print(cnNetwork);
}
