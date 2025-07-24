#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <convolutional-neural-network.h>

typedef ConvolutionalNeuralNetworkCreateInfo CreateInfo;

struct convolutional_neural_network {
     size_t input_neuron_count;
     size_t output_neuron_count;
     size_t hidden_layer_count;
     size_t *hidden_neuron_count;
     Matrix **hidden_layer;
     double learning_rate;
};

static int _compare_double(const void *data1, const void *data2);
static void _multiply_double(void *data, const void *data1, const void *data2);
static void _print_double(void *data);

ConvolutionalNeuralNetwork *ConvolutionalNeuralNetwork_Create(const CreateInfo *cInfo)
{
     ConvolutionalNeuralNetwork *cnNetwork = (ConvolutionalNeuralNetwork *)malloc(sizeof (ConvolutionalNeuralNetwork));
     assert(cnNetwork);
     cnNetwork->hidden_layer = (Matrix **)malloc((cInfo->hidden_layer_count + 1) * sizeof (Matrix *));
     assert(cnNetwork->hidden_layer);
     cnNetwork->hidden_layer[0] = Matrix_Create(sizeof (double), cInfo->input_neuron_count, cInfo->hidden_neuron_count[0]);
     for (size_t i = 1; i < cInfo->hidden_layer_count - 1; i++) {
          cnNetwork->hidden_layer[i] = Matrix_Create(sizeof (double), cInfo->hidden_neuron_count[i - 1], cInfo->hidden_neuron_count[i]);
     }
     cnNetwork->hidden_layer[cInfo->hidden_layer_count - 1] = Matrix_Create(sizeof(double),
                                                                            cInfo->hidden_neuron_count[cInfo->hidden_layer_count - 1],
                                                                            cInfo->output_neuron_count);
     for (size_t i = 0; i < Matrix_GetRowCount(cnNetwork->hidden_layer[0]); i++) {
          for (size_t j = 0; j < Matrix_GetColumnCount(cnNetwork->hidden_layer[0]); j++) {
               double weight = cInfo->weight_initilization(cInfo->input_neuron_count, cInfo->hidden_neuron_count[0]);
               Matrix_SetData(cnNetwork->hidden_layer[0], i, j, &weight);
          }
     }
     for (size_t i = 1; i < cInfo->hidden_layer_count - 1; i++) {
          for (size_t i = 0; i < Matrix_GetRowCount(cnNetwork->hidden_layer[i]); i++) {
               for (size_t j = 0; j < Matrix_GetColumnCount(cnNetwork->hidden_layer[i]); j++) {
                 double weight = cInfo->weight_initilization(cInfo->hidden_neuron_count[i - 1], cInfo->hidden_neuron_count[i]);
                 Matrix_SetData(cnNetwork->hidden_layer[i], i, j, &weight);
               }
          }
     }
     for (size_t i = 0; i < Matrix_GetRowCount(cnNetwork->hidden_layer[cInfo->hidden_layer_count - 1]); i++) {
          for (size_t j = 0; j < Matrix_GetRowCount(cnNetwork->hidden_layer[cInfo->hidden_layer_count - 1]); j++) {
               double weight = cInfo->weight_initilization(cInfo->hidden_neuron_count[cInfo->hidden_layer_count - 1],
                                                           cInfo->output_neuron_count);
               Matrix_SetData(cnNetwork->hidden_layer[cInfo->hidden_layer_count - 1], i, j, &weight);
          }
     }
     cnNetwork->input_neuron_count = cInfo->input_neuron_count;
     cnNetwork->output_neuron_count = cInfo->output_neuron_count;
     cnNetwork->hidden_layer_count = cInfo->hidden_layer_count;
     cnNetwork->hidden_neuron_count = cInfo->hidden_neuron_count;
     cnNetwork->learning_rate = cInfo->learning_rate;
     return cnNetwork;
}

void ConvolutionalNeuralNetwork_Destroy(ConvolutionalNeuralNetwork *cnNetwork)
{
     for (size_t i = 0; i < cnNetwork->hidden_layer_count; i++) {
          Matrix_Destroy(cnNetwork->hidden_layer[i]);
     }
     free(cnNetwork->hidden_layer);
     free(cnNetwork);
}

void ConvolutionalNeuralNetwork_Train(ConvolutionalNeuralNetwork *cnnetowrk, Matrix *ilayer, Matrix *olayer,
                                      double (*aFunction)(double weight))
{
     Matrix **matrix = (Matrix **)malloc((cnnetowrk->hidden_layer_count + 1) * sizeof (Matrix *));
     assert(matrix);
     matrix[0] = Matrix_Multiplication(ilayer, cnnetowrk->hidden_layer[0], _multiply_double);
     assert(matrix[0]);
     // matrix_activation(matrix[0], activate);
     for (size_t i = 0; i < Matrix_GetRowCount(matrix[0]); i++) {
          for (size_t j = 0; j < Matrix_GetColumnCount(matrix[0]); j++) {
               double weight = aFunction(*(double *)Matrix_GetData(matrix[0], i, j));
               Matrix_SetData(matrix[0], i, j, &weight);
          }
     }
     for (size_t i = 1; i < cnnetowrk->hidden_layer_count - 1; i++) {
          matrix[i] = Matrix_Multiplication(matrix[i - 1], cnnetowrk->hidden_layer[i], _multiply_double);
          assert(matrix[i]);
          for (size_t j = 0; j < Matrix_GetRowCount(matrix[i]); j++) {
               for (size_t k = 0; k < Matrix_GetColumnCount(matrix[i]); k++) {
                    double weight = aFunction(*(double *)Matrix_GetData(matrix[i], j, k));
                    Matrix_SetData(matrix[i], j, k, &weight);
               }
          }
     }
     matrix[cnnetowrk->hidden_layer_count - 1] = Matrix_Multiplication(matrix[cnnetowrk->hidden_layer_count - 1], olayer, _multiply_double);
     for (size_t i = 0; i < Matrix_GetRowCount(matrix[cnnetowrk->hidden_layer_count - 1]); i++) {
          for (size_t j = 0; j < Matrix_GetColumnCount(matrix[cnnetowrk->hidden_layer_count - 1]); j++) {
               double weight = aFunction(*(double *)Matrix_GetData(matrix[cnnetowrk->hidden_layer_count - 1], i, j));
               Matrix_SetData(matrix[cnnetowrk->hidden_layer_count - 1], i, j, &weight);
          }
     }
}

#if 0
void ConvolutionalNeuralNetwork_Save(ConvolutionalNeuralNetwork *cnNetwork, const char* dName)
{
     FILE *file = fopen(dName, "w+b");
     assert(file);
     fwrite(&cnNetwork->input_neuron_count, sizeof (size_t), 1, file);
     fwrite(&cnNetwork->output_neuron_count, sizeof (size_t), 1, file);
     fwrite(&cnNetwork->hidden_layer_count, sizeof (size_t), 1, file);
     fwrite(cnNetwork->hidden_neuron_count, sizeof (size_t), cnNetwork->hidden_layer_count, file);
     for (size_t i = 0; i < cnNetwork->hidden_layer_count; i++) {
          matrix_save(cnNetwork->hidden_layer[i], file);
     }
     fwrite(&cnNetwork->learning_rate, sizeof (double), 1, file);
     fclose(file);
}

ConvolutionalNeuralNetwork *ConvolutionalNeuralNetwork_Load(const char *dName)
{
     ConvolutionalNeuralNetwork *cnNetwork = (ConvolutionalNeuralNetwork *)malloc(sizeof (ConvolutionalNeuralNetwork));
     assert(cnNetwork);
     FILE *file = fopen(dName, "r+b");
     assert(file);
     fread(&cnNetwork->input_neuron_count, sizeof (size_t), 1, file);
     fread(&cnNetwork->output_neuron_count, sizeof (size_t), 1, file);
     fread(&cnNetwork->hidden_layer_count, sizeof (size_t), 1, file);
     cnNetwork->hidden_neuron_count = (size_t *)malloc(cnNetwork->hidden_layer_count * sizeof (size_t));
     assert(convolutional_neural_network->hidden_neuron_count);
     fread(cnNetwork->hidden_neuron_count, sizeof (size_t), cnNetwork->hidden_layer_count, file);
     cnNetwork->hidden_layer = (Matrix **)malloc(cnNetwork->hidden_layer_count * sizeof (Matrix *));
     assert(cnNetwork->hidden_layer);
     cnNetwork->hidden_layer[0] = matrix_load(file, cnNetwork->input_neuron_count, cnNetwork->hidden_neuron_count[0]);
     assert(cnNetwork->hidden_layer[0]);
     for (size_t i = 1; i < cnNetwork->hidden_layer_count - 1; i++) {
          cnNetwork->hidden_layer[i] = matrix_load(file, cnNetwork->hidden_neuron_count[i - 1], cnNetwork->hidden_neuron_count[i]);
          assert(convolutional_neural_network->hidden_layer[i]);
     }
     cnNetwork->hidden_layer[cnNetwork->hidden_layer_count - 1] =
          matrix_load(file, cnNetwork->hidden_neuron_count[cnNetwork->hidden_layer_count - 1], cnNetwork->output_neuron_count);
     assert(cnNetwork->hidden_layer[cnNetwork->hidden_layer_count - 1]);
     fread(&cnNetwork->learning_rate, sizeof (double), 1, file);
     return cnNetwork;
}
#endif

void ConvolutionalNeuralNetwork_Print(ConvolutionalNeuralNetwork *cnNetwork)
{
     for (size_t i = 0; i < cnNetwork->hidden_layer_count; i++) {
          printf("Hidden Layer [%zu] :\n", i);
          Matrix_Traverse(cnNetwork->hidden_layer[i], _print_double);
     }
}

static void _print_double(void *data)
{
     printf("%f ", *(double *)data);
}

static void _multiply_double(void *data, const void *data1, const void *data2)
{
     *(double *)data = (*(double *)data1) * (*(double *)data2);
}

static int _compare_double(const void *data1, const void *data2)
{
     int compare = 0;
     if (*(double *)data1 < *(double *)data2) {
          compare = -1;
     } else if (*(double *)data1 > *(double *)data2) {
          compare = 1;
     }
     return compare;
}
