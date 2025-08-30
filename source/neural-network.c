#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <neural-network.h>

typedef NeuralNetworkCreateInfo CreateInfo;
typedef NeuralNetworkLayer Layer;

struct neural_network {
    size_t input_neuron_count;
    size_t output_neuron_count;
    size_t hidden_layer_count;
    size_t *hidden_neuron_count;
    Layer **hidden_layer;
    double learning_rate;
};

static int _CompareDouble(const void *data1, const void *data2);
static void _MultiplyDouble(void *data, const void *data1, const void *data2);
static void _PrintDouble(void *data);

NeuralNetwork *NeuralNetwork_Create(const CreateInfo *cInfo)
{
    NeuralNetwork *nNetwork = (NeuralNetwork *)malloc(sizeof (NeuralNetwork));
    assert(nNetwork);
    nNetwork->hidden_layer = (Layer **)malloc((cInfo->hidden_layer_count + 1) * sizeof (Layer *));
    assert(nNetwork->hidden_layer);
    nNetwork->hidden_layer[0] =
        Matrix_Create(sizeof (double), cInfo->input_neuron_count, cInfo->hidden_neuron_count[0]);
    for (size_t i = 1; i < cInfo->hidden_layer_count - 1; i++) {
        nNetwork->hidden_layer[i] =
            Matrix_Create(sizeof (double), cInfo->hidden_neuron_count[i - 1], cInfo->hidden_neuron_count[i]);
    }
    nNetwork->hidden_layer[cInfo->hidden_layer_count - 1] =
        Matrix_Create(sizeof(double), cInfo->hidden_neuron_count[cInfo->hidden_layer_count - 1],
                    cInfo->output_neuron_count);
    int lrCount = Matrix_GetRowCount(nNetwork->hidden_layer[0]);
    int lcCount = Matrix_GetColumnCount(nNetwork->hidden_layer[0]);
    for (size_t i = 0; i < lrCount; i++) {
        for (size_t j = 0; j < lcCount; j++) {
            double weight =
                cInfo->weight_initilization(cInfo->input_neuron_count, cInfo->hidden_neuron_count[0]);
            Matrix_SetData(nNetwork->hidden_layer[0], i, j, &weight);
        }
    }
    for (size_t i = 1; i < cInfo->hidden_layer_count - 1; i++) {
        lrCount = Matrix_GetRowCount(nNetwork->hidden_layer[i]);
        lcCount = Matrix_GetColumnCount(nNetwork->hidden_layer[i]);
        for (size_t i = 0; i < lrCount; i++) {
            for (size_t j = 0; j < lcCount; j++) {
                double weight =
                        cInfo->weight_initilization(cInfo->hidden_neuron_count[i - 1],
                                                    cInfo->hidden_neuron_count[i]);
                Matrix_SetData(nNetwork->hidden_layer[i], i, j, &weight);
            }
        }
    }
    lrCount = Matrix_GetRowCount(nNetwork->hidden_layer[cInfo->hidden_layer_count - 1]);
    lcCount = Matrix_GetColumnCount(nNetwork->hidden_layer[cInfo->hidden_layer_count - 1]);
    for (size_t i = 0; i < lrCount; i++) {
        for (size_t j = 0; j < lcCount; j++) {
            double weight =
                cInfo->weight_initilization(cInfo->hidden_neuron_count[cInfo->hidden_layer_count - 1],
                                            cInfo->output_neuron_count);
            Matrix_SetData(nNetwork->hidden_layer[cInfo->hidden_layer_count - 1], i, j, &weight);
        }
    }
    nNetwork->input_neuron_count = cInfo->input_neuron_count;
    nNetwork->output_neuron_count = cInfo->output_neuron_count;
    nNetwork->hidden_layer_count = cInfo->hidden_layer_count;
    nNetwork->hidden_neuron_count = cInfo->hidden_neuron_count;
    nNetwork->learning_rate = cInfo->learning_rate;
    return nNetwork;
}

void NeuralNetwork_Destroy(NeuralNetwork *nNetwork)
{
    for (size_t i = 0; i < nNetwork->hidden_layer_count; i++) {
        Matrix_Destroy(nNetwork->hidden_layer[i]);
    }
    free(nNetwork->hidden_layer);
    free(nNetwork);
}

void NeuralNetwork_Train(NeuralNetwork *nNetwork, Layer *ilayer, Layer *lOutput,
                         double (*aFunction)(double weight))
{
    Layer **layer = (Layer **)malloc((nNetwork->hidden_layer_count + 1) * sizeof (Layer *));
    assert(layer);
    layer[0] = Matrix_Multiplication(ilayer, nNetwork->hidden_layer[0], _MultiplyDouble);
    assert(layer[0]);
    // matrix_activation(layer[0], activate);
    int lrCount = Matrix_GetRowCount(layer[0]);
    int lcCount = Matrix_GetColumnCount(layer[0]);
    for (size_t i = 0; i < lrCount; i++) {
        for (size_t j = 0; j < lcCount; j++) {
            double weight = aFunction(*(double *)Matrix_GetData(layer[0], i, j));
            Matrix_SetData(layer[0], i, j, &weight);
        }
    }
    for (size_t i = 1; i < nNetwork->hidden_layer_count - 1; i++) {
        layer[i] = Matrix_Multiplication(layer[i - 1], nNetwork->hidden_layer[i], _MultiplyDouble);
        assert(layer[i]);
        lrCount = Matrix_GetRowCount(layer[i]);
        lcCount = Matrix_GetColumnCount(layer[i]);
        for (size_t j = 0; j < lrCount; j++) {
            for (size_t k = 0; k < lcCount; k++) {
                double weight = aFunction(*(double *)Matrix_GetData(layer[i], j, k));
                Matrix_SetData(layer[i], j, k, &weight);
            }
        }
    }
    layer[nNetwork->hidden_layer_count - 1] =
        Matrix_Multiplication(layer[nNetwork->hidden_layer_count - 1], lOutput, _MultiplyDouble);
    lrCount = Matrix_GetRowCount(layer[nNetwork->hidden_layer_count - 1]);
    lcCount = Matrix_GetColumnCount(layer[nNetwork->hidden_layer_count - 1]);
    for (size_t i = 0; i < lrCount; i++) {
        for (size_t j = 0; j < lcCount; j++) {
            double weight =
                aFunction(*(double *)Matrix_GetData(layer[nNetwork->hidden_layer_count - 1], i, j));
            Matrix_SetData(layer[nNetwork->hidden_layer_count - 1], i, j, &weight);
        }
    }
}

#if 0
void NeuralNetwork_Save(NeuralNetwork *nNetwork, const char* dName)
{
    FILE *file = fopen(dName, "w+b");
    assert(file);
    fwrite(&nNetwork->input_neuron_count, sizeof (size_t), 1, file);
    fwrite(&nNetwork->output_neuron_count, sizeof (size_t), 1, file);
    fwrite(&nNetwork->hidden_layer_count, sizeof (size_t), 1, file);
    fwrite(nNetwork->hidden_neuron_count, sizeof (size_t), nNetwork->hidden_layer_count, file);
    for (size_t i = 0; i < nNetwork->hidden_layer_count; i++) {
        matrix_save(nNetwork->hidden_layer[i], file);
    }
    fwrite(&nNetwork->learning_rate, sizeof (double), 1, file);
    fclose(file);
}

NeuralNetwork *NeuralNetwork_Load(const char *dName)
{
    NeuralNetwork *nNetwork = (NeuralNetwork *)malloc(sizeof (NeuralNetwork));
    assert(nNetwork);
    FILE *file = fopen(dName, "r+b");
    assert(file);
    fread(&nNetwork->input_neuron_count, sizeof (size_t), 1, file);
    fread(&nNetwork->output_neuron_count, sizeof (size_t), 1, file);
    fread(&nNetwork->hidden_layer_count, sizeof (size_t), 1, file);
    nNetwork->hidden_neuron_count = (size_t *)malloc(nNetwork->hidden_layer_count * sizeof (size_t));
    assert(convolutional_neural_network->hidden_neuron_count);
    fread(nNetwork->hidden_neuron_count, sizeof (size_t), nNetwork->hidden_layer_count, file);
    nNetwork->hidden_layer = (Layer **)malloc(nNetwork->hidden_layer_count * sizeof (Layer *));
    assert(nNetwork->hidden_layer);
    nNetwork->hidden_layer[0] = matrix_load(file, nNetwork->input_neuron_count, nNetwork->hidden_neuron_count[0]);
    assert(nNetwork->hidden_layer[0]);
    for (size_t i = 1; i < nNetwork->hidden_layer_count - 1; i++) {
        nNetwork->hidden_layer[i] = matrix_load(file, nNetwork->hidden_neuron_count[i - 1], nNetwork->hidden_neuron_count[i]);
        assert(convolutional_neural_network->hidden_layer[i]);
    }
    nNetwork->hidden_layer[nNetwork->hidden_layer_count - 1] =
        matrix_load(file, nNetwork->hidden_neuron_count[nNetwork->hidden_layer_count - 1], nNetwork->output_neuron_count);
    assert(nNetwork->hidden_layer[nNetwork->hidden_layer_count - 1]);
    fread(&nNetwork->learning_rate, sizeof (double), 1, file);
    return nNetwork;
}
#endif

void NeuralNetwork_Print(NeuralNetwork *nNetwork)
{
    for (size_t i = 0; i < nNetwork->hidden_layer_count; i++) {
        printf("Hidden Layer [%zu] :\n", i);
        Matrix_Traverse(nNetwork->hidden_layer[i], _PrintDouble);
    }
}

static void _PrintDouble(void *data)
{
    printf("%f ", *(double *)data);
}

static void _MultiplyDouble(void *data, const void *data1, const void *data2)
{
    *(double *)data = (*(double *)data1) * (*(double *)data2);
}

static int _CompareDouble(const void *data1, const void *data2)
{
    int compare = 0;
    if (*(double *)data1 < *(double *)data2) {
        compare = -1;
    } else if (*(double *)data1 > *(double *)data2) {
        compare = 1;
    }
    return compare;
}
