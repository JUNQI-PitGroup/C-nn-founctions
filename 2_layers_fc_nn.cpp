#include <stdlib.h>
#include <stdio.h>

#include "nn_function.h"

#define inputLength 19

// This is a dense neural network with 2 layers, weight updated by SGD
#define layer1_neuronNum 128   // ⬇
#define layer2_neuronNum 1    // ⬇

static double weightTensor_1[layer1_neuronNum][inputLength + 1] = { 0 }; // 第一层权重矩阵
static double weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] = { 0 }; // 第二层权重矩阵

static double linear1_outputTensor[layer1_neuronNum] = { 0 };
static double activate1_outputTensor[layer1_neuronNum] = { 0 };
static double linear2_outputTensor[layer2_neuronNum] = { 0 };
static double activate2_outputTensor[layer2_neuronNum] = { 0 };

static void UpdateWeights(double* inputTensor, double* labelTensor, double learningRate) {
    // 反向传播
    double layer2_grad[layer2_neuronNum]{};
    MSE_LossDerivative(layer2_grad, activate2_outputTensor, labelTensor, layer2_neuronNum);
    TanhVectorDerivative(layer2_grad, layer2_neuronNum, linear2_outputTensor);

    double layer1_grad[layer1_neuronNum]{};
    LinearVectorDerivative(layer2_grad, layer1_grad, layer1_neuronNum, layer2_neuronNum, &weightTensor_2[0][0]);
    ReLuVectorDerivative(layer1_grad, layer1_neuronNum, linear1_outputTensor);

    // 更新权重
    UpdateLinearWeight(&weightTensor_2[0][0], activate1_outputTensor, layer2_grad, layer2_neuronNum, layer1_neuronNum, learningRate);
    UpdateLinearWeight(&weightTensor_1[0][0], inputTensor, layer1_grad, layer1_neuronNum, inputLength, learningRate);
}

static void Forward(double* inputTensor, double* outputTensor) {
    // 第一层
    for (int i = 0; i < layer1_neuronNum; i++) {
        linear1_outputTensor[i] = Linear(inputTensor, inputLength, weightTensor_1[i]);
        activate1_outputTensor[i] = ReLU(linear1_outputTensor[i]);
    }
    // 第二层
    for (int i = 0; i < layer2_neuronNum; i++) {
        linear2_outputTensor[i] = Linear(activate1_outputTensor, layer1_neuronNum, weightTensor_2[i]);
        activate2_outputTensor[i] = Tanh(linear2_outputTensor[i]);
    }
    for (int i = 0; i < layer2_neuronNum; i++) outputTensor[i] = activate2_outputTensor[i];
}

// 以下是对外函数

void UpdateWeights_2Layers_NN(double* inputTensor, double* labelTensor, double learningRate) {
    UpdateWeights(
        inputTensor,
        labelTensor,
        learningRate);
}

void Forward_2Layers_NN(double* inputTensor, double* outputTensor) {
    Forward(inputTensor, outputTensor);
}

void Randomized_2Layers_NN_Weight(int seed) {
    srand(seed);
    // ReLU 神经元用 He 初始化权重，Sigmoid 神经元用 Xavier 初始化权重
    HeInitialize(&weightTensor_1[0][0], layer1_neuronNum, inputLength + 1);
    XavierInitialize(&weightTensor_2[0][0], layer2_neuronNum, layer1_neuronNum + 1);
    printf("Weight Randomized.\n");
}

void Print_2Layers_NN_Weight() {
    printf("static double weightTensor_1[layer1_neuronNum][inputLength + 1] = \n");
    PrintWeightTensor((double*)weightTensor_1, layer1_neuronNum, inputLength + 1);
    printf("static double weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] = \n");
    PrintWeightTensor((double*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1);
}

void Save_2Layers_NN_Weight() {
    Save_WeightTensor((double*)weightTensor_1, layer1_neuronNum, inputLength + 1, "./2_layers_fc_nn_saved_weight/2_layers_fc_nn_saved_weight_layer_1_tensor.txt");
    Save_WeightTensor((double*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1, "./2_layers_fc_nn_saved_weight/2_layers_fc_nn_saved_weight_layer_2_tensor.txt");
}

void Load_2Layers_NN_Weight() {
    Load_WeightTensor((double*)weightTensor_1, layer1_neuronNum, inputLength + 1, "./2_layers_fc_nn_saved_weight/2_layers_fc_nn_saved_weight_layer_1_tensor.txt");
    Load_WeightTensor((double*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1, "./2_layers_fc_nn_saved_weight/2_layers_fc_nn_saved_weight_layer_2_tensor.txt");
    printf("Loaded 2 Layers FC NN Weight. (%d -> %d)\n", 
        layer1_neuronNum, layer2_neuronNum);
}
