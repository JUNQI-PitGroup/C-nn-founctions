#include <stdlib.h>
#include <stdio.h>

#include "nn_function.h"


# define inputLength 14

// This is a dense neural network with 1 layers, weight updated by SGD
# define layer1_neuronNum 1   // ⬇

static float weightTensor_1[layer1_neuronNum][inputLength + 1] = { 0 }; // 第一层权重矩阵

static float linear1_outputTensor[layer1_neuronNum] = { 0 };
static float activate1_outputTensor[1] = { 0 };

static void UpdateWeights(float* inputTensor, float* labelTensor, float learningRate) {
    // 计算第一层梯度
    float layer1_grad[1];
    MSE_LossDerivative(layer1_grad, activate1_outputTensor, labelTensor, layer1_neuronNum);
    TanhVectorDerivative(layer1_grad, layer1_neuronNum, linear1_outputTensor);

    // 更新第一层权重和偏置
    UpdateLinearWeight(&weightTensor_1[0][0], inputTensor, layer1_grad, layer1_neuronNum, inputLength, learningRate);
}

static void Forward(float* inputTensor, float* outputTensor) {
    // 第一层
    LinearVector(inputTensor, inputLength, &weightTensor_1[0][0], layer1_neuronNum, linear1_outputTensor);
    TanhVector(linear1_outputTensor, layer1_neuronNum, activate1_outputTensor);

    CopyVector(activate1_outputTensor, outputTensor, layer1_neuronNum);
    return;
}

// 以下是对外函数

void UpdateWeights_1Layers_NN(float* inputTensor, float* labelTensor, float learningRate) {
    UpdateWeights(inputTensor, labelTensor, learningRate);
}

void Forward_1Layers_NN(float* inputTensor, float* outputTensor) {
    Forward(inputTensor, outputTensor);
}

void Randomized_1Layers_NN_Weight(int seed) {
    srand(seed);
    // ReLU 神经元用 He 初始化权重，Sigmoid 神经元用 Xavier 初始化权重
    HeInitialize(&weightTensor_1[0][0], layer1_neuronNum, inputLength + 1);
    printf("Weight Randomized.\n");
}

void Print_1Layers_NN_Weight() {
    printf("static float weightTensor_1[layer1_neuronNum][inputLength + 1] = \n");
    PrintWeightTensor((float*)weightTensor_1, layer1_neuronNum, inputLength + 1);
}

void Save_1Layers_NN_Weight() {
    Save_WeightTensor((float*)weightTensor_1, layer1_neuronNum, inputLength + 1, "./1_layers_fc_nn_saved_weight/1_layers_fc_nn_saved_weight_layer_1_tensor.txt");
}

void Load_1Layers_NN_Weight() {
    Load_WeightTensor((float*)weightTensor_1, layer1_neuronNum, inputLength + 1, "./1_layers_fc_nn_saved_weight/1_layers_fc_nn_saved_weight_layer_1_tensor.txt");
    printf("Loaded 1 Layers FC NN Weight. (%d)\n",
        layer1_neuronNum);
}
