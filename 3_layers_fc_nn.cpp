#include <stdlib.h>
#include <stdio.h>

#include "nn_function.h"


# define inputLength 19

// This is a dense neural network with 3 layers, weight updated by SGD
# define layer1_neuronNum 128   // ⬇
# define layer2_neuronNum 24    // ⬇
# define layer3_neuronNum 1     // ⬇

static double weightTensor_1[layer1_neuronNum][inputLength + 1] = { 0 }; // 第一层权重矩阵
static double weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] = { 0 }; // 第二层权重矩阵
static double weightTensor_3[layer3_neuronNum][layer2_neuronNum + 1] = { 0 }; // 第三层权重矩阵

static double linear1_outputTensor[layer1_neuronNum] = { 0 };
static double activate1_outputTensor[layer1_neuronNum] = { 0 };
static double linear2_outputTensor[layer2_neuronNum] = { 0 };
static double activate2_outputTensor[layer2_neuronNum] = { 0 };
static double linear3_outputTensor[layer3_neuronNum] = { 0 };
static double activate3_outputTensor[layer3_neuronNum] = { 0 };

static void UpdateWeights(double* inputTensor, double* labelTensor, double learningRate) {
    // 反向传播
    double layer3_grad[layer3_neuronNum]{};
    MSE_LossDerivative(layer3_grad, activate3_outputTensor, labelTensor, layer3_neuronNum);
    TanhVectorDerivative(layer3_grad, layer3_neuronNum, linear3_outputTensor);

    double layer2_grad[layer2_neuronNum]{};
    LinearVectorDerivative(layer3_grad, layer2_grad, layer2_neuronNum, layer3_neuronNum, &weightTensor_3[0][0]);
    ReLuVectorDerivative(layer2_grad, layer2_neuronNum, linear2_outputTensor);

    double layer1_grad[layer1_neuronNum]{};
    LinearVectorDerivative(layer2_grad, layer1_grad, layer1_neuronNum, layer2_neuronNum, &weightTensor_2[0][0]);
    ReLuVectorDerivative(layer1_grad, layer1_neuronNum, linear1_outputTensor);

    // 更新权重
    UpdateLinearWeight(&weightTensor_3[0][0], activate2_outputTensor, layer3_grad, layer3_neuronNum, layer2_neuronNum, learningRate);
    UpdateLinearWeight(&weightTensor_2[0][0], activate1_outputTensor, layer2_grad, layer2_neuronNum, layer1_neuronNum, learningRate);
    UpdateLinearWeight(&weightTensor_1[0][0], inputTensor, layer1_grad, layer1_neuronNum, inputLength, learningRate);
}

static void Forward(double* inputTensor, double* outputTensor) {
    for (int i = 0; i < layer1_neuronNum; i++) {
        linear1_outputTensor[i] = Linear(inputTensor, inputLength, weightTensor_1[i]);
        activate1_outputTensor[i] = ReLU(linear1_outputTensor[i]);
    }
    for (int i = 0; i < layer2_neuronNum; i++) {
        linear2_outputTensor[i] = Linear(activate1_outputTensor, layer1_neuronNum, weightTensor_2[i]);
        activate2_outputTensor[i] = ReLU(linear2_outputTensor[i]);
    }
    for (int i = 0; i < layer3_neuronNum; i++) {
        linear3_outputTensor[i] = Linear(activate2_outputTensor, layer2_neuronNum, weightTensor_3[i]);
        activate3_outputTensor[i] = Tanh(linear3_outputTensor[i]);
    }
    for (int i = 0; i < layer3_neuronNum; i++) outputTensor[i] = activate3_outputTensor[i];
}

// 以下是对外函数

void UpdateWeights_3Layers_NN(double* inputTensor, double* labelTensor, double learningRate) {
    UpdateWeights(
        inputTensor,
        labelTensor,
        learningRate);
}

void Forward_3Layers_NN(double* inputTensor, double* outputTensor) {
    Forward(inputTensor, outputTensor);
}

void Randomized_3Layers_NN_Weight(int seed) {
    srand(seed);
    // ReLU 神经元用 He 初始化权重，Sigmoid 神经元用 Xavier 初始化权重
    HeInitialize(&weightTensor_1[0][0], layer1_neuronNum, inputLength + 1);
    HeInitialize(&weightTensor_2[0][0], layer2_neuronNum, layer1_neuronNum + 1);
    XavierInitialize(&weightTensor_3[0][0], layer3_neuronNum, layer2_neuronNum + 1);
    printf("Weight Randomized.\n");
}

void Print_3Layers_NN_Weight() {
    printf("static double weightTensor_1[layer1_neuronNum][inputLength + 1] = \n");
    PrintWeightTensor((double*)weightTensor_1, layer1_neuronNum, inputLength + 1);
    printf("static double weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] = \n");
    PrintWeightTensor((double*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1);
    printf("static double weightTensor_3[layer3_neuronNum][layer2_neuronNum + 1] = \n");
    PrintWeightTensor((double*)weightTensor_3, layer3_neuronNum, layer2_neuronNum + 1);
}

void Save_3Layers_NN_Weight() {
    Save_WeightTensor((double*)weightTensor_1, layer1_neuronNum, inputLength + 1, "./3_layers_fc_nn_saved_weight/3_layers_fc_nn_saved_weight_layer_1_tensor.txt");
    Save_WeightTensor((double*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1, "./3_layers_fc_nn_saved_weight/3_layers_fc_nn_saved_weight_layer_2_tensor.txt");
    Save_WeightTensor((double*)weightTensor_3, layer3_neuronNum, layer2_neuronNum + 1, "./3_layers_fc_nn_saved_weight/3_layers_fc_nn_saved_weight_layer_3_tensor.txt");
}

void Load_3Layers_NN_Weight() {
    Load_WeightTensor((double*)weightTensor_1, layer1_neuronNum, inputLength + 1, "./3_layers_fc_nn_saved_weight/3_layers_fc_nn_saved_weight_layer_1_tensor.txt");
    Load_WeightTensor((double*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1, "./3_layers_fc_nn_saved_weight/3_layers_fc_nn_saved_weight_layer_2_tensor.txt");
    Load_WeightTensor((double*)weightTensor_3, layer3_neuronNum, layer2_neuronNum + 1, "./3_layers_fc_nn_saved_weight/3_layers_fc_nn_saved_weight_layer_3_tensor.txt");
    printf("Loaded 3 Layers Dense NN Weight. (%d -> %d -> %d)\n",
        layer1_neuronNum, layer2_neuronNum, layer3_neuronNum);
}