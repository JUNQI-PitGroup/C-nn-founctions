#include <stdlib.h>
#include <stdio.h>

#include "nn_function.h"


#define inputLength 24 * 24

#define layer1_neuronNum 512
#define layer2_neuronNum 128
#define layer3_neuronNum 32
#define layer4_neuronNum 16
#define layer5_neuronNum 8
#define layer6_neuronNum 3

static float weightTensor_1[layer1_neuronNum][inputLength + 1] = { 0 }; // 第一层权重矩阵
static float weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] = { 0 }; // 第二层权重矩阵
static float weightTensor_3[layer3_neuronNum][layer2_neuronNum + 1] = { 0 }; // 第三层权重矩阵
static float weightTensor_4[layer4_neuronNum][layer3_neuronNum + 1] = { 0 }; // 输四层权重矩阵
static float weightTensor_5[layer5_neuronNum][layer4_neuronNum + 1] = { 0 }; // 第五层权重矩阵
static float weightTensor_6[layer6_neuronNum][layer5_neuronNum + 1] = { 0 }; // 输出层权重矩阵

// 分开保存线性输出和激活输出
static float linear1_outputTensor[layer1_neuronNum] = { 0 };
static float activate1_outputTensor[layer1_neuronNum] = { 0 };
static float linear2_outputTensor[layer2_neuronNum] = { 0 };
static float activate2_outputTensor[layer2_neuronNum] = { 0 };
static float linear3_outputTensor[layer3_neuronNum] = { 0 };
static float activate3_outputTensor[layer3_neuronNum] = { 0 };
static float linear4_outputTensor[layer4_neuronNum] = { 0 };
static float activate4_outputTensor[layer4_neuronNum] = { 0 };
static float linear5_outputTensor[layer5_neuronNum] = { 0 };
static float activate5_outputTensor[layer5_neuronNum] = { 0 };
static float linear6_outputTensor[layer6_neuronNum] = { 0 };
static float activate6_outputTensor[layer6_neuronNum] = { 0 };

static void UpdateWeights(float* inputTensor, float* labelTensor, float learningRate) {
    // 反向传播
    float layer6_grad[layer6_neuronNum]{};
    SoftmaxAndCrossEntropyLossDerivative(activate6_outputTensor, labelTensor, layer6_grad, layer6_neuronNum);

    float layer5_grad[layer5_neuronNum]{};
    LinearVectorDerivative(layer6_grad, layer5_grad, layer5_neuronNum, layer6_neuronNum, &weightTensor_6[0][0]);
    ReLuVectorDerivative(layer5_grad, layer5_neuronNum, linear5_outputTensor);

    float layer4_grad[layer4_neuronNum]{};
    LinearVectorDerivative(layer5_grad, layer4_grad, layer4_neuronNum, layer5_neuronNum, &weightTensor_5[0][0]);
    ReLuVectorDerivative(layer4_grad, layer4_neuronNum, linear4_outputTensor);

    float layer3_grad[layer3_neuronNum]{};
    LinearVectorDerivative(layer4_grad, layer3_grad, layer3_neuronNum, layer4_neuronNum, &weightTensor_4[0][0]);
    ReLuVectorDerivative(layer3_grad, layer3_neuronNum, linear3_outputTensor);

    float layer2_grad[layer2_neuronNum]{};
    LinearVectorDerivative(layer3_grad, layer2_grad, layer2_neuronNum, layer3_neuronNum, &weightTensor_3[0][0]);
    ReLuVectorDerivative(layer2_grad, layer2_neuronNum, linear2_outputTensor);

    float layer1_grad[layer1_neuronNum]{};
    LinearVectorDerivative(layer2_grad, layer1_grad, layer1_neuronNum, layer2_neuronNum, &weightTensor_2[0][0]);
    ReLuVectorDerivative(layer1_grad, layer1_neuronNum, linear1_outputTensor);

    // 更新权重
    UpdateLinearWeight(&weightTensor_6[0][0], activate5_outputTensor, layer6_grad, layer6_neuronNum, layer5_neuronNum, learningRate);
    UpdateLinearWeight(&weightTensor_5[0][0], activate4_outputTensor, layer5_grad, layer5_neuronNum, layer4_neuronNum, learningRate);
    UpdateLinearWeight(&weightTensor_4[0][0], activate3_outputTensor, layer4_grad, layer4_neuronNum, layer3_neuronNum, learningRate);
    UpdateLinearWeight(&weightTensor_3[0][0], activate2_outputTensor, layer3_grad, layer3_neuronNum, layer2_neuronNum, learningRate);
    UpdateLinearWeight(&weightTensor_2[0][0], activate1_outputTensor, layer2_grad, layer2_neuronNum, layer1_neuronNum, learningRate);
    UpdateLinearWeight(&weightTensor_1[0][0], inputTensor, layer1_grad, layer1_neuronNum, inputLength, learningRate);
}

static void Forward(float* inputTensor, float* outputTensor) {
    // 第一层有 layer1_neuronNum 个神经元
    for (int i = 0; i < layer1_neuronNum; i++) {
        linear1_outputTensor[i] = Linear(inputTensor, inputLength, weightTensor_1[i]);
        activate1_outputTensor[i] = ReLU(linear1_outputTensor[i]);
    }
    // 第二层有 layer2_neuronNum 个神经元
    for (int i = 0; i < layer2_neuronNum; i++) {
        linear2_outputTensor[i] = Linear(activate1_outputTensor, layer1_neuronNum, weightTensor_2[i]);
        activate2_outputTensor[i] = ReLU(linear2_outputTensor[i]);
    }
    // 第三层有 layer3_neuronNum 个神经元
    for (int i = 0; i < layer3_neuronNum; i++) {
        linear3_outputTensor[i] = Linear(activate2_outputTensor, layer2_neuronNum, weightTensor_3[i]);
        activate3_outputTensor[i] = ReLU(linear3_outputTensor[i]);
    }
    // 第四层有 layer4_neuronNum 个神经元
    for (int i = 0; i < layer4_neuronNum; i++) {
        linear4_outputTensor[i] = Linear(activate3_outputTensor, layer3_neuronNum, weightTensor_4[i]);
        activate4_outputTensor[i] = ReLU(linear4_outputTensor[i]);
    }
    // 第五层有 layer5_neuronNum 个神经元
    for (int i = 0; i < layer5_neuronNum; i++) {
        linear5_outputTensor[i] = Linear(activate4_outputTensor, layer4_neuronNum, weightTensor_5[i]);
        activate5_outputTensor[i] = ReLU(linear5_outputTensor[i]);
    }
    // 第六层有 layer6_neuronNum 个神经元
    for (int i = 0; i < layer6_neuronNum; i++) {
        linear6_outputTensor[i] = Linear(activate5_outputTensor, layer5_neuronNum, weightTensor_6[i]);
    }
    Softmax(linear6_outputTensor, activate6_outputTensor, layer6_neuronNum);

    for (int i = 0; i < layer6_neuronNum; i++) outputTensor[i] = activate6_outputTensor[i];

    return;
}

// 以下是对外函数

void UpdateWeights_6Layers_NN(float* inputTensor, float* labelTensor, float learningRate) {
    UpdateWeights(inputTensor, labelTensor, learningRate);
}

void Forward_6Layers_NN(float* inputTensor, float* outputTensor) {
    Forward(inputTensor, outputTensor);
}

void Randomized_6Layers_NN_Weight(int seed) {
    srand(seed);
    // ReLU 神经元用 He 初始化权重，Sigmoid 神经元用 Xavier 初始化权重
    HeInitialize(&weightTensor_1[0][0], layer1_neuronNum, inputLength + 1);
    HeInitialize(&weightTensor_2[0][0], layer2_neuronNum, layer1_neuronNum + 1);
    HeInitialize(&weightTensor_3[0][0], layer3_neuronNum, layer2_neuronNum + 1);
    HeInitialize(&weightTensor_4[0][0], layer4_neuronNum, layer3_neuronNum + 1);
    HeInitialize(&weightTensor_5[0][0], layer5_neuronNum, layer4_neuronNum + 1);
    XavierInitialize(&weightTensor_6[0][0], layer6_neuronNum, layer5_neuronNum + 1);
    printf("Weight Randomized.\n");
}

void Print_6Layers_NN_Weight() {
    printf("static float weightTensor_1[layer1_neuronNum][inputLength + 1] = \n");
    PrintWeightTensor((float*)weightTensor_1, layer1_neuronNum, inputLength + 1);
    printf("static float weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] = \n");
    PrintWeightTensor((float*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1);
    printf("static float weightTensor_3[layer3_neuronNum][layer2_neuronNum + 1] = \n");
    PrintWeightTensor((float*)weightTensor_3, layer3_neuronNum, layer2_neuronNum + 1);
    printf("static float weightTensor_4[layer4_neuronNum][layer3_neuronNum + 1] = \n");
    PrintWeightTensor((float*)weightTensor_4, layer4_neuronNum, layer3_neuronNum + 1);
    printf("static float weightTensor_5[layer5_neuronNum][layer4_neuronNum + 1] = \n");
    PrintWeightTensor((float*)weightTensor_5, layer5_neuronNum, layer4_neuronNum + 1);
    printf("static float weightTensor_6[layer6_neuronNum][layer5_neuronNum + 1] = \n");
    PrintWeightTensor((float*)weightTensor_6, layer6_neuronNum, layer5_neuronNum + 1);
}

void Save_6Layers_NN_Weight() {
    Save_WeightTensor((float*)weightTensor_1, layer1_neuronNum, inputLength + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_1_tensor.txt");
    Save_WeightTensor((float*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_2_tensor.txt");
    Save_WeightTensor((float*)weightTensor_3, layer3_neuronNum, layer2_neuronNum + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_3_tensor.txt");
    Save_WeightTensor((float*)weightTensor_4, layer4_neuronNum, layer3_neuronNum + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_4_tensor.txt");
    Save_WeightTensor((float*)weightTensor_5, layer5_neuronNum, layer4_neuronNum + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_5_tensor.txt");
    Save_WeightTensor((float*)weightTensor_6, layer6_neuronNum, layer5_neuronNum + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_6_tensor.txt");
}

void Load_6Layers_NN_Weight() {
    Load_WeightTensor((float*)weightTensor_1, layer1_neuronNum, inputLength + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_1_tensor.txt");
    Load_WeightTensor((float*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_2_tensor.txt");
    Load_WeightTensor((float*)weightTensor_3, layer3_neuronNum, layer2_neuronNum + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_3_tensor.txt");
    Load_WeightTensor((float*)weightTensor_4, layer4_neuronNum, layer3_neuronNum + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_4_tensor.txt");
    Load_WeightTensor((float*)weightTensor_5, layer5_neuronNum, layer4_neuronNum + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_5_tensor.txt");
    Load_WeightTensor((float*)weightTensor_6, layer6_neuronNum, layer5_neuronNum + 1, "./6_layers_fc_nn_saved_weight/6_layers_fc_nn_saved_weight_layer_6_tensor.txt");
    printf("Loaded 5 Layers Dense NN Weight. (%d -> %d -> %d -> %d -> %d -> %d)\n",
        layer1_neuronNum, layer2_neuronNum, layer3_neuronNum, layer4_neuronNum, layer5_neuronNum, layer6_neuronNum);
}