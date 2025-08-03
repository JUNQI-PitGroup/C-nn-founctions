#include <stdlib.h>
#include <stdio.h>

#include "nn_function.h"


# define inputLength 449

// This is a dense neural network with 4 layers, weight updated by SGD
# define layer1_neuronNum 768   // ⬇
# define layer2_neuronNum 96    // ⬇
# define layer3_neuronNum 32    // ⬇
# define layer4_neuronNum 1     // ⬇

static float weightTensor_1[layer1_neuronNum][inputLength + 1] {}; // 第一层权重矩阵
static float weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] {}; // 第二层权重矩阵
static float weightTensor_3[layer3_neuronNum][layer2_neuronNum + 1] {}; // 第三层权重矩阵
static float weightTensor_4[layer4_neuronNum][layer3_neuronNum + 1] {}; // 输出层权重矩阵

static float linear1_outputTensor[layer1_neuronNum] {};
static float activate1_outputTensor[layer1_neuronNum] {};
static float linear2_outputTensor[layer2_neuronNum] {};
static float activate2_outputTensor[layer2_neuronNum] {};
static float linear3_outputTensor[layer3_neuronNum] {};
static float activate3_outputTensor[layer3_neuronNum] {};
static float linear4_outputTensor[layer4_neuronNum] {};
static float activate4_outputTensor[layer4_neuronNum] {};

static void UpdateWeights(float* inputTensor, float* labelTensor, float learningRate) {
    // 反向传播
    float layer4_grad[layer4_neuronNum]{};
    MSE_LossDerivative(layer4_grad, activate4_outputTensor, labelTensor, layer4_neuronNum);
	TanhVectorDerivative(layer4_grad, layer4_neuronNum, linear4_outputTensor);

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
    UpdateLinearWeight_AVX2(&weightTensor_4[0][0], activate3_outputTensor, layer4_grad, layer4_neuronNum, layer3_neuronNum, learningRate);
    UpdateLinearWeight_AVX2(&weightTensor_3[0][0], activate2_outputTensor, layer3_grad, layer3_neuronNum, layer2_neuronNum, learningRate);
    UpdateLinearWeight_AVX2(&weightTensor_2[0][0], activate1_outputTensor, layer2_grad, layer2_neuronNum, layer1_neuronNum, learningRate);
    UpdateLinearWeight_AVX2(&weightTensor_1[0][0], inputTensor, layer1_grad, layer1_neuronNum, inputLength, learningRate);
}

static void Forward(float* inputTensor, float* outputTensor) {
    // 第一层有 layer1_neuronNum 个神经元
    for (int i = 0; i < layer1_neuronNum; i++) {
        linear1_outputTensor[i] = Linear_AVX2(inputTensor, inputLength, weightTensor_1[i]);
        activate1_outputTensor[i] = ReLU(linear1_outputTensor[i]);
    }
    // 第二层有 layer2_neuronNum 个神经元
    for (int i = 0; i < layer2_neuronNum; i++) {
        linear2_outputTensor[i] = Linear_AVX2(activate1_outputTensor, layer1_neuronNum, weightTensor_2[i]);
        activate2_outputTensor[i] = ReLU(linear2_outputTensor[i]);
    }
    // 第三层有 layer3_neuronNum 个神经元
    for (int i = 0; i < layer3_neuronNum; i++) {
        linear3_outputTensor[i] = Linear_AVX2(activate2_outputTensor, layer2_neuronNum, weightTensor_3[i]);
        activate3_outputTensor[i] = ReLU(linear3_outputTensor[i]);
    }
    // 第四层有 layer4_neuronNum 个神经元
    for (int i = 0; i < layer4_neuronNum; i++) {
        linear4_outputTensor[i] = Linear_AVX2(activate3_outputTensor, layer3_neuronNum, weightTensor_4[i]);
        activate4_outputTensor[i] = Tanh(linear4_outputTensor[i]);
    }

    for (int i = 0; i < layer4_neuronNum; i++) outputTensor[i] = activate4_outputTensor[i];

    return;
}

// 以下是对外函数

void UpdateWeights_4Layers_NN(float* inputTensor, float* labelTensor, float learningRate) {
    UpdateWeights(
        inputTensor,
        labelTensor,
        learningRate);
}

void Forward_4Layers_NN(float* inputTensor, float* outputTensor) {
    Forward(inputTensor, outputTensor);
}

void Randomized_4Layers_NN_Weight(int seed) {
    srand(seed);
    // ReLU 神经元用 He 初始化权重，Sigmoid 神经元用 Xavier 初始化权重
    HeInitialize(&weightTensor_1[0][0], layer1_neuronNum, inputLength + 1);
    HeInitialize(&weightTensor_2[0][0], layer2_neuronNum, layer1_neuronNum + 1);
    HeInitialize(&weightTensor_3[0][0], layer3_neuronNum, layer2_neuronNum + 1);
    XavierInitialize(&weightTensor_4[0][0], layer4_neuronNum, layer3_neuronNum + 1);
    printf("Weight Randomized.\n");
}

void Print_4Layers_NN_Weight() {
    printf("static float weightTensor_1[layer1_neuronNum][inputLength + 1] = \n");
    PrintWeightTensor((float*)weightTensor_1, layer1_neuronNum, inputLength + 1);
    printf("static float weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] = \n");
    PrintWeightTensor((float*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1);
    printf("static float weightTensor_3[layer3_neuronNum][layer2_neuronNum + 1] = \n");
    PrintWeightTensor((float*)weightTensor_3, layer3_neuronNum, layer2_neuronNum + 1);
    printf("static float weightTensor_4[layer4_neuronNum][layer3_neuronNum + 1] = \n");
    PrintWeightTensor((float*)weightTensor_4, layer4_neuronNum, layer3_neuronNum + 1);
}

void Save_4Layers_NN_Weight() {
    Save_WeightTensor((float*)weightTensor_1, layer1_neuronNum, inputLength + 1, "./4_layers_fc_nn_saved_weight/4_layers_fc_nn_saved_weight_layer_1_tensor.txt");
    Save_WeightTensor((float*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1, "./4_layers_fc_nn_saved_weight/4_layers_fc_nn_saved_weight_layer_2_tensor.txt");
    Save_WeightTensor((float*)weightTensor_3, layer3_neuronNum, layer2_neuronNum + 1, "./4_layers_fc_nn_saved_weight/4_layers_fc_nn_saved_weight_layer_3_tensor.txt");
    Save_WeightTensor((float*)weightTensor_4, layer4_neuronNum, layer3_neuronNum + 1, "./4_layers_fc_nn_saved_weight/4_layers_fc_nn_saved_weight_layer_4_tensor.txt");
}

void Load_4Layers_NN_Weight() {
    Load_WeightTensor((float*)weightTensor_1, layer1_neuronNum, inputLength + 1, "./4_layers_fc_nn_saved_weight/4_layers_fc_nn_saved_weight_layer_1_tensor.txt");
    Load_WeightTensor((float*)weightTensor_2, layer2_neuronNum, layer1_neuronNum + 1, "./4_layers_fc_nn_saved_weight/4_layers_fc_nn_saved_weight_layer_2_tensor.txt");
    Load_WeightTensor((float*)weightTensor_3, layer3_neuronNum, layer2_neuronNum + 1, "./4_layers_fc_nn_saved_weight/4_layers_fc_nn_saved_weight_layer_3_tensor.txt");
    Load_WeightTensor((float*)weightTensor_4, layer4_neuronNum, layer3_neuronNum + 1, "./4_layers_fc_nn_saved_weight/4_layers_fc_nn_saved_weight_layer_4_tensor.txt");
    printf("Loaded 4 Layers Dense NN Weight. (%d -> %d -> %d -> %d)\n",
        layer1_neuronNum, layer2_neuronNum, layer3_neuronNum, layer4_neuronNum);
}
