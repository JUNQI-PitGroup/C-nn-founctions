#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn_function.h"

// ������ṹ
// Padding -> Conv -> Maxpooling -> Padding -> Conv -> Maxpooling -> Padding -> Conv -> Maxpooling -> Flatten -> FC -> FC -> Softmax

static double filter_1[16][1][3][3], filter_1_bias[16]{}; // Filter[FilterNum][Channels][W][H], Bias[FilterNum]
static double paddedInputTensor_1[1][26][26]{}; // PaddedTensor[Channels][W][H]
static double conv1_outputTensor[16][24][24]{}; // ConvOutputTensor[FilterNum][W][H]
static double pooling1_outputTensor[16][12][12]{};

static double filter_2[32][16][3][3], filter_2_bias[32]{}; // Filter[FilterNum][Channels][W][H], Bias[FilterNum]
static double paddedInputTensor_2[16][14][14]{}; // PaddedTensor[Channels][W][H]
static double conv2_outputTensor[32][12][12]{}; // ConvOutputTensor[FilterNum][W][H]
static double pooling2_outputTensor[32][6][6]{};

static double filter_3[32][32][3][3], filter_3_bias[32]{}; // Filter[FilterNum][Channels][W][H], Bias[FilterNum]
static double paddedInputTensor_3[32][8][8]{}; // PaddedTensor[Channels][W][H]
static double conv3_outputTensor[32][6][6]{}; // ConvOutputTensor[FilterNum][W][H]
static double pooling3_outputTensor[32][3][3]{};

static double flattenOutputTensor[32 * 3 * 3]{}; // FlattenedTensor[Channels * W * H]

static const int fc1_Num = 96, fc2_Num = 4;
static double fc1_outputTensor[fc1_Num]{};
static double fc2_outputTensor[fc2_Num]{};

static double weightTensor_1[fc1_Num][32 * 3 * 3 + 1] = { 0 }; // [�������Ԫ��][�������Ԫ�� + 1]
static double weightTensor_2[fc2_Num][fc1_Num + 1] = { 0 };

// UpdateWeights �еĸ����ݶȷ���
static double pooling3_grad[32][6][6]{};
static double conv3_grad[32][6][6]{}, filter3_grad[32][32][3][3]{}, filter3_bias_grad[32]{};
static double pooling2_grad[32][12][12]{};
static double conv2_grad[16][12][12]{}, filter2_grad[32][16][3][3]{}, filter2_bias_grad[32]{};
static double pooling1_grad[16][24][24]{};
static double conv1_grad[1][24][24]{}, filter1_grad[16][1][3][3]{}, filter1_bias_grad[16]{};

static void UpdateWeights(double* predictedTensor, double* labelTensor, double learningRate) {
    // ���� CrosEntropyLoss �� Softmax ���ݶȣ�Ҳ���� Loss �Եڶ��� Linear ���ݶ�
    double linear1_grad[fc2_Num]{};
    SoftmaxAndCrossEntropyLossDerivative(predictedTensor, labelTensor, linear1_grad, fc2_Num);
    
    // ����Ե�һ�� Linear �ݶ�
    double linear2_grad[fc1_Num]{};
    LinearVectorDerivative(linear1_grad, linear2_grad, fc1_Num, fc2_Num, &weightTensor_2[0][0]);
    ReLuVectorDerivative(linear2_grad, fc1_Num, fc1_outputTensor);

    // ����� flatten ���ݶȣ����� flatten ��û���κ��£����ñ����ݶȾ��������൱�ڶ� pooling3 ���ݶ�
    double* flatten_grad = (double*)calloc(32 * 3 * 3, sizeof(double));
    if (flatten_grad == NULL) { printf("Memory allocation for flatten_grad failed.\n"); return; }
    LinearVectorDerivative(linear2_grad, flatten_grad, 32 * 3 * 3, fc1_Num, &weightTensor_1[0][0]);

    // ����� conv ReLU3 ���ݶ�
    // double pooling3_grad[32][6][6]{};
    MaxPoolingDerivatives(&conv3_outputTensor[0][0][0], flatten_grad, &pooling3_grad[0][0][0], 32, 6, 6, 2, 2, 2, 2);
    free(flatten_grad);

    // ����� conv3 ���ݶ�
    ReLuVectorDerivative(&pooling3_grad[0][0][0], 32 * 6 * 6, &conv3_outputTensor[0][0][0]);

    // ����� maxpooling2 ���ݶ� �� �� filter3 ���ݶ�
    // double conv3_grad[32][6][6]{}, filter3_grad[32][32][3][3]{}, filter3_bias_grad[32]{};
    ConvDerivative(&pooling3_grad[0][0][0], 6, 6, &paddedInputTensor_3[0][0][0], 32, 8, 8, &filter_3[0][0][0][0], 32, 3, 3, 1, 1, 1,
        &conv3_grad[0][0][0], &filter3_grad[0][0][0][0], filter3_bias_grad);

    // ����� conv ReLU2 ���ݶ�
    // double pooling2_grad[32][12][12]{};
    MaxPoolingDerivatives(&conv2_outputTensor[0][0][0], &conv3_grad[0][0][0], &pooling2_grad[0][0][0], 32, 12, 12, 2, 2, 2, 2);

    // ����� conv2 ���ݶ�
    ReLuVectorDerivative(&pooling2_grad[0][0][0], 32 * 12 * 12, &conv2_outputTensor[0][0][0]);

    // ����� maxpooling1 ���ݶ� �� �� filter2 ���ݶ�
    // double conv2_grad[16][12][12]{}, filter2_grad[32][16][3][3]{}, filter2_bias_grad[32]{};
    ConvDerivative(&pooling2_grad[0][0][0], 12, 12, &paddedInputTensor_2[0][0][0], 16, 14, 14, &filter_2[0][0][0][0], 32, 3, 3, 1, 1, 1,
        &conv2_grad[0][0][0], &filter2_grad[0][0][0][0], filter2_bias_grad);

    // ����� conv ReLU1 ���ݶ�
    // double pooling1_grad[16][24][24]{};
    MaxPoolingDerivatives(&conv1_outputTensor[0][0][0], &conv2_grad[0][0][0], &pooling1_grad[0][0][0], 16, 24, 24, 2, 2, 2, 2);

    // ����� conv2 ���ݶ�
    ReLuVectorDerivative(&pooling1_grad[0][0][0], 16 * 24 * 24, &conv1_outputTensor[0][0][0]);

    // ����� ��������ͼ ���ݶ� �� �� filter1 ���ݶ�
    // double conv1_grad[1][24][24]{}, filter1_grad[16][1][3][3]{}, filter1_bias_grad[16]{};
    ConvDerivative(&pooling1_grad[0][0][0], 24, 24, &paddedInputTensor_1[0][0][0], 1, 26, 26, &filter_1[0][0][0][0], 16, 3, 3, 1, 1, 1,
        &conv1_grad[0][0][0], &filter1_grad[0][0][0][0], filter1_bias_grad);

    // ����Ȩ�� fc1
    UpdateLinearWeight(&weightTensor_1[0][0], flattenOutputTensor, linear2_grad, fc1_Num, 32 * 3 * 3, learningRate);
    // ����Ȩ�� fc2
    UpdateLinearWeight(&weightTensor_2[0][0], &fc1_outputTensor[0], linear1_grad, fc2_Num, fc1_Num, learningRate);
    // ����Ȩ�� filter1
    UpdateFilterAndBias(&filter_1[0][0][0][0], filter_1_bias, &filter1_grad[0][0][0][0], filter1_bias_grad, 16, 1, 3, 3, learningRate);
    // ����Ȩ�� filter2
    UpdateFilterAndBias(&filter_2[0][0][0][0], filter_2_bias, &filter2_grad[0][0][0][0], filter2_bias_grad, 32, 16, 3, 3, learningRate);
    // ����Ȩ�� filter3
    UpdateFilterAndBias(&filter_3[0][0][0][0], filter_3_bias, &filter3_grad[0][0][0][0], filter3_bias_grad, 32, 32, 3, 3, learningRate);
}

static void Forward(double* inputTensor, double* outputTensor) {
    // Current Tensor Shape (Channels * W * H) = 1, 24, 24

    // (inputTensor[], paddedTensor[], Channels, unpaddedW, unpaddedH, padW, padH)
    Padding(inputTensor, &paddedInputTensor_1[0][0][0], 1, 24, 24, 1, 1);
    /* 
    (inputTensor[], InputChannels, inputW, inputH, 
    filter[], filterNum, filterBias[], filterW, filterH, filterStride, 
    outputTensor[], outputW, outputH)
    */
    Conv(&paddedInputTensor_1[0][0][0], 1, 26, 26,
        &filter_1[0][0][0][0], 16, &filter_1_bias[0], 3, 3, 1,
        &conv1_outputTensor[0][0][0], 24, 24);

    // (inputTensor[], Channels * W * H, outputTensor[])
    ReLuVector(&conv1_outputTensor[0][0][0], 16 * 24 * 24, &conv1_outputTensor[0][0][0]);

    // (inputTensor[], outputTensor[], Channels, inputW, inputH, KernelW, KernelH, StrideW, StrideH)
    MaxPooling(&conv1_outputTensor[0][0][0], &pooling1_outputTensor[0][0][0], 16, 24, 24, 2, 2, 2, 2);

    // Current Tensor Shape (Channels * W * H) = 16, 12, 12

    // (inputTensor[], paddedTensor[], Channels, unpaddedW, unpaddedH, padW, padH)
    Padding(&pooling1_outputTensor[0][0][0], &paddedInputTensor_2[0][0][0], 16, 12, 12, 1, 1);
    /*
    (inputTensor[], InputChannels, inputW, inputH,
    filter[], filterNum, filterBias[], filterW, filterH, filterStride,
    outputTensor[], outputW, outputH)
    */
    Conv(&paddedInputTensor_2[0][0][0], 16, 14, 14,
        &filter_2[0][0][0][0], 32, &filter_2_bias[0], 3, 3, 1,
        &conv2_outputTensor[0][0][0], 12, 12);

    // (inputTensor[], Channels * W * H, outputTensor[])
    ReLuVector(&conv2_outputTensor[0][0][0], 32 * 12 * 12, &conv2_outputTensor[0][0][0]);

    // (inputTensor[], outputTensor[], Channels, inputW, inputH, KernelW, KernelH, StrideW, StrideH)
    MaxPooling(&conv2_outputTensor[0][0][0], &pooling2_outputTensor[0][0][0], 32, 12, 12, 2, 2, 2, 2);

    // Current Tensor Shape (Channels * W * H) = 32, 6, 6

   // (inputTensor[], paddedTensor[], Channels, unpaddedW, unpaddedH, padW, padH)
    Padding(&pooling2_outputTensor[0][0][0], &paddedInputTensor_3[0][0][0], 32, 6, 6, 1, 1);
    /*
    (inputTensor[], InputChannels, inputW, inputH,
    filter[], filterNum, filterBias[], filterW, filterH, filterStride,
    outputTensor[], outputW, outputH)
    */
    Conv(&paddedInputTensor_3[0][0][0], 32, 8, 8,
        &filter_3[0][0][0][0], 32, &filter_3_bias[0], 3, 3, 1,
        &conv3_outputTensor[0][0][0], 6, 6);

    // (inputTensor[], Channels * W * H, outputTensor[])
    ReLuVector(&conv3_outputTensor[0][0][0], 32 * 6 * 6, &conv3_outputTensor[0][0][0]);

    // (inputTensor[], outputTensor[], Channels, inputW, inputH, KernelW, KernelH, StrideW, StrideH)
    MaxPooling(&conv3_outputTensor[0][0][0], &pooling3_outputTensor[0][0][0], 32, 6, 6, 2, 2, 2, 2);
    
    // Current Tensor Shape (Channels * W * H) = 32, 3, 3

    // (inputTensor[], outputTensor[], inputChannels, outputW, outputH)
    Flatten(&pooling3_outputTensor[0][0][0], &flattenOutputTensor[0], 32, 3, 3);

    // Current Tensor Shape (Length) = 32 * 3 * 3

    // fc1_Num ����Ԫȫ���Ӳ�
    for (int i = 0; i < fc1_Num; i++) {
        fc1_outputTensor[i] = Linear(&flattenOutputTensor[0], 32 * 3 * 3, weightTensor_1[i]);
        fc1_outputTensor[i] = ReLU(fc1_outputTensor[i]);
    }

    // Current Tensor Shape (Length) = 128

    // 3 ����Ԫ�����
    for (int i = 0; i < fc2_Num; i++) {
        fc2_outputTensor[i] = Linear(&fc1_outputTensor[0], fc1_Num, weightTensor_2[i]);
    }
    Softmax(fc2_outputTensor, outputTensor, fc2_Num);

    // Current Tensor Shape (Length) = 3
}

// �����Ƕ��⺯��

void UpdateWeights_3Layers_CNN(double* predictedTensor, double* labelTensor, double learningRate) {
    UpdateWeights(predictedTensor, labelTensor, learningRate);
}

void Forward_3Layers_CNN(double* inputTensor, double* outputTensor) {
    Forward(inputTensor, outputTensor);
}

void Randomized_3Layers_CNN_Weight(int seed) {
    srand(seed);
    // (filter[], filterNum, channels, filterW, filterH) ��Bias ����Ҫ�������Ĭ�� 0��
    RandomizeFilterTensor(&filter_1[0][0][0][0], 16, 1, 3, 3);
    RandomizeFilterTensor(&filter_2[0][0][0][0], 32, 16, 3, 3);
    RandomizeFilterTensor(&filter_3[0][0][0][0], 32, 32, 3, 3);

    HeInitialize(&weightTensor_1[0][0], fc1_Num, 32 * 3 * 3 + 1); // �ʺϳ�ʼ�� ReLU �����
    XavierInitialize(&weightTensor_2[0][0], fc2_Num, fc1_Num + 1);// �ʺϳ�ʼ�� Softmax �����������������������֮�䱣���ݶȵ��ȶ�
    printf("Weight Randomized.\n");
}

void Print_3Layers_CNN_Weight() {

}

void Save_3Layers_CNN_Weight() {

}

void Load_3Layers_CNN_Weight() {

}