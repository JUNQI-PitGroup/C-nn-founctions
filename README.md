This a pure C language deep leaning framework, programed when I was in college.
You can lean how to use network file such as 3_layers_fc_nn in file "PointIsInCircle.cpp", "PointIsInCircle.cpp" is a simple example for using nn flie.
All nn functions are in file "nn_function.cpp", such as linear, relu, tanh, sigmoid, Conv2d, LearningRateDecay, and their derivatives, and update weights functions. 


you can train a network easily, like using pytorch in python: 
1. define your data set:
  float dataVector[10000][5]; float labelVector[10000][1]; // suppose you have 10000 samples, input length = 5, output length = 1
2. feedforward your data vector and backward and upgrade weights (you can use AVX2 version, it's 10 times faster)
  float predictedVector[10000][1];
  for epochs{
    for (n = 0; n < BatchNum; n++){
      Forward_2Layers_NN(DataVector[n], predictedVector[n]); // Feedforward
      UpdateWeights_2Layers_NN(floatInputTensor, labelTensor[i], learningRate); // Backward and upgrade weights
      PrintProgressBar("training...", n + 1, batchSize, progressBarLength); // show process
    }
   float batchLoss = MSE_BatchLoss(&predictedVector[0][0], &labelVector[0][0], 1, batchSize);
   if (EarlyStop(batchLoss, epoch, patience)) break;
   printf("Epoch %d/%d  BatchLoss = %.9f  lr = %f\n", epoch, initialEpoch, batchLoss, learningRate);
   }
   
  Save_2Layers_NN_Weight() 
  // you need to create new folders named "2_layers_fc_nn_saved_weight" to contain weights
  Load_2Layers_NN_Weight()
  
  
This is a training process sample:
predicting...  [==================================================] 5000/5000  for 0.1s
Epoch 0/120  BatchLoss = 0.543209255  lr = 0.000000
training...  [==================================================] 5000/5000  for 0.3s
Epoch 1/120  BatchLoss = 0.534400463  lr = 0.010000
training...  [==================================================] 5000/5000  for 0.3s
Epoch 2/120  BatchLoss = 0.391641200  lr = 0.010000
training...  [==================================================] 5000/5000  for 0.3s
Epoch 3/120  BatchLoss = 0.222769946  lr = 0.010000


How to build a network Function and upgrade it's weights:

#include <stdlib.h>
#include <stdio.h>

#include "nn_function.h"

#define inputLength 449

// This is a dense neural network with 2 layers, weight updated by SGD
#define layer1_neuronNum 512   // ⬇
#define layer2_neuronNum 1     // ⬇

static float weightTensor_1[layer1_neuronNum][inputLength + 1] = { 0 }; // to save the first layer weights
static float weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] = { 0 }; // to save the second layer weights

static float linear1_outputTensor[layer1_neuronNum] = { 0 };
static float activate1_outputTensor[layer1_neuronNum] = { 0 };
static float linear2_outputTensor[layer2_neuronNum] = { 0 };
static float activate2_outputTensor[layer2_neuronNum] = { 0 };

static void Forward(float* inputTensor, float* outputTensor) {
    // first layer
    for (int i = 0; i < layer1_neuronNum; i++) {
        linear1_outputTensor[i] = Linear_AVX2(inputTensor, inputLength, weightTensor_1[i]);
        activate1_outputTensor[i] = ReLU(linear1_outputTensor[i]);
    }
    // second layer
    for (int i = 0; i < layer2_neuronNum; i++) {
        linear2_outputTensor[i] = Linear_AVX2(activate1_outputTensor, layer1_neuronNum, weightTensor_2[i]);
        activate2_outputTensor[i] = Tanh(linear2_outputTensor[i]);
    }
    for (int i = 0; i < layer2_neuronNum; i++) outputTensor[i] = activate2_outputTensor[i];
}

that's very easy

then backward and upgrade weights

static void UpdateWeights(float* inputTensor, float* labelTensor, float learningRate) {
    // backward
    float layer2_grad[layer2_neuronNum]{};
    MSE_LossDerivative(layer2_grad, activate2_outputTensor, labelTensor, layer2_neuronNum);
    TanhVectorDerivative(layer2_grad, layer2_neuronNum, linear2_outputTensor);

    float layer1_grad[layer1_neuronNum]{};
    LinearVectorDerivative(layer2_grad, layer1_grad, layer1_neuronNum, layer2_neuronNum, &weightTensor_2[0][0]);
    ReLuVectorDerivative(layer1_grad, layer1_neuronNum, linear1_outputTensor);

    // upgrade weights
    UpdateLinearWeight_AVX2(&weightTensor_2[0][0], activate1_outputTensor, layer2_grad, layer2_neuronNum, layer1_neuronNum, learningRate);
    UpdateLinearWeight_AVX2(&weightTensor_1[0][0], inputTensor, layer1_grad, layer1_neuronNum, inputLength, learningRate);
}
