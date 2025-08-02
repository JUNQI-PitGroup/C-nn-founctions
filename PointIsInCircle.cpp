#include <stdio.h>
#include <stdlib.h>

#include "nn_function.h"
#include "3_layers_fc_nn.h"

#define inputLength 2
#define batchNum 20000

float dataTensor[batchNum][2]{};
float labelTensor[batchNum][1]{};
float predictedTensor[batchNum][1]{};

const int progressBarLength = 20;

void PointIsInCircle() {
    Randomized_3Layers_NN_Weight(45);
	for (int i = 0; i < batchNum; i++) {
        dataTensor[i][0] = (float)(rand() % 100) / 100.0f * 2 - 1; // x in [-1, 1]
        dataTensor[i][1] = (float)(rand() % 100) / 100.0f * 2 - 1; // y in [-1, 1]
		// Check if the point is inside the circle of radius 0.5 centered at (0,0)
		if (dataTensor[i][0] + dataTensor[i][1] <= 1.0f) {
            labelTensor[i][0] = 1.0f; // Inside the circle
		} else {
            labelTensor[i][0] = 0; // Outside the circle
		}
	}
    // 可自定义参数
    const int batchSize = batchNum;
    const float initialLearningRate = 0.01;
    const int initialEpoch = 200;
    const int patience = 200;
    const float learningRateDecayTo = 0.5; // 学习率衰减到原来的 learningRateDecayTo 倍

    for (int i = 0; i < batchSize; i++) {
        float floatInputTensor[inputLength] = { 0 };
        for (int j = 0; j < inputLength; j++)
            floatInputTensor[j] = (float)dataTensor[i][j];
        Forward_3Layers_NN(floatInputTensor, predictedTensor[i]);
        PrintProgressBar("predicting...", i + 1, batchSize, progressBarLength);
    }
    float batchLoss = MSE_BatchLoss(&predictedTensor[0][0], &labelTensor[0][0], 1, batchSize);
    printf("Epoch %d/%d  BatchLoss = %.9f  lr = %f\n", 0, initialEpoch, batchLoss, 0.0f);


    for (int epoch = 1; epoch <= initialEpoch; epoch++) {
        float learningRate = LearningRateDecay(epoch, initialLearningRate, initialEpoch, learningRateDecayTo);
        for (int i = 0; i < batchSize; i++) {
            float floatInputTensor[inputLength] = { 0 };
            for (int j = 0; j < inputLength; j++)
                floatInputTensor[j] = (float)dataTensor[i][j];
            Forward_3Layers_NN(floatInputTensor, predictedTensor[i]);
            //printf("Loss = %f\n", Loss(predictedTensor[i][0], labelTensor[i][0]));
            UpdateWeights_3Layers_NN(floatInputTensor, labelTensor[i], learningRate);
            PrintProgressBar("training...", i + 1, batchSize, progressBarLength);
        }
        float batchLoss = MSE_BatchLoss(&predictedTensor[0][0], &labelTensor[0][0], 1, batchSize);

        if (EarlyStop(batchLoss, epoch, patience)) break;

        printf("Epoch %d/%d  BatchLoss = %.9f  lr = %f\n", epoch, initialEpoch, batchLoss, learningRate);
    }
}