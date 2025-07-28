#include <stdio.h>
#include <stdlib.h>

#include "nn_function.h"
#include "3_layers_fc_nn.h"

#define inputLength 2
#define batchNum 20000

double dataTensor[batchNum][2]{};
double labelTensor[batchNum][1]{};
double predictedTensor[batchNum][1]{};

const int progressBarLength = 20;

void PointIsInCircle() {
    Randomized_3Layers_NN_Weight(45);
	for (int i = 0; i < batchNum; i++) {
        dataTensor[i][0] = (double)(rand() % 100) / 100.0 * 2 - 1; // x in [-1, 1]
        dataTensor[i][1] = (double)(rand() % 100) / 100.0 * 2 - 1; // y in [-1, 1]
		// Check if the point is inside the circle of radius 0.5 centered at (0,0)
		if (dataTensor[i][0] + dataTensor[i][1] <= 1.0) {
            labelTensor[i][0] = 1.0; // Inside the circle
		} else {
            labelTensor[i][0] = 0; // Outside the circle
		}
	}
    // 可自定义参数
    const int batchSize = batchNum;
    const double initialLearningRate = 0.01;
    const int initialEpoch = 200;
    const int patience = 50;
    const double learningRateDecayTo = 0.5; // 学习率衰减到原来的 learningRateDecayTo 倍

    for (int i = 0; i < batchSize; i++) {
        double doubleInputTensor[inputLength] = { 0 };
        for (int j = 0; j < inputLength; j++)
            doubleInputTensor[j] = (double)dataTensor[i][j];
        Forward_3Layers_NN(doubleInputTensor, predictedTensor[i]);
        PrintProgressBar("predicting...", i + 1, batchSize, progressBarLength);
    }
    double batchLoss = MSE_BatchLoss(&predictedTensor[0][0], &labelTensor[0][0], 1, batchSize);
    printf("Epoch %d/%d  BatchLoss = %.9f  lr = %f\n", 0, initialEpoch, batchLoss, 0.0);


    for (int epoch = 1; epoch <= initialEpoch; epoch++) {
        double learningRate = LearningRateDecay(epoch, initialLearningRate, initialEpoch, learningRateDecayTo);
        for (int i = 0; i < batchSize; i++) {
            double doubleInputTensor[inputLength] = { 0 };
            for (int j = 0; j < inputLength; j++)
                doubleInputTensor[j] = (double)dataTensor[i][j];
            Forward_3Layers_NN(doubleInputTensor, predictedTensor[i]);
            //printf("Loss = %f\n", Loss(predictedTensor[i][0], labelTensor[i][0]));
            UpdateWeights_3Layers_NN(doubleInputTensor, labelTensor[i], learningRate);
            PrintProgressBar("training...", i + 1, batchSize, progressBarLength);
        }
        double batchLoss = MSE_BatchLoss(&predictedTensor[0][0], &labelTensor[0][0], 1, batchSize);

        if (EarlyStop(batchLoss, epoch, patience)) break;

        printf("Epoch %d/%d  BatchLoss = %.9f  lr = %f\n", epoch, initialEpoch, batchLoss, learningRate);
    }
}
