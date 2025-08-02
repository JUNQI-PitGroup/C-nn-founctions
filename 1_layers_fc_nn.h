#pragma once


// This is a dense neural network with 1 layers, weight updated by SGD


void UpdateWeights_1Layers_NN(float* inputTensor, float* labelTensor, float learningRate);

void Forward_1Layers_NN(float* inputTensor, float* outputTensor);

void Randomized_1Layers_NN_Weight(int seed);

void Print_1Layers_NN_Weight();

void Save_1Layers_NN_Weight();

void Load_1Layers_NN_Weight();