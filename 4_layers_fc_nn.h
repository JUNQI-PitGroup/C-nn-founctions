#pragma once


// This is a dense neural network with 4 layers, weight updated by SGD


void UpdateWeights_4Layers_NN(float* inputTensor, float* labelTensor, float learningRate);

void Forward_4Layers_NN(float* inputTensor, float* outputTensor);

void Randomized_4Layers_NN_Weight(int seed);

void Print_4Layers_NN_Weight();

void Save_4Layers_NN_Weight();

void Load_4Layers_NN_Weight();