#pragma once


// This is a dense neural network with 2 layers, weight updated by SGD


void UpdateWeights_2Layers_NN(float* inputTensor, float* labelTensor, float learningRate);

void Forward_2Layers_NN(float* inputTensor, float* outputTensor);

void Randomized_2Layers_NN_Weight(int seed);

void Print_2Layers_NN_Weight();

void Save_2Layers_NN_Weight();

void Load_2Layers_NN_Weight();