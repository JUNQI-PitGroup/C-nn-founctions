#pragma once


// This is a dense neural network with 6 layers, weight updated by SGD


void UpdateWeights_6Layers_NN(float* inputTensor, float* labelTensor, float learningRate);

void Forward_6Layers_NN(float* inputTensor, float* outputTensor);

void Randomized_6Layers_NN_Weight(int seed);

void Print_6Layers_NN_Weight();

void Save_6Layers_NN_Weight();

void Load_6Layers_NN_Weight();