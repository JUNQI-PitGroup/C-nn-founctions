#pragma once


// This is a dense neural network with 5 layers, weight updated by SGD


void UpdateWeights_5Layers_NN(double* inputTensor, double* labelTensor, double learningRate);

void Forward_5Layers_NN(double* inputTensor, double* outputTensor);

void Randomized_5Layers_NN_Weight(int seed);

void Print_5Layers_NN_Weight();

void Save_5Layers_NN_Weight();

void Load_5Layers_NN_Weight();