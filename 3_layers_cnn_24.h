#pragma once


void UpdateWeights_3Layers_CNN(float* predictedTensor, float* labelTensor, float learningRate);

void Forward_3Layers_CNN(float* inputTensor, float* outputTensor);

void Randomized_3Layers_CNN_Weight(int seed);

void Print_3Layers_CNN_Weight();

void Save_3Layers_CNN_Weight();

void Load_3Layers_CNN_Weight();