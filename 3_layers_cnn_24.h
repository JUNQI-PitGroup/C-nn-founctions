#pragma once


void UpdateWeights_3Layers_CNN(double* predictedTensor, double* labelTensor, double learningRate);

void Forward_3Layers_CNN(double* inputTensor, double* outputTensor);

void Randomized_3Layers_CNN_Weight(int seed);

void Print_3Layers_CNN_Weight();

void Save_3Layers_CNN_Weight();

void Load_3Layers_CNN_Weight();