#pragma once


double LearningRateDecay(int epoch, double initialLearningRate, int initialEpoch, double decayRate);

bool EarlyStop(double batchLoss, int epoch, int patience);

void RandomizeWeightTensor(double* weightTensor, int rows, int cols);
void XavierInitialize(double* weightTensor, int rows, int cols);
void HeInitialize(double* weightTensor, int rows, int cols);
void RandomizeFilterTensor(double* filterTensor, int kernels_1_Num, int channels, int filterWidth, int filterHeight);

void UpdateLinearWeight(double* weightTensor, double* fcInputTensor, double* layerGradTensor, int neuronNum, int inputLength, double learningRate);
void UpdateFilterAndBias(double* filterTensor, double* filterBias, double* dFilterTensor, double* dFilterBias,
    int filterNum, int filterChannels, int filterWidth, int filterHeight, double learningRate);

double Linear(double* inputTensor, int inputSize, double* weightTensor);
double ReLU(double x);
double Sigmoid(double x);
double Tanh(double x);
void Softmax(double* inputTensor, double* outputTensor, int length);
double NoGrad(double x);

void LinearVector(double* inputTensor, int inputLength, double* weightTensor, int outputLength, double* outputTensor);
void ReLuVector(double* inputTensor, int length, double* outputTensor);
void SigmoidVector(double* inputTensor, int length, double* outputTensor);
void TanhVector(double* inputTensor, int length, double* outputTensor);
void NoGradVector(double* inputTensor, int length, double* outputTensor);

double ReLUDerivative(double x);
double SigmoidDerivative(double x);
double TanhDerivative(double x);

void LinearVectorDerivative(double* inputDerivativeTensor, double* outputDerivativeTensor, int linearInputLength, int linearOutputLength, double* weightTensor);
void ReLuVectorDerivative(double* DerivativeTensor, int length, double* reluInputTensor);
void SigmoidVectorDerivative(double* inputTensor, int length, double* outputTensor);
void TanhVectorDerivative(double* DerivativeTensor, int length, double* tanhInputTensor);


void Padding(double* inputTensor, double* paddedTensor, int inputWidth, int inputHeight, int inputChannels, int addWidth, int addHeight);

void Conv(double* inputTensor, int inputWidth, int inputHeight, int inputChannels,
    double* filterTensor, int filterNum, double* filterBias, int filterWidth, int filterHeight, int stride,
    double* outputTensor, int outputWidth, int outputHeight);

void ConvDerivative(double* dOutput, int outputWidth, int outputHeight,
    double* paddedInputTensor, int paddedInputChannels, int paddedInputWidth, int paddedInputHeight,
    double* filterTensor, int filterNum, int filterWidth, int filterHeight, int stride, int paddingW, int paddingH,
    double* dUnpaddedInputTensor, double* dFilterTensor, double* dFilterBias);

void MaxPooling(double* inputTensor, double* outputTensor, int inputChannels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);
void MinPooling(double* inputTensor, double* outputTensor, int inputChannels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);
void AvgPooling(double* inputTensor, double* outputTensor, int inputChannels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);

void MaxPoolingDerivatives(double* inputTensor, double* inputDerivativeTensor, double* outputDerivativeTensor,
    int Channels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);
void MinPoolingDerivatives(double* inputTensor, double* inputDerivativeTensor, double* outputDerivativeTensor,
    int Channels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);
void AvgPoolingDerivatives(double* inputTensor, double* inputDerivativeTensor, double* outputDerivativeTensor,
    int Channels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);

void Flatten(double* inputTensor, double* outputTensor, int inputChannels, int inputWidth, int inputHeight);

double MSE_Loss(double* prediction, double* label, int inputLength);
double MSE_BatchLoss(double* predictedTensor, double* labelTensor, int eachLength, int batchSize);
void MSE_LossDerivative(double* gradTensor, double* prediction, double* label, int inputLength);

double CrossEntropyLoss(double* predictedTensor, double* labelTensor, int classNum);
double CrossEntropyBatchLoss(double* predictedTensor, double* labelTensor, int batchSize, int classNum);
void SoftmaxAndCrossEntropyLossDerivative(double* softmaxOutput, double* labels, double* lossAndSoftmaxDerivativeTensor, int classNum);

// other tools
void CopyVector(double* srcTensor, double* dstTensor, int length);
void RandomizeVector(double* vector, int length);

void PrintWeightTensor(double* matrix, int rows, int cols);
void PrintTensor1D(double* matrix, int length);
void PrintTensor2D(double* matrix, int rows, int cols); 
void PrintTensor3D(double* matrix, int channels, int rows, int cols);
void PrintTensor4D(double* matrix, int num, int channels, int rows, int cols);

void PrintProgressBar(const char* info, int progress, int total, int barLength);

void Save_WeightTensor(double* weightTensor, int rows, int cols, const char* filename);
void Load_WeightTensor(double* weightTensor, int rows, int cols, const char* filename);


