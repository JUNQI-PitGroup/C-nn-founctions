#pragma once


float LearningRateDecay(int epoch, float initialLearningRate, int initialEpoch, float decayRate);

bool EarlyStop(float batchLoss, int epoch, int patience);

void RandomizeWeightTensor(float* weightTensor, int rows, int cols);
void XavierInitialize(float* weightTensor, int rows, int cols);
void HeInitialize(float* weightTensor, int rows, int cols);
void RandomizeFilterTensor(float* filterTensor, int kernels_1_Num, int channels, int filterWidth, int filterHeight);

void UpdateLinearWeight(float* weightTensor, float* fcInputTensor, float* layerGradTensor, int neuronNum, int inputLength, float learningRate);
void UpdateLinearWeight_AVX2(float* weightTensor, float* fcInputTensor, float* layerGradTensor,
    int neuronNum, int inputLength, float learningRate);
void UpdateFilterAndBias(float* filterTensor, float* filterBias, float* dFilterTensor, float* dFilterBias,
    int filterNum, int filterChannels, int filterWidth, int filterHeight, float learningRate);

float Linear(float* inputTensor, int inputSize, float* weightTensor);
float Linear_AVX2(float* inputTensor, int inputSize, float* weightTensor);
inline float ReLU(float x);
inline float LeakyReLU(float x);
float Sigmoid(float x);
float Tanh(float x);
void Softmax(float* inputTensor, float* outputTensor, int length);
float NoGrad(float x);

void LinearVector(float* inputTensor, int inputLength, float* weightTensor, int outputLength, float* outputTensor);
void LinearVector_AVX2(float* inputTensor, int inputLength, float* weightTensor, int outputLength, float* outputTensor);
void ReLuVector(float* inputTensor, int length, float* outputTensor);
void SigmoidVector(float* inputTensor, int length, float* outputTensor);
void TanhVector(float* inputTensor, int length, float* outputTensor);
void NoGradVector(float* inputTensor, int length, float* outputTensor);

float ReLUDerivative(float x);
float SigmoidDerivative(float x);
float TanhDerivative(float x);

void LinearVectorDerivative(float* inputDerivativeTensor, float* outputDerivativeTensor, 
    int linearInputLength, int linearOutputLength, float* weightTensor);
void ReLuVectorDerivative(float* DerivativeTensor, int length, float* reluInputTensor);
void SigmoidVectorDerivative(float* inputTensor, int length, float* outputTensor);
void TanhVectorDerivative(float* DerivativeTensor, int length, float* tanhInputTensor);


void Padding(float* inputTensor, float* paddedTensor, int inputWidth, int inputHeight, int inputChannels, int addWidth, int addHeight);

void Conv(float* inputTensor, int inputWidth, int inputHeight, int inputChannels,
    float* filterTensor, int filterNum, float* filterBias, int filterWidth, int filterHeight, int stride,
    float* outputTensor, int outputWidth, int outputHeight);

void ConvDerivative(float* dOutput, int outputWidth, int outputHeight,
    float* paddedInputTensor, int paddedInputChannels, int paddedInputWidth, int paddedInputHeight,
    float* filterTensor, int filterNum, int filterWidth, int filterHeight, int stride, int paddingW, int paddingH,
    float* dUnpaddedInputTensor, float* dFilterTensor, float* dFilterBias);

void MaxPooling(float* inputTensor, float* outputTensor, int inputChannels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);
void MinPooling(float* inputTensor, float* outputTensor, int inputChannels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);
void AvgPooling(float* inputTensor, float* outputTensor, int inputChannels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);

void MaxPoolingDerivatives(float* inputTensor, float* inputDerivativeTensor, float* outputDerivativeTensor,
    int Channels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);
void MinPoolingDerivatives(float* inputTensor, float* inputDerivativeTensor, float* outputDerivativeTensor,
    int Channels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);
void AvgPoolingDerivatives(float* inputTensor, float* inputDerivativeTensor, float* outputDerivativeTensor,
    int Channels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight);

void Flatten(float* inputTensor, float* outputTensor, int inputChannels, int inputWidth, int inputHeight);

float MSE_Loss(float* prediction, float* label, int inputLength);
float MSE_BatchLoss(float* predictedTensor, float* labelTensor, int eachLength, int batchSize);
void MSE_LossDerivative(float* gradTensor, float* prediction, float* label, int inputLength);

float CrossEntropyLoss(float* predictedTensor, float* labelTensor, int classNum);
float CrossEntropyBatchLoss(float* predictedTensor, float* labelTensor, int batchSize, int classNum);
void SoftmaxAndCrossEntropyLossDerivative(float* softmaxOutput, float* labels, float* lossAndSoftmaxDerivativeTensor, int classNum);

// other tools
void CopyVector(float* srcTensor, float* dstTensor, int length);
void RandomizeVector(float* vector, int length);

void PrintWeightTensor(float* matrix, int rows, int cols);
void PrintTensor1D(float* matrix, int length);
void PrintTensor2D(float* matrix, int rows, int cols); 
void PrintTensor3D(float* matrix, int channels, int rows, int cols);
void PrintTensor4D(float* matrix, int num, int channels, int rows, int cols);

void PrintProgressBar(const char* info, int progress, int total, int barLength);

void Save_WeightTensor(float* weightTensor, int rows, int cols, const char* filename);
void Load_WeightTensor(float* weightTensor, int rows, int cols, const char* filename);



