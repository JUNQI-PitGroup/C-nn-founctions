# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <chrono>


// training functions
double LearningRateDecay(int epoch, double initialLearningRate, int initialEpoch, double decayRate) {
    if (initialEpoch < 20)
        return initialLearningRate;
    double k = ((1 - decayRate) * initialLearningRate) / (initialEpoch / 2.0);
    if (epoch < 3 * initialEpoch / 4 && epoch >= initialEpoch / 4)
        return initialLearningRate - k * (epoch - initialEpoch / 4.0); // decay when epoch is between 1/4 and 3/4 of initialEpoch
    else if (epoch < initialEpoch / 4)
        return initialLearningRate;
    else
        return initialLearningRate * decayRate;
}

double lowestLoss = DBL_MAX; // 无穷大
int lowestLossEpoch = 0;
bool EarlyStop(double batchLoss, int epoch, int patience) {
    if (epoch == 1) {
        lowestLoss = batchLoss;
        lowestLossEpoch = epoch;
        return false;
    }
    if (batchLoss < lowestLoss * 0.995) {
        lowestLoss = batchLoss;
        lowestLossEpoch = epoch;
    }
    if (epoch - lowestLossEpoch > patience) {
        printf("Early stopping at epoch %d   BatchLoss = %.9f\n", epoch, batchLoss);
        return true;
    }
    return false;
}

// weight initialization functions
void RandomizeWeightTensor(double* weightTensor, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols + 1; j++)
            weightTensor[i * cols + j] = ((rand() % 100) / 1000.0) + 0.1;
}
void XavierInitialize(double* weightTensor, int rows, int cols) {
    // Xavier initialization for Sigmoid or Tanh
    double scale = sqrt(6.0 / (rows + cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weightTensor[i * cols + j] = ((double)rand() / RAND_MAX) * 2 * scale - scale;
        }
    }
}
void HeInitialize(double* weightTensor, int rows, int cols) {
    // He initialization for ReLU
    double scale = sqrt(2.0 / cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weightTensor[i * cols + j] = ((double)rand() / RAND_MAX) * 2 * scale - scale;
        }
    }
}
void RandomizeFilterTensor(double* filterTensor, int kernelsNum, int channels, int filterWidth, int filterHeight) {
    int kernelSize = kernelsNum * filterWidth * filterHeight * channels;
    for (int i = 0; i < kernelSize; i++)
        filterTensor[i] = ((rand() % 2000) / 10000.0) - 0.1;
}

// weight update functions
void UpdateLinearWeight(double* weightTensor, double* fcInputTensor, double* layerGradTensor, int neuronNum, int inputLength, double learningRate) {
    for (int j = 0; j < neuronNum; j++) {
        for (int k = 0; k < inputLength; k++) {
            weightTensor[j * (inputLength + 1) + k] -= learningRate * layerGradTensor[j] * fcInputTensor[k];
        }
        weightTensor[j * (inputLength + 1) + inputLength] -= learningRate * layerGradTensor[j]; // bias
    }
}
void UpdateFilterAndBias(double* filterTensor, double* filterBias, double* dFilterTensor, double* dFilterBias, 
    int filterNum, int filterChannels, int filterWidth, int filterHeight, double learningRate) {
    int oneFilterSize = filterWidth * filterHeight * filterChannels;
    for (int filterIndex = 0; filterIndex < filterNum; filterIndex++) {
        for (int pos = 0; pos < oneFilterSize; pos++) {
            int filterPos = filterIndex * oneFilterSize + pos;
            filterTensor[filterPos] -= learningRate * dFilterTensor[filterPos];
        }
        dFilterBias[filterIndex] -= learningRate * dFilterBias[filterIndex];
    }
}

// activation functions and their derivatives
double Linear(double* inputTensor, int inputSize, double* weightTensor) {
    double sum = 0;
    for (int i = 0; i < inputSize; i++) {
        sum += inputTensor[i] * weightTensor[i];
    }
    sum += weightTensor[inputSize]; // bias
    return sum;
}
double ReLU(double x) {
    return x > 0 ? x : 0.0;
}
double Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double Tanh(double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}
double NoGrad(double x) {
    return 1.0;
}

void Softmax(double* inputTensor, double* outputTensor, int length) {
    double maxVal = inputTensor[0];
    for (int i = 1; i < length; i++) {
        if (inputTensor[i] > maxVal) {
            maxVal = inputTensor[i];
        }
    }

    double sumExp = 0.0;
    for (int i = 0; i < length; i++) {
        sumExp += exp(inputTensor[i] - maxVal); // 减去最大值以提高数值稳定性
    }

    for (int i = 0; i < length; i++) {
        outputTensor[i] = exp(inputTensor[i] - maxVal) / sumExp;
    }
    /*
    在 Forward 函数中的使用方法：
    // 计算第二层神经元的原始输出
    for (int i = 0; i < layer2_neuronNum; i++) {
        layer2_outputTensor[i] = Linear(layer1_outputTensor, layer1_neuronNum, weightTensor_2[i]);
    }

    // 使用Softmax处理第二层输出
    Softmax(layer2_outputTensor, outputTensor, layer2_neuronNum);
    */
}

void LinearVector(double* inputTensor, int inputLength, double* weightTensor, int outputLength, double* outputTensor) {
    // inputTensor 是输入向量，inputLength 是输入向量的维度，weightTensor 是权重矩阵，outputLength 是输出向量的维度，outputTensor 是输出向量
    for (int i = 0; i < outputLength; i++) {
        outputTensor[i] = Linear(inputTensor, inputLength, weightTensor + i * (inputLength + 1));
    }
}
void ReLuVector(double* inputTensor, int length, double* outputTensor) {
    for (int i = 0; i < length; i++) {
        outputTensor[i] = ReLU(inputTensor[i]);
    }
}
void SigmoidVector(double* inputTensor, int length, double* outputTensor) {
    for (int i = 0; i < length; i++) {
        outputTensor[i] = Sigmoid(inputTensor[i]);
    }
}
void TanhVector(double* inputTensor, int length, double* outputTensor) {
    for (int i = 0; i < length; i++) {
        outputTensor[i] = Tanh(inputTensor[i]);
    }
}
void NoGradVector(double* inputTensor, int length, double* outputTensor) {
    for (int i = 0; i < length; i++) {
        outputTensor[i] = NoGrad(inputTensor[i]);
    }
}

double ReLUDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}
double SigmoidDerivative(double x) {
    return Sigmoid(x) * (1.0 - Sigmoid(x));
}
double TanhDerivative(double x) {
    return 1.0 - Tanh(x) * Tanh(x);
}

void LinearVectorDerivative(double* inputDerivativeTensor, double* outputDerivativeTensor, int linearInputLength, int linearOutputLength, double* weightTensor) {
    // inputTensor 是 Loss 相对 Linear 的梯度，outputTensor 是我们要求的梯度（Loss 对 Linear 上一层的梯度）
    // linearInputLength 是输入的维度，outputLength 是输出的维度（本层神经元数），weightTensor 是本层权重
    for (int j = 0; j < linearInputLength; j++) {
        outputDerivativeTensor[j] = 0;
        for (int k = 0; k < linearOutputLength; k++) {
            outputDerivativeTensor[j] += inputDerivativeTensor[k] * weightTensor[k * (linearInputLength + 1) + j];
        }
    }
}
void ReLuVectorDerivative(double* DerivativeTensor, int length, double* reluInputTensor) {
    // DerivativeTensor 是 Loss 相对 ReLU 前一层的梯度，reluInputTensor 是输入到 ReLU 的梯度
    for (int i = 0; i < length; i++) {
        DerivativeTensor[i] *= ReLUDerivative(reluInputTensor[i]);
    }
}
void SigmoidVectorDerivative(double* DerivativeTensor, int length, double* sigmoidInputTensor) {
    for (int i = 0; i < length; i++) {
        DerivativeTensor[i] *= SigmoidDerivative(sigmoidInputTensor[i]);
    }
}
void TanhVectorDerivative(double* DerivativeTensor, int length, double* tanhInputTensor) {
    for (int i = 0; i < length; i++) {
        DerivativeTensor[i] *= TanhDerivative(tanhInputTensor[i]);
    }
}

void Padding(double* inputTensor, double* paddedTensor, int inputChannels, int inputWidth, int inputHeight, int addWidth, int addHeight) {
    // 在输入的图像的 上下左右 周围添加 0 像素
    // paddedTensor 应该提前初始化为 全0 后再输入到这个函数
    int inputOneChannelSize = inputWidth * inputHeight;
    int paddedWidth = inputWidth + addWidth * 2;
    int paddedHeight = inputHeight + addHeight * 2;
    int paddedOneChannelSize = paddedWidth * paddedHeight;
    for (int channels = 0; channels < inputChannels; channels++) {
        for (int Rows = 0; Rows < inputHeight; Rows++) {
            for (int Cols = 0; Cols < inputWidth; Cols++) {
                paddedTensor[channels * paddedOneChannelSize + (Rows + addHeight) * paddedWidth + Cols + addWidth] =
                    inputTensor[inputOneChannelSize * channels + Rows * inputWidth + Cols];
            }
        }
    }
}

// convolution functions and their derivatives
void Conv(double* inputTensor, int inputChannels, int inputWidth, int inputHeight,
          double* filterTensor, int filterNum, double* filterBias, int filterWidth, int filterHeight, int stride,
          double* outputTensor, int outputWidth, int outputHeight) {
    // 输入 所有 通道的特征图，输入 所有 卷积核，输出 所有 通道的特征图
    // 卷积核可以是任意大小 卷积核的定义：double filter_n[卷积核个数][输入特征图的通道数][卷积核长][卷积核宽]; double filterBias[卷积核个数];
    // 输出特征图的定义：double layer_n_output[卷积核数][输出特征图的行数][输出特征图的列数];
    int inputOneChannelSize = inputWidth * inputHeight;
    int filterArea = filterWidth * filterHeight;
    int oneFilterSize = inputChannels * filterArea;
    int outputOneChannelSize = outputWidth * outputHeight;

    int rightLimit = inputWidth - filterWidth + 1;
    int bottomLimit = inputHeight - filterHeight + 1;
    for (int filterIndex = 0; filterIndex < filterNum; filterIndex++) {
        for (int inputRows = 0, outputRows = 0; inputRows < bottomLimit; inputRows += stride, outputRows++) {
            for (int inputCols = 0, outputCols = 0; inputCols < rightLimit; inputCols += stride, outputCols++) {
                double sum = 0.0;
                for (int dRows = 0; dRows < filterHeight; dRows++) {
                    for (int dCols = 0; dCols < filterWidth; dCols++) {
                        for (int channels = 0; channels < inputChannels; channels++) {
                            sum += inputTensor[channels * inputOneChannelSize + (inputRows + dRows) * inputWidth + inputCols + dCols] *
                                filterTensor[filterIndex * oneFilterSize + filterArea * channels + dRows * filterWidth + dCols];
                        }
                    }
                }
                outputTensor[filterIndex * outputOneChannelSize + outputRows * outputWidth + outputCols] = sum + filterBias[filterIndex];
            }
        }
    }
}

void ConvDerivative(double* dOutput, int outputWidth, int outputHeight,
    double* paddedInputTensor, int paddedInputChannels, int paddedInputWidth, int paddedInputHeight,
    double* filterTensor, int filterNum, int filterWidth, int filterHeight, int stride, int paddingW, int paddingH,
    double* dUnpaddedInputTensor, double* dFilterTensor, double* dFilterBias) {
    /*
        dOutput 是从 Loss 传来的梯度（dLoss / dConv），dUnpaddedInputTensor 是我们要求得的 Loss 对卷积层上一层的梯度
        dFilterTensor 是 Loss 对卷积核的梯度，dFilterBias 是 Loss 对卷积核偏置项的梯度
        如果卷积前有 padding，那么就写 W 方向和 H 方向上增加的 0 的数量，如果卷积前没 padding，那么就 paddingW 和 paddingH 为 0
    */
    // 每一个卷积核都有 paddedInputChannels 个通道，每个通道的梯度都是上一层的某个卷积核的结果，所以
    // dUnpaddedInputTensor 的维度是 (unpaddedInputHeight * unpaddedInputWidth, inputChannels)
    // dFilterTensor 的维度是 (filterHeight * filterWidth, inputChannels)

    // dUnpaddedInputTensor, dFilterTensor 和 dFilterBias 都要初始化为 0 
    int dUnpaddedInputTensorLength = (paddedInputWidth - 2 * paddingW) * (paddedInputHeight - 2 * paddingH) * paddedInputChannels;
    for (int i = 0; i < dUnpaddedInputTensorLength; i++) dUnpaddedInputTensor[i] = 0.0;
    int dFilterTensorLength = filterWidth * filterHeight * paddedInputChannels * filterNum;
    for (int i = 0; i < dFilterTensorLength; i++) dFilterTensor[i] = 0.0;
    int dFilterBiasLength = filterNum;
    for (int i = 0; i < dFilterBiasLength; i++) dFilterBias[i] = 0.0;

    int filterArea = filterWidth * filterHeight;

    int unpaddedInputWidth = paddedInputWidth - 2 * paddingW;
    int unpaddedInputHeight = paddedInputHeight - 2 * paddingH;
    int unpaddedInputArea = unpaddedInputWidth * unpaddedInputHeight;

    int oneFilterSize = paddedInputChannels * filterArea;
    int outputOneChannelSize = outputWidth * outputHeight;

    // 计算梯度
    for (int filterIndex = 0; filterIndex < filterNum; filterIndex++) { // 注意，这里的 filterIndex 是指 卷积核的索引，同样也是输出的通道索引
        for (int outputRows = 0; outputRows < outputHeight; outputRows++) {
            for (int outputCols = 0; outputCols < outputWidth; outputCols++) {
                int inputRows = outputRows * stride;
                int inputCols = outputCols * stride;
                double dOutputValue = dOutput[filterIndex * outputOneChannelSize + outputRows * outputWidth + outputCols];

                // 更新卷积核和输入张量的梯度
                for (int dRows = 0; dRows < filterHeight; dRows++) {
                    for (int dCols = 0; dCols < filterWidth; dCols++) {
                        // 计算填充后的索引
                        int paddedInputRow = inputRows + dRows;
                        int paddedInputCol = inputCols + dCols;
                        int unpaddedInputRow = paddedInputRow - paddingH;
                        int unpaddedInputCol = paddedInputCol - paddingW;

                        // 判断是否在非 padding 区域内
                        if (paddedInputRow >= paddingH && paddedInputRow < paddedInputHeight - paddingH &&
                            paddedInputCol >= paddingW && paddedInputCol < paddedInputWidth - paddingW) {

                            for (int channels = 0; channels < paddedInputChannels; channels++) {
                                // 计算未填充输入的实际索引
                                int unpaddedInputIndex = channels * unpaddedInputArea + unpaddedInputRow * unpaddedInputWidth + unpaddedInputCol;
                                int filterPos = filterIndex * oneFilterSize + channels * filterArea + dRows * filterWidth + dCols;

                                // 更新卷积核的梯度
                                dFilterTensor[filterPos] += paddedInputTensor[paddedInputRow * paddedInputWidth + paddedInputCol] * dOutputValue;
                                // 更新未填充输入的梯度
                                dUnpaddedInputTensor[unpaddedInputIndex] += filterTensor[filterPos] * dOutputValue;
                            }
                        }
                    }
                }
                // 更新偏置项的梯度
                dFilterBias[filterIndex] += dOutputValue;
            }
        }
    }
}

// pooling functions and their derivatives
void MaxPooling(double* inputTensor, double* outputTensor, int inputChannels, int inputWidth, int inputHeight, 
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight) {
    // 参数（输入矩阵，输出矩阵，通道数，输入宽度，输入高度，池化宽度，池化高度，步长宽度，步长高度）
    int poolingRows = (inputHeight - poolingKernelHeight) / strideHeight + 1;
    int poolingCols = (inputWidth - poolingKernelWidth) / strideWidth + 1;
    int inputOneChannelSize = inputHeight * inputWidth;
    int outputOneChannelSize = poolingRows * poolingCols;

    for (int Channels = 0; Channels < inputChannels; Channels++) {
        for (int Rows = 0; Rows < poolingRows; Rows++) {
            for (int Cols = 0; Cols < poolingCols; Cols++) {
                double maxVal = inputTensor[inputOneChannelSize * Channels + Rows * strideHeight * inputWidth + Cols * strideWidth];
                int maxIndex = inputOneChannelSize * Channels + Rows * strideHeight * inputWidth + Cols * strideWidth;
                for (int dRows = 0; dRows < poolingKernelHeight; dRows++) {
                    for (int dCols = 0; dCols < poolingKernelWidth; dCols++) {
                        int index = inputOneChannelSize * Channels + (Rows * strideHeight + dRows) * inputWidth + Cols * strideWidth + dCols;
                        if (inputTensor[index] > maxVal) {
                            maxVal = inputTensor[index];
                            maxIndex = index;
                        }
                    }
                }
                outputTensor[outputOneChannelSize * Channels + Rows * poolingCols + Cols] = maxVal;
            }
        }
    }
    return;
}
void MinPooling(double* inputTensor, double* outputTensor, int inputChannels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight) {
    // 参数（输入矩阵，输出矩阵，通道数，输入宽度，输入高度，池化宽度，池化高度，步长宽度，步长高度）
    int poolingRows = (inputHeight - poolingKernelHeight) / strideHeight + 1;
    int poolingCols = (inputWidth - poolingKernelWidth) / strideWidth + 1;
    int inputOneChannelSize = inputHeight * inputWidth;
    int outputOneChannelSize = poolingRows * poolingCols;

    for (int Channels = 0; Channels < inputChannels; Channels++) {
        for (int Rows = 0; Rows < poolingRows; Rows++) {
            for (int Cols = 0; Cols < poolingCols; Cols++) {
                double minVal = inputTensor[inputOneChannelSize * Channels + Rows * strideHeight * inputWidth + Cols * strideWidth];
                int minIndex = inputOneChannelSize * Channels + Rows * strideHeight * inputWidth + Cols * strideWidth;
                for (int dRows = 0; dRows < poolingKernelHeight; dRows++) {
                    for (int dCols = 0; dCols < poolingKernelWidth; dCols++) {
                        int index = inputOneChannelSize * Channels + (Rows * strideHeight + dRows) * inputWidth + Cols * strideWidth + dCols;
                        if (inputTensor[index] < minVal) {
                            minVal = inputTensor[index];
                            minIndex = index;
                        }
                    }
                }
                outputTensor[outputOneChannelSize * Channels + Rows * poolingCols + Cols] = minVal;
            }
        }
    }
    return;
}
void AvgPooling(double* inputTensor, double* outputTensor, int inputChannels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight) {
    // 参数（输入矩阵，输出矩阵，通道数，输入宽度，输入高度，池化宽度，池化高度，步长宽度，步长高度）
    int poolingRows = (inputHeight - poolingKernelHeight) / strideHeight + 1;
    int poolingCols = (inputWidth - poolingKernelWidth) / strideWidth + 1;
    int inputOneChannelSize = inputHeight * inputWidth;
    int outputOneChannelSize = poolingRows * poolingCols;
    int poolingArea = poolingKernelWidth * poolingKernelHeight;

    for (int Channels = 0; Channels < inputChannels; Channels++) {
        for (int Rows = 0; Rows < poolingRows; Rows++) {
            for (int Cols = 0; Cols < poolingCols; Cols++) {
                double sum = 0.0;
                for (int dRows = 0; dRows < poolingKernelHeight; dRows++) {
                    for (int dCols = 0; dCols < poolingKernelWidth; dCols++) {
                        int index = inputOneChannelSize * Channels + (Rows * strideHeight + dRows) * inputWidth + Cols * strideWidth + dCols;
                        sum += inputTensor[index];
                    }
                }
                outputTensor[outputOneChannelSize * Channels + Rows * poolingCols + Cols] = sum / poolingArea;
            }
        }
    }
    return;
}

void MaxPoolingDerivatives(double* inputTensor, double* inputDerivativeTensor, double* outputDerivativeTensor,
    int Channels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight) {
    /* 
        inputTensor: 前向传播时输入到池化层矩阵，
        inputDerivativeTensor: 反向传播时传向池化层前的导数矩阵，
        outputDerivativeTensor: 反向传播时 Loss 对池化层前一层的导数矩阵，也就是我们要求的导数矩阵
        
        前向传播时的通道数，前向传播时的输入宽度，前向传播时的输入高度，
        池化宽度，池化高度，步长宽度，步长高度）
    */
    // outputDerivativeTensor 初始化为全 0

    int outputDerivativeTensorLength = Channels * inputWidth * inputHeight;
    for (int i = 0; i < outputDerivativeTensorLength; i++) outputDerivativeTensor[i] = 0.0;

    int poolingRows = (inputHeight - poolingKernelHeight) / strideHeight + 1;
    int poolingCols = (inputWidth - poolingKernelWidth) / strideWidth + 1;
    int inputOneChannelSize = inputHeight * inputWidth;
    int outputOneChannelSize = poolingRows * poolingCols;

    for (int c = 0; c < Channels; c++) {
        for (int i = 0; i < poolingRows; i++) {
            for (int j = 0; j < poolingCols; j++) {
                int maxIndex = inputOneChannelSize * c + i * strideHeight * inputWidth + j * strideWidth;
                double maxVal = inputTensor[maxIndex];
                for (int di = 0; di < poolingKernelHeight; di++) {
                    for (int dj = 0; dj < poolingKernelWidth; dj++) {
                        int index = inputOneChannelSize * c + (i * strideHeight + di) * inputWidth + j * strideWidth + dj;
                        if (index >= inputOneChannelSize * c && index < inputOneChannelSize * (c + 1)) {
                            if (inputTensor[index] > maxVal) {
                                maxVal = inputTensor[index];
                                maxIndex = index;
                            }
                        }
                    }
                }
                outputDerivativeTensor[maxIndex] += inputDerivativeTensor[outputOneChannelSize * c + i * poolingCols + j];
            }
        }
    }
    return;
}
void MinPoolingDerivatives(double* inputTensor, double* inputDerivativeTensor, double* outputDerivativeTensor,
    int Channels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight) {
    /*
        inputTensor: 前向传播时输入到池化层矩阵，
        inputDerivativeTensor: 反向传播时传向池化层前的导数矩阵，
        outputDerivativeTensor: 反向传播时 Loss 对池化层前一层的导数矩阵，也就是我们要求的导数矩阵

        前向传播时的通道数，前向传播时的输入宽度，前向传播时的输入高度，
        池化宽度，池化高度，步长宽度，步长高度）
    */
    // outputDerivativeTensor 要提前初始化为全 0
    int poolingRows = (inputHeight - poolingKernelHeight) / strideHeight + 1;
    int poolingCols = (inputWidth - poolingKernelWidth) / strideWidth + 1;
    int inputOneChannelSize = inputHeight * inputWidth;
    int outputOneChannelSize = poolingRows * poolingCols;

    for (int c = 0; c < Channels; c++) {
        for (int i = 0; i < poolingRows; i++) {
            for (int j = 0; j < poolingCols; j++) {
                int minIndex = inputOneChannelSize * c + i * strideHeight * inputWidth + j * strideWidth;
                double minVal = inputTensor[minIndex];
                for (int di = 0; di < poolingKernelHeight; di++) {
                    for (int dj = 0; dj < poolingKernelWidth; dj++) {
                        int index = inputOneChannelSize * c + (i * strideHeight + di) * inputWidth + j * strideWidth + dj;
                        if (index >= inputOneChannelSize * c && index < inputOneChannelSize * (c + 1)) {
                            if (inputTensor[index] < minVal) {
                                minVal = inputTensor[index];
                                minIndex = index;
                            }
                        }
                    }
                }
                outputDerivativeTensor[minIndex] += inputDerivativeTensor[outputOneChannelSize * c + i * poolingCols + j];
            }
        }
    }
    return;
}
void AvgPoolingDerivatives(double* inputTensor, double* inputDerivativeTensor, double* outputDerivativeTensor,
    int Channels, int inputWidth, int inputHeight,
    int poolingKernelWidth, int poolingKernelHeight, int strideWidth, int strideHeight) {
    /*
        inputTensor: 前向传播时输入到池化层矩阵，
        inputDerivativeTensor: 反向传播时传向池化层前的导数矩阵，
        outputDerivativeTensor: 反向传播时 Loss 对池化层前一层的导数矩阵，也就是我们要求的导数矩阵

        前向传播时的通道数，前向传播时的输入宽度，前向传播时的输入高度，
        池化宽度，池化高度，步长宽度，步长高度）
    */
    // outputDerivativeTensor 要提前初始化为全 0
    int poolingRows = (inputHeight - poolingKernelHeight) / strideHeight + 1;
    int poolingCols = (inputWidth - poolingKernelWidth) / strideWidth + 1;
    int inputOneChannelSize = inputHeight * inputWidth;
    int outputOneChannelSize = poolingRows * poolingCols;
    int poolingArea = poolingKernelWidth * poolingKernelHeight;

    for (int c = 0; c < Channels; c++) {
        for (int i = 0; i < poolingRows; i++) {
            for (int j = 0; j < poolingCols; j++) {
                int outputIndex = outputOneChannelSize * c + i * poolingCols + j;
                double gradient = inputDerivativeTensor[outputIndex] / poolingArea;
                for (int di = 0; di < poolingKernelHeight; di++) {
                    for (int dj = 0; dj < poolingKernelWidth; dj++) {
                        int inputIndex = inputOneChannelSize * c + (i * strideHeight + di) * inputWidth + j * strideWidth + dj;
                        outputDerivativeTensor[inputIndex] += gradient;
                    }
                }
            }
        }
    }
    return;
}

// faltten 3D tensor to 1D tensor
void Flatten(double* inputTensor, double* outputTensor, int inputChannels, int inputWidth, int inputHeight) {
    int inputOneChannelSize = inputHeight * inputWidth;
    int outputOneChannelSize = inputChannels * inputOneChannelSize;
    for (int i = 0; i < outputOneChannelSize; i++) {
        outputTensor[i] = inputTensor[i];
    }
    return;
}

// loss functions and their derivatives
double MSE_Loss(double* prediction, double* label, int inputLength) {
    double sum = 0;
    for (int i = 0; i < inputLength; i++)
        sum += (prediction[i] - label[i]) * (prediction[i] - label[i]);
    return sum;
}
double MSE_BatchLoss(double* predictedTensor, double* labelTensor, int eachLength, int batchSize) {
    double sum = 0;
    for (int i = 0; i < batchSize; i++) {
        sum += MSE_Loss(predictedTensor + i * eachLength, labelTensor + i * eachLength, eachLength);
    }
    return sum / batchSize;
}
void MSE_LossDerivative(double* gradTensor, double* prediction, double* label, int inputLength) {
    for (int i = 0; i < inputLength; i++)
        gradTensor[i] = 2 * (prediction[i] - label[i]);
}

# define EPSILON 1e-15 // 用于防止log(0)的问题
double CrossEntropyLoss(double* predictedTensor, double* labelTensor, int classNum) {
    double loss = 0.0;

    for (int i = 0; i < classNum; i++) {
        // 防止预测值为 0 或 1 导致的数值问题
        double clampedPredict = fmax(fmin(predictedTensor[i], 1.0 - EPSILON), EPSILON);

        // 计算交叉熵损失
        loss -= labelTensor[i] * log(clampedPredict);
    }
    return loss;
}
double CrossEntropyBatchLoss(double* predictedTensor, double* labelTensor, int batchSize, int classNum) {
    double loss = 0.0;
    for (int i = 0; i < batchSize; i++) {
        loss += CrossEntropyLoss(predictedTensor + i * classNum, labelTensor + i * classNum, classNum);
    }
    return loss / batchSize;
}

void SoftmaxAndCrossEntropyLossDerivative(double* softmaxOutput, double* labels, double* lossAndSoftmaxDerivativeTensor, int classNum) {
    // softmaxOutput 是一个长度为 classNum 的数组，求得 Loss 对 softmax 前面 LinearOutput 的导数
    for (int i = 0; i < classNum; i++) {
        lossAndSoftmaxDerivativeTensor[i] = softmaxOutput[i] - labels[i]; // Label - Predict
    }
}

// other tools
void CopyVector(double* srcTensor, double* dstTensor, int length) {
    for (int i = 0; i < length; i++) {
        dstTensor[i] = srcTensor[i];
    }
}

void RandomizeVector(double* vector, int length) {
    for (int i = 0; i < length; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

void PrintWeightTensor(double* matrix, int rows, int cols) {
    printf("{\n");
    for (int i = 0; i < rows; i++) {
        printf("{ ");
        for (int j = 0; j < cols; j++) {
            if (matrix[i * cols + j] >= 0.0) printf(" ");
            printf("%.4f", matrix[i * cols + j]);
            if (j != cols - 1)
                printf(",");
        }
        printf(" },\n");
    }
    printf(" };\n");
}
void PrintTensor1D(double* matrix, int length) {
    for (int i = 0; i < length; i++) {
        if (matrix[i] >= 0.0) printf(" ");
        printf("%.2f", matrix[i]);
        if (i != length - 1)
            printf(",");
    }
    printf("\n");
}
void PrintTensor2D(double* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix[i * cols + j] >= 0.0) printf(" ");
            printf("%.2f", matrix[i * cols + j]);
            if (j != cols - 1)
                printf(",");
        }
        printf("\n");
    }
}
void PrintTensor3D(double* matrix, int channels, int rows, int cols) {
    for (int c = 0; c < channels; c++) {
        printf("Channel: %d:\n", c);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[c * rows * cols + i * cols + j] >= 0.0) printf(" ");
                printf("%.2f", matrix[c * rows * cols + i * cols + j]);
                if (j != cols - 1)
                    printf(",");
            }
            printf("\n");
        }
    }
}
void PrintTensor4D(double* matrix, int num, int channels, int rows, int cols) {
    for (int n = 0; n < num; n++) {
        printf("Num: %d:\n", n);
        for (int c = 0; c < channels; c++) {
            printf("Channel: %d:\n", c);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (matrix[n * channels * rows * cols + c * rows * cols + i * cols + j] >= 0.0) printf(" ");
                    printf("%.2f", matrix[n * channels * rows * cols + c * rows * cols + i * cols + j]);
                    if (j != cols - 1)
                        printf(",");
                }
                printf("\n");
            }
        }
    }
}
static int lastPos = -1;
static double t1 = 0.0;
static double gettime()
{
    // 获取当前时间点
    auto now = std::chrono::high_resolution_clock::now();

    // 转换为毫秒
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    // 返回当前时间（毫秒）
    return static_cast<double>(duration);
}
// 会覆盖之前的打印信息，请把要打印的内容放在进度条的后面，progress为当前进度（从1开始到total），total为总进度
void PrintProgressBar(const char* info, int progress, int total, int barLength) {
    if (barLength <= 0) return;
    if (progress <= 1) t1 = gettime();
    int pos = (int)(barLength * (float)progress / total);
    if (pos == lastPos) return;
    printf("\r%s  [", info);
    for (int i = 0; i < barLength; i++) {
        if (i < pos)
            printf("=");
        else
            printf(" ");
    }
    double t2 = gettime();
    printf("] %d/%d  for %.1fs", progress, total, (t2 - t1) / 1000.0);
    lastPos = pos;
    if (progress == total) {
        printf("\n");
        lastPos = -1;
    }
    fflush(stdout);

}

// file tools
void Save_WeightTensor(double* weightTensor, int rows, int cols, const char* filename) {
    FILE* fp = NULL;
    errno_t err = fopen_s(&fp, filename, "w"); // 注意这里传递的是 &fp，并改为文本模式
    if (fp == NULL) {
        printf("File Stream Error: can not open file %s for writing.\n", filename);
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fprintf(fp, "%.16f ", weightTensor[i * cols + j]); // 以文本形式写入，保留小数点后16位
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}
void Load_WeightTensor(double* weightTensor, int rows, int cols, const char* filename) {
    FILE* fp = NULL;
    errno_t err = fopen_s(&fp, filename, "r"); // 注意这里传递的是 &fp，并改为文本模式
    if (fp == NULL) {
        printf("File Stream Error: can not open file %s for reading.\n", filename);
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fscanf_s(fp, "%lf", &weightTensor[i * cols + j]); // 以文本形式读取
        }
    }
    fclose(fp);
}

