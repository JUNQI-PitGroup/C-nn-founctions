#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdexcept>

#include "3_layers_cnn_24.h"
#include "nn_function.h"

# define CLASS_NUM 4
# define Channels 1
# define BatchSize 12
# define Width 24
const int oneNum = 3; // 一个数字有几张图

static float cnnDataTensor[BatchSize][Channels][Width][Width]{};
static float labelTensor[BatchSize][CLASS_NUM]{};
static float cnnOutputTensor[BatchSize][CLASS_NUM]{};

static void dataNorm(unsigned char* data, float* tensor, int width, int height) {
	for (int w = 0; w < width; w++) {
		for (int h = 0; h < height; h++) {
			tensor[h * width + w] = 1 - (float)data[h * width * 3 + w * 3 + 0] / 255.0f;
		}
	}
}

void TrainPic() {
	Randomized_3Layers_CNN_Weight(45);
	/*RandomizeVector(&cnnDataTensor[0][0][0][0], BatchSize * Channels * Width * Width);*/

	// 加载图像
	int width, height, channels;
	for (int n = 0; n < BatchSize; n++) {
		char a[] = "num_pic//pic_num_";
		char b[] = ".bmp";
		char* c = (char*)calloc(sizeof(a) + sizeof(b) + 1 + 2 - 2, sizeof(char));
		if (c == NULL) { printf("calloc failed\n"); return; }
		c[sizeof(a) + sizeof(b) + 1 + 2 - 1 - 2] = '\0';
		for (int i = 0; i < sizeof(a) - 1; ++i) c[i] = a[i];
		for (int i = 0; i < sizeof(b) - 1; ++i) c[sizeof(a) - 1 + i + 2] = b[i];
		c[sizeof(a) - 1] = '0' + n / oneNum;
		c[sizeof(a) - 1 + 1] = 'a' + n % oneNum;
		unsigned char* data = stbi_load(c, &width, &height, &channels, 0);
		dataNorm(data, &cnnDataTensor[n][0][0][0], width, height);
		labelTensor[n][n / oneNum] = 1.0f;
		printf("Loaded Pic Path \"%s\"  Label ", c);
		PrintTensor1D(labelTensor[n], CLASS_NUM);
	}
	printf("testing before traing\n");
	Forward_3Layers_CNN(&cnnDataTensor[0][0][0][0], cnnOutputTensor[0]);
	PrintTensor1D(cnnOutputTensor[0], 3);
	// Print_3Layers_CNN_Weight();
	const float initialLearningRate = 0.05, learningRateDecayTo = 0.2;
	const int initialEpoch = 401;
	for (int epoch = 0; epoch < initialEpoch; epoch++) {
		float learningRate = LearningRateDecay(epoch, initialLearningRate, initialEpoch, learningRateDecayTo);
		for (int batch = 0; batch < BatchSize; batch++) {
			Forward_3Layers_CNN(&cnnDataTensor[batch][0][0][0], cnnOutputTensor[batch]);
			UpdateWeights_3Layers_CNN(&cnnOutputTensor[batch][0], labelTensor[batch], learningRate);
		}
		if (epoch % 20 == 0) {
			float batchLoss = CrossEntropyBatchLoss(&cnnOutputTensor[0][0], &labelTensor[0][0], BatchSize, CLASS_NUM);
			printf("Epoch %d/%d  BatchLoss = %f  lr = %f\n", epoch, initialEpoch, batchLoss, learningRate);
		}
	}
	Print_3Layers_CNN_Weight();
	printf("testing \n\n");
	static float testTensor[1][24][24]{};
	char testPath[] = "num_pic//test_image.bmp";
	unsigned char* data = stbi_load(testPath, &width, &height, &channels, 0);
	dataNorm(data, &testTensor[0][0][0], width, height);
	static float testOutputTensor[CLASS_NUM]{};
	Forward_3Layers_CNN(&testTensor[0][0][0], testOutputTensor);
	PrintTensor1D(testOutputTensor, CLASS_NUM);
	
	return;
}