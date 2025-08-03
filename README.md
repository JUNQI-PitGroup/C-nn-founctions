轻量级纯 C 语言深度学习框架
这是我在大学期间使用纯 C 语言编写的轻量级深度学习框架，支持简单的全连接神经网络和卷积神经网络的前向传播、反向传播与训练。
框架不依赖任何第三方库，适合学习深度学习基本原理或在资源受限环境中部署简单模型。

文件结构说明
nn_function .cpp / .h：所有神经网络函数的实现，包括：

激活函数及其导数：ReLU、Tanh、Sigmoid, Softmax ...

线性层（全连接层）：Linear, Linear_AVX2

卷积网络：Conv2d, padding, MaxPooling ...

学习率调度器：LearningRateDecay

损失函数及其导数：MSE（均方误差）, CrossEntropyLoss（交叉熵误差）

权重更新函数（含 AVX2 加速版本）

等等 几十个函数

PointIsInCircle.cpp：简单示例，用于演示如何使用训练好的网络文件（例如 3_layers_fc_nn.cpp）判断一个点是否在圆内。

快速入门
你可以像在 PyTorch 中一样快速搭建网络并训练，只需几步：

1. 准备数据
    float dataVector[10000][5];      // 输入样本（假设一共 10000 个，输入维度为 5）

    float labelVector[10000][1];     // 对应标签（输出维度为 1）

2. 网络前向 + 反向传播 + 权重更新（支持 AVX2 加速）

    float predictedVector[10000][1];
    
    for (epoch = 0; epoch < initialEpoch; epoch++) {
    
        for (n = 0; n < BatchNum; n++) {
        
            Forward_2Layers_NN(DataVector[n], predictedVector[n]); // 前向传播
            
            UpdateWeights_2Layers_NN(DataVector[n], labelVector[n], learningRate); // 反向传播并更新权重
            
            PrintProgressBar("training...", n + 1, batchSize, progressBarLength); // 显示训练进度
            
        }
        
        float batchLoss = MSE_BatchLoss(&predictedVector[0][0], &labelVector[0][0], 1, batchSize);
        
        if (EarlyStop(batchLoss, epoch, patience)) break;
        
        printf("Epoch %d/%d  BatchLoss = %.9f  lr = %f\n", epoch, initialEpoch, batchLoss, learningRate);
        
    }

3. 保存 / 加载模型参数
   
训练前请手动创建目录 2_layers_fc_nn_saved_weight/

    Save_2Layers_NN_Weight();  // 保存权重文件
    
    Load_2Layers_NN_Weight();  // 加载权重文件

训练示例输出

    predicting...  [====================] 5000/5000  for 0.1 s
    
    training...  [====================] 5000/5000  for 0.3 s

如何搭建神经网络（以 2 层网络为例）

    #define inputLength 449
    
    #define layer1_neuronNum 512
    
    #define layer2_neuronNum 1
    
    static float weightTensor_1[layer1_neuronNum][inputLength + 1] = { 0 }; // 第一层权重
    
    static float weightTensor_2[layer2_neuronNum][layer1_neuronNum + 1] = { 0 }; // 第二层权重
    
    // 前向传播
    
    static void Forward(float* inputTensor, float* outputTensor) {
    
        for (int i = 0; i < layer1_neuronNum; i++) {
        
            linear1_outputTensor[i] = Linear_AVX2(inputTensor, inputLength, weightTensor_1[i]);
            
            activate1_outputTensor[i] = ReLU(linear1_outputTensor[i]);
            
        }
        
        for (int i = 0; i < layer2_neuronNum; i++) {
        
            linear2_outputTensor[i] = Linear_AVX2(activate1_outputTensor, layer1_neuronNum, weightTensor_2[i]);
            
            activate2_outputTensor[i] = Tanh(linear2_outputTensor[i]);
            
            outputTensor[i] = activate2_outputTensor[i];
            
        }
        
    }

如何更新权重

    static void UpdateWeights(float* inputTensor, float* labelTensor, float learningRate) {
    
        float layer2_grad[layer2_neuronNum]{};
        
        MSE_LossDerivative(layer2_grad, activate2_outputTensor, labelTensor, layer2_neuronNum);
        
        TanhVectorDerivative(layer2_grad, layer2_neuronNum, linear2_outputTensor);
    
        float layer1_grad[layer1_neuronNum]{};
        
        LinearVectorDerivative(layer2_grad, layer1_grad, layer1_neuronNum, layer2_neuronNum, &weightTensor_2[0][0]);
        
        ReLuVectorDerivative(layer1_grad, layer1_neuronNum, linear1_outputTensor);
    
        UpdateLinearWeight_AVX2(&weightTensor_2[0][0], activate1_outputTensor, layer2_grad, layer2_neuronNum, layer1_neuronNum, learningRate);
        
        UpdateLinearWeight_AVX2(&weightTensor_1[0][0], inputTensor, layer1_grad, layer1_neuronNum, inputLength, learningRate);
        
    }

特性:

🧠 支持全连接层、ReLU/Tanh/Sigmoid 激活函数

🧮 支持 MSE 损失函数及导数计算

⚡ 支持 AVX2 加速（速度可提升 10 倍）

💾 支持权重保存/加载

🛠️ 代码结构清晰、模块化，便于扩展

适用人群

想深入理解神经网络底层实现逻辑的学习者

对 PyTorch/TensorFlow 抽象过高感到困惑的开发者

想在嵌入式设备或无 Python 环境中运行神经网络的工程人员

如需进一步使用说明或集成方法，欢迎参考示例文件 PointIsInCircle.cpp 或查看 nn_function.cpp 中的函数注释。





