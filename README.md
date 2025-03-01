# Generative-Models

## 📚 目录

- [深度学习基础知识](#深度学习必“懂”知识)
- [深度学习三种架构](#深度学习三种架构)
  - [CNN](#CNN)  
  - [RNN](#RNN)
  - [Transformer](#Transformer)
- [生成模型技术路线](#生成模型技术路线)
  - [VAE](#VAE)
  - [GAN](#GAN)
  - [Difussion](#Difussion)
  - [Autoregressive](#Autoregressive)
- [Sora技术原理](#Sora技术原理)


### 1.1 Sigmoid 激活函数实现
![image](https://github.com/user-attachments/assets/02cf363d-cd7a-4203-8dbb-3833bcc96ff8)


```python 
import math

def sigmoid(z: float) -> float:
    result= 1 / (1 + math.exp(-z))
    return round(result, 4)

if __name__ == "__main__":
    z = float(input())
    print(f"{sigmoid(z):.4f}")
```

### 1.2 Softmax 激活函数实现
![image](https://github.com/user-attachments/assets/e8799da9-e3db-4f16-a563-ae2a86a4b3c8)

```python 
import numpy as np
import math

def softmax(scores: list[float]) -> list[float]:
    exp_scores = [math.exp(score) for score in scores]
    sum_exp_scores = sum(exp_scores)

    probabilities = [round(score / sum_exp_scores, 4) for score in exp_scores]
    
    return probabilities

if __name__ == "__main__":
    scores = np.array(eval(input()))
    print(softmax(scores))
```

### 1.3 （具有反向传播）单神经元
使用基础Python语法,没有依赖NumPy等库，通过循环逐个处理样本

```python 
import math
import numpy as np
def single_neuron_model(features, labels, weights, bias):
    
    probabilities = []

    for cur_feature in features:

        z = sum(weight * feature for weight, feature in zip(weights, cur_feature)) + bias
        prob = 1 / (1 + math.exp(-z))
        probabilities.append(round(prob, 4))

    mse = round(sum((prob - label)**2 for prob, label in zip(probabilities, labels)) / len(labels), 4)
    
    return probabilities, mse

def single_neuron_model(features, labels, weights, bias):
    
    features = np.array(features)
    labels = np.array(labels)
    weights = np.array(weights)

    z = np.dot(features, weights) + bias
    prob = 1 / (1 + np.exp(-z))
    probabilities = np.round(prob, 4).tolist()
    mse = np.round(np.mean((prob - labels)**2), 4)


    
    return probabilities, mse


if __name__ == "__main__":
    features = np.array(eval(input()))
    labels = np.array(eval(input()))
    weights = np.array(eval(input()))
    bias = float(input())
    print(single_neuron_model(features, labels, weights, bias))
```

如果涉及反向传播：
![image](https://github.com/user-attachments/assets/b5d437c7-ea03-46b6-bf9d-4fb56d6e63db)

```python

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    features = np.array(features)
    labels = np.array(labels)
    weight = np.array(initial_weights)
    bias = initial_bias
    mse_value = []
    for _ in range(epochs):
        z = np.dot(features, weight) + bias
        prob = sigmoid(z)

        mse = np.mean((prob - labels)**2)
        mse_value.append(round(mse, 4))

        weight_gradient = (2 / len(labels)) * np.dot(features.T, (prob - labels) * (prob * (1 - prob))) 
        bias_gradient = (2 / len(labels)) * np.sum((prob - labels)*(prob * (1 - prob)))

        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient
        updated_weight = np.round(weight, 4)
        updated_bias = round(bias, 4)

    return updated_weight.tolist(), updated_bias, mse_value

if __name__ == "__main__":
    features = np.array(eval(input()))
    labels = np.array(eval(input()))
    initial_weights = np.array(eval(input()))
    initial_bias = float(input())
    learning_rate = float(input())
    epochs = int(input())
    print(train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs))


```

### 1.4 Log Softmax函数的实现

```python
import numpy as np


def log_softmax(scores: list) -> np.ndarray:
    scores = scores - np.max(scores)
    prob = scores - np.log(np.sum(np.exp(scores)))

    return prob

if __name__ == "__main__":
    scores = eval(input())
    print(log_softmax(scores))
```

### 1.5 熵、KL散度、交叉熵
![image](https://github.com/user-attachments/assets/959bf521-4046-49c5-a024-99795f02a0b4)
![image](https://github.com/user-attachments/assets/50982527-7e33-4b32-ac8f-b1af5feee25e)
![image](https://github.com/user-attachments/assets/407066ad-2417-4b42-9bd2-fab20df13679)

![image](https://github.com/user-attachments/assets/47c379dc-f7bf-44de-9fb2-7c222a95cded)

```python
import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    kl_1 = np.log(sigma_q / sigma_p)
    kl_2 = (sigma_p**2 + (mu_p - mu_q)**2)/ (2*sigma_q**2)

    kl_div = kl_1 + kl_2 - 0.5
    return kl_div


if __name__ == "__main__":
    mu_p, sigma_p, mu_q, sigma_q = map(float, input().split())
    print(kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q))
```

### 1.6 优化算法

梯度下降是一种优化算法，用于通过迭代更新参数来最小化目标函数（如损失函数），解决模型训练中的参数优化问题。基本方法是根据目标函数的梯度方向调整参数，常见变种包括：标准梯度下降（逐步沿负梯度方向更新）、随机梯度下降（SGD）（每次使用一个样本计算梯度，提升效率）、动量法（引入历史梯度累积，减少震荡）、NAG（Nesterov加速梯度）（在预估位置计算梯度，提前感知变化）、自适应方法（如 AdaGrad、RMSProp、Adam，通过动态调整学习率加速收敛）。这些方法旨在提高收敛速度、稳定性和优化效果。
![image](https://github.com/user-attachments/assets/d397c91a-8037-48b2-a4cd-5746ad8cecaa)
![image](https://github.com/user-attachments/assets/927b7fc2-6a24-43bd-a62a-5ea02729dfb7)
![image](https://github.com/user-attachments/assets/5fdff66e-8fee-47ec-b985-4c767ac883f8)
![image](https://github.com/user-attachments/assets/dadc7dd9-1d0e-4911-bd19-c9c707aa212e)
![image](https://github.com/user-attachments/assets/2bb7a265-3aa7-4d9b-99c4-99f364d8028d)
![image](https://github.com/user-attachments/assets/bb1e90d6-e3d5-4d18-99bc-1cbfdd48d89f)
![image](https://github.com/user-attachments/assets/3fa80ab1-596a-4987-b5a7-afcb8e291430)
![image](https://github.com/user-attachments/assets/5841d2d8-4306-43a4-b129-fc72f92edf57)
![image](https://github.com/user-attachments/assets/46ca7274-a82c-4671-ba39-58d3690df9f5)


```python

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, num_iterations + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return x

```

### 1.7 什么是 Dropout？它是如何工作的？Dropout 的保留概率如何选择？是否可以动态调整？
- Dropout 是一种正则化技术，用于防止神经网络的过拟合。它通过在训练过程中随机“丢弃”一部分神经元（即将其输出置为零），从而减少神经元之间的相互依赖，提高模型的泛化能力。
- 通常选择 0.5，但可以根据任务调整。动态 Dropout（如 DropConnect）可以在训练过程中调整保留概率。

### 1.8 什么是 L1 和 L2 正则化？它们的区别是什么？
L1 和 L2 正则化是两种常用的正则化方法，用于防止模型过拟合。它们通过在损失函数中添加权重的惩罚项，限制模型的复杂度。
- L1 正则化：在损失函数中添加权重的绝对值之和，促使权重稀疏化（即部分权重变为零），适合特征选择，适用于高维稀疏数据
- L2 正则化：在损失函数中添加权重的平方和，促使权重趋向于较小的值，但不会完全为零

## CNN手撕及面试
### 1. 什么是卷积神经网络（CNN）？它与普通神经网络有何不同？

与普通神经网络（如全连接网络）不同，CNN利用了图像的局部相关性和参数共享，减少了参数数量，提高了训练效率。

### 2. 什么是卷积操作？为什么需要卷积？
卷积操作是通过一个小的滤波器（kernel）在输入数据上滑动，计算点积以提取局部特征。卷积的作用是提取局部模式（如边缘、纹理等），并通过层层叠加提取更高层次的特征。

### 3. 什么是池化（Pooling）？有哪些常见的池化方法？

池化是一种下采样操作，用于减少特征图的尺寸，降低计算复杂度，同时保留重要特征。
常见的池化方法：最大池化（Max Pooling）：取池化窗口中的最大值。平均池化（Average Pooling）：取池化窗口中的平均值

### 4. 什么是填充（Padding）？为什么需要填充？
填充是在输入数据的边缘添加额外的像素（通常为0），以控制输出特征图的大小。
作用：保持输出特征图的尺寸（"same" padding）。提高边缘区域的特征提取能力


###  5. 什么是参数共享？为什么它对CNN很重要？

参数共享的核心思想是利用图像的平移不变性。同一个卷积核在整个输入特征图上滑动时使用相同的参数。

### 6. CNN的感受野（Receptive Field）是什么？为什么重要？

感受野是指卷积神经网络中某一层的一个神经元在输入图像上对应的区域大小。感受野越大，神经元能够捕获的上下文信息越多。
深层网络通过叠加卷积层可以扩大感受野，从而提取全局特征

### 7.什么是转置卷积（Transposed Convolution）？它的作用是什么？
转置卷积是一种上采样操作，用于将低分辨率特征图恢复到高分辨率。
作用：常用于生成对抗网络（GAN）和图像分割任务中，用于生成高分辨率图像或恢复原始尺寸。
转置卷积并不是简单的反向卷积，而是通过插值和卷积操作实现上采样。


### 8. CNN中常见的网络架构有哪些？
- AlexNet：引入ReLU、Dropout和数据增强，赢得ImageNet比赛。
- VGG：使用小卷积核（3x3）堆叠，结构简单但参数量大。
- GoogLeNet（Inception）：引入Inception模块，减少计算量。在同一层中并行使用不同大小的卷积核（如1×1、3×3、5×5）和池化操作。通过多尺度特征提取，捕获不同大小的特征。
- ResNet：引入残差连接（skip connection），解决深层网络的梯度消失问题。残差的核心思想是：让网络学习输入与输出之间的差值（残差），而不是直接学习输出本身。残差连接使得网络更容易学习到恒等映射或接近恒等映射的函数。
即使增加了网络深度，新的层可以选择学习“零映射”（即不改变输入），从而避免退化问题。
- 现代架构如EfficientNet（同时调整网络的深度、宽度和分辨率，找到最优的缩放比例）、MobileNet（将标准卷积分解为深度卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution））等，注重参数效率和计算效率

### 9. CNN的常见问题及解决方法
问题1：过拟合：数据增强（如翻转、旋转、裁剪）。正则化（如L2正则化、Dropout）。
问题2：梯度消失或梯度爆炸：使用ReLU激活函数。使用残差网络（ResNet）或批归一化（Batch Normalization）。

### 10. 手撕二维卷积
```python
import numpy as np  

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):  
    """  
    实现一个简单的2D卷积操作。  

    参数：  
    - input_matrix: 输入矩阵（二维数组），表示输入的图像或特征图。  
    - kernel: 卷积核（二维数组），用于提取特征。  
    - padding: 填充大小，在输入矩阵的边缘添加的零的数量。  
    - stride: 步幅，卷积核在输入矩阵上滑动的步长。  

    返回：  
    - output_matrix: 输出矩阵（二维数组），表示卷积操作后的特征图。  
    """  
    # 获取输入矩阵的高度和宽度  
    input_height, input_width = input_matrix.shape  

    # 获取卷积核的高度和宽度  
    kernel_height, kernel_width = kernel.shape  

    # 对输入矩阵进行填充，填充模式为常数（值为0）  
    # 填充的大小为 (padding, padding) 在高度和宽度方向上分别添加  
    padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')  

    # 获取填充后的输入矩阵的高度和宽度  
    input_height_padded, input_width_padded = padded_input.shape  

    # 计算输出矩阵的高度和宽度  
    # 输出尺寸公式：((输入尺寸 + 2*填充 - 卷积核尺寸) // 步幅) + 1  
    output_height = (input_height_padded - kernel_height) // stride + 1  
    output_width = (input_width_padded - kernel_width) // stride + 1  

    # 初始化输出矩阵，大小为 (output_height, output_width)，初始值为0  
    output_matrix = np.zeros((output_height, output_width))  

    # 遍历输出矩阵的每个位置  
    for i in range(output_height):  # 遍历输出矩阵的行  
        for j in range(output_width):  # 遍历输出矩阵的列  
            # 提取输入矩阵中与当前卷积核位置对应的区域  
            # 区域的起始位置由 (i*stride, j*stride) 决定  
            # 区域的大小与卷积核相同  
            region = padded_input[i*stride:i*stride + kernel_height, j*stride:j*stride + kernel_width]  
            
            # 计算区域与卷积核的逐元素乘积的和，并赋值给输出矩阵的当前位置  
            output_matrix[i, j] = np.sum(region * kernel)  

    # 返回卷积操作后的输出矩阵  
    return output_matrix
```

## RNN手撕及面试
![image](https://github.com/user-attachments/assets/83395ace-d874-447b-8cb1-2e7f0494f93f)
![image](https://github.com/user-attachments/assets/9f687b18-034f-4b78-9112-c618e62888d5)
![image](https://github.com/user-attachments/assets/b3ae7939-2bcb-423e-9c00-7e78cdb1d987)
![image](https://github.com/user-attachments/assets/0a891299-0747-48cb-892a-2c784141627c)
![image](https://github.com/user-attachments/assets/5e636d51-d093-4360-9f0b-22d656cf5ca6)
![image](https://github.com/user-attachments/assets/ea2cc8b0-c47b-4316-a155-dc92cc3a6e8d)
```python
def rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b):
    h = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)
    for x in input_sequence:
        x = np.array(x)
        h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + b)
    final_hidden_state = np.round(h, 4)
    return final_hidden_state.tolist()
```
```python
class LSTM:  
    def __init__(self, input_size, hidden_size):  
        """  
        初始化 LSTM 的权重和偏置。  

        参数：  
        - input_size: 输入特征的维度。  
        - hidden_size: 隐藏状态的维度（即 LSTM 单元的数量）。  
        """  
        self.input_size = input_size  
        self.hidden_size = hidden_size  

        # 初始化遗忘门 (Forget Gate) 的权重和偏置  
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)  # 遗忘门的权重矩阵  
        self.bf = np.zeros((hidden_size, 1))  # 遗忘门的偏置  

        # 初始化输入门 (Input Gate) 的权重和偏置  
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)  # 输入门的权重矩阵  
        self.bi = np.zeros((hidden_size, 1))  # 输入门的偏置  

        # 初始化候选记忆单元 (Candidate Cell State) 的权重和偏置  
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)  # 候选记忆单元的权重矩阵  
        self.bc = np.zeros((hidden_size, 1))  # 候选记忆单元的偏置  

        # 初始化输出门 (Output Gate) 的权重和偏置  
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)  # 输出门的权重矩阵  
        self.bo = np.zeros((hidden_size, 1))  # 输出门的偏置  

    def forward(self, x, initial_hidden_state, initial_cell_state):  
        """  
        前向传播，计算 LSTM 的输出。  

        参数：  
        - x: 输入序列，形状为 (序列长度, 输入特征维度)。  
        - initial_hidden_state: 初始隐藏状态，形状为 (hidden_size, 1)。  
        - initial_cell_state: 初始记忆单元状态，形状为 (hidden_size, 1)。  

        返回：  
        - outputs: 所有时间步的隐藏状态，形状为 (序列长度, hidden_size, 1)。  
        - h: 最后一个时间步的隐藏状态。  
        - c: 最后一个时间步的记忆单元状态。  
        """  
        h = initial_hidden_state  # 当前时间步的隐藏状态  
        c = initial_cell_state  # 当前时间步的记忆单元状态  
        outputs = []  # 用于存储每个时间步的隐藏状态  

        # 遍历输入序列的每个时间步  
        for t in range(len(x)):  
            xt = x[t].reshape(-1, 1)  # 当前时间步的输入，形状为 (input_size, 1)  
            concat = np.vstack((h, xt))  # 将隐藏状态和输入拼接，形状为 (hidden_size + input_size, 1)  

            # 遗忘门 (Forget Gate)  
            # 计算遗忘门的激活值，决定记忆单元中哪些信息需要遗忘  
            ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)  

            # 输入门 (Input Gate)  
            # 计算输入门的激活值，决定当前输入中哪些信息需要写入记忆单元  
            it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)  
            # 计算候选记忆单元的值，表示当前时间步的新信息  
            c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)  

            # 更新记忆单元状态 (Cell State)  
            # 遗忘门控制遗忘旧信息，输入门控制写入新信息  
            c = ft * c + it * c_tilde  

            # 输出门 (Output Gate)  
            # 计算输出门的激活值，决定记忆单元中哪些信息需要输出  
            ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)  

            # 更新隐藏状态 (Hidden State)  
            # 隐藏状态由输出门控制，并结合当前记忆单元的值  
            h = ot * np.tanh(c)  

            # 将当前时间步的隐藏状态存储到输出列表中  
            outputs.append(h)  

        # 返回所有时间步的隐藏状态，以及最后一个时间步的隐藏状态和记忆单元状态  
        return np.array(outputs), h, c  

    def sigmoid(self, x):  
        """  
        Sigmoid 激活函数。  

        参数：  
        - x: 输入值。  

        返回：  
        - Sigmoid 函数的输出值。  
        """  
        return 1 / (1 + np.exp(-x))  
```


## Transformer手撕及面试
- [Transformer1](https://zhuanlan.zhihu.com/p/438625445)  
- [Transformer2](https://zhuanlan.zhihu.com/p/363466672)  
- [Transformer3](https://zhuanlan.zhihu.com/p/148656446)

```python
import torch
import math
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        :param d_model: 词嵌入的维度
        :param vocab: 词表的大小
        """
        super(Embeddings, self).__init__()

        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.lut(x)

        return x * math.sqrt(self.d_model)
```

```python
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)

        # 创建位置矩阵
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]

        # 创建除数项
        div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000) / d_model)

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加批次维度并注册为buffer
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return x
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head


        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):

        batch_size, seq_length = query.size(0), query.size(1)

        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)

        Q = Q.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.n_head, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.bool()
            scores = scores.masked_fill(~mask, float('-inf'))

        attention_weight = F.softmax(scores, dim = -1)

        att_out = torch.matmul(attention_weight, V)
        att_out = att_out.transpose(1, 2).contiguous()
        att_out = att_out.view(batch_size, seq_length, self.d_model)

        output = self.out_linear(att_out)

        return output
if __name__ == "__main__":
    d_model = 512
    n_head = 8
    batch_size = 16
    seq_length = 32
    attention = MultiHeadAttention(d_model, n_head)
    query = torch.rand(batch_size, seq_length, d_model)
    key = torch.rand(batch_size, seq_length, d_model)
    value = torch.rand(batch_size, seq_length, d_model)
    output = attention(query, key, value)
    print("输入形状:", query.shape)
    print("输出形状:", output.shape)
```

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.d_model, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.d_model)
        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)


        return x

```

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        var = x.var(-1, unbiased = False, keepdim = True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
```

## VAE变分自编码器推导

- AE：Autoencoder是一种自监督的神经网络，用于学习数据的高效表示。其主要目标是通过压缩数据并尝试重构它来捕捉数据的关键特征由于Autoencoder（自编码器）学习的是恒等函数，当网络的参数数量超过数据本身的复杂度时，存在过拟合的风险。
- VAE：在VAE中，输入数据被映射到一个潜在的隐向量的分布上，这个分布通常假设为正态分布，其参数由输入数据决定。因此，VAE的关键在于学习输入数据的概率分布特性，而不仅仅是确定性的映射关系。（因为采样的参数和网络参数绑定在一起，因此需要重参数化技巧，通俗点就是神经网络输出近似后验分布的均值和方差）
- VQ-VAE（Vector Quantized Variational Autoencoder）是一种基于离散潜在空间的生成模型，它结合了变分自动编码器（VAE）和矢量量化（Vector Quantization）的思想。VQ-VAE 的核心目标是用离散的潜在表示来对数据进行建模，同时避免传统 VAE 在潜在空间连续分布上的模糊性问题。停止梯度在 VQ-VAE 中的意义：1. 解决离散操作不可导的问题：离散操作（如矢量量化）通过停止梯度让梯度流仅作用于连续变量，从而实现模型的端到端优化。2. 分离编码器和代码表的优化：停止梯度让编码器的输出 $z_e(x)$ 和代码表的嵌入 $e_k$ 优化过程解耦，使两者能够独立训练，最终收敛到合理的表示。
![image](https://github.com/user-attachments/assets/4ba95eea-c9ab-4a89-af81-7793c2d8bca2)
![image](https://github.com/user-attachments/assets/e508c84a-27a4-4554-a365-b98fe78d864d)
![image](https://github.com/user-attachments/assets/7d2cafd5-2876-4597-a300-f4dccd5deec5)
![image](https://github.com/user-attachments/assets/916f93e3-44a0-4ca3-a019-43395099035f)
![image](https://github.com/user-attachments/assets/9c022246-b226-44d0-ac83-b662b0990dd6)

## GAN生成对抗网络推导
![image](https://github.com/user-attachments/assets/8d384490-5f9c-4ad5-b4c8-f2b5171da1a8)
![image](https://github.com/user-attachments/assets/5ec9bee4-2e65-404e-81d5-1b2f477c14dc)
![image](https://github.com/user-attachments/assets/dc3e5455-22ab-4273-96a2-a2a2b2988f10)
![image](https://github.com/user-attachments/assets/3fe144f3-5199-43aa-9bd9-6216687072bf)

## Diffusion Model 推导

见论文：[Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)

## Diffusion Policy 推导


