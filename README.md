# Generative-Models

## 📚 目录

- [深度学习必“懂”知识](#深度学习必“懂”知识)
- [深度学习三种架构](#深度学习三种架构)
  - [CNN](#CNN)  
  - [RNN](#RNN)
  - [Transformer](#Transformer)
- [生成模型技术路线](#生成模型技术路线)
  - [VAE](#VAE)
  - [GAN](#GAN)
  - [Difussion](#Difussion)
  - [Autoregressive](#Autoregressive)
- [Chatgpt](#Chatgpt)
- [Bert](#Bert)
- [VLA](#VLA)
- [Deepseek](#Deepseek)

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


## Transformer 面试
- [](https://zhuanlan.zhihu.com/p/438625445)  
- [](https://zhuanlan.zhihu.com/p/363466672)  
- [](https://zhuanlan.zhihu.com/p/148656446) 





