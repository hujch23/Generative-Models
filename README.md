# Generative-Models

## 📚 目录

- [深度学习基础](#深度学习基础)  
- [CNN](#CNN)  
- [RNN](#RNN)
- [Transformer](#Transformer)
- [VAE](#VAE)
- [Difussion](#Difussion)
- [Autoregressive](#Autoregressive)
- [Chatgpt](#Chatgpt)
- [Bert](#Bert)
- [VLA](#VLA)
- [Deepseek](#Deepseek)

## 1.1 Sigmoid 激活函数实现
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

## 1.2 Softmax 激活函数实现
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

## 1.3 （具有反向传播）单神经元
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

## 1.4 Log Softmax函数的实现

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

## 1.5 熵、KL散度、交叉熵
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







