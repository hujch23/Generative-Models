# Generative-Models

## 📚 目录

- [手撕代码](#手撕代码)  
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
f(x) = 1 / (1 + e^(-x))

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
![image](https://github.com/user-attachments/assets/f3486876-2c98-42ea-b977-63b0011a291b)
import numpy as np
import math

```python 
def softmax(scores: list[float]) -> list[float]:
    exp_scores = [math.exp(score) for score in scores]
    sum_exp_scores = sum(exp_scores)

    probabilities = [round(score / sum_exp_scores, 4) for score in exp_scores]
    
    return probabilities

if __name__ == "__main__":
    scores = np.array(eval(input()))
    print(softmax(scores))
```
