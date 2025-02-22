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

## Sigmoid 激活函数实现
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
