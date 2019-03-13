# Bit-Flips Attack:

This repository constains a Pytorch implementation of the paper "[Bit-Flip Attack: Crushing Neural Network with Progressive Bit Search]()"

If you find this project useful to you, please cite [our work]():

```
sad
```


## Usage

```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

## Weight quantization
For the case that directly quantize the deep neural network without retraining it, we add the function ```--optimize_step``` to optimize the step-size of quantizer to minimize the loss (e.g., mean-square-error loss) between quantized weight and its full precision base. It is intriguing to find out that:

- directly apply the uniform quantizer can achieve higher accuracy (close to the full precision baseline) without optimize the quantizer, for high-bit quantization (e.g., 8-bit). 

- On the contrary, for the low-bit quantization (e.g., 4-bit), directly quantize the weight causes significant accuracy loss. With the ```--optimize_step``` enabled, accuracy can partially recover without retraining.   