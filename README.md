<!---
# Learning A Spiking Neural Network for Efficient Image Deraining
[![GoogleDrive](https://img.shields.io/badge/Data-GoogleDrive-brightgreen)](https://drive.google.com/drive/folders/1KRR_L276nviPT9JFPL9zfBiZVKJO6dM1?usp=drive_link)
[![BaiduPan](https://img.shields.io/badge/Data-BaiduPan-brightgreen)](https://pan.baidu.com/s/1TlgoslD-hIzySDL8l6gekw?pwd=pu2t)
--->
<div align="center">

# 【UNDER REVIEW CIKM'2025🔥】SpikeRain: Towards Energy-Efficient Single Image Deraining with Spiking Neural Networks
</div>

> SpikeRain: Towards Energy-Efficient Single Image Deraining with Spiking Neural Networks
> 
>  
>
> 
>
> Primary contact: 

> **Abstract:** 
With the growing deployment of vision systems on edge devices, there is a pressing need for energy-efficient, temporally-aware image deraining models. To address this need, we propose SpikeRain, a spiking neural network (SNN) architecture that achieves competitive deraining performance with significantly lower computational overhead than conventional artificial neural network (ANN) based methods. Unlike such traditional ANN-based models that rely on dense activations and high memory requirements, SpikeRain exploits the event-driven and sparse firing nature of spiking neurons to perform efficient temporal integration and contextual learning across multiple time steps. SpikeRain hinges on an encoder-decoder architecture enhanced with spiking-native modules. A Dense Spiking Residual Block (DSRB) facilitates efficient temporal integration and deep feature reuse, improving detail preservation. On the other hand, a Multi-Dimensional Spiking Attention (MDSA) module captures intricate dependencies across temporal, channel, and spatial dimensions, effectively suppressing rain streaks. Finally, an Adaptive Residual Feature Enhancement (ARFE) module applies gated attention to selectively refine salient features, boosting the perceptual quality of the restored output. Extensive experiments on synthetic and real-world deraining benchmarks show that SpikeRain consistently achieves very competitive PSNR and SSIM scores compared to existing deraining methods, while requiring significantly fewer parameters, FLOPs, and energy. SpikeRain reduces the number of parameters by approximately 40\% and FLOPs by 88.98\% while maintaining a comparable energy level to the existing SNN-based deraining method, owing to its spike-driven architectural efficiency. Overall, results expose the strong potential of SNNs for energy-efficient single image deraining, making SpikeRain a compelling solution for real-time image restoration on neuromorphic devices and low-power processing platforms.


<!---
## News

- **July 4, 2023:** Paper submitted. 
- **Sep 13, 2023:** The basic version is released, including codes, pre-trained models on the Sate 1k dataset, and the used dataset.
- **Sep 14, 2023:** RICE dataset updated.
  ** Sep 15, 2023:** The [visual results on Sate 1K](https://pan.baidu.com/s/1dToHnHI9GVaHQ3-I6OIbpA?pwd=rs1k) and [real-world dataset RSSD300](https://pan.baidu.com/s/1OZUWj8eo6EmP5Rh8DE1mrA?pwd=8ad5) are updated.-->


## Preparation

## Datasets
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain12</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>Rain1200</th>
    <th>RW_Rain</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="https://pan.baidu.com/s/1mrXshB3Y0qO205aRR_lZaw?pwd=9ojl 提取码：9ojl">Download</a> </td>
    <td align="center"> <a href="https://pan.baidu.com/s/1wzvNW7UgLsSZbLd_lFfpyA?pwd=s6v0 提取码：s6v0 ">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1pPF4CL7rvKRVvkVkJ0C8RA?pwd=k2iv 提取码：k2iv">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1RlzbJa8XtCBUHXrJfxqiIQ?pwd=ajnz 提取码：ajnz">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1GM85dkgxf8CyuwlhOV4e8Q?pwd=23vf 提取码：23vf">Download</a> </td>
  </tr>
</tbody>
</table>


### Install

We test the code on PyTorch 1.9.1 + CUDA 11.1 + cuDNN 8.0.5.

1. Create a new conda environment
```
conda create -n ESDNet python=3.8
conda activate ESDNet 
```

2. Install dependencies
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install matplotlib scikit-image opencv-python numpy einops math natsort tqdm lpips time tensorboardX
```
### 🛠️ Training, Testing and Evaluation

### Train
You should change the path to yours in the `Train.py` file.  Then run the following script to test the trained model:


```sh
python train.py
```

### Test
You should change the path to yours in the `Test.py` file.  Then run the following script to test the trained model:

```sh
python test.py
```


### Evaluation
You should change the path to yours in the `dataset_load.py` file.  Then run the following script to test the trained model:

```sh
python evaluation.py
```
It is recommended that you can download the visual deraining results and retest the quantitative results on your own device and environment.



## 👍 Acknowledgment

This code is based on the [Restormer](https://github.com/swz30/Restormer), [spikingjelly](https://github.com/fangwei123456/spikingjelly) and [C2PNet](https://github.com/YuZheng9/C2PNet). Thanks for their awesome work.
