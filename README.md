# ScaleViM-PDD: Multi-Scale EfficientViM with Physical Decoupling and Dual-Domain Fusion for Remote Sensing Image Dehazing

abstract:Remote sensing images are often degraded by atmospheric haze, which not only reduces image quality but also complicates information extraction, particularly in high-level visual analysis tasks such as object detection and scene classification. State-space models (SSMs) have recently emerged as a powerful paradigm for vision tasks, showing great promise due to their computational efficiency and robust capacity to model global dependencies. However, most existing learning-based dehazing methods lack physical interpretability, leading to weak generalization. Furthermore, they typically rely on spatial features while neglecting crucial frequency domain information, resulting in incomplete feature representation. To address these challenges, we propose ScaleViM-PDD, a novel network that enhances an SSM backbone with two key innovations: a Multi-scale EfficientViM with Physical Decoupling (ScaleViM-P) module and a Dual-Domain Fusion (DD Fusion) module. The ScaleViM-P module synergistically integrates a Physical Decoupling block within a multi-scale EfficientViM architecture. This design enables the network to mitigate haze interference in a physically grounded manner at each representational scale while simultaneously capturing global contextual information to adaptively handle complex haze distributions. To further address detail loss, the DD Fusion module replaces conventional skip connections by incorporating a novel Frequency Domain Module (FDM) alongside channel and position attention. This allows for a more effective fusion of spatial and frequency features, significantly improving the recovery of fine-grained details, including color and texture information. Extensive experiments on nine publicly available remote sensing datasets demonstrate that ScaleViM-PDD consistently surpasses state-of-the-art baselines in both qualitative and quantitative evaluations, highlighting its strong generalization ability.


## 🧠 Network Architecture

![Network Architecture](image/ScaleVIM-PDD.png)

---

### 1.🚀 Getting Started

We train and test the code on **PyTorch 1.13.0 + CUDA 11.7**. The detailed configuration is mentioned in the paper.

### 2.Create a new conda environment
<pre lang="markdown"> 
conda create -n ScaleViM-PDD python=3.8 
conda activate ScaleViM-PDD  </pre>

###  3.⚠️notice
The current open source code is less readable, but it can be trained and tested. You only need to modify the path. Note: modify the key image size parameters. We are currently accelerating the compilation of a more readable version.

## 4.📦 Available Resources

While the code is being finalized, you can access the following components:

- 🔹 **model weights**  
  [📥 Download](ScaleVIM-PDD
Link: https://pan.baidu.com/s/18dS3bJoZM4-shghvlJrKuQ Extraction code: 1230)


- 🔹 **RSID dataset (used for training and evaluation)**  
  [📥 Download](https://drive.google.com/drive/folders/1abSw9GWyyOJINWCRNHBUoJBBw3FCttaS?usp=drive_link)

## 5.🙏 Acknowledgment

This work is currently under second peer review in the TGRS journal. Our project is based on **[EfficientViM]([https://github.com/nachifur/RDDM](https://github.com/mlvlab/EfficientViM))**, and we are very grateful for this excellent work. Their contributions laid the foundation for advances in image restoration.

---

Stay tuned for the full release, including training/inference code and detailed documentation. If you have any questions, please feel free to contact us at aaron@ahut.edu.cn
