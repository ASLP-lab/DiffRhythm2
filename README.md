<p align="center">
    <img src="src/diffrhythm2_logo.png" width="400"/>
<p>

<p align="center">
   <h1>Di♪♪Rhythm 2: Efficient And High Fidelity Song Generation Via Block Flow Matching</h1>
</p>

<div style='display:flex; gap: 0.25rem; '>
  <a href='https://arxiv.org/pdf/2510.22950'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
  <a href='https://aslp-lab.github.io/DiffRhythm2.github.io'><img src='https://img.shields.io/badge/Project-Demo-green'></a>
  <a href='https://huggingface.co/ASLP-lab/DiffRhythm2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow'></a>
  <a href='https://github.com/xiaomi-research/diffrhythm2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow'></a>
  <a href="https://github.com/xiaomi-research/diffrhythm2/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue?&color=blue"/></a>
</div>

Yuepeng Jiang, Huakang Chen, Ziqian Ning, Jixun Yao, Zerui Han, Di Wu, Meng Meng, Jian Luan, Zhonghua Fu, Lei Xie†

<!-- <p align="center">
  <b>DiffRhythm 2</b> &nbsp;&nbsp;|&nbsp;&nbsp;
  📑 <a href="https://arxiv.org/abs/2503.01183">Paper</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  🎵 <a href="https://aslp-lab.github.io/DiffRhythm.github.io/">Demo</a>
</p> -->

DiffRhythm 2 (Chinese: 谛韵, Dì Yùn) is the next-generation open-sourced music generation framework that advances the original DiffRhythm with a semi-autoregressive diffusion architecture. It is capable of generating full-length songs with precise lyric alignment and coherent musical structures. The name inherits the essence of DiffRhythm — “Diff” reflects its diffusion-based generative backbone, while “Rhythm” emphasizes its dedication to musicality and temporal flow. The Chinese name 谛韵 (Dì Yùn) continues this dual symbolism: “谛” (attentive listening) represents perceptual awareness, and “韵” (melodic charm) captures the expressive beauty of music.

<p align="center">
    <img src="src/model2.png" width="80%"/>
<p>

<p align="center">
    <img src="src/model1.png" width="80%"/>
<p>

## 📢 News and Updates

* **2025.10.30** 🚀 We released the [DiffRhythm2 paper](https://arxiv.org/pdf/2510.22950), demo code and model weights.

## 📋 TODOs
- [ ] Support Colab.
- [ ] Gradio support.
- [ ] Song extension.
- [ ] Instrumental music generation.
- [x] Release code and weights.
- [x] Release paper to Arxiv.

## 🔨 Inference

Following the steps below to clone the repository and install the environment.

```bash 
# clone and enter the repositry
git clone https://github.com/xiaomi-research/diffrhythm2.git
cd diffrhythm2

# install the environment
## espeak-ng
# For Debian-like distribution (e.g. Ubuntu, Mint, etc.)
sudo apt-get install espeak-ng
# For RedHat-like distribution (e.g. CentOS, Fedora, etc.) 
sudo yum install espeak-ng
# For MacOS
brew install espeak-ng
# For Windows
# Please visit https://github.com/espeak-ng/espeak-ng/releases to download .msi installer

## install requirements
pip install -r requirements.txt
```

On Linux you can now simply use the inference script:
```bash
# For inference using a reference WAV file
bash inference.sh
```

Weights will be automatically downloaded from Hugging Face upon the first run.

Example files of lyrics and reference audio can be found in `example`. 


## 📜 License & Disclaimer

DiffRhythm 2 (code and weights) is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). This open-source license allows you to freely use, modify, and distribute the model, as long as you include the appropriate copyright notice and disclaimer.

We do not make any profit from this model. Our goal is to provide a high-quality base model for music generation, fostering innovation in AI music and contributing to the advancement of human creativity. We hope that DiffRhythm 2 will serve as a foundation for further research and development in the field of AI-generated music.

DiffRhythm 2 enables the creation of original music across diverse genres, supporting applications in artistic creation, education, and entertainment. While designed for positive use cases, potential risks include unintentional copyright infringement through stylistic similarities, inappropriate blending of cultural musical elements, and misuse for generating harmful content. To ensure responsible deployment, users must implement verification mechanisms to confirm musical originality, disclose AI involvement in generated works, and obtain permissions when adapting protected styles.
