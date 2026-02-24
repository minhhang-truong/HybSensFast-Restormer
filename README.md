# GuidedHybSensUIR (Enhanced with Restormer & Prior Fusion)

This repository is an enhanced version of the **GuidedHybSensUIR** model. We have modified the original architecture by integrating Restormer blocks and a multi-scale prior concatenation mechanism within the U-Net stages to refine the restoration process.

---

# 🚀 Key Enhancements
This version introduces specific structural modifications to the original GuidedHybSensUIR:
* **Restormer Integration:** We have integrated Restormer blocks within the U-Net architecture. By leveraging Multi-Dconv Head Transposed Attention (MDTA), the model better captures global context compared to standard convolutions.
* **Prior-Guided Concatenation:** A `concat` strategy is implemented between the physical priors and the feature maps across different layers of the U-Net. This ensures that the prior information directly guides the feature extraction and reconstruction phases.
* **Refined Color Balance:** These modifications lead to a slight improvement in color balance and stability in underwater scenes, helping to mitigate common color casts more effectively than the baseline.

---
# 📜 Original Work
This project is built upon the research:
[Underwater Image Restoration Through a Prior Guided Hybrid Sense Approach and Extensive Benchmark Analysis](https://ieeexplore.ieee.org/document/10824878)
<div>
<span class="author-block">
  Xiaojiao Guo<sup> 👨‍💻‍ </sup>
</span>,
  <span class="author-block">
    <a href='https://cxh.netlify.app/'>Xuhang Chen</a><sup> 👨‍💻‍ </sup>
  </span>,
  <span class="author-block">
    Shuqiang Wang<sup> 📮</sup>
  </span>,
  <span class="author-block">
    <a href='https://cmpun.github.io/'>Chi-Man Pun</a><sup> 📮</sup>
  </span>
  ( 👨‍💻‍ Equal contributions, 📮 Corresponding author)
</div>

<b>University of Macau, SIAT CAS, Huizhou Univeristy, Baoshan Univeristy</b>

In <b>_IEEE Transactions on Circuits and Systems for Video Technology_</b>

# 🔮 Benchmark Dataset

[Kaggle](https://www.kaggle.com/datasets/xuhangc/underwaterbenchmarkdataset)

If you need visualization results for the original work, you may contact Dr.Guo via yc27441@um.edu.mo

# ⚙️ Usage

## Training
You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```
If you have difficulties with the usage of `accelerate`, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

## Inference

Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in `config.yml`.

```bash
python infer.py
```

# Citation

If you find this code or the base model useful, please cite the original paper:
```bib
@ARTICLE{10824878,
  author={Guo, Xiaojiao and Chen, Xuhang and Wang, Shuqiang and Pun, Chi-Man},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Underwater Image Restoration Through a Prior Guided Hybrid Sense Approach and Extensive Benchmark Analysis}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Image restoration;Image color analysis;Benchmark testing;Deep learning;Green products;Distortion;Transformers;Adaptation models;Training;Lighting;Underwater image restoration;image enhancement;prior guided attention;efficient Transformer;multi-scales hybridization},
  doi={10.1109/TCSVT.2025.3525593}
}
```
