# PALM-LAY: A Multi-Script Cross-Regional Dataset for Layout Analysis of Palm Leaf Manuscripts

[![Paper DOI](https://img.shields.io/badge/Paper-ICDAR_DALL_2025-blue)](link-to-paper)
[![Dataset License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## 📖 Overview

**PALM-LAY** is the **first unified, cross-regional annotated dataset** specifically designed for **layout analysis** of historical palm leaf manuscripts.  

Palm leaf manuscripts are among Asia’s oldest written heritage, covering religious scriptures, literary epics, astrology charts, and indigenous medical treatises. Their **script diversity, complex layouts, and physical degradation** pose unique challenges for document image analysis.

PALM-LAY addresses this gap by providing:

- **566 manuscript pages** curated from **six distinct collections**.  
- **6,334 annotated layout regions** across **seven categories**:
  - `MainRegion`
  - `ParagraphRegion`
  - `TextLineRegion`
  - `SymbolicMark`
  - `PhysicalDamage`
  - `Illustration`
  - `Other`  

The dataset enables benchmarking of state-of-the-art object detection models for **cross-script generalization** and **multi-class layout segmentation**.

---

## 🌍 Scripts & Sources

The dataset spans **six palm leaf traditions** from South and Southeast Asia:

- **Tamil** (India) – Naladiyar, Tholkapppiyam, Thirikadugam  
- **Jathakam** (India, Malayalam) – astrological horoscopes  
- **Kambaramayanam** (India) – Tamil epic manuscripts  
- **Khmer** (Cambodia) – Sleukrith and archive collections  
- **Sundanese** (Indonesia, Java) – 15th century manuscripts  
- **Balinese** (Indonesia, Bali) – ornate Balinese Lontar collections  

<p align="center">
  <img src="figures/fig1_examples.png" width="80%"><br>
  <em>Fig. 1 – Examples of palm leaf manuscripts across six scripts and regions.</em>
</p>

---

## 🗂️ Layout Annotation Schema

Each image is annotated using **seven consistent region categories**:

| Category          | Description |
|-------------------|-------------|
| **MainRegion**    | Primary content area; whole text block |
| **ParagraphRegion** | Grouped blocks of related text lines |
| **TextLineRegion** | Individual horizontal lines of text |
| **SymbolicMark** | Section dividers, religious symbols |
| **PhysicalDamage** | Binding holes, cracks, fading |
| **Illustration** | Deities, animals, cultural drawings |
| **Other** | Non-original (stamps, labels, notes) |

<p align="center">
  <img src="figures/fig2_layout_categories.png" width="80%"><br>
  <em>Fig. 2 – Visualization of annotated regions across scripts.</em>
</p>

---

## 📊 Dataset Statistics

| Script            | Pages | Train | Test |
|-------------------|------:|------:|-----:|
| Tamil             | 101   | 81    | 20   |
| Jathakam          | 108   | 86    | 22   |
| Kambaramayanam    | 41    | 33    | 8    |
| Khmer             | 155   | 124   | 31   |
| Balinese          | 100   | 80    | 20   |
| Sundanese         | 61    | 49    | 12   |
| **Total**         | **566** | **453** | **113** |

- **Total regions annotated:** 6,334  
- **Illustrations:** concentrated in Balinese and Tamil subsets  
- **PhysicalDamage:** consistently present across all collections  

<p align="center">
  <img src="figures/fig3_annotation_workflow.png" width="80%"><br>
  <em>Fig. 3 – Annotation workflow and quality control pipeline.</em>
</p>

---

## ⚙️ Benchmarking Experiments

Experiments:

1. **Script-specific performance** – trained & tested per script.  
2. **Cross-script generalization** – trained on combined dataset, tested across all scripts.

### Key Findings

- **High accuracy** on `MainRegion`, `ParagraphRegion`, and `TextLineRegion`.  
- **Lower accuracy** on small/rare categories (`SymbolicMark`, `PhysicalDamage`, `Other`).  
- **YOLO series** excelled on small-object detection.  
- **Transformer-based DETR** models handled large, structured regions well.  
- **Cross-script training** improved underrepresented categories, showing feature transfer across scripts.

<p align="center">
  <img src="figures/fig4_detection_results.png" width="80%"><br>
  <em>Fig. 4 – Sample detection outputs on different scripts.</em>
</p>

---
## 🙏 Acknowledgements

Manuscript collections sourced from **India, Cambodia and Indonesia** under open Creative Commons licenses.

Annotation team included students and researchers from **Cambodia, China, and Indonesia**.

**Funding Support:** This is part of the **PALM-WORLD project** is supported by **The World Academy of Sciences (Italy), the Chinese Academy of Sciences (China), One-to-Many Research (Cambodia), and the National Natural Science Foundation of China (China)**.




## 🔗 References
Nair, B.B., Rani, N.S. (2023). *HMPLMD: Handwritten Malayalam palm leaf manuscript dataset.
Jailingeswari, I., Gopinathan, S. (2024). *Tamil handwritten palm leaf manuscript dataset (TH-PLMD). 
Valy, D., Verleysen, M., Chhun, S., Burie, J.C. (2017). *A new Khmer palm leaf manuscript dataset for document analysis and recognition: Sleukrith Set.
Suryani, M., Paulus, E., Hadi, S., Darsa, U.A., Burie, J.C. (2017). *The handwritten Sundanese palm leaf manuscript dataset from 15th century.*
Kesiman, M.W.A., Burie, J.C., Wibawantara, G.N.M.A., Sunarya, I.M.G., Ogier, J.M. (2016). *AMADI_LontarSet: The first handwritten Balinese palm leaf manuscripts dataset.
Kesiman, M.W.A., Valy, D., Burie, J.C., Paulus, E., Suryani, M., Hadi, S., Verleysen, M., Chhun, S., Ogier, J.M. (2018). *ICFHR 2018 competition on document image analysis tasks for Southeast Asian palm leaf manuscripts.

## 📚 Citation

If you find **PALM-LAY** interesting and useful for your research, please cite:

```bibtex
@inproceedings{thuon2025palmlay,
  title     = {PALM-LAY: A Multi-Script Cross-Regional Dataset for Layout Analysis of Palm Leaf Manuscripts},
  author    = {Thuon, Nimol and Du, Jun and Theang, Panhapin and Thuon, Ratana},
  booktitle = {Proceedings of the International Conference on Document Analysis and Recognition (ICDAR)},
  year      = {2025},
  publisher = {Springer},
}

