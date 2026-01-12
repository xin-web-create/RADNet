# RADNet: Retina-inspired Dehazing Network


RADNet is a **Retina-inspired CNN Architecture for Dehazing** designed to address three persistent challenges in single-image dehazing: texture/edge loss, non-uniform haze adaptation, and inconsistent restoration under mixed illumination. Inspired by the center-surround contrast encoding and parallel ON/OFF pathways of the retina, RADNet introduces three novel modules to enhance dehazing performance.

---

## 🎯 Key Features

- **RFC Module (Retina-inspired Feature Convolution)**  
  Combines ON-type, OFF-type, and center-difference convolutions with a standard 3×3 convolution to enhance local contrast and preserve edges.

- **CSPCA Module (Channel-Spatial-Pixel Cooperative Attention)**  
  Jointly modulates features across channel, spatial, and pixel dimensions for adaptive, location-aware haze removal.

- **ON/OFF Dual-Branch Fusion**  
  Processes the original image and its intensity-inverted counterpart in parallel, integrating them via attention-based fusion for robust recovery in both bright and dark regions.








## 📊 Results

### Synthetic Benchmarks (RESIDE SOTS)

| Method        | SOTS-Indoor (PSNR/SSIM) | SOTS-Outdoor (PSNR/SSIM) |
|---------------|-------------------------|--------------------------|
| DCP           | 16.62 / 0.818           | 19.13 / 0.815            |
| FFA-Net       | 36.39 / 0.989           | 33.57 / 0.984            |
| MSBDN         | 33.67 / 0.985           | 33.48 / 0.982            |
| **RADNet**    | **36.83 / 0.993**       | **32.61 / 0.980**        |


---

## 🔬 Ablation Study (Dense-Haze)

| Variant               | PSNR (dB) | SSIM   |
|-----------------------|-----------|--------|
| w/o RFC               | 16.22     | 0.520  |
| w/o CSPCA             | 16.16     | 0.531  |
| w/o ON/OFF            | 15.99     | 0.567  |
| **Baseline**       | **16.89** | **0.607** |


---

## 🙏 Acknowledgements

This work is inspired by biological vision processing and builds upon prior research in CNN-based dehazing. We thank the authors of RESIDE, O-HAZE, and Dense-Haze for providing benchmark datasets.

---

