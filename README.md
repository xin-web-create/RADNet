# RADNet: Image Dehazing Network Based on Retinal Neuromorphic Inspiration

Single-image dehazing is challenging due to spatially non-uniform haze, complex illumination, and fine structure loss. We present RADNet, a retina-inspired convolutional neural network that translates computational principles of early visual processing into a multi-stage restoration pipeline. It consists of three core components: a Retina-inspired Feature Convolution (RFC) module with ON-type, OFF-type, and center-difference filters; a Channel–Spatial–Pixel Cooperative Attention (CSPCA) module; and an attention-based ON/OFF Fusion module. Inspired by the luminance-difference extraction and fusion capability of retinal ON/OFF neurons, parallel ON and OFF branches separately extract bright-path and dark-path information, which are integrated at multiple scales. Experiments indicate that RADNet achieves competitive performance while maintaining a compact model size.

> Code is available at https://github.com/xin-web-create/RADNet.

## 🎯 Key Features

- **RFC Module (Retina-inspired Feature Convolution)**&#8203;
  Mimics the center–surround antagonistic receptive fields of retinal bipolar/ganglion cells. It applies four parallel convolutions — ON-type (center-positive, surround-negative), OFF-type (center-negative, surround-positive), center-difference, and a standard 3×3.

- **CSPCA Module (Channel–Spatial–Pixel Cooperative Attention)**&#8203;
  Jointly computes channel attention, multi-dilated spatial attention, and pixel attention, which are fused cooperatively to enable cross-dimensional interaction for adaptive, location-aware haze removal.

- **ON/OFF Dual-Branch Fusion**
  Processes the original image and its intensity-inverted counterpart (1 − I) in parallel. An attention-based fusion module reweights and mixes the two branches through branch-wise channel gating and a residual connection, enabling robust recovery in both bright and dark regions.

## 📊 Results

### Synthetic Benchmark (RESIDE SOTS)

| Method | SOTS-Indoor (PSNR / SSIM) | SOTS-Outdoor (PSNR / SSIM) | Params (M) | FLOPs (G) |
|--------|---------------------------|----------------------------|------------|-----------|
| DCP | 16.62 / 0.818 | 19.13 / 0.815 | – | – |
| DehazeNet | 19.82 / 0.821 | 24.75 / 0.927 | 0.01 | 0.58 |
| AOD-Net | 20.51 / 0.816 | 24.14 / 0.920 | 0.002 | 0.12 |
| GridDehazeNet | 32.16 / 0.984 | 30.86 / 0.982 | 0.96 | 21.5 |
| MSBDN | 33.67 / 0.985 | 33.48 / 0.982 | 31.35 | 41.54 |
| FFA-Net | 36.39 / 0.989 | 33.57 / 0.984 | 4.456 | 287.8 |
| AECR-Net | 37.17 / 0.990 | – | 2.611 | 52.2 |
| DeHamer | 36.63 / 0.988 | 35.18 / 0.986 | 132.50 | 60.3 |
| Fourmer | 37.32 / 0.990 | – | 1.29 | 20.6 |
| **RADNet (Ours)**&#8203; | **36.83 / 0.993** | **32.61 / 0.980** | **5.822** | **41.77** |



## 🙏 Acknowledgements

This work is inspired by biological vision processing and builds upon prior research in CNN-based dehazing. We thank the authors of RESIDE, O-Haze, NH-Haze, and Dense-Haze for providing the benchmark datasets.
