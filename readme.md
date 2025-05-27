# Synthetic Seismogram Denoising with CNNs and Frequency Decomposition

This script generates synthetic seismograms, adds random noise, decomposes them using three different frequency‐decomposition techniques (Empirical Mode Decomposition, Continuous Wavelet Transform, and FFT), and trains a small convolutional neural network (CNN) to denoise each representation. Finally, it compares the validation mean absolute error (MAE) of the three approaches.

---

## Table of Contents

- [Features](#features)  
- [Dependencies](#dependencies)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [Script Structure](#script-structure)  
- [Output](#output)  
- [Author](#author)  
- [License](#license)  

---

## Features

- **Synthetic Seismogram Generation** – combines multiple frequency‐modulated chirp signals with an envelope to mimic realistic seismic waveforms.  
- **Noise Injection** – adds Gaussian noise to create “noisy” inputs.  
- **Frequency Decompositions**  
  - Empirical Mode Decomposition (EMD) via PyEMD  
  - Continuous Wavelet Transform (CWT)  
  - Fast Fourier Transform (FFT)  
- **CNN Denoiser** – trains a small encoder–decoder model on each decomposition to learn a mapping from noisy to clean.  
- **Performance Comparison** – plots validation MAE for all three methods side by side.

---

## Dependencies

This script was developed and tested on Python 3.8+. You’ll need:

- [NumPy](https://numpy.org/)  
- [SciPy](https://scipy.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [PyEMD](https://github.com/laszukdawid/PyEMD)  
- [TensorFlow 2.x](https://www.tensorflow.org/)  

You can install them via pip:

```bash
pip install numpy scipy matplotlib pyemd tensorflow

## Installation

1. Clone the repository (or simply place seismogram_random.py in your working directory).
2. Install dependencies as shown above.

## Configuration

At the top of seismogram_random.py, you can adjust:


| Variable     | Description                                                | Default                                                                |
| ------------ | ---------------------------------------------------------- | ---------------------------------------------------------------------- |
| `N_total`    | Total number of synthetic examples to generate             | `100`                                                                  |
| `N_train`    | Number of examples used for training (rest for validation) | `80`                                                                   |
| `out_dir`    | Directory for saving any figures or model checkpoints      | `'/Users/asifashraf/Documents/Manuscripts/Seismogram_machineLearning'` |
| `dt`         | Sampling interval (s)                                      | `0.005`                                                                |
| `start_time` | Start time of synthetic record (s)                         | `0`                                                                    |
| `end_time`   | End time of synthetic record (s)                           | `50`                                                                   |

