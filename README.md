# Spatial-Spectral Entropy-based Quality (SSEQ) index

This is my implementation of the **SSEQ index**. I wasn't able to find a fully implemented Python version of this index, so I decided to use [Aca4peop's code](https://github.com/Aca4peop/SSEQ-Python) as a starting point and then add my own modifications.

The full details of SSEQ can be found in the paper: [No-reference image quality assessment based on spatial and spectral entropies (Liu et al.)](https://doi.org/10.1016/j.image.2014.06.006). The original MATLAB implementation is [here](https://github.com/utlive/SSEQ).

## Highlights

Vectorized implementation of:

- Patch spatial entropy
- DCT for spectral entropy

## Results

Every dataset was split into a training and a test set. I used the training sets with K-fold cross-validation to get the best parameters for each SVR model. The following are the results on each test set:

| Dataset  | LCC    | SROCC  |
|----------|--------|--------|
| csiq     | 0,8493 | 0,7913 |
| kadid10k | 0,6075 | 0,5716 |
| koniq10k | 0,5745 | 0,5573 |
| tid2013  | 0,7892 | 0,7204 |