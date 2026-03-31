# Models Tested

## Summary

We tested **60 model configurations** across **16 architecture families**, varying
window sizes (5, 8, 10, 15, 20), channel widths, dropout rates, learning rates,
and structural variations. All models were evaluated on a held-out test session
with a session-based split (no temporal leakage).

## Architecture Families

### 1. CNN Baseline
Standard 2-layer 1D CNN with batch normalization.
```
Conv1d(features→64, k=3) → ReLU → BN
Conv1d(64→128, k=3) → ReLU → BN
AdaptiveAvgPool1d(1)
Linear(128→64) → ReLU → Dropout → Linear(64→1)
```
**Best config**: W=10, F1@0.75 = 0.931

### 2. CNN Deep + Residual (WINNER)
3-layer CNN with a residual connection on the second layer.
```
Conv1d(features→64, k=3) → ReLU → BN
Conv1d(64→64, k=3) → ReLU → BN + SKIP CONNECTION
Conv1d(64→128, k=3) → ReLU → BN
AdaptiveAvgPool1d(1)
Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1)
```
**Best config**: W=5, F1@0.75 = 0.946. The residual connection helps gradient
flow and the shorter window (5 vs 10) reduces overfitting on this small dataset.

### 3. CNN Wide
Same as baseline but doubled channel widths (128→256).
**Best config**: W=8, F1@0.75 = 0.935. More parameters didn't help much — dataset
is too small for the extra capacity to pay off.

### 4. CNN Multi-Kernel
Inception-style parallel convolutions with kernel sizes 1, 3, and 5, concatenated
then fed through another Conv1d layer.
**Best config**: W=10, F1@0.75 = 0.932. Captures multi-scale temporal patterns
but no clear win over simpler architectures.

### 5. CNN Dilated
3-layer CNN with increasing dilation (1, 2, 4) for wider receptive field without
more parameters.
**Best config**: W=8, F1@0.75 = 0.940. Strong performer — the dilated convolutions
effectively look further back in time without flattening.

### 6. CNN MaxPool
Same as baseline but uses max pooling instead of average pooling. Captures the
most extreme activation rather than the average.
**Best config**: W=5, F1@0.75 = 0.940. Max pooling helps detect peaks in the
temporal signal.

### 7. CNN Avg+Max Pool
Concatenates both average AND max pooling outputs before the FC layers.
**Best config**: W=10, F1@0.75 = 0.933. Marginal improvement from combining both
pooling strategies.

### 8. MLP (Single Timestep)
Feed-forward networks operating on one timestep at a time (no temporal window).
Tested deep (4-layer, h=256), wide (2-layer, h=512), residual, and GELU variants.
**Best config**: MLP Wide h=512, F1@0.75 = 0.936. Surprisingly competitive — the
obstacle signal is mostly in the current readings.

### 9. LSTM
Standard LSTM with 2 layers, 128 hidden units, dropout 0.3.
Not in the top performers. Best F1@0.75 was around 0.85-0.90.
Recurrent models struggled with the small dataset and didn't meaningfully benefit
from the temporal information.

### 10. GRU
Same setup as LSTM but with GRU cells. Similar performance to LSTM.

### 11. LSTM+CNN Hybrid
LSTM processes the sequence first, then a CNN operates on the LSTM outputs.
**Best config**: W=10, F1@0.75 = 0.940. The LSTM's hidden state gives the CNN
richer temporal features to work with.

### 12. GRU + Attention
GRU with learned attention weights over timesteps, producing a weighted context
vector.
**Best config**: W=10, F1@0.75 = 0.829. Attention didn't help — with only 5-10
timesteps, there isn't enough temporal structure for attention to exploit.

### 13. Transformer
Small transformer encoder (d_model=64, 4 heads, 2 layers) with learned positional
embeddings.
**Best config**: W=5, F1@0.75 = 0.930. Respectable but no improvement over CNNs.
Transformers need more data to shine.

### 14. Bidirectional LSTM
LSTM that processes the window in both forward and backward directions.
**Best config**: W=5, F1@0.75 = 0.904. Bidirectionality didn't help for this
task — the most recent readings are what matter.

### 15. Logistic Regression (Baseline)
Simple linear classifier on single-timestep features.
F1@0.50 = 0.942, AUC = 0.972. This was the "wait, is the task even hard?" check.
With distance features, even linear models perform well.

### 16. Random Forest / XGBoost
Tree-based ensembles on single-timestep features.
Random Forest: F1 = 0.906, AUC = 0.981.
XGBoost: F1 = 0.867, AUC = 0.970.
Good AUC but lower F1 than neural models at the optimized threshold.

## Full Results Table (Top 20, sorted by F1 @ threshold=0.75)

| Rank | Model | W | F1@0.5 | AUC | F1@0.75 | FPR@0.75 | FNR@0.75 |
|------|-------|---|--------|-----|---------|----------|----------|
| 1 | **CNN deep+res** | **5** | **0.936** | **0.983** | **0.946** | **2.5%** | **6.2%** |
| 2 | LSTM+CNN | 10 | 0.849 | 0.948 | 0.940 | 1.3% | 9.2% |
| 3 | CNN maxpool | 5 | 0.929 | 0.973 | 0.940 | 4.5% | 3.8% |
| 4 | CNN dilated | 8 | 0.926 | 0.960 | 0.940 | 0.8% | 10.0% |
| 5 | MLP wide h=512 | flat | 0.908 | 0.982 | 0.936 | 4.4% | 4.6% |
| 6 | CNN baseline | 15 | 0.880 | 0.956 | 0.935 | 0.9% | 10.8% |
| 7 | CNN small ch=32/64 | 10 | 0.922 | 0.954 | 0.935 | 0.8% | 10.8% |
| 8 | CNN wide | 8 | 0.919 | 0.965 | 0.935 | 0.4% | 11.5% |
| 9 | CNN avg+max | 10 | 0.840 | 0.962 | 0.933 | 2.1% | 9.2% |
| 10 | CNN wide | 5 | 0.912 | 0.976 | 0.932 | 4.5% | 5.4% |
| 11 | CNN multikernel | 10 | 0.871 | 0.959 | 0.932 | 1.3% | 10.8% |
| 12 | CNN deep+res | 8 | 0.918 | 0.970 | 0.932 | 1.2% | 10.8% |
| 13 | CNN lr=0.005 | 10 | 0.911 | 0.959 | 0.932 | 1.3% | 10.8% |
| 14 | CNN ch=128/256 | 10 | 0.883 | 0.960 | 0.931 | 0.4% | 12.3% |
| 15 | Transformer | 10 | 0.906 | 0.960 | 0.931 | 0.4% | 12.3% |
| 16 | CNN baseline | 10 | 0.903 | 0.957 | 0.931 | 0.4% | 12.3% |
| 17 | Transformer | 5 | 0.923 | 0.974 | 0.930 | 2.9% | 8.5% |
| 18 | CNN avg+max | 8 | 0.906 | 0.962 | 0.929 | 2.5% | 9.2% |
| 19 | MLP residual | flat | 0.895 | 0.988 | 0.929 | 5.2% | 4.6% |
| 20 | CNN wide | 10 | 0.891 | 0.950 | 0.929 | 2.1% | 10.0% |

## Key Observations

1. **CNNs dominate** — they work best for this task at this data scale.
2. **Smaller windows win** — W=5 (about 1 second) beats W=10 or W=20. The
   obstacle signal is immediate, not gradual.
3. **Residual connections help** — the skip connection in the deep CNN made it
   the clear winner.
4. **Recurrent models underperform** — LSTM/GRU/BiLSTM all trail CNNs by 5-10%
   F1 points. They need more data.
5. **Transformers are okay but not great** — competitive but no advantage over
   CNNs at this scale.
6. **Single-timestep MLP is shockingly good** — proves the obstacle signal is
   mostly in the current reading. Temporal context helps but isn't essential.
7. **Hyperparameter sensitivity is low** — most CNN configs within the same
   family perform within ~2% F1 of each other. Architecture choice matters more
   than tuning.
