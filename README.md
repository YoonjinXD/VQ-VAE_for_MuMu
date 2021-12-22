# VQ-VAE for MuMu dataset

1. Download dataset
```
sh download.sh
```

2. Download available album images. It takes time(hours).
```
python download.py
```

3. Check & Test Utilizer
```
python utilize.py
```

4. To train new model
```
python train_vq-vae.py
```

(+)
The uploaded model '128_16_128_128-step8000.pt' means
```python
num_embeddings(K) = 128
embedding_dim(D) = 16
num_hiddens = 128
num_residual_hiddens = 128
```
of hyperparameters are used and saved after step 8000.
batch size was 128.
