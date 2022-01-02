# VQ-VAE for MuMu dataset

1. Download dataset
```
sh download.sh
```

2. Download available album images. It takes time(hours).
```
python download.py
```

3. Inference
```
python inference.py --infer_type encoding --img_dir input.png --latent_dir input.pt
python inference.py --infer_type decoding --latent_dir input.pt --save_dir results.png
```

4. To train new model
```
python train_vq-vae.py
```

(+)
The uploaded model '128_16_128_128-step8000.pt' refers to
```python
num_embeddings(K) = 128
embedding_dim(D) = 16
num_hiddens = 128
num_residual_hiddens = 128
```
The model trained in step number 8000.
