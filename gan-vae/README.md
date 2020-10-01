This repository is for gan-vae reward function training of paper "Guided Dialog Policy Training without Adversarial Learning in the Loop"

### VAE-GAN training: 
```sh
python -u mwoz_gan_vae.py --max_epoch 400 --data_dir ./data/multiwoz --init_lr 0.0005 --vae_loss bce --l2_lambda 0.000001 --early_stop True --round_for_disc False --gan_type wgan --op rmsprop --gan_ratio 3 

```
Requirements: pytorch-1.0, cuda-9.2, nltk-3.2.5, Python-2.7

* The dataset and pretrained reward function can be found [here](https://drive.google.com/file/d/1RdcG4nHlS4NqDtNWUscTU-mp4NZyWwOA/view?usp=sharing). Copy the downloaded dataset to path './data/' and unzip them. Copy pretrained models to path './logs/'. You can reuse the pretrained vae model: './logs/2019-09-06T10:50:18.034181-mwoz_gan_vae.py'.

This code is based on the source code of Tiancheng's LAED work.
