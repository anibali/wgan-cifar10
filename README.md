# wgan-cifar10

An unofficial implementation of (improved) WGAN in PyTorch for CIFAR-10 image
data.

![Generated CIFAR-10 samples](/docs/samples.png)

## Requirements

* A modern NVIDIA GPU
* [Docker](https://docs.docker.com/engine/installation/)
* [Docker Compose](https://docs.docker.com/compose/install/)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/wiki/Installation)
* [nvidia-docker-compose](https://github.com/eywalker/nvidia-docker-compose)

## Usage

```sh
nvidia-docker-compose run --rm pytorch bin/train.py
```

Outputs from training are written to files in `out/`.

There are a number of command line options which can be used to configure
the training process:

```
--epochs N      number of epochs to train (default=1000)
--gen-iters N   generator iterations per epoch (default=100)
--disc-iters N  discriminator iterations per generator iteration (default=5)
--batch-size N  input batch size (default=64)
--disc-lr LR    discriminator learning rate (default=2e-4)
--gen-lr LR     generator learning rate (default=2e-4)
--unimproved    disable gradient penalty and use weight clipping instead
```

## References

* [Wasserstein GAN](https://arxiv.org/abs/1701.07875), Arjovsky et al.
* [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028),
  Gulrajani et al.

## Related repositories

* [martinarjovsky/WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN) -
  Official repository for the original WGAN paper (PyTorch).
* [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training) -
  Official repository for the improved WGAN training paper (TensorFlow).
* [caogang/wgan-gp](https://github.com/caogang/wgan-gp) - Unofficial partial
  port of the improved WGAN code (PyTorch).
