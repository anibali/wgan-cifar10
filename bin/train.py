#!/usr/bin/env python3

import sys

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
import torchvision.utils

from wgan.data import Cifar10Dataset
from wgan.model import Generator, Discriminator
from wgan import lipschitz

def calculate_discriminator_gradients(discriminator, generator, real_var, lipschitz_constraint):
    for param in discriminator.parameters():
        param.requires_grad = True

    discriminator.zero_grad()
    lipschitz_constraint.prepare_discriminator()

    real_out = discriminator(real_var).mean()
    real_out.backward(torch.cuda.FloatTensor([-1]))

    batch_size = real_var.size(0)
    noise = torch.randn(batch_size, 128).type_as(real_var.data)
    noise = Variable(noise, volatile=True)

    gen_out = generator(noise)
    fake_var = Variable(gen_out.data)
    fake_out = discriminator(fake_var).mean()
    fake_out.backward(torch.cuda.FloatTensor([1]))

    loss_penalty = lipschitz_constraint.calculate_loss_penalty(real_var.data, fake_var.data)

    disc_loss = fake_out - real_out + loss_penalty
    return disc_loss

def calculate_generator_gradients(discriminator, generator, batch_size):
    for param in discriminator.parameters():
        param.requires_grad = False

    generator.zero_grad()

    noise = torch.randn(batch_size, 128).cuda()
    noise = Variable(noise)

    fake_var = generator(noise)
    fake_out = discriminator(fake_var).mean()
    fake_out.backward(torch.cuda.FloatTensor([-1]))

    gen_loss = -fake_out
    return gen_loss

def loop_data_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch

def main():
    batch_size = 64
    epochs = 1000
    gen_iters = 100 # Generator iterations per epoch
    disc_iters = 5 # Discriminator iterations per generator iteration

    train_data = Cifar10Dataset('/data/dlds/cifar-10')
    train_loader = DataLoader(train_data, batch_size, num_workers=4,
        pin_memory=True, drop_last=True)
    inf_train_data = loop_data_loader(train_loader)

    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    lipschitz_constraint = lipschitz.GradientPenalty(discriminator)

    optim_gen = optim.Adam(generator.parameters(), lr=2e-4, betas=(0, 0.9))
    optim_disc = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0, 0.9))

    for epoch in range(epochs):
        for gen_iter in range(gen_iters):
            for disc_iter in range(disc_iters):
                batch = next(inf_train_data)
                real_var = Variable(batch['input'].cuda())

                disc_loss = calculate_discriminator_gradients(
                    discriminator, generator, real_var, lipschitz_constraint)
                optim_disc.step()

            gen_loss = calculate_generator_gradients(discriminator, generator, batch_size)
            print('[{:3d}|{:2d}] disc_loss={:0.4f} gen_loss={:0.4f}'.format(
                epoch, gen_iter, disc_loss.data[0], gen_loss.data[0]))

            samples = torchvision.utils.save_image((generator.last_output.data.cpu() + 1) / 2,
                'out/samples.png', nrow=8, range=(-1, 1))
            samples = torchvision.utils.save_image((batch['input'] + 1) / 2,
                'out/real.png', nrow=8)

            optim_gen.step()

if __name__ == '__main__':
    main()
