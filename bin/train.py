#!/usr/bin/env python3

'''
Executable script for training a Wasserstein GAN on CIFAR-10 data.
'''

import os
import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
import torchvision.utils

from wgan.data import Cifar10Dataset
from wgan.model import Generator, Discriminator
from wgan import lipschitz, progress

def calculate_disc_gradients(discriminator, generator, real_var, lipschitz_constraint):
    '''Calculate gradients and loss for the discriminator.'''

    # Enable gradient calculations for discriminator parameters
    for param in discriminator.parameters():
        param.requires_grad = True

    # Set discriminator parameter gradients to zero
    discriminator.zero_grad()

    lipschitz_constraint.prepare_discriminator()

    real_out = discriminator(real_var).mean()
    real_out.backward(torch.cuda.FloatTensor([-1]))

    # Sample Gaussian noise input for the generator
    noise = torch.randn(real_var.size(0), 128).type_as(real_var.data)
    noise = Variable(noise, volatile=True)

    gen_out = generator(noise)
    fake_var = Variable(gen_out.data)
    fake_out = discriminator(fake_var).mean()
    fake_out.backward(torch.cuda.FloatTensor([1]))

    loss_penalty = lipschitz_constraint.calculate_loss_penalty(real_var.data, fake_var.data)

    disc_loss = fake_out - real_out + loss_penalty
    return disc_loss

def calculate_gen_gradients(discriminator, generator, batch_size):
    '''Calculate gradients and loss for the generator.'''

    # Disable gradient calculations for discriminator parameters
    for param in discriminator.parameters():
        param.requires_grad = False

    # Set generator parameter gradients to zero
    generator.zero_grad()

    # Sample Gaussian noise input for the generator
    noise = torch.randn(batch_size, 128).cuda()
    noise = Variable(noise)

    fake_var = generator(noise)
    fake_out = discriminator(fake_var).mean()
    fake_out.backward(torch.cuda.FloatTensor([-1]))

    gen_loss = -fake_out
    return gen_loss

def loop_data_loader(data_loader):
    '''Create an infinitely looping generator for a data loader.'''

    while True:
        for batch in data_loader:
            yield batch

def parse_args():
    '''Parse command-line arguments.'''

    parser = argparse.ArgumentParser(description='WGAN model trainer.')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
        help='number of epochs to train (default=1000)')
    parser.add_argument('--gen-iters', type=int, default=100, metavar='N',
        help='generator iterations per epoch (default=100)')
    parser.add_argument('--disc-iters', type=int, default=5, metavar='N',
        help='discriminator iterations per generator iteration (default=5)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
        help='input batch size (default=64)')
    parser.add_argument('--disc-lr', type=float, default=2e-4, metavar='LR',
        help='discriminator learning rate (default=2e-4)')
    parser.add_argument('--gen-lr', type=float, default=2e-4, metavar='LR',
        help='generator learning rate (default=2e-4)')
    parser.add_argument('--unimproved', default=False, action='store_true',
        help='disable gradient penalty and use weight clipping instead')

    args = parser.parse_args()

    return args

def main():
    '''Main entrypoint function for training.'''

    # Parse command-line arguments
    args = parse_args()

    # Create directory for saving outputs
    os.makedirs('out', exist_ok=True)

    # Initialise CIFAR-10 data loader
    train_loader = DataLoader(Cifar10Dataset('data/cifar-10'),
        args.batch_size, num_workers=4, pin_memory=True, drop_last=True)
    inf_train_data = loop_data_loader(train_loader)

    # Build neural network models and copy them onto the GPU
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    # Select which Lipschitz constraint to use
    if args.unimproved:
        lipschitz_constraint = lipschitz.WeightClipping(discriminator)
    else:
        lipschitz_constraint = lipschitz.GradientPenalty(discriminator)

    # Initialise the parameter optimisers
    optim_gen = optim.Adam(generator.parameters(), lr=2e-4, betas=(0, 0.9))
    optim_disc = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0, 0.9))

    # Run the main training loop
    for epoch in range(args.epochs):
        avg_disc_loss = 0
        avg_gen_loss = 0

        for gen_iter in range(args.gen_iters):
            # Train the discriminator (aka critic)
            for _ in range(args.disc_iters):
                batch = next(inf_train_data)
                real_var = Variable(batch['input'].cuda())

                disc_loss = calculate_disc_gradients(
                    discriminator, generator, real_var, lipschitz_constraint)
                avg_disc_loss += disc_loss.data[0]
                optim_disc.step()

            # Train the generator
            gen_loss = calculate_gen_gradients(discriminator, generator, args.batch_size)
            avg_gen_loss += gen_loss.data[0]
            optim_gen.step()

            # Save generated images
            torchvision.utils.save_image((generator.last_output.data.cpu() + 1) / 2,
                'out/samples.png', nrow=8, range=(-1, 1))

            # Advance the progress bar
            progress.bar(gen_iter + 1, args.gen_iters,
                prefix='Epoch {:4d}'.format(epoch), length=30)

        # Calculate mean losses
        avg_disc_loss /= args.gen_iters * args.disc_iters
        avg_gen_loss /= args.gen_iters

        # Print loss metrics for the last batch of the epoch
        print('Epoch {:4d}: disc_loss={:8.4f}, gen_loss={:8.4f}'.format(
            epoch, avg_disc_loss, avg_gen_loss))

        # Save the discriminator weights and optimiser state
        torch.save({
            'epoch': epoch + 1,
            'model_state': discriminator.state_dict(),
            'optim_state': optim_disc.state_dict(),
        }, os.path.join('out', 'discriminator.pth'))

        # Save the generator weights and optimiser state
        torch.save({
            'epoch': epoch + 1,
            'model_state': generator.state_dict(),
            'optim_state': optim_gen.state_dict(),
        }, os.path.join('out', 'generator.pth'))

if __name__ == '__main__':
    main()
