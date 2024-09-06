"""
Title: Real Time Simulation of a spiking neural network
author: Nick Pellegrin
date: 10/21/2021
"""
# IMPORTS --------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils

import matplotlib.pyplot as plt
import numpy as np
import itertools





# DATA LOADER -----------------------------------------------------------------------------------------
batch_size = 128
data_path = '/tmp/data/mnist'

dtype = torch.float
device = torch.device("cpu")

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)  





# NEURON PARAMETERS ----------------------------------------------------------------------------------
beta = 0.5
spike_grad = surrogate.fast_sigmoid(slope=25)


# NETWORK HYPERPARAMETERS ----------------------------------------------------------------------------
# num_inputs = 784
# num_hidden = 1000
# num_outputs = 10
pop_outputs = 500



# TRAINING PARAMETERS --------------------------------------------------------------------------------
num_steps = 50



# NETWORK DEFINITION ---------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # initialize layers
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, stride=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, kernel_size=5, stride=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64*4*4, pop_outputs)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)


    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem3


    
    # def step(self, spk_inputs, mem_potentials):
    #     cur1 = self.fc1(spk_inputs)
    #     spk1, mem_potentials[0] = self.lif1(cur1, mem_potentials[0])
    #     cur2 = self.fc2(spk1)
    #     spk2, mem_potentials[1] = self.lif2(cur2, mem_potentials[1])
    #     return spk2

    # def simulate(self, num_steps):
    #     mem1 = self.lif1.init_leaky()
    #     mem2 = self.lif2.init_leaky()
    #     mem_potentials = [mem1, mem2]

    #     out_rec = []

    #     for step in range(num_steps):
    #         spk_in = spikegen.rate_conv(torch.rand((1, num_inputs))).unsqueeze(1)
    #         spk_out = self.step(spk_in, mem_potentials)
    #         out_rec.append(spk_out)
        
    #     return torch.stack(out_rec)





# TRAINING --------------------------------------------------------------------------------------------
def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


def batch_accuracy(train_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            utils.reset(net)
            spk_rec, _ = forward_pass(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets, population_code=True, num_classes=10) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total


def train(net, train_loader, test_loader, num_steps):
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=10)
    num_epochs = 1
    loss_hist = []
    test_acc_hist = []
    counter = 0

    # Outer training loop
    for epoch in range(num_epochs):

        # Training loop
        for data, targets in iter(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, _ = forward_pass(net, num_steps, data)

            # initialize the loss & sum over time
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            if counter % 50 == 0:
                with torch.no_grad():
                    net.eval()

                    # Test set forward pass
                    test_acc = batch_accuracy(test_loader, net, num_steps)
                    print(f"Iteration {counter}, Test Acc: {test_acc * 100:.2f}%\n")
                    test_acc_hist.append(test_acc.item())

            counter += 1


# MAIN ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    net = Net()
    train(net, train_loader, test_loader, num_steps)


    # output = net.simulate(200)
    # print(output.shape)
    # output = output.squeeze(1)
    # output = output.squeeze(1)
    # print(output.shape)

    # fig = plt.figure(facecolor="w", figsize=(10, 5))
    # ax = fig.add_subplot(111)
    # splt.raster(output, ax, s=100, c="black", marker='|', linewidths=2.5)
    # plt.title("Output Layer")
    # plt.xlabel("Time step")
    # plt.show()
