"""
Title: Real Time Simulation of a spiking neural network
author: Nick Pellegrin
date: 10/21/2021
"""
# IMPORTS --------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import matplotlib.pyplot as plt



# NEURON PARAMETERS ----------------------------------------------------------------------------------
beta = 0.85

# NETWORK HYPERPARAMETERS ----------------------------------------------------------------------------
num_inputs = 784
num_hidden = 1000
num_outputs = 10


# NETWORK DEFINITION ---------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def step(self, spk_inputs, mem_potentials):
        cur1 = self.fc1(spk_inputs)
        spk1, mem_potentials[0] = self.lif1(cur1, mem_potentials[0])
        cur2 = self.fc2(spk1)
        spk2, mem_potentials[1] = self.lif2(cur2, mem_potentials[1])
        return spk2

    def simulate(self, num_steps):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_potentials = [mem1, mem2]

        out_rec = []

        for step in range(num_steps):
            spk_in = spikegen.rate_conv(torch.rand((1, num_inputs))).unsqueeze(1)
            spk_out = self.step(spk_in, mem_potentials)
            out_rec.append(spk_out)
        
        return torch.stack(out_rec)



# MAIN ------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    net = Net()

    output = net.simulate(200)
    print(output.shape)
    output = output.squeeze(1)
    output = output.squeeze(1)
    print(output.shape)

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(output, ax, s=100, c="black", marker='|', linewidths=2.5)
    plt.title("Output Layer")
    plt.xlabel("Time step")
    plt.show()
