import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import matplotlib.pyplot as plt

alpha = 0.9
beta = 0.85

# batch_size = 128

num_inputs = 784
num_hidden = 1000
num_outputs = 10

num_steps = 200
spk_in = spikegen.rate_conv(torch.rand((num_steps, 784))).unsqueeze(1)


# Define Network
class Net(nn.Module):
   def __init__(self):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

   def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []  # Record the output trace of spikes
        mem2_rec = []  # Record the output trace of membrane potential

        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)

net = Net()

output, mem_rec = net.forward(spk_in)
output = output.squeeze(1)
print(output.shape)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
#  s: size of scatter points; c: color of scatter points
splt.raster(output, ax, s=100, c="black", marker='|', linewidths=2.5)
plt.title("Output Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()