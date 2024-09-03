import torch
from snntorch import spikegen
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt

spike_data = spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)
# sample contains 784 neurons, and each neuron has 200 time steps
print(spike_data.size())
# >>> torch.Size([200, 1, 784])


fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)

#  s: size of scatter points; c: color of scatter points
splt.raster(spike_data, ax, s=1.5, c="black")
plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()