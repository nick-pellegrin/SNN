import torch
from snntorch import spikegen
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt

spike_in = spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)
# sample contains 784 neurons, and each neuron has 200 time steps
# >>> torch.Size([200, 1, 784])

spike_data = spike_in.squeeze(1)
print(spike_data.shape)


fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data, ax, s=1.5, c="black", marker='|', linewidths=0.5)
plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()