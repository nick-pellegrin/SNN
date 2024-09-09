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








