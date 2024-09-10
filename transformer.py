import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn

class SpikingTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=2, num_classes=10, beta=0.9, spike_grad=None):
        super(SpikingTransformer, self).__init__()

        # Multihead self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        
        # Spiking neurons in the feedforward layers
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Feedforward network within the transformer block
        self.ffn1 = nn.Linear(d_model, 4 * d_model)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.ffn2 = nn.Linear(4 * d_model, d_model)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Classifier
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # src: [seq_len, batch_size, d_model]

        # Self-attention
        attn_output, _ = self.self_attn(src, src, src)
        
        # Add & Norm
        src = self.norm1(src + attn_output)

        # Feedforward + spiking dynamics (LIF neurons)
        ff_output = F.relu(self.ffn1(src))
        ff_output, _ = self.lif1(ff_output)  # Spiking behavior in feedforward layer
        ff_output = F.relu(self.ffn2(ff_output))
        ff_output, _ = self.lif2(ff_output)  # Another spiking neuron layer
        src = self.norm2(src + ff_output)

        # Output classification
        out = self.fc_out(src[-1])  # Take the last time step for classification

        return out

# Example usage
model = SpikingTransformer(d_model=128, nhead=8, num_classes=10, beta=0.9)
src = torch.rand(10, 32, 128)  # Sequence length of 10, batch size of 32, embedding size of 128
out = model(src)
print(out.shape)  # Should output [32, 10] -> batch size x num_classes