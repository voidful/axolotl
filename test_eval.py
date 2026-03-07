import torch
import numpy as np

loss = torch.tensor(0.4)
logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
labels = torch.tensor([1, 0])

# mimic prediction step return
output = (loss, logits, labels)

# what HF does:
# if loss is None, it doesn't add to all_losses
if output[0] is not None:
    print(" loss is not none, shape:", output[0].shape)
    
