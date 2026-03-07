import torch
loss = torch.tensor([0.5])
batch_size = 4
try:
    print(loss.repeat(batch_size).shape)
except Exception as e:
    print("Error:", e)
