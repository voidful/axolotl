import torch
from transformers import Trainer, TrainingArguments

loss = torch.tensor([0.5])
# Let's inspect HF Trainer's gather logic under mock Distributed
try:
    from accelerate.utils import gather_object
    print(gather_object([loss]))
except Exception as e:
    print(e)
