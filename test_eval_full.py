import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __len__(self): return 10
    def __getitem__(self, i): return {"input_ids": torch.tensor([1,2,3]), "labels": torch.tensor([1,2,3])}

class DummyModel(torch.nn.Module):
    def forward(self, input_ids, labels=None, **kwargs):
        class Output:
            loss = torch.tensor([0.5])
            logits = torch.ones(3, 10)
        return Output()
    
model = DummyModel()
trainer = Trainer(model=model, args=TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=2))
metrics = trainer.evaluate(eval_dataset=DummyDataset())
print("EVAL METRICS:", metrics)
