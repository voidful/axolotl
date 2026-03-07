from transformers import Trainer, TrainingArguments
import torch
class DummyM(torch.nn.Module):
    def forward(self, input_ids, labels=None):
        return torch.tensor([0.4]), torch.ones(2, 5)

t = Trainer(model=DummyM(), args=TrainingArguments("tmp", prediction_loss_only=True))
# The Trainer drops logits/labels when prediction_loss_only is True
loss, logits, labels = t.prediction_step(t.model, {"input_ids": torch.tensor([[1,2]])}, prediction_loss_only=True)
print(loss, logits, labels)
