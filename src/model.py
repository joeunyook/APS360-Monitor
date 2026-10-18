"""DistilBERT monitor: encode each chunk, pool over chunks, classify the trajectory."""
import torch
import torch.nn as nn
from transformers import AutoModel


class TrajectoryMonitor(nn.Module):
    def __init__(self, name="distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # input_ids: (num_chunks, seq_len) for a single trajectory
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        chunk_emb = hidden[:, 0]                     # [CLS] embedding per chunk
        traj_emb = chunk_emb.mean(dim=0)             # pool over chunks
        return self.head(traj_emb).squeeze(-1)       # logit for P(unsafe)


def train_step(model, optimizer, input_ids, attention_mask, label):
    logit = model(input_ids, attention_mask)
    loss = nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(float(label)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
