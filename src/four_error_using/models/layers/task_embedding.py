"""TaskEmbedding — learnable per-task context vector.

A lookup table of `num_tasks` vectors (one per task id 0..num_tasks-1). The
selected vector is concatenated onto the visual feature to condition the head
on which task produced the window.
"""
import torch.nn as nn


class TaskEmbedding(nn.Module):
    def __init__(self, num_tasks=8, task_emb_dim=32):
        super().__init__()
        self.embedding_dim = task_emb_dim
        self.embedding = nn.Embedding(num_embeddings=num_tasks, embedding_dim=task_emb_dim)

    def forward(self, task_id):
        return self.embedding(task_id)
