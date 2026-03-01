```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserInteraction:
    user_id: int
    item_id: int
    rating: float
    timestamp: float
    feedback_type: str = "explicit"


@dataclass
class RLHFFeedback:
    user_id: int
    item_a_id: int
    item_b_id: int
    preferred_item: int
    confidence: float = 1.0


class InteractionDataset(Dataset):
    def __init__(self, interactions: List[UserInteraction], num_items: int):
        self.interactions = interactions
        self.num_items = num_items

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        interaction = self.interactions[idx]
        return {
            "user_id": torch.tensor(interaction.user_id, dtype=torch.long),
            "item_id": torch.tensor(interaction.item_id, dtype=torch.long),
            "rating": torch.tensor(interaction.rating, dtype=torch.float),
        }


class PreferenceDataset(Dataset):
    def __init__(self, preferences: List[RLHFFeedback]):
        self.preferences = preferences

    def __len__(self) -> int:
        return len(self.preferences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pref = self.preferences[idx]
        return {
            "user_id": torch.tensor(pref.user_id, dtype=torch.long),
            "item_a": torch.tensor(pref.item_a_id, dtype=torch.long),
            "item_b": torch.tensor(pref.item_b_id, dtype=torch.long),
            "preferred": torch.tensor(
                1 if pref.preferred_item == pref.item_a_id else 0,
                dtype=torch.float,
            ),
            "confidence": torch.tensor(pref.confidence, dtype=torch.float),
        }


class NeuralCollaborativeFilter(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.2,
    ):
        super(NeuralCollaborativeFilter, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # GMF embeddings
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim),
            ])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Output layer
        self.output_layer = nn.Linear(embedding_dim + hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        for embedding in [
            self.gmf_user_embedding,
            self.gmf_item_embedding,
            self.mlp_user_embedding,
            self.mlp_item_embedding,
        ]:
            nn.init.normal_(embedding.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # GMF path
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user * gmf_item

        # MLP path
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Concatenate and predict
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.sigmoid(self.output_layer(combined)).squeeze(-1)

        # Return both prediction and combined embedding for reward model
        return prediction, combined

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        gmf_emb = self.gmf_user_embedding(user_ids)
        mlp_emb = self.mlp_user_embedding(user_ids)
        return torch.cat([gmf_emb, mlp_emb], dim=-1)

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        gmf_emb = self.gmf_item_embedding(item_ids)
        mlp_emb = self.mlp_item_embedding(item_ids)
        return torch.cat([gmf_emb, mlp_emb], dim=-1)


class RewardModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.1,
    ):
        super(RewardModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings).squeeze(-1)


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        user_embedding_dim: int,
        item_embedding_dim: int,
        hidden_dim: int = 128,
        dropout