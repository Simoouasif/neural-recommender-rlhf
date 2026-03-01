```python
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from collections import deque
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    gae_lambda: float = 0.95
    gamma: float = 0.99
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    normalize_advantages: bool = True
    
    # Training
    learning_rate: float = 1e-5
    actor_learning_rate: float = 1e-5
    critic_learning_rate: float = 1e-4
    warmup_steps: int = 100
    total_steps: int = 10000
    rollout_steps: int = 256
    batch_size: int = 32
    
    # KL divergence penalty
    kl_coef: float = 0.1
    kl_target: float = 0.02
    kl_horizon: int = 10000
    adaptive_kl: bool = True
    
    # Reward model
    reward_model_path: Optional[str] = None
    reward_scale: float = 1.0
    reward_clip: float = 5.0
    
    # Reference model
    ref_model_path: Optional[str] = None
    
    # Logging
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    output_dir: str = "./rlhf_output"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RolloutBuffer:
    states: List[torch.Tensor] = field(default_factory=list)
    actions: List[torch.Tensor] = field(default_factory=list)
    rewards: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    dones: List[torch.Tensor] = field(default_factory=list)
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None
    
    def __len__(self):
        return len(self.states)


class RecommendationEnvironment:
    def __init__(
        self,
        num_items: int,
        num_users: int,
        item_embeddings: Optional[torch.Tensor] = None,
        user_embeddings: Optional[torch.Tensor] = None,
        interaction_matrix: Optional[torch.Tensor] = None,
        config: Optional[PPOConfig] = None
    ):
        self.num_items = num_items
        self.num_users = num_users
        self.config = config or PPOConfig()
        
        self.item_embeddings = item_embeddings
        self.user_embeddings = user_embeddings
        self.interaction_matrix = interaction_matrix
        
        self.current_user = None
        self.recommended_items = set()
        self.step_count = 0
        self.max_steps = 10
        
    def reset(self, user_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if user_id is None:
            user_id = np.random.randint(0, self.num_users)
        
        self.current_user = user_id
        self.recommended_items = set()
        self.step_count = 0
        
        state = self._get_state()
        return state
    
    def step(self, action: int) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        reward = self._compute_reward(action)
        self.recommended_items.add(action)
        self.step_count += 1
        
        done = self.step_count >= self.max_steps
        next_state = self._get_state()
        
        info = {
            "user_id": self.current_user,
            "item_id": action,
            "step": self.step_count,
            "diversity_score": self._compute_diversity(),
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> Dict[str, torch.Tensor]:
        state = {}
        
        if self.user_embeddings is not None:
            state["user_embedding"] = self.user_embeddings[self.current_user]
        else:
            state["user_id"] = torch.tensor(self.current_user, dtype=torch.long)
        
        if self.item_embeddings is not None:
            state["item_embeddings"] = self.item_embeddings
        
        recommended_mask = torch.zeros(self.num_items)
        for item in self.recommended_items:
            recommended_mask[item] = 1.0
        state["recommended_mask"] = recommended_mask
        
        state["step"] = torch.tensor(self.step_count, dtype=torch.float)
        
        return state
    
    def _compute_reward(self, action: int) -> float:
        if action in self.recommended_items:
            return -1.0
        
        reward = 0.0
        
        if self.interaction_matrix is not None:
            if self.interaction_matrix[self.current_user, action] > 0:
                reward += self.interaction_matrix[self.current_user, action].item()
        
        diversity_bonus = self._compute_diversity_for_item(action)
        reward += 0.1 * diversity_bonus
        
        return reward
    
    def _compute_diversity(self) -> float:
        if len(self.recommended_items) < 2 or self.item_embeddings is None:
            return 1.0
        
        items = list(self.recommended_items)
        embeddings = self.item_embeddings[items]
        
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(0),
            embeddings.unsqueeze(1),
            dim=2
        )
        
        n = len(items)
        diversity = 1.0 - (similarities.sum() - n) / (n * (n - 1) + 1e-8)
        return diversity.item()
    
    def _compute_diversity_for_item(self, item_id: int) -> float:
        if len(self.recommended_items) == 0 or self.item_embeddings is None:
            return 1.0
        
        item_emb = self.item_embeddings[item_id]
        existing_embs = self.item_embeddings[list(self.recommended_items)]
        
        similarities = F.cosine_similarity(
            item_emb.unsqueeze(0),
            existing_embs,
            dim=1
        )
        
        diversity = 1.0 - similarities.mean().item()
        return diversity


class Actor