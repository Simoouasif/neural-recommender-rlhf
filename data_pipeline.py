```python
"""
Data pipeline for neural-recommender-rlhf.
Handles data loading, preprocessing, and preparation for training.
"""

import os
import json
import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data pipeline."""
    data_dir: str = "data"
    train_file: str = "train.json"
    val_file: str = "val.json"
    test_file: str = "test.json"
    preference_file: str = "preferences.json"
    
    # Item/User parameters
    num_users: int = 10000
    num_items: int = 50000
    embedding_dim: int = 64
    max_seq_len: int = 50
    
    # Data split parameters
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Batch parameters
    batch_size: int = 256
    num_workers: int = 4
    
    # Negative sampling
    num_negatives: int = 4
    
    # Preference data
    min_rating_for_positive: float = 4.0
    
    # Padding
    pad_token_id: int = 0
    
    random_seed: int = 42


@dataclass
class InteractionRecord:
    """Single user-item interaction record."""
    user_id: int
    item_id: int
    rating: float
    timestamp: int
    context: Optional[Dict[str, Any]] = None


@dataclass
class PreferenceRecord:
    """Preference comparison record for RLHF."""
    user_id: int
    chosen_item_id: int
    rejected_item_id: int
    chosen_context: Optional[List[int]] = None
    rejected_context: Optional[List[int]] = None
    confidence: float = 1.0


class DataProcessor:
    """Processes raw data into model-ready format."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.user2idx: Dict[Any, int] = {}
        self.item2idx: Dict[Any, int] = {}
        self.idx2user: Dict[int, Any] = {}
        self.idx2item: Dict[int, Any] = {}
        self.item_features: Optional[np.ndarray] = None
        
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
    
    def build_vocabularies(
        self,
        interactions: List[InteractionRecord]
    ) -> None:
        """Build user and item vocabularies from interactions."""
        users = sorted(set(r.user_id for r in interactions))
        items = sorted(set(r.item_id for r in interactions))
        
        # Reserve 0 for padding
        self.user2idx = {u: i + 1 for i, u in enumerate(users)}
        self.item2idx = {it: i + 1 for i, it in enumerate(items)}
        self.idx2user = {v: k for k, v in self.user2idx.items()}
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        
        logger.info(
            f"Built vocabularies: {len(self.user2idx)} users, "
            f"{len(self.item2idx)} items"
        )
    
    def encode_user(self, user_id: Any) -> int:
        return self.user2idx.get(user_id, 0)
    
    def encode_item(self, item_id: Any) -> int:
        return self.item2idx.get(item_id, 0)
    
    def build_user_sequences(
        self,
        interactions: List[InteractionRecord]
    ) -> Dict[int, List[int]]:
        """Build chronological item sequences per user."""
        user_interactions = defaultdict(list)
        for record in interactions:
            user_idx = self.encode_user(record.user_id)
            item_idx = self.encode_item(record.item_id)
            user_interactions[user_idx].append((record.timestamp, item_idx))
        
        user_sequences = {}
        for user_idx, timestamped_items in user_interactions.items():
            timestamped_items.sort(key=lambda x: x[0])
            user_sequences[user_idx] = [item for _, item in timestamped_items]
        
        return user_sequences
    
    def create_sequential_samples(
        self,
        user_sequences: Dict[int, List[int]],
        all_item_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Create (context, target, negatives) samples from sequences."""
        samples = []
        
        for user_idx, sequence in user_sequences.items():
            if len(sequence) < 2:
                continue
            
            for i in range(1, len(sequence)):
                context = sequence[max(0, i - self.config.max_seq_len):i]
                target = sequence[i]
                
                # Negative sampling
                negatives = self._sample_negatives(
                    target,
                    set(sequence),
                    all_item_ids,
                    self.config.num_negatives
                )
                
                samples.append({
                    "user_id": user_idx,
                    "context": context,
                    "target": target,
                    "negatives": negatives,
                })
        
        return samples
    
    def _sample_negatives(
        self,
        positive_item: int,
        user_history: set,
        all_items: List[int],
        num_negatives: int
    ) -> List[int]:
        """Sample negative items not in user history."""
        negatives = []
        attempts = 0
        max_attempts = num_negatives * 10
        
        while len(negatives) < num_negatives and attempts < max_attempts:
            candidate = random.choice(all_items)
            if candidate not in user_history and candidate != positive_item:
                negatives.append(candidate)
            attempts += 1
        
        # Pad if needed
        while len(negatives) < num_negatives:
            negatives.append(random.choice(all_items))
        
        return negatives
    
    def create_preference_pairs(
        self,
        interactions: List[InteractionRecord],
        user_sequences: Dict[int, List[int]]
    ) -> List[PreferenceRecord]:
        """Create preference pairs from rating data."""
        preference_records = []
        
        user_ratings = defaultdict(list)
        for record in interactions:
            user_idx = self.encode_user(record.user_id)
            item_idx = self.encode_item(record.item_id)
            user_ratings[user_idx].append((item_idx, record.rating))
        
        for user_idx, rated_items in user_ratings.items():
            positives = [
                (item, r) for item, r in rated_items
                if r >= self.config.min_rating_for_positive
            ]
            negatives = [
                (item, r) for item, r in rated_items
                if r < self.config.min_rating_for_positive
            ]
            
            if not positives or not negatives:
                continue
            
            context = user_sequences.get(user_idx, [])
            context = context[-self.config.max_seq_len:]
            
            for _ in range(min(len(