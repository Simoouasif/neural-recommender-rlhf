```python
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import logging
from pathlib import Path
import threading
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    EXPLICIT_RATING = "explicit_rating"
    THUMBS_UP_DOWN = "thumbs_up_down"
    CLICK = "click"
    PURCHASE = "purchase"
    SKIP = "skip"
    DWELL_TIME = "dwell_time"
    COMPARISON = "comparison"
    RANKING = "ranking"


class FeedbackSignal(Enum):
    POSITIVE = 1
    NEGATIVE = -1
    NEUTRAL = 0


@dataclass
class RecommendationContext:
    session_id: str
    user_id: str
    timestamp: float
    recommendation_id: str
    item_ids: List[str]
    model_version: str
    context_features: Dict[str, Any] = field(default_factory=dict)
    position_in_list: Optional[int] = None


@dataclass
class UserFeedback:
    feedback_id: str
    feedback_type: FeedbackType
    signal: FeedbackSignal
    recommendation_context: RecommendationContext
    item_id: str
    timestamp: float
    raw_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["feedback_type"] = self.feedback_type.value
        data["signal"] = self.signal.value
        data["recommendation_context"]["item_ids"] = self.recommendation_context.item_ids
        return data


@dataclass
class ComparisonFeedback:
    feedback_id: str
    user_id: str
    session_id: str
    timestamp: float
    preferred_item_id: str
    rejected_item_id: str
    recommendation_context: RecommendationContext
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class RankingFeedback:
    feedback_id: str
    user_id: str
    session_id: str
    timestamp: float
    ranked_items: List[str]
    original_order: List[str]
    recommendation_context: RecommendationContext
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


class FeedbackNormalizer:
    def __init__(self):
        self.rating_scales = {
            5: lambda x: (x - 3) / 2,
            10: lambda x: (x - 5.5) / 4.5,
            100: lambda x: (x - 50) / 50,
        }
        self.dwell_time_thresholds = {
            "short": 5,
            "medium": 30,
            "long": 120,
        }

    def normalize_rating(self, rating: float, scale: int = 5) -> float:
        if scale in self.rating_scales:
            return self.rating_scales[scale](rating)
        return (2 * rating / scale) - 1

    def normalize_dwell_time(self, dwell_seconds: float) -> FeedbackSignal:
        if dwell_seconds < self.dwell_time_thresholds["short"]:
            return FeedbackSignal.NEGATIVE
        elif dwell_seconds < self.dwell_time_thresholds["medium"]:
            return FeedbackSignal.NEUTRAL
        else:
            return FeedbackSignal.POSITIVE

    def normalize_click(self, clicked: bool, position: int = 0) -> float:
        position_discount = 1.0 / (1 + 0.1 * position)
        return (1.0 if clicked else -0.5) * position_discount


class RewardCalculator:
    def __init__(self):
        self.feedback_weights = {
            FeedbackType.PURCHASE: 1.0,
            FeedbackType.EXPLICIT_RATING: 0.8,
            FeedbackType.COMPARISON: 0.7,
            FeedbackType.RANKING: 0.6,
            FeedbackType.DWELL_TIME: 0.5,
            FeedbackType.CLICK: 0.4,
            FeedbackType.THUMBS_UP_DOWN: 0.6,
            FeedbackType.SKIP: 0.3,
        }
        self.decay_factor = 0.95

    def calculate_reward(self, feedback: UserFeedback) -> float:
        weight = self.feedback_weights.get(feedback.feedback_type, 0.5)
        signal_value = feedback.signal.value
        raw_contribution = (feedback.raw_value or 0) * 0.2 if feedback.raw_value else 0
        time_decay = self._apply_time_decay(feedback.timestamp)
        reward = weight * (signal_value + raw_contribution) * time_decay
        return reward

    def calculate_comparison_reward(self, feedback: ComparisonFeedback) -> Dict[str, float]:
        confidence = feedback.confidence
        preferred_reward = confidence * 1.0
        rejected_reward = confidence * -1.0
        return {
            feedback.preferred_item_id: preferred_reward,
            feedback.rejected_item_id: rejected_reward,
        }

    def calculate_ranking_reward(self, feedback: RankingFeedback) -> Dict[str, float]:
        rewards = {}
        n = len(feedback.ranked_items)
        for rank, item_id in enumerate(feedback.ranked_items):
            ideal_reward = (n - rank) / n
            if item_id in feedback.original_order:
                original_rank = feedback.original_order.index(item_id)
                rank_diff = original_rank - rank
                rank_bonus = rank_diff / n
                rewards[item_id] = ideal_reward + rank_bonus * 0.2
            else:
                rewards[item_id] = ideal_reward
        return rewards

    def _apply_time_decay(self, timestamp: float) -> float:
        age_hours = (time.time() - timestamp) / 3600
        return self.decay_factor ** age_hours


class FeedbackStorage:
    def __init__(self, storage_path: str = "feedback_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._feedback_buffer: List[Dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self._buffer_size = 100

    def save_feedback(self, feedback: UserFeedback) -> bool:
        try:
            feedback_dict = feedback.to_dict()
            with self._buffer_lock:
                self._feedback_buffer.append(feedback_dict)
                if len(self._feedback_buffer) >= self._buffer_size:
                    self._flush_buffer()
            return True
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return False

    def save_comparison_feedback(self, feedback: ComparisonFeedback) -> bool:
        try:
            feedback_dict = feedback.to_dict()
            file_path = self.storage_path / "comparison_feedback.jsonl"
            with open(file_path, "a") as f:
                f.write(json.dumps(feedback_dict) + "\n")
            return True
        except Exception as e:
            logger.error