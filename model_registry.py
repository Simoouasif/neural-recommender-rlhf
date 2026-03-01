```python
"""
Model Registry for Neural Recommender RLHF System.
Manages model versioning, storage, loading, and lifecycle management.
"""

import os
import json
import shutil
import hashlib
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class ModelType(Enum):
    """Types of models in the RLHF pipeline."""
    REWARD_MODEL = "reward_model"
    POLICY_MODEL = "policy_model"
    REFERENCE_MODEL = "reference_model"
    VALUE_MODEL = "value_model"
    CRITIC_MODEL = "critic_model"
    EMBEDDING_MODEL = "embedding_model"
    ENSEMBLE = "ensemble"


@dataclass
class ModelMetrics:
    """Performance metrics for a registered model."""
    ndcg_at_10: Optional[float] = None
    ndcg_at_20: Optional[float] = None
    hit_rate_at_10: Optional[float] = None
    hit_rate_at_20: Optional[float] = None
    mrr: Optional[float] = None
    coverage: Optional[float] = None
    diversity: Optional[float] = None
    novelty: Optional[float] = None
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    kl_divergence: Optional[float] = None
    human_preference_score: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput_qps: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetrics":
        custom = data.pop("custom_metrics", {})
        instance = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        instance.custom_metrics = custom
        return instance

    def is_better_than(self, other: "ModelMetrics", primary_metric: str = "ndcg_at_10") -> bool:
        """Compare this model's metrics against another."""
        self_val = getattr(self, primary_metric, None) or self.custom_metrics.get(primary_metric)
        other_val = getattr(other, primary_metric, None) or other.custom_metrics.get(primary_metric)
        if self_val is None or other_val is None:
            return False
        return self_val > other_val


@dataclass
class ModelVersion:
    """Represents a specific version of a registered model."""
    model_id: str
    version: str
    model_type: ModelType
    status: ModelStatus
    created_at: str
    updated_at: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metrics: Optional[ModelMetrics] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    architecture_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    checkpoint_path: Optional[str] = None
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    checksum: Optional[str] = None
    model_size_mb: Optional[float] = None
    framework: str = "pytorch"
    framework_version: str = ""
    created_by: str = "system"
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["model_type"] = self.model_type.value
        data["status"] = self.status.value
        if self.metrics:
            data["metrics"] = self.metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        data = data.copy()
        data["model_type"] = ModelType(data["model_type"])
        data["status"] = ModelStatus(data["status"])
        if data.get("metrics"):
            data["metrics"] = ModelMetrics.from_dict(data["metrics"])
        return cls(**data)

    @property
    def version_id(self) -> str:
        return f"{self.model_id}:{self.version}"

    def is_deployable(self) -> bool:
        return self.status in (ModelStatus.STAGING, ModelStatus.PRODUCTION)


class ModelRegistry:
    """
    Central registry for managing neural recommender RLHF models.
    
    Supports model versioning, lifecycle management, promotion workflows,
    and integration with training pipelines.
    """

    REGISTRY_FILE = "registry.json"
    MODELS_DIR = "models"
    ARTIFACTS_DIR = "artifacts"
    METADATA_FILE = "metadata.json"

    def __init__(
        self,
        registry_root: str = "./model_registry",
        auto_save: bool = True,
        max_versions_per_model: int = 50,
    ):
        self.registry_root = Path(registry_root)
        self.auto_save = auto_save
        self.max_versions_per_model = max_versions_per_model
        self._lock = threading.RLock()
        self._registry: Dict[str, Dict[str, ModelVersion]] = {}
        self._production_models: Dict[str, str] = {}
        self._initialize_storage()
        self._load_registry()

    def _initialize_storage(self) -> None:
        """Create directory structure for registry."""
        dirs = [
            self.registry_root,
            self.registry_root / self.MODELS_DIR,
            self.registry_root / self.ARTIFACTS_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Registry storage initialized at {self.registry_root}")

    def _load_registry(self) -> None:
        """Load registry state from disk."""
        registry_file = self.registry_root / self.REGISTRY_FILE
        if registry_file.exists():
            try:
                with open(registry_file, "r") as f:
                    data = json.load(f)
                self._registry = {}
                for model_id, versions in data.get("models", {}).items():
                    self._registry[model_id] = {
                        v: ModelVersion.from_dict(vdata)
                        for v, vdata in versions.items()
                    }
                self._production_models = data.get("production_models", {})
                logger.info(f"Loaded registry with {len(self._registry)} models")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self._registry = {}
                self._production_models = {}
        else:
            logger.info("No existing registry found, starting fresh")

    def _save_registry(self) -> None:
        """Persist registry state to disk."""
        if not self.auto_save:
            return
        registry_file = self.registry_root / self.REGISTRY_FILE
        data = {
            "models": {
                model_id: {v: mv.to_dict() for v, mv in versions