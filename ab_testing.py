```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import hashlib
import random
from enum import Enum
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class VariantType(Enum):
    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class Variant:
    name: str
    variant_type: VariantType
    model_config: Dict[str, Any]
    traffic_percentage: float
    description: str = ""
    
    def __post_init__(self):
        if not 0 < self.traffic_percentage <= 100:
            raise ValueError(f"Traffic percentage must be between 0 and 100, got {self.traffic_percentage}")


@dataclass
class MetricResult:
    metric_name: str
    variant_name: str
    values: List[float] = field(default_factory=list)
    
    @property
    def mean(self) -> float:
        return np.mean(self.values) if self.values else 0.0
    
    @property
    def std(self) -> float:
        return np.std(self.values) if self.values else 0.0
    
    @property
    def count(self) -> int:
        return len(self.values)
    
    @property
    def sem(self) -> float:
        return stats.sem(self.values) if len(self.values) > 1 else 0.0


@dataclass
class StatisticalTestResult:
    metric_name: str
    control_variant: str
    treatment_variant: str
    test_type: str
    statistic: float
    p_value: float
    confidence_level: float
    is_significant: bool
    effect_size: float
    relative_improvement: float
    confidence_interval: Tuple[float, float]
    power: float
    sample_size_control: int
    sample_size_treatment: int
    

@dataclass
class ExperimentConfig:
    experiment_id: str
    name: str
    description: str
    variants: List[Variant]
    metrics: List[str]
    primary_metric: str
    confidence_level: float = 0.95
    minimum_detectable_effect: float = 0.05
    minimum_sample_size: int = 1000
    max_duration_days: int = 30
    sequential_testing: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        total_traffic = sum(v.traffic_percentage for v in self.variants)
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Total traffic percentage must equal 100, got {total_traffic}")
        
        if self.primary_metric not in self.metrics:
            raise ValueError(f"Primary metric '{self.primary_metric}' must be in metrics list")


class UserAssignment:
    def __init__(self, experiment_id: str, variants: List[Variant], salt: str = ""):
        self.experiment_id = experiment_id
        self.variants = variants
        self.salt = salt
        self._build_assignment_buckets()
    
    def _build_assignment_buckets(self):
        self.buckets = []
        cumulative = 0
        for variant in self.variants:
            cumulative += variant.traffic_percentage
            self.buckets.append((cumulative, variant.name))
    
    def assign_user(self, user_id: str) -> str:
        hash_key = f"{self.experiment_id}_{user_id}_{self.salt}"
        hash_value = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 100.0
        
        for threshold, variant_name in self.buckets:
            if bucket < threshold:
                return variant_name
        
        return self.buckets[-1][1]


class MetricsCollector:
    def __init__(self):
        self.metrics: Dict[str, Dict[str, MetricResult]] = defaultdict(dict)
        self.events: List[Dict[str, Any]] = []
    
    def record_event(
        self,
        experiment_id: str,
        variant_name: str,
        user_id: str,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ):
        if timestamp is None:
            timestamp = datetime.now()
        
        event = {
            "experiment_id": experiment_id,
            "variant_name": variant_name,
            "user_id": user_id,
            "metric_name": metric_name,
            "value": value,
            "timestamp": timestamp.isoformat()
        }
        self.events.append(event)
        
        key = f"{experiment_id}_{variant_name}"
        if metric_name not in self.metrics[key]:
            self.metrics[key][metric_name] = MetricResult(
                metric_name=metric_name,
                variant_name=variant_name
            )
        self.metrics[key][metric_name].values.append(value)
        
        logger.debug(f"Recorded event: {event}")
    
    def get_metric_result(
        self,
        experiment_id: str,
        variant_name: str,
        metric_name: str
    ) -> Optional[MetricResult]:
        key = f"{experiment_id}_{variant_name}"
        return self.metrics.get(key, {}).get(metric_name)
    
    def get_all_metrics(
        self,
        experiment_id: str
    ) -> Dict[str, Dict[str, MetricResult]]:
        result = {}
        for key, metrics in self.metrics.items():
            if key.startswith(f"{experiment_id}_"):
                variant_name = key[len(f"{experiment_id}_"):]
                result[variant_name] = metrics
        return result
    
    def to_dataframe(self, experiment_id: str) -> pd.DataFrame:
        filtered_events = [
            e for e in self.events 
            if e["experiment_id"] == experiment_id
        ]
        return pd.DataFrame(filtered_events)


class StatisticalAnalyzer:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def t_test(
        self,
        control_result: MetricResult,
        treatment_result: MetricResult,
        alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        control_values = np.array(control_result.values)
        treatment_values = np.array(treatment_result.values)
        
        if len(control_values) < 2 or len(treatment_values) < 2:
            raise ValueError("Need at least 2 samples for t-test")
        
        t_stat, p_value = stats.ttest_ind(
            control_values,
            treatment_values,
            alternative=alternative,
            equal_var=False
        )
        
        is_significant = p_value < self.alpha
        
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        effect_size = self._cohens_d(control_values, treatment_values)
        
        relative_improvement = (
            (treatment_mean - control_mean) / abs(control_mean)
            if control_mean != 0 else 0.0
        )