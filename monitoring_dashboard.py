```python
#!/usr/bin/env python3
"""
Monitoring Dashboard for Neural Recommender RLHF System
Real-time monitoring of training metrics, reward model performance, and system health
"""

import os
import sys
import time
import json
import logging
import threading
import datetime
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import psutil

# Try to import optional dependencies
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time"""
    timestamp: float
    epoch: int
    step: int
    
    # Training metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    reward_loss: float = 0.0
    total_loss: float = 0.0
    
    # RLHF specific metrics
    mean_reward: float = 0.0
    reward_std: float = 0.0
    kl_divergence: float = 0.0
    entropy: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    
    # Reward model metrics
    reward_accuracy: float = 0.0
    reward_correlation: float = 0.0
    preference_accuracy: float = 0.0
    
    # Recommendation metrics
    ndcg_at_10: float = 0.0
    hit_rate_at_10: float = 0.0
    mrr: float = 0.0
    diversity_score: float = 0.0
    novelty_score: float = 0.0
    coverage: float = 0.0
    
    # Human feedback metrics
    feedback_count: int = 0
    positive_feedback_ratio: float = 0.0
    feedback_quality_score: float = 0.0
    
    # System metrics
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    throughput_samples_per_sec: float = 0.0
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Gradient metrics
    gradient_norm: float = 0.0
    gradient_max: float = 0.0


@dataclass
class AlertConfig:
    """Configuration for monitoring alerts"""
    kl_divergence_max: float = 0.5
    reward_drop_threshold: float = 0.1
    gradient_norm_max: float = 10.0
    gpu_memory_threshold: float = 0.95
    cpu_threshold: float = 0.90
    ram_threshold: float = 0.90
    min_feedback_quality: float = 0.3
    ndcg_drop_threshold: float = 0.05


@dataclass
class Alert:
    """Monitoring alert"""
    severity: str  # 'info', 'warning', 'critical'
    metric: str
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


class MetricsBuffer:
    """Thread-safe circular buffer for metrics"""
    
    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self._buffer: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
    
    def append(self, snapshot: MetricSnapshot):
        with self._lock:
            self._buffer.append(snapshot)
    
    def get_recent(self, n: int = 100) -> List[MetricSnapshot]:
        with self._lock:
            buffer_list = list(self._buffer)
            return buffer_list[-n:] if len(buffer_list) >= n else buffer_list
    
    def get_all(self) -> List[MetricSnapshot]:
        with self._lock:
            return list(self._buffer)
    
    def get_latest(self) -> Optional[MetricSnapshot]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None
    
    def __len__(self):
        with self._lock:
            return len(self._buffer)


class AlertManager:
    """Manages monitoring alerts"""
    
    def __init__(self, config: AlertConfig, max_alerts: int = 100):
        self.config = config
        self.alerts: deque = deque(maxlen=max_alerts)
        self._lock = threading.Lock()
        self.alert_callbacks: List = []
        self._alert_counts: Dict[str, int] = defaultdict(int)
        self._cooldown: Dict[str, float] = {}
        self.cooldown_period: float = 60.0  # seconds
    
    def add_callback(self, callback):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def _should_alert(self, metric: str) -> bool:
        """Check if we should alert (cooldown logic)"""
        now = time.time()
        last_alert = self._cooldown.get(metric, 0)
        if now - last_alert > self.cooldown_period:
            self._cooldown[metric] = now
            return True
        return False
    
    def check_metrics(self, snapshot: MetricSnapshot) -> List[Alert]:
        """Check metrics against thresholds and generate alerts"""
        new_alerts = []
        
        # KL divergence check
        if snapshot.kl_divergence > self.config.kl_divergence_max:
            if self._should_alert('kl_divergence'):
                alert = Alert(
                    severity='warning' if snapshot.kl_divergence < self.config.kl_divergence_max * 2 else 'critical',
                    metric='kl_divergence',
                    message=f'KL divergence too high: {snapshot.kl_divergence:.4f}',
                    value=snapshot.kl_divergence,
                    threshold=self.config.kl_divergence_max
                )
                new_alerts.append(alert)
        
        # Gradient norm check
        if snapshot.gradient_norm > self.config.gradient_norm_max:
            if self._should_alert('gradient_norm'):
                alert = Alert(
                    severity='warning',
                    metric='gradient_norm',
                    message=f'Gradient norm exploding: {snapshot.gradient_norm:.4f}',
                    value=snapshot.gradient_norm,
                    threshold=self.config.gradient_norm_max
                )
                new_alerts.append(alert)
        
        # GPU memory check
        if snapshot.gpu_memory_total > 0:
            gpu_util = snapshot.gpu_memory_used / snapshot.gpu_memory_total
            if gpu_util > self.config.gpu_memory_threshold:
                if self._should_alert('gpu_memory'):
                    alert = Alert(
                        severity='critical' if gpu_util > 0.98