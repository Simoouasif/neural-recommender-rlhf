```python
"""
Safety filter for neural-recommender-rlhf system.
Filters and validates recommendations for safety and appropriateness.
"""

import re
import logging
from typing import Any, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for safety filtering."""
    
    # Content filtering
    enable_content_filter: bool = True
    enable_toxicity_filter: bool = True
    enable_bias_filter: bool = True
    enable_privacy_filter: bool = True
    
    # Thresholds
    toxicity_threshold: float = 0.7
    bias_threshold: float = 0.6
    confidence_threshold: float = 0.1
    
    # Score bounds
    min_score: float = -10.0
    max_score: float = 10.0
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_recommendations_per_request: int = 100
    
    # Diversity constraints
    enable_diversity_filter: bool = True
    min_diversity_score: float = 0.1
    
    # Blocked categories and keywords
    blocked_categories: list = field(default_factory=list)
    blocked_keywords: list = field(default_factory=list)
    
    # PII patterns
    enable_pii_filter: bool = True


@dataclass
class FilterResult:
    """Result of safety filtering."""
    
    passed: bool
    filtered_scores: Optional[np.ndarray]
    filtered_indices: Optional[np.ndarray]
    violations: list
    warnings: list
    metadata: dict
    
    def __post_init__(self):
        if self.violations is None:
            self.violations = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class ContentFilter:
    """Filters content based on keywords and categories."""
    
    # Common harmful keyword patterns
    DEFAULT_HARMFUL_PATTERNS = [
        r'\b(hate|violence|abuse|harass)\w*\b',
        r'\b(explicit|adult|nsfw)\w*\b',
        r'\b(spam|scam|fraud|phish)\w*\b',
        r'\b(illegal|illicit|contraband)\w*\b',
    ]
    
    # PII patterns
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.harmful_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.DEFAULT_HARMFUL_PATTERNS
        ]
        
        self.blocked_keyword_patterns = [
            re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            for kw in self.config.blocked_keywords
        ]
        
        self.pii_patterns = {
            name: re.compile(pattern)
            for name, pattern in self.PII_PATTERNS.items()
        }
    
    def check_text(self, text: str) -> tuple[bool, list]:
        """
        Check text for content violations.
        
        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        violations = []
        
        if not isinstance(text, str):
            return True, []
        
        # Check harmful patterns
        if self.config.enable_content_filter:
            for pattern in self.harmful_patterns:
                if pattern.search(text):
                    violations.append(f"Harmful content pattern detected: {pattern.pattern}")
        
        # Check blocked keywords
        for pattern in self.blocked_keyword_patterns:
            if pattern.search(text):
                violations.append(f"Blocked keyword detected")
        
        # Check PII
        if self.config.enable_pii_filter:
            for pii_type, pattern in self.pii_patterns.items():
                if pattern.search(text):
                    violations.append(f"PII detected: {pii_type}")
        
        return len(violations) == 0, violations
    
    def check_category(self, category: str) -> tuple[bool, list]:
        """Check if category is blocked."""
        violations = []
        
        if category in self.config.blocked_categories:
            violations.append(f"Blocked category: {category}")
        
        return len(violations) == 0, violations


class ToxicityFilter:
    """
    Toxicity scoring filter.
    
    In production, this would integrate with a toxicity detection model.
    This implementation provides a rule-based approximation.
    """
    
    TOXIC_KEYWORDS = {
        'hate': 0.9,
        'kill': 0.8,
        'attack': 0.6,
        'harm': 0.7,
        'threat': 0.75,
        'abuse': 0.8,
        'harass': 0.8,
        'discriminat': 0.85,
    }
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.threshold = config.toxicity_threshold
    
    def score(self, text: str) -> float:
        """
        Compute toxicity score for text.
        
        Returns:
            Float in [0, 1] where 1 is most toxic
        """
        if not isinstance(text, str) or not text:
            return 0.0
        
        text_lower = text.lower()
        max_score = 0.0
        
        for keyword, score in self.TOXIC_KEYWORDS.items():
            if keyword in text_lower:
                max_score = max(max_score, score)
        
        return max_score
    
    def is_toxic(self, text: str) -> tuple[bool, float]:
        """
        Check if text is toxic.
        
        Returns:
            Tuple of (is_toxic, toxicity_score)
        """
        score = self.score(text)
        return score >= self.threshold, score


class BiasFilter:
    """
    Bias detection filter for recommendations.
    
    Checks for demographic bias, representation bias, etc.
    """
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.threshold = config.bias_threshold
    
    def check_demographic_parity(
        self,
        recommendations: np.ndarray,
        protected_attributes: Optional[dict] = None
    ) -> tuple[bool, list]:
        """
        Check recommendations for demographic parity.
        
        Args:
            recommendations: Array of recommended item indices
            protected_attributes: Dict mapping item index to protected group
            
        Returns:
            Tuple of (passes_check, list_of_warnings)
        """
        warnings = []
        
        if protected_attributes is None or len(recommendations) == 0:
            return True, warnings
        
        # Count representation by group
        group_counts = {}
        total =