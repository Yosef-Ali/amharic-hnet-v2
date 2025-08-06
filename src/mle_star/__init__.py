"""
MLE-STAR Integration for Amharic H-Net
Google MLE-STAR machine learning engineering agent integration

This module provides MLE-STAR capabilities for automated machine learning
engineering tasks specifically adapted for Amharic language processing
and H-Net architecture optimization.

Components:
- Web-based model discovery
- Two-loop refinement system (outer: ablation studies, inner: iterative refinement)
- Ensemble methods with meta-learners
- Integrated performance evaluation system
"""

from .web_model_discovery import (
    WebModelDiscovery,
    ModelDiscoveryResult,
    SearchQuery
)

from .refinement_loops import (
    MLEStarRefinementEngine,
    ComponentVariation,
    AblationResult,
    RefinementResult,
    ComponentInterface,
    ChunkingComponentInterface,
    AttentionComponentInterface
)

from .ensemble_methods import (
    MLEStarEnsembleManager,
    EnsembleCandidate,
    EnsembleWeight,
    CulturalAwareMetaLearner,
    WeightOptimizer,
    DynamicEnsembleSelector
)

from .integrated_evaluation import (
    IntegratedEvaluationSystem,
    MLEStarKaggleStyleEvaluator,
    MLEStarMetrics,
    IntegratedEvaluationResult
)

__version__ = "1.0.0"
__author__ = "MLE-STAR Integration Team"

__all__ = [
    # Web Model Discovery
    "WebModelDiscovery",
    "ModelDiscoveryResult", 
    "SearchQuery",
    
    # Refinement Loops
    "MLEStarRefinementEngine",
    "ComponentVariation",
    "AblationResult", 
    "RefinementResult",
    "ComponentInterface",
    "ChunkingComponentInterface",
    "AttentionComponentInterface",
    
    # Ensemble Methods
    "MLEStarEnsembleManager",
    "EnsembleCandidate",
    "EnsembleWeight",
    "CulturalAwareMetaLearner",
    "WeightOptimizer",
    "DynamicEnsembleSelector",
    
    # Integrated Evaluation
    "IntegratedEvaluationSystem",
    "MLEStarKaggleStyleEvaluator",
    "MLEStarMetrics",
    "IntegratedEvaluationResult"
]