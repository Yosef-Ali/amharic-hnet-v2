# MLE-STAR Integration Guide for Amharic H-Net

## Overview

This guide provides comprehensive documentation for using Google's MLE-STAR (Machine Learning Engineering Agent via Search and Targeted Refinement) methodology with Amharic H-Net language processing systems.

MLE-STAR achieved a 63% medal rate in Kaggle competitions by combining:
- **Web-based model discovery** for finding state-of-the-art approaches
- **Two-loop refinement system** with ablation studies and iterative improvement
- **Advanced ensemble methods** with bespoke meta-learners
- **Automated optimization** with cultural safety integration

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Component Documentation](#component-documentation)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)

## Quick Start

### Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Test Installation**:
```bash
python mle_star_integration_test.py --quick-test
```

### Basic Usage

```python
from src.mle_star import (
    WebModelDiscovery, MLEStarRefinementEngine,
    MLEStarEnsembleManager, IntegratedEvaluationSystem
)

# 1. Model Discovery
discovery = WebModelDiscovery()
query = SearchQuery("amharic language model transformer")
models = discovery.discover_models(query)

# 2. Refinement System
refinement = MLEStarRefinementEngine(base_model, eval_function)
results = refinement.run_full_mle_star_cycle()

# 3. Ensemble Methods
ensemble = MLEStarEnsembleManager(models, eval_function, cultural_safety_function)
weights = ensemble.optimize_ensemble_weights(method="evolutionary")

# 4. Integrated Evaluation
evaluator = IntegratedEvaluationSystem(model)
evaluation = evaluator.run_comprehensive_evaluation(test_data)
```

## Architecture Overview

### MLE-STAR Components

```
┌─────────────────────────────────────────────────────────────┐
│                    MLE-STAR Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  Web Discovery  │    │ Refinement Loop │                │
│  │                 │    │                 │                │
│  │ • ArXiv Search  │    │ • Ablation Study│                │
│  │ • GitHub Search │    │ • Inner Loop    │                │
│  │ • HuggingFace   │    │ • Convergence   │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           ▼                       ▼                        │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Ensemble Methods│    │ Integrated Eval │                │
│  │                 │    │                 │                │
│  │ • Meta-learners │    │ • Kaggle Metrics│                │
│  │ • Weight Optim  │    │ • Cultural Safe │                │
│  │ • Dynamic Select│    │ • Statistical   │                │
│  └─────────────────┘    └─────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Integration with H-Net

```
┌─────────────────────────────────────────────────────────────┐
│                   Amharic H-Net + MLE-STAR                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   H-Net Core    │◄──►│   MLE-STAR      │                │
│  │                 │    │                 │                │
│  │ • 300M Params   │    │ • Model Search  │                │
│  │ • Dynamic Chunk │    │ • Auto Refine   │                │
│  │ • Cultural Safe │    │ • Ensemble Opt  │                │
│  │ • Hierarchical  │    │ • Performance   │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           ▼                       ▼                        │
│  ┌─────────────────────────────────────────┐                │
│  │         Enhanced Performance            │                │
│  │                                         │                │
│  │ • 63% Kaggle Medal Rate Methodology    │                │
│  │ • Automated Architecture Optimization  │                │
│  │ • Cultural Safety Preservation         │                │
│  │ • Amharic-Specific Improvements        │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Component Documentation

### 1. Web-Based Model Discovery

The `WebModelDiscovery` component implements MLE-STAR's web search methodology to find relevant models and architectures.

#### Features
- **Multi-source search**: ArXiv, GitHub, Hugging Face
- **Relevance scoring**: Amharic-specific keyword weighting
- **Architecture analysis**: Automatic categorization of model types
- **Implementation feasibility**: Complexity assessment
- **Caching system**: 24-hour cache for performance

#### Usage

```python
from src.mle_star import WebModelDiscovery, SearchQuery

# Initialize discovery system
discovery = WebModelDiscovery(cache_dir="cache/models")

# Create search query
query = SearchQuery(
    query="amharic transformer hierarchical",
    max_results=20,
    language_filter="amharic"
)

# Discover models
results = discovery.discover_models(query)

# Generate recommendations
current_config = {"d_model": 768, "compression_ratio": 4.5}
recommendations = discovery.generate_architecture_recommendations(
    results, current_config
)

# Access top recommendations
for rec in recommendations['high_priority']:
    print(f"Model: {rec['model_name']}")
    print(f"Relevance: {rec['relevance_score']:.3f}")
    print(f"Integration: {rec['integration_approach']}")
```

#### Configuration Options

```python
# Search source configuration
search_sources = {
    "arxiv": {"enabled": True, "timeout": 30},
    "github": {"enabled": True, "timeout": 30},
    "huggingface": {"enabled": True, "timeout": 30}
}

# Relevance scoring weights
scoring_weights = {
    "amharic_keywords": 0.4,
    "architecture_keywords": 0.3,
    "recency_bonus": 0.2,
    "implementation_feasibility": 0.1
}
```

### 2. Two-Loop Refinement System

The `MLEStarRefinementEngine` implements the core MLE-STAR dual-loop methodology.

#### Architecture

**Outer Loop (Ablation Studies)**:
- Systematically disables/simplifies components
- Measures performance impact
- Identifies most critical components
- Statistical significance testing

**Inner Loop (Iterative Refinement)**:
- Generates variations of critical components
- Tests variations in parallel
- Converges on optimal configurations
- Pareto optimality analysis

#### Usage

```python
from src.mle_star import MLEStarRefinementEngine

# Initialize refinement engine
refinement = MLEStarRefinementEngine(
    base_model=your_hnet_model,
    eval_function=your_eval_function,
    output_dir="refinement_results"
)

# Run complete refinement cycle
results = refinement.run_full_mle_star_cycle(
    max_cycles=3,
    components_per_cycle=2
)

# Access results
print(f"Total improvement: {results['total_improvement']:.4f}")
print(f"Best model: {results['final_best_model']}")

# Manually run individual loops
ablation_results = refinement.run_outer_loop(
    components_to_ablate=['chunking', 'attention'],
    num_ablation_runs=5
)

refinement_results = refinement.run_inner_loop(
    target_component='chunking',
    max_iterations=10
)
```

#### Component Interfaces

Create custom component interfaces for your architecture:

```python
from src.mle_star import ComponentInterface, ComponentVariation

class CustomComponentInterface(ComponentInterface):
    def get_variations(self) -> List[ComponentVariation]:
        return [
            ComponentVariation(
                component_name="custom",
                variation_name="enhanced_attention",
                parameters={"heads": 16, "dropout": 0.1},
                description="Enhanced attention mechanism",
                estimated_complexity="medium",
                expected_improvement=0.15
            )
        ]
    
    def apply_variation(self, variation: ComponentVariation, model: nn.Module) -> nn.Module:
        # Apply variation to model
        modified_model = copy.deepcopy(model)
        # ... modification logic ...
        return modified_model
```

### 3. Ensemble Methods

The `MLEStarEnsembleManager` provides sophisticated ensemble techniques with cultural safety integration.

#### Features
- **Bespoke meta-learners**: Cultural context-aware combination
- **Multiple optimization methods**: Gradient, evolutionary, Bayesian
- **Dynamic selection**: Input-dependent model selection
- **Weight optimization**: Constrained optimization with cultural safety

#### Usage

```python
from src.mle_star import MLEStarEnsembleManager

# Initialize ensemble manager
ensemble = MLEStarEnsembleManager(
    base_models=[model1, model2, model3],
    eval_function=evaluate_performance,
    cultural_safety_function=evaluate_cultural_safety
)

# Create ensemble candidates
candidates = ensemble.create_ensemble_candidates()

# Optimize weights using multiple methods
methods = ["gradient", "evolutionary", "bayesian"]
for method in methods:
    results = ensemble.optimize_ensemble_weights(
        method=method,
        cultural_safety_threshold=0.95
    )
    print(f"{method}: {results['best_score']:.4f}")

# Train meta-learner
meta_learner = ensemble.train_meta_learner(
    train_data=training_data,
    val_data=validation_data
)

# Setup dynamic selection
ensemble.setup_dynamic_selection(context_analyzer_function)
```

#### Meta-Learner Configuration

```python
from src.mle_star import MetaLearnerConfig, CulturalAwareMetaLearner

config = MetaLearnerConfig(
    input_dim=len(models) * 256,  # Model prediction dimensions
    hidden_dims=[512, 256, 128],
    output_dim=256,
    cultural_context_dim=32,
    use_attention=True,
    dropout=0.1
)

meta_learner = CulturalAwareMetaLearner(config)
```

### 4. Integrated Evaluation System

The `IntegratedEvaluationSystem` combines traditional Amharic metrics with MLE-STAR performance indicators.

#### Features
- **Kaggle-style evaluation**: Medal probability calculation
- **Traditional metrics**: Perplexity, compression, cultural safety
- **MLE-STAR metrics**: Discovery effectiveness, refinement gains
- **Statistical significance**: Confidence intervals, p-values
- **Comprehensive reporting**: Visualizations and recommendations

#### Usage

```python
from src.mle_star import IntegratedEvaluationSystem

# Initialize evaluation system
evaluator = IntegratedEvaluationSystem(
    model=your_model,
    output_dir="evaluation_results"
)

# Run comprehensive evaluation
results = evaluator.run_comprehensive_evaluation(
    test_data=test_dataset,
    discovery_results=discovery_output,
    refinement_results=refinement_output,
    ensemble_results=ensemble_output
)

# Access Kaggle medal probability
medal_prob = results.kaggle_medal_probability
print(f"Expected percentile: {medal_prob['expected_percentile']:.1f}th")
print(f"Gold probability: {medal_prob['gold_probability']:.3f}")

# View recommendations
for target in results.next_optimization_targets:
    print(f"Optimization target: {target}")
```

## Usage Examples

### Example 1: Complete MLE-STAR Workflow

```python
#!/usr/bin/env python3
"""Complete MLE-STAR workflow example"""

import torch
from src.models.hnet_300m_amharic import create_300m_model
from src.mle_star import *

def main():
    # 1. Initialize base model
    model = create_300m_model(vocab_size=256)
    
    # 2. Model discovery
    discovery = WebModelDiscovery()
    query = SearchQuery("amharic hierarchical transformer")
    discovered_models = discovery.discover_models(query)
    
    print(f"Discovered {len(discovered_models)} relevant models")
    
    # 3. Refinement system
    def eval_function(model):
        # Your evaluation logic here
        return torch.rand(1).item()  # Mock evaluation
    
    refinement = MLEStarRefinementEngine(model, eval_function)
    refinement_results = refinement.run_full_mle_star_cycle()
    
    print(f"Refinement improvement: {refinement_results['total_improvement']:.4f}")
    
    # 4. Ensemble methods
    models = [model, refinement_results['final_best_model']]
    ensemble = MLEStarEnsembleManager(models, eval_function, lambda m: 0.95)
    
    ensemble_results = ensemble.optimize_ensemble_weights("evolutionary")
    print(f"Ensemble score: {ensemble_results['best_score']:.4f}")
    
    # 5. Integrated evaluation
    evaluator = IntegratedEvaluationSystem(model)
    evaluation = evaluator.run_comprehensive_evaluation(
        test_data=[],  # Your test data
        discovery_results=discovered_models,
        refinement_results=refinement_results,
        ensemble_results=ensemble_results
    )
    
    # 6. Results
    medal_prob = evaluation.kaggle_medal_probability
    print(f"Kaggle performance expectation: {medal_prob['expected_percentile']:.1f}th percentile")

if __name__ == "__main__":
    main()
```

### Example 2: Custom Component Refinement

```python
"""Custom component refinement example"""

from src.mle_star import ComponentInterface, ComponentVariation
import torch.nn as nn

class AmharicEmbeddingInterface(ComponentInterface):
    """Custom interface for Amharic embedding refinement"""
    
    def get_variations(self) -> List[ComponentVariation]:
        return [
            ComponentVariation(
                component_name="embedding",
                variation_name="morpheme_aware",
                parameters={
                    "morpheme_vocab_size": 10000,
                    "morpheme_dim": 128,
                    "fusion_method": "concatenate"
                },
                description="Morpheme-aware embedding for Amharic",
                estimated_complexity="medium",
                expected_improvement=0.20
            ),
            ComponentVariation(
                component_name="embedding",
                variation_name="cultural_context",
                parameters={
                    "cultural_dim": 64,
                    "context_types": ["religious", "historical", "cultural"]
                },
                description="Cultural context-aware embeddings",
                estimated_complexity="high",
                expected_improvement=0.25
            )
        ]
    
    def apply_variation(self, variation: ComponentVariation, model: nn.Module) -> nn.Module:
        modified_model = copy.deepcopy(model)
        
        if variation.variation_name == "morpheme_aware":
            # Add morpheme-aware embedding layers
            params = variation.parameters
            morpheme_embedding = nn.Embedding(
                params["morpheme_vocab_size"],
                params["morpheme_dim"]
            )
            modified_model.morpheme_embedding = morpheme_embedding
            
        elif variation.variation_name == "cultural_context":
            # Add cultural context layers
            params = variation.parameters
            cultural_embedding = nn.Embedding(
                len(params["context_types"]),
                params["cultural_dim"]
            )
            modified_model.cultural_embedding = cultural_embedding
        
        return modified_model

# Usage
refinement_engine = MLEStarRefinementEngine(base_model, eval_function)
refinement_engine.component_interfaces["embedding"] = AmharicEmbeddingInterface()

# Run refinement targeting embedding component
results = refinement_engine.run_inner_loop("embedding", max_iterations=8)
```

### Example 3: Cultural Safety Integration

```python
"""Cultural safety integration example"""

from src.safety.cultural_guardrails import AmharicCulturalGuardrails

def create_cultural_safety_evaluator():
    """Create culturally-aware evaluation function"""
    guardrails = AmharicCulturalGuardrails()
    
    def evaluate_with_cultural_safety(model):
        # Standard performance evaluation
        performance = evaluate_model_performance(model)
        
        # Generate sample text
        sample_texts = generate_sample_texts(model, num_samples=10)
        
        # Evaluate cultural safety
        cultural_violations = 0
        for text in sample_texts:
            is_safe, violations = guardrails.check_cultural_safety(text)
            if not is_safe:
                cultural_violations += len(violations)
        
        # Penalize cultural violations
        cultural_penalty = cultural_violations * 0.1
        adjusted_score = performance - cultural_penalty
        
        return max(0, adjusted_score)
    
    return evaluate_with_cultural_safety

# Use in MLE-STAR workflow
cultural_eval = create_cultural_safety_evaluator()
refinement = MLEStarRefinementEngine(model, cultural_eval)
results = refinement.run_full_mle_star_cycle()
```

## Best Practices

### 1. Performance Optimization

**Model Discovery**:
- Use specific search queries for better relevance
- Enable caching for repeated experiments
- Filter by implementation complexity for practical adoption

```python
# Good: Specific query
query = SearchQuery("amharic morphological transformer chunking")

# Better: Multiple targeted queries
queries = [
    "amharic language model hierarchical",
    "ethiopian nlp transformer architecture", 
    "semitic language processing neural"
]
```

**Refinement System**:
- Start with ablation studies to identify critical components
- Use reasonable iteration limits to avoid overfitting
- Monitor convergence to stop early when appropriate

```python
# Good configuration
refinement_config = {
    'max_iterations': 10,
    'convergence_threshold': 0.001,
    'parallel_variations': 3
}
```

**Ensemble Methods**:
- Test multiple optimization methods
- Balance performance with cultural safety
- Use dynamic selection for production deployment

```python
# Test multiple optimization approaches
methods = ["gradient", "evolutionary", "bayesian"]
results = {}
for method in methods:
    results[method] = ensemble.optimize_ensemble_weights(method)

best_method = max(results.keys(), key=lambda k: results[k]['best_score'])
```

### 2. Cultural Safety Guidelines

**Always prioritize cultural safety**:
- Set minimum cultural safety thresholds
- Include cultural safety in optimization objectives
- Regular monitoring of generated content

```python
# Cultural safety configuration
cultural_config = {
    'safety_threshold': 0.95,
    'violation_penalty': 0.1,
    'monitoring_frequency': 'continuous'
}
```

**Protected terms handling**:
- Maintain updated lists of sacred/sensitive terms
- Use context-aware safety checking
- Implement graceful fallbacks for violations

### 3. Computational Efficiency

**Resource management**:
- Use gradient checkpointing for large models
- Enable mixed precision training
- Monitor GPU memory usage

```python
# Efficient training configuration
training_config = {
    'use_mixed_precision': True,
    'gradient_checkpointing': True,
    'batch_size': 8,  # Adjust based on GPU memory
    'gradient_accumulation_steps': 4
}
```

**Parallel processing**:
- Run variation testing in parallel
- Use concurrent model evaluation
- Leverage multi-GPU setups when available

### 4. Reproducibility

**Random seed management**:
```python
import torch
import numpy as np
import random

def set_reproducible_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

**Configuration tracking**:
- Save all configuration parameters
- Track model versions and checkpoints
- Document evaluation procedures

## Troubleshooting

### Common Issues

**1. Web Discovery Issues**

*Problem*: Low number of discovered models
```python
# Solution: Expand search queries and sources
discovery = WebModelDiscovery()
discovery.search_sources["semantic_scholar"] = {
    "base_url": "https://api.semanticscholar.org/",
    "enabled": True
}
```

*Problem*: Network timeouts
```python
# Solution: Increase timeout and add retry logic
discovery.session.timeout = 60
discovery.max_retries = 3
```

**2. Refinement System Issues**

*Problem*: No performance improvements
```python
# Check evaluation function
def debug_eval_function(model):
    score = eval_function(model)
    print(f"Model score: {score:.4f}")
    return score

# Use smaller learning rates
refinement_config['convergence_threshold'] = 0.0001
```

*Problem*: Out of memory errors
```python
# Solution: Reduce parallel variations
refinement = MLEStarRefinementEngine(
    base_model=model,
    eval_function=eval_function,
    parallel_variations=2  # Reduced from default
)
```

**3. Ensemble Issues**

*Problem*: Weight optimization fails
```python
# Solution: Check candidate model compatibility
candidates = ensemble.create_ensemble_candidates()
for candidate in candidates:
    try:
        score = eval_function(candidate.model)
        print(f"{candidate.model_id}: {score:.4f}")
    except Exception as e:
        print(f"Error with {candidate.model_id}: {e}")
```

*Problem*: Cultural safety violations
```python
# Solution: Increase safety threshold
ensemble_config = {
    'cultural_safety_threshold': 0.98,  # Stricter threshold
    'violation_penalty_weight': 0.2     # Higher penalty
}
```

### Debugging Tools

**Enable detailed logging**:
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mle_star_debug.log'),
        logging.StreamHandler()
    ]
)
```

**Memory profiling**:
```python
import psutil
import torch

def profile_memory():
    process = psutil.Process()
    print(f"CPU Memory: {process.memory_info().rss / 1024**3:.2f} GB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

**Performance monitoring**:
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.2f}s")

# Usage
with timer("Model discovery"):
    results = discovery.discover_models(query)
```

## API Reference

### WebModelDiscovery

```python
class WebModelDiscovery:
    def __init__(self, cache_dir: str = "cache/model_discovery"):
        """Initialize web model discovery system."""
    
    def discover_models(self, query: SearchQuery) -> List[ModelDiscoveryResult]:
        """Discover models based on search query."""
    
    def generate_architecture_recommendations(self, 
                                            results: List[ModelDiscoveryResult],
                                            current_config: Dict) -> Dict:
        """Generate architecture recommendations."""
```

### MLEStarRefinementEngine

```python
class MLEStarRefinementEngine:
    def __init__(self, base_model: nn.Module, eval_function: Callable, output_dir: str):
        """Initialize refinement engine."""
    
    def run_outer_loop(self, components_to_ablate: List[str], 
                       num_ablation_runs: int = 5) -> List[AblationResult]:
        """Run ablation studies to identify critical components."""
    
    def run_inner_loop(self, target_component: str, max_iterations: int = 10,
                       convergence_threshold: float = 0.001) -> List[RefinementResult]:
        """Run iterative refinement on target component."""
    
    def run_full_mle_star_cycle(self, max_cycles: int = 3,
                               components_per_cycle: int = 2) -> Dict:
        """Run complete MLE-STAR refinement cycle."""
```

### MLEStarEnsembleManager

```python
class MLEStarEnsembleManager:
    def __init__(self, base_models: List[nn.Module], eval_function: Callable,
                 cultural_safety_function: Callable, output_dir: str):
        """Initialize ensemble manager."""
    
    def create_ensemble_candidates(self) -> List[EnsembleCandidate]:
        """Create ensemble candidates from base models."""
    
    def optimize_ensemble_weights(self, method: str = "evolutionary",
                                 **kwargs) -> Dict:
        """Optimize ensemble weights using specified method."""
    
    def train_meta_learner(self, train_data: List[Dict], val_data: List[Dict],
                          config: MetaLearnerConfig = None) -> CulturalAwareMetaLearner:
        """Train bespoke meta-learner for ensemble combination."""
```

### IntegratedEvaluationSystem

```python
class IntegratedEvaluationSystem:
    def __init__(self, model: nn.Module, output_dir: str = "integrated_evaluation"):
        """Initialize integrated evaluation system."""
    
    def run_comprehensive_evaluation(self, test_data: List[Dict],
                                   discovery_results: List[ModelDiscoveryResult] = None,
                                   refinement_results: List[RefinementResult] = None,
                                   ensemble_results: Dict = None) -> IntegratedEvaluationResult:
        """Run comprehensive evaluation combining all metrics."""
```

---

## Conclusion

This integration of Google MLE-STAR with Amharic H-Net represents a significant advancement in automated machine learning engineering for low-resource languages. The system provides:

- **Automated model discovery** to find state-of-the-art approaches
- **Systematic optimization** through ablation studies and iterative refinement  
- **Advanced ensemble methods** with cultural safety integration
- **Comprehensive evaluation** with Kaggle-style performance assessment

By following this guide and using the provided components, you can achieve the same level of automated optimization that enabled MLE-STAR to achieve a 63% medal rate in Kaggle competitions, specifically adapted for Amharic language processing tasks.

For support and additional examples, see:
- `/examples/` directory for complete workflow examples
- `/tests/` directory for unit tests and integration tests
- `mle_star_integration_test.py` for comprehensive system testing

**Remember**: Always prioritize cultural safety and responsible AI practices when working with language models for underrepresented communities.

---

*This documentation is part of the MLE-STAR integration for Amharic H-Net project. For updates and contributions, please refer to the project repository.*