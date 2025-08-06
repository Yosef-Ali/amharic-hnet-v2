#!/usr/bin/env python3
"""
MLE-STAR Two-Loop Refinement System
Implementation of Google MLE-STAR's dual-loop refinement methodology

This module implements the core MLE-STAR refinement approach:
- Outer Loop: Ablation studies to identify performance-critical components
- Inner Loop: Iterative generation and testing of component variations

Adapted for Amharic H-Net architecture optimization with cultural safety integration.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import copy
import time
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import concurrent.futures
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComponentVariation:
    """Container for component variation configuration."""
    component_name: str
    variation_name: str
    parameters: Dict[str, Any]
    description: str
    estimated_complexity: str  # "low", "medium", "high"
    expected_improvement: float  # 0.0 to 1.0


@dataclass
class AblationResult:
    """Container for ablation study results."""
    component_name: str
    baseline_performance: float
    ablated_performance: float
    performance_impact: float  # baseline - ablated
    relative_impact: float     # impact / baseline
    confidence_interval: Tuple[float, float]
    statistical_significance: float  # p-value


@dataclass
class RefinementResult:
    """Container for refinement iteration results."""
    iteration: int
    variation: ComponentVariation
    performance: float
    improvement: float
    training_time: float
    convergence_metrics: Dict[str, float]
    cultural_safety_score: float
    is_pareto_optimal: bool = False


class ComponentInterface(ABC):
    """Abstract interface for components that can be refined."""
    
    @abstractmethod
    def get_variations(self) -> List[ComponentVariation]:
        """Return list of possible variations for this component."""
        pass
    
    @abstractmethod
    def apply_variation(self, variation: ComponentVariation, model: nn.Module) -> nn.Module:
        """Apply variation to model and return modified model."""
        pass
    
    @abstractmethod
    def get_baseline_config(self) -> Dict[str, Any]:
        """Return baseline configuration for this component."""
        pass


class ChunkingComponentInterface(ComponentInterface):
    """Interface for dynamic chunking component variations."""
    
    def get_variations(self) -> List[ComponentVariation]:
        """Generate variations for chunking component."""
        variations = [
            ComponentVariation(
                component_name="chunking",
                variation_name="boundary_threshold_adaptive",
                parameters={"adaptive_threshold": True, "threshold_range": [0.3, 0.7]},
                description="Adaptive boundary threshold based on content complexity",
                estimated_complexity="medium",
                expected_improvement=0.15
            ),
            ComponentVariation(
                component_name="chunking",
                variation_name="multi_scale_boundary_detection",
                parameters={"scales": [1, 2, 4, 8], "fusion_method": "attention"},
                description="Multi-scale boundary detection with attention fusion",
                estimated_complexity="high",
                expected_improvement=0.25
            ),
            ComponentVariation(
                component_name="chunking",
                variation_name="morpheme_aware_chunking",
                parameters={"morpheme_weights": True, "boundary_bias": 0.2},
                description="Amharic morpheme-aware boundary detection",
                estimated_complexity="medium",
                expected_improvement=0.30
            ),
            ComponentVariation(
                component_name="chunking",
                variation_name="compression_ratio_dynamic",
                parameters={"dynamic_ratio": True, "min_ratio": 2.0, "max_ratio": 8.0},
                description="Dynamic compression ratio based on content type",
                estimated_complexity="low",
                expected_improvement=0.10
            )
        ]
        return variations
    
    def apply_variation(self, variation: ComponentVariation, model: nn.Module) -> nn.Module:
        """Apply chunking variation to model."""
        # This would modify the DynamicSemanticChunker in the actual model
        modified_model = copy.deepcopy(model)
        
        if variation.variation_name == "boundary_threshold_adaptive":
            # Implement adaptive threshold logic
            if hasattr(modified_model, 'dynamic_chunker'):
                modified_model.dynamic_chunker.use_adaptive_threshold = True
                modified_model.dynamic_chunker.threshold_range = variation.parameters["threshold_range"]
        
        elif variation.variation_name == "multi_scale_boundary_detection":
            # Implement multi-scale detection
            if hasattr(modified_model, 'dynamic_chunker'):
                modified_model.dynamic_chunker.detection_scales = variation.parameters["scales"]
                modified_model.dynamic_chunker.fusion_method = variation.parameters["fusion_method"]
        
        elif variation.variation_name == "morpheme_aware_chunking":
            # Implement morpheme-aware processing
            if hasattr(modified_model, 'dynamic_chunker'):
                modified_model.dynamic_chunker.morpheme_aware = True
                modified_model.dynamic_chunker.boundary_bias = variation.parameters["boundary_bias"]
        
        return modified_model
    
    def get_baseline_config(self) -> Dict[str, Any]:
        """Return baseline chunking configuration."""
        return {
            "boundary_threshold": 0.5,
            "detection_scales": [1, 2, 4],
            "morpheme_aware": False,
            "dynamic_ratio": False
        }


class AttentionComponentInterface(ComponentInterface):
    """Interface for attention mechanism variations."""
    
    def get_variations(self) -> List[ComponentVariation]:
        """Generate variations for attention component."""
        variations = [
            ComponentVariation(
                component_name="attention",
                variation_name="flash_attention_optimized",
                parameters={"use_flash_attention": True, "memory_efficient": True},
                description="Flash attention with memory optimization",
                estimated_complexity="medium",
                expected_improvement=0.20
            ),
            ComponentVariation(
                component_name="attention",
                variation_name="hierarchical_attention",
                parameters={"hierarchical_levels": 3, "cross_level_attention": True},
                description="Multi-level hierarchical attention mechanism",
                estimated_complexity="high",
                expected_improvement=0.25
            ),
            ComponentVariation(
                component_name="attention",
                variation_name="cultural_aware_attention",
                parameters={"cultural_bias_correction": True, "amharic_position_encoding": True},
                description="Culturally-aware attention for Amharic text",
                estimated_complexity="medium",
                expected_improvement=0.18
            )
        ]
        return variations
    
    def apply_variation(self, variation: ComponentVariation, model: nn.Module) -> nn.Module:
        """Apply attention variation to model."""
        modified_model = copy.deepcopy(model)
        
        if variation.variation_name == "flash_attention_optimized":
            # Enable flash attention if available
            if hasattr(modified_model, 'hierarchical_backbone'):
                for layer in modified_model.hierarchical_backbone.transformer_layers:
                    if hasattr(layer, 'self_attn'):
                        layer.self_attn.use_flash_attention = True
        
        return modified_model
    
    def get_baseline_config(self) -> Dict[str, Any]:
        """Return baseline attention configuration."""
        return {
            "use_flash_attention": False,
            "hierarchical_levels": 1,
            "cultural_aware": False
        }


class MLEStarRefinementEngine:
    """
    MLE-STAR Two-Loop Refinement Engine
    
    Implements the dual-loop methodology:
    1. Outer Loop: Systematic ablation studies to identify critical components
    2. Inner Loop: Iterative refinement of the most impactful component
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 eval_function: Callable,
                 output_dir: str = "mle_star_refinement"):
        """
        Initialize MLE-STAR refinement engine.
        
        Args:
            base_model: Base H-Net model to refine
            eval_function: Function to evaluate model performance
            output_dir: Directory to save refinement results
        """
        self.base_model = base_model
        self.eval_function = eval_function
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Component interfaces
        self.component_interfaces = {
            "chunking": ChunkingComponentInterface(),
            "attention": AttentionComponentInterface()
        }
        
        # Refinement state
        self.baseline_performance = None
        self.ablation_results = []
        self.refinement_history = []
        self.best_model = None
        self.best_performance = 0.0
        
        # Thread safety
        self.lock = Lock()
        
        logger.info(f"MLE-STAR Refinement Engine initialized")
        logger.info(f"Available components: {list(self.component_interfaces.keys())}")
    
    def run_outer_loop(self, 
                      components_to_ablate: Optional[List[str]] = None,
                      num_ablation_runs: int = 5) -> List[AblationResult]:
        """
        Run outer loop: ablation studies to identify critical components.
        
        Args:
            components_to_ablate: List of component names to ablate (None for all)
            num_ablation_runs: Number of runs per ablation for statistical significance
            
        Returns:
            List of ablation results ranked by impact
        """
        logger.info("🔄 Starting MLE-STAR Outer Loop (Ablation Studies)")
        
        if components_to_ablate is None:
            components_to_ablate = list(self.component_interfaces.keys())
        
        # Establish baseline performance
        if self.baseline_performance is None:
            logger.info("Establishing baseline performance...")
            self.baseline_performance = self._evaluate_model_performance(self.base_model)
            logger.info(f"Baseline performance: {self.baseline_performance:.4f}")
        
        ablation_results = []
        
        for component_name in components_to_ablate:
            logger.info(f"Running ablation study for component: {component_name}")
            
            ablation_performances = []
            
            # Run multiple ablation experiments for statistical significance
            for run in range(num_ablation_runs):
                logger.info(f"  Ablation run {run + 1}/{num_ablation_runs}")
                
                # Create ablated model (component disabled/simplified)
                ablated_model = self._create_ablated_model(component_name)
                
                # Evaluate ablated model
                ablated_performance = self._evaluate_model_performance(ablated_model)
                ablation_performances.append(ablated_performance)
                
                logger.info(f"    Performance: {ablated_performance:.4f}")
            
            # Calculate statistics
            mean_ablated_perf = np.mean(ablation_performances)
            std_ablated_perf = np.std(ablation_performances)
            
            performance_impact = self.baseline_performance - mean_ablated_perf
            relative_impact = performance_impact / self.baseline_performance if self.baseline_performance > 0 else 0
            
            # Calculate confidence interval
            confidence_interval = (
                mean_ablated_perf - 1.96 * std_ablated_perf / np.sqrt(num_ablation_runs),
                mean_ablated_perf + 1.96 * std_ablated_perf / np.sqrt(num_ablation_runs)
            )
            
            # Statistical significance (simplified t-test)
            t_stat = performance_impact / (std_ablated_perf / np.sqrt(num_ablation_runs)) if std_ablated_perf > 0 else 0
            p_value = 2 * (1 - self._approximate_t_cdf(abs(t_stat), num_ablation_runs - 1))
            
            ablation_result = AblationResult(
                component_name=component_name,
                baseline_performance=self.baseline_performance,
                ablated_performance=mean_ablated_perf,
                performance_impact=performance_impact,
                relative_impact=relative_impact,
                confidence_interval=confidence_interval,
                statistical_significance=p_value
            )
            
            ablation_results.append(ablation_result)
            
            logger.info(f"  Impact: {performance_impact:.4f} ({relative_impact:.2%})")
            logger.info(f"  Statistical significance: p={p_value:.4f}")
        
        # Sort by impact (descending)
        ablation_results.sort(key=lambda x: x.performance_impact, reverse=True)
        self.ablation_results = ablation_results
        
        # Save ablation results
        self._save_ablation_results(ablation_results)
        
        logger.info("🎯 Outer Loop completed")
        logger.info(f"Most impactful component: {ablation_results[0].component_name} "
                   f"(impact: {ablation_results[0].performance_impact:.4f})")
        
        return ablation_results
    
    def run_inner_loop(self, 
                      target_component: str,
                      max_iterations: int = 10,
                      convergence_threshold: float = 0.001,
                      parallel_variations: int = 3) -> List[RefinementResult]:
        """
        Run inner loop: iterative refinement of target component.
        
        Args:
            target_component: Component to refine
            max_iterations: Maximum refinement iterations
            convergence_threshold: Convergence threshold for improvement
            parallel_variations: Number of variations to test in parallel
            
        Returns:
            List of refinement results
        """
        logger.info(f"🔧 Starting MLE-STAR Inner Loop for component: {target_component}")
        
        if target_component not in self.component_interfaces:
            raise ValueError(f"Unknown component: {target_component}")
        
        component_interface = self.component_interfaces[target_component]
        available_variations = component_interface.get_variations()
        
        logger.info(f"Available variations: {len(available_variations)}")
        
        refinement_results = []
        current_best_model = copy.deepcopy(self.base_model)
        current_best_performance = self.baseline_performance
        
        for iteration in range(max_iterations):
            logger.info(f"Inner Loop Iteration {iteration + 1}/{max_iterations}")
            
            # Select top variations to test (based on expected improvement and complexity)
            variations_to_test = self._select_variations_for_iteration(
                available_variations, 
                parallel_variations,
                iteration
            )
            
            # Test variations in parallel
            iteration_results = self._test_variations_parallel(
                variations_to_test,
                current_best_model,
                iteration
            )
            
            refinement_results.extend(iteration_results)
            
            # Find best variation in this iteration
            best_iteration_result = max(iteration_results, key=lambda x: x.performance)
            
            logger.info(f"  Best variation: {best_iteration_result.variation.variation_name}")
            logger.info(f"  Performance: {best_iteration_result.performance:.4f}")
            logger.info(f"  Improvement: {best_iteration_result.improvement:.4f}")
            
            # Check for improvement
            if best_iteration_result.performance > current_best_performance:
                improvement = best_iteration_result.performance - current_best_performance
                
                if improvement > convergence_threshold:
                    # Update current best
                    current_best_model = self._apply_variation_to_model(
                        best_iteration_result.variation,
                        current_best_model
                    )
                    current_best_performance = best_iteration_result.performance
                    
                    logger.info(f"  ✅ New best model found! Improvement: {improvement:.4f}")
                    
                    # Update global best if better
                    if current_best_performance > self.best_performance:
                        self.best_model = copy.deepcopy(current_best_model)
                        self.best_performance = current_best_performance
                        logger.info(f"  🏆 Global best updated: {self.best_performance:.4f}")
                else:
                    logger.info(f"  Improvement below threshold ({convergence_threshold}), converged")
                    break
            else:
                logger.info(f"  No improvement found in this iteration")
                
                # Check for early stopping
                if iteration > 3:  # Allow some iterations before early stopping
                    recent_improvements = [r.improvement for r in refinement_results[-parallel_variations:]]
                    if all(imp <= convergence_threshold for imp in recent_improvements):
                        logger.info(f"  Early stopping: no significant improvements")
                        break
        
        # Identify Pareto optimal solutions
        self._identify_pareto_optimal_solutions(refinement_results)
        
        # Save refinement results
        self._save_refinement_results(refinement_results)
        
        self.refinement_history.extend(refinement_results)
        
        logger.info("🎯 Inner Loop completed")
        logger.info(f"Best performance achieved: {self.best_performance:.4f}")
        
        return refinement_results
    
    def run_full_mle_star_cycle(self, 
                               max_cycles: int = 3,
                               components_per_cycle: int = 2) -> Dict[str, Any]:
        """
        Run complete MLE-STAR refinement cycle.
        
        Args:
            max_cycles: Maximum number of outer-inner loop cycles
            components_per_cycle: Number of top components to refine per cycle
            
        Returns:
            Complete refinement results
        """
        logger.info("🚀 Starting Full MLE-STAR Refinement Cycle")
        
        cycle_results = {
            'cycles': [],
            'final_best_model': None,
            'final_performance': 0.0,
            'total_improvement': 0.0,
            'refinement_summary': {}
        }
        
        initial_performance = self.baseline_performance or self._evaluate_model_performance(self.base_model)
        
        for cycle in range(max_cycles):
            logger.info(f"🔄 MLE-STAR Cycle {cycle + 1}/{max_cycles}")
            
            cycle_start_time = time.time()
            
            # Outer Loop: Ablation Studies
            ablation_results = self.run_outer_loop()
            
            # Select top components for refinement
            top_components = [r.component_name for r in ablation_results[:components_per_cycle]]
            
            cycle_refinement_results = []
            
            # Inner Loop: Refine each top component
            for component in top_components:
                logger.info(f"Refining component: {component}")
                component_results = self.run_inner_loop(component)
                cycle_refinement_results.extend(component_results)
            
            cycle_time = time.time() - cycle_start_time
            
            # Record cycle results
            cycle_result = {
                'cycle_number': cycle + 1,
                'ablation_results': ablation_results,
                'refinement_results': cycle_refinement_results,
                'best_performance_in_cycle': max([r.performance for r in cycle_refinement_results]),
                'cycle_time': cycle_time
            }
            
            cycle_results['cycles'].append(cycle_result)
            
            # Check for convergence across cycles
            if cycle > 0:
                prev_best = cycle_results['cycles'][-2]['best_performance_in_cycle']
                current_best = cycle_result['best_performance_in_cycle']
                cycle_improvement = current_best - prev_best
                
                if cycle_improvement < 0.001:  # Minimal improvement threshold
                    logger.info("Convergence achieved across cycles")
                    break
        
        # Finalize results
        cycle_results['final_best_model'] = self.best_model
        cycle_results['final_performance'] = self.best_performance
        cycle_results['total_improvement'] = self.best_performance - initial_performance
        cycle_results['refinement_summary'] = self._generate_refinement_summary()
        
        # Generate comprehensive report
        self._generate_final_report(cycle_results)
        
        logger.info("🏁 Full MLE-STAR Cycle completed")
        logger.info(f"Total improvement: {cycle_results['total_improvement']:.4f}")
        
        return cycle_results
    
    def _create_ablated_model(self, component_name: str) -> nn.Module:
        """Create model with specified component ablated (disabled/simplified)."""
        ablated_model = copy.deepcopy(self.base_model)
        
        # Simplified ablation - could be more sophisticated
        if component_name == "chunking" and hasattr(ablated_model, 'dynamic_chunker'):
            # Disable dynamic chunking, use fixed chunking
            ablated_model.dynamic_chunker = None
        elif component_name == "attention" and hasattr(ablated_model, 'hierarchical_backbone'):
            # Simplify attention mechanism
            for layer in ablated_model.hierarchical_backbone.transformer_layers:
                if hasattr(layer, 'self_attn'):
                    layer.self_attn.num_heads = min(4, layer.self_attn.num_heads)  # Reduce attention heads
        
        return ablated_model
    
    def _evaluate_model_performance(self, model: nn.Module) -> float:
        """Evaluate model performance using the provided evaluation function."""
        try:
            with torch.no_grad():
                performance = self.eval_function(model)
            return performance
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return 0.0
    
    def _select_variations_for_iteration(self, 
                                       available_variations: List[ComponentVariation],
                                       num_variations: int,
                                       iteration: int) -> List[ComponentVariation]:
        """Select variations to test in current iteration."""
        # Sort by expected improvement and complexity
        scored_variations = []
        
        for var in available_variations:
            # Score based on expected improvement and inverse complexity
            complexity_penalty = {"low": 0.0, "medium": 0.1, "high": 0.2}[var.estimated_complexity]
            score = var.expected_improvement - complexity_penalty
            scored_variations.append((score, var))
        
        # Sort by score and select top variations
        scored_variations.sort(key=lambda x: x[0], reverse=True)
        
        # Add some exploration (random selections) in later iterations
        exploration_factor = min(0.3, iteration * 0.1)
        num_random = int(num_variations * exploration_factor)
        num_top = num_variations - num_random
        
        selected = [var for _, var in scored_variations[:num_top]]
        
        if num_random > 0:
            remaining = [var for _, var in scored_variations[num_top:]]
            if remaining:
                selected.extend(np.random.choice(remaining, min(num_random, len(remaining)), replace=False))
        
        return selected[:num_variations]
    
    def _test_variations_parallel(self, 
                                variations: List[ComponentVariation],
                                base_model: nn.Module,
                                iteration: int) -> List[RefinementResult]:
        """Test multiple variations in parallel."""
        results = []
        
        def test_single_variation(variation):
            try:
                start_time = time.time()
                
                # Apply variation to model
                modified_model = self._apply_variation_to_model(variation, base_model)
                
                # Evaluate performance
                performance = self._evaluate_model_performance(modified_model)
                
                # Calculate improvement
                improvement = performance - self.baseline_performance
                
                # Calculate cultural safety score (simplified)
                cultural_safety_score = self._evaluate_cultural_safety(modified_model)
                
                training_time = time.time() - start_time
                
                result = RefinementResult(
                    iteration=iteration,
                    variation=variation,
                    performance=performance,
                    improvement=improvement,
                    training_time=training_time,
                    convergence_metrics={"loss_reduction": improvement},
                    cultural_safety_score=cultural_safety_score
                )
                
                with self.lock:
                    logger.info(f"    {variation.variation_name}: {performance:.4f} (+{improvement:.4f})")
                
                return result
                
            except Exception as e:
                logger.error(f"Error testing variation {variation.variation_name}: {e}")
                return None
        
        # Run in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(variations)) as executor:
            future_to_variation = {
                executor.submit(test_single_variation, var): var 
                for var in variations
            }
            
            for future in concurrent.futures.as_completed(future_to_variation):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        return results
    
    def _apply_variation_to_model(self, variation: ComponentVariation, model: nn.Module) -> nn.Module:
        """Apply variation to model using appropriate component interface."""
        component_interface = self.component_interfaces[variation.component_name]
        return component_interface.apply_variation(variation, model)
    
    def _evaluate_cultural_safety(self, model: nn.Module) -> float:
        """Evaluate cultural safety of model (simplified implementation)."""
        # This would use the cultural guardrails system
        # For now, return a placeholder score
        return 0.95  # Assume high cultural safety
    
    def _identify_pareto_optimal_solutions(self, results: List[RefinementResult]):
        """Identify Pareto optimal solutions in the results."""
        # Sort by performance
        sorted_results = sorted(results, key=lambda x: x.performance, reverse=True)
        
        pareto_front = []
        
        for result in sorted_results:
            is_dominated = False
            
            for other in sorted_results:
                if (other.performance >= result.performance and 
                    other.cultural_safety_score >= result.cultural_safety_score and
                    other.training_time <= result.training_time and
                    (other.performance > result.performance or 
                     other.cultural_safety_score > result.cultural_safety_score or
                     other.training_time < result.training_time)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                result.is_pareto_optimal = True
                pareto_front.append(result)
        
        logger.info(f"Identified {len(pareto_front)} Pareto optimal solutions")
    
    def _approximate_t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF (simplified implementation)."""
        # Very simplified approximation - in practice, use scipy.stats
        return 0.5 + 0.5 * np.tanh(t / 2)
    
    def _save_ablation_results(self, results: List[AblationResult]):
        """Save ablation results to JSON file."""
        results_data = []
        for result in results:
            results_data.append({
                'component_name': result.component_name,
                'baseline_performance': result.baseline_performance,
                'ablated_performance': result.ablated_performance,
                'performance_impact': result.performance_impact,
                'relative_impact': result.relative_impact,
                'confidence_interval': result.confidence_interval,
                'statistical_significance': result.statistical_significance
            })
        
        with open(self.output_dir / 'ablation_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def _save_refinement_results(self, results: List[RefinementResult]):
        """Save refinement results to JSON file."""
        results_data = []
        for result in results:
            results_data.append({
                'iteration': result.iteration,
                'variation_name': result.variation.variation_name,
                'component_name': result.variation.component_name,
                'performance': result.performance,
                'improvement': result.improvement,
                'training_time': result.training_time,
                'cultural_safety_score': result.cultural_safety_score,
                'is_pareto_optimal': result.is_pareto_optimal,
                'parameters': result.variation.parameters
            })
        
        with open(self.output_dir / 'refinement_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def _generate_refinement_summary(self) -> Dict[str, Any]:
        """Generate summary of refinement process."""
        if not self.refinement_history:
            return {}
        
        summary = {
            'total_iterations': len(self.refinement_history),
            'components_refined': list(set(r.variation.component_name for r in self.refinement_history)),
            'best_performance': self.best_performance,
            'total_improvement': self.best_performance - (self.baseline_performance or 0),
            'pareto_optimal_count': sum(1 for r in self.refinement_history if r.is_pareto_optimal),
            'average_training_time': np.mean([r.training_time for r in self.refinement_history]),
            'best_variations': []
        }
        
        # Find best variation per component
        component_best = {}
        for result in self.refinement_history:
            comp = result.variation.component_name
            if comp not in component_best or result.performance > component_best[comp].performance:
                component_best[comp] = result
        
        for comp, result in component_best.items():
            summary['best_variations'].append({
                'component': comp,
                'variation': result.variation.variation_name,
                'performance': result.performance,
                'improvement': result.improvement
            })
        
        return summary
    
    def _generate_final_report(self, cycle_results: Dict[str, Any]):
        """Generate comprehensive final report."""
        report_path = self.output_dir / 'mle_star_final_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# MLE-STAR Refinement Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Initial Performance**: {self.baseline_performance:.4f}\n")
            f.write(f"- **Final Performance**: {cycle_results['final_performance']:.4f}\n")
            f.write(f"- **Total Improvement**: {cycle_results['total_improvement']:.4f}\n")
            f.write(f"- **Number of Cycles**: {len(cycle_results['cycles'])}\n\n")
            
            f.write("## Cycle Details\n\n")
            for i, cycle in enumerate(cycle_results['cycles']):
                f.write(f"### Cycle {i+1}\n\n")
                f.write(f"- **Duration**: {cycle['cycle_time']:.2f} seconds\n")
                f.write(f"- **Best Performance**: {cycle['best_performance_in_cycle']:.4f}\n")
                
                f.write("\n#### Top Ablation Results\n\n")
                for result in cycle['ablation_results'][:3]:
                    f.write(f"- **{result.component_name}**: Impact {result.performance_impact:.4f} "
                           f"(p={result.statistical_significance:.4f})\n")
                
                f.write(f"\n#### Refinement Results\n\n")
                pareto_results = [r for r in cycle['refinement_results'] if r.is_pareto_optimal]
                f.write(f"- **Total Variations Tested**: {len(cycle['refinement_results'])}\n")
                f.write(f"- **Pareto Optimal Solutions**: {len(pareto_results)}\n")
                
                f.write("\n")
        
        logger.info(f"Final report saved to: {report_path}")


def main():
    """Example usage of MLE-STAR Refinement Engine."""
    # This would typically be called with a real model and evaluation function
    print("🚀 MLE-STAR Two-Loop Refinement System")
    print("=" * 60)
    print("This is the core implementation of Google MLE-STAR's")
    print("dual-loop refinement methodology for Amharic H-Net optimization")
    print("=" * 60)


if __name__ == "__main__":
    main()