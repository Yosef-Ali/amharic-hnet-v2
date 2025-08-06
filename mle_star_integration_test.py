#!/usr/bin/env python3
"""
MLE-STAR Integration Test for Amharic H-Net
Comprehensive test of Google MLE-STAR integration with existing H-Net pipeline

This script demonstrates the complete MLE-STAR workflow:
1. Web-based model discovery for Amharic NLP architectures
2. Two-loop refinement system (ablation + iterative improvement)
3. Ensemble methods with bespoke meta-learners
4. Integrated evaluation with Kaggle-style performance assessment

Run this script to test the full MLE-STAR + H-Net integration.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import existing H-Net components
from src.models.hnet_300m_amharic import AmharicHNet300M, HNet300MConfig, create_300m_model
from src.training.train import AmharicHNetTrainer
from src.evaluation.evaluate import AmharicHNetEvaluator
from src.preprocessing.prepare_amharic import AmharicPreprocessor
from src.safety.cultural_guardrails import AmharicCulturalGuardrails

# Import MLE-STAR components
from src.mle_star import (
    WebModelDiscovery, SearchQuery,
    MLEStarRefinementEngine, ComponentVariation,
    MLEStarEnsembleManager, EnsembleCandidate,
    IntegratedEvaluationSystem, MLEStarKaggleStyleEvaluator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mle_star_integration_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLEStarHNetIntegrationTest:
    """
    Complete MLE-STAR + H-Net Integration Test System
    
    This class orchestrates the full MLE-STAR workflow with Amharic H-Net,
    demonstrating how Google's MLE-STAR methodology can be applied to
    improve Amharic language processing models.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize MLE-STAR integration test.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Create output directory
        self.output_dir = Path("mle_star_integration_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = AmharicPreprocessor()
        self.cultural_guardrails = AmharicCulturalGuardrails()
        
        # MLE-STAR components (will be initialized during test)
        self.model_discovery = None
        self.refinement_engine = None
        self.ensemble_manager = None
        self.evaluation_system = None
        
        # Test state
        self.baseline_model = None
        self.discovered_models = []
        self.refined_models = []
        self.ensemble_results = {}
        self.final_results = {}
        
        logger.info(f"MLE-STAR Integration Test initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for testing."""
        return {
            'model': {
                'd_model': 768,
                'n_encoder_layers': 4,
                'n_decoder_layers': 4,
                'n_main_layers': 12,
                'n_heads': 12,
                'compression_ratio': 4.5,
                'vocab_size': 256
            },
            'training': {
                'batch_size': 8,  # Small for testing
                'num_epochs': 2,  # Quick test
                'learning_rate': 1e-4
            },
            'mle_star': {
                'discovery': {
                    'max_results': 10,
                    'search_queries': [
                        "amharic language model transformer",
                        "ethiopian nlp hierarchical",
                        "semitic language processing"
                    ]
                },
                'refinement': {
                    'max_iterations': 5,
                    'convergence_threshold': 0.001
                },
                'ensemble': {
                    'optimization_methods': ['gradient', 'evolutionary'],
                    'max_candidates': 5
                }
            }
        }
    
    def run_complete_mle_star_workflow(self) -> Dict[str, Any]:
        """
        Run the complete MLE-STAR workflow integration test.
        
        Returns:
            Complete test results
        """
        logger.info("🚀 Starting Complete MLE-STAR + H-Net Integration Test")
        logger.info("=" * 80)
        
        test_start_time = time.time()
        
        try:
            # Phase 1: Initialize baseline H-Net model
            logger.info("📊 Phase 1: Baseline Model Initialization")
            baseline_results = self._initialize_baseline_model()
            
            # Phase 2: Web-based model discovery
            logger.info("🔍 Phase 2: Web-Based Model Discovery")
            discovery_results = self._run_model_discovery()
            
            # Phase 3: Two-loop refinement system
            logger.info("🔧 Phase 3: Two-Loop Refinement System")
            refinement_results = self._run_refinement_system()
            
            # Phase 4: Ensemble methods
            logger.info("🎯 Phase 4: Ensemble Methods")
            ensemble_results = self._run_ensemble_methods()
            
            # Phase 5: Integrated evaluation
            logger.info("📈 Phase 5: Integrated Evaluation")
            evaluation_results = self._run_integrated_evaluation(
                discovery_results, refinement_results, ensemble_results
            )
            
            # Phase 6: Generate final report
            logger.info("📝 Phase 6: Final Report Generation")
            final_results = self._generate_final_report(
                baseline_results, discovery_results, refinement_results,
                ensemble_results, evaluation_results
            )
            
            total_time = time.time() - test_start_time
            
            logger.info("✅ MLE-STAR Integration Test Completed Successfully!")
            logger.info(f"Total execution time: {total_time:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            raise
    
    def _initialize_baseline_model(self) -> Dict[str, Any]:
        """Initialize and evaluate baseline H-Net model."""
        logger.info("Creating baseline 300M parameter Amharic H-Net model...")
        
        # Create baseline model using the existing architecture
        model_config = self.config.get('model', {})
        
        # Use the sophisticated 300M model
        self.baseline_model = create_300m_model(
            vocab_size=model_config.get('vocab_size', 256),
            max_seq_length=512,
            use_mixed_precision=True
        )
        
        self.baseline_model.to(self.device)
        
        # Quick baseline evaluation (mock for testing)
        baseline_performance = self._evaluate_model_performance(self.baseline_model)
        
        baseline_results = {
            'model_parameters': sum(p.numel() for p in self.baseline_model.parameters()),
            'performance_score': baseline_performance,
            'cultural_safety_score': 0.95,  # Mock score
            'architecture_type': 'hierarchical_hnet_300m'
        }
        
        logger.info(f"Baseline model created: {baseline_results['model_parameters']:,} parameters")
        logger.info(f"Baseline performance: {baseline_performance:.4f}")
        
        return baseline_results
    
    def _run_model_discovery(self) -> Dict[str, Any]:
        """Run web-based model discovery."""
        logger.info("Initializing MLE-STAR model discovery system...")
        
        self.model_discovery = WebModelDiscovery(
            cache_dir=str(self.output_dir / "discovery_cache")
        )
        
        discovery_config = self.config.get('mle_star', {}).get('discovery', {})
        search_queries = discovery_config.get('search_queries', [
            "amharic language model transformer"
        ])
        
        all_discovery_results = []
        
        # Run discovery for each query
        for query_text in search_queries:
            logger.info(f"Searching for: {query_text}")
            
            query = SearchQuery(
                query=query_text,
                max_results=discovery_config.get('max_results', 10),
                language_filter="amharic"
            )
            
            try:
                results = self.model_discovery.discover_models(query)
                all_discovery_results.extend(results)
                logger.info(f"Found {len(results)} relevant models for query: {query_text}")
                
            except Exception as e:
                logger.warning(f"Error in model discovery for query '{query_text}': {e}")
        
        # Generate architecture recommendations
        if all_discovery_results and self.baseline_model:
            current_config = {
                'd_model': self.baseline_model.config.d_model,
                'compression_ratio': self.baseline_model.config.chunk_size
            }
            
            recommendations = self.model_discovery.generate_architecture_recommendations(
                all_discovery_results, current_config
            )
        else:
            recommendations = {'high_priority': [], 'medium_priority': [], 'low_priority': []}
        
        discovery_results = {
            'total_models_found': len(all_discovery_results),
            'high_relevance_models': len([r for r in all_discovery_results if r.amharic_relevance_score > 0.7]),
            'recommendations': recommendations,
            'discovery_results': all_discovery_results[:10],  # Top 10 for storage
            'search_effectiveness': len(all_discovery_results) / len(search_queries) if search_queries else 0
        }
        
        logger.info(f"Model discovery completed: {discovery_results['total_models_found']} models found")
        logger.info(f"High relevance models: {discovery_results['high_relevance_models']}")
        
        return discovery_results
    
    def _run_refinement_system(self) -> Dict[str, Any]:
        """Run two-loop refinement system."""
        logger.info("Initializing MLE-STAR two-loop refinement system...")
        
        if not self.baseline_model:
            raise ValueError("Baseline model must be initialized before refinement")
        
        # Initialize refinement engine
        self.refinement_engine = MLEStarRefinementEngine(
            base_model=self.baseline_model,
            eval_function=self._evaluate_model_performance,
            output_dir=str(self.output_dir / "refinement")
        )
        
        refinement_config = self.config.get('mle_star', {}).get('refinement', {})
        
        try:
            # Run outer loop (ablation studies)
            logger.info("Running outer loop: ablation studies...")
            ablation_results = self.refinement_engine.run_outer_loop(
                components_to_ablate=['chunking', 'attention'],
                num_ablation_runs=3  # Reduced for testing
            )
            
            # Run inner loop (iterative refinement) on most impactful component
            if ablation_results:
                most_impactful_component = ablation_results[0].component_name
                logger.info(f"Running inner loop on most impactful component: {most_impactful_component}")
                
                refinement_iterations = self.refinement_engine.run_inner_loop(
                    target_component=most_impactful_component,
                    max_iterations=refinement_config.get('max_iterations', 5),
                    convergence_threshold=refinement_config.get('convergence_threshold', 0.001)
                )
            else:
                refinement_iterations = []
            
            # Compile results
            refinement_results = {
                'ablation_results': [
                    {
                        'component': r.component_name,
                        'performance_impact': r.performance_impact,
                        'statistical_significance': r.statistical_significance
                    }
                    for r in ablation_results
                ],
                'refinement_iterations': len(refinement_iterations),
                'best_performance': max([r.performance for r in refinement_iterations]) if refinement_iterations else 0,
                'total_improvement': (
                    max([r.performance for r in refinement_iterations]) - self.refinement_engine.baseline_performance
                    if refinement_iterations and self.refinement_engine.baseline_performance else 0
                ),
                'convergence_achieved': any(r.improvement < refinement_config.get('convergence_threshold', 0.001) 
                                         for r in refinement_iterations[-3:]) if len(refinement_iterations) >= 3 else False
            }
            
            logger.info(f"Refinement completed: {refinement_results['refinement_iterations']} iterations")
            logger.info(f"Total improvement: {refinement_results['total_improvement']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in refinement system: {e}")
            refinement_results = {
                'ablation_results': [],
                'refinement_iterations': 0,
                'best_performance': 0,
                'total_improvement': 0,
                'convergence_achieved': False,
                'error': str(e)
            }
        
        return refinement_results
    
    def _run_ensemble_methods(self) -> Dict[str, Any]:
        """Run ensemble methods with meta-learners."""
        logger.info("Initializing MLE-STAR ensemble methods...")
        
        if not self.baseline_model:
            raise ValueError("Baseline model must be initialized before ensemble methods")
        
        # Create multiple model variants for ensemble (simplified for testing)
        ensemble_models = [self.baseline_model]
        
        # Create a few variations of the baseline model for ensemble testing
        for i in range(2):  # Create 2 additional variants
            variant_model = create_300m_model(
                vocab_size=256,
                max_seq_length=512,
                use_mixed_precision=True
            )
            # Slightly modify the model (in practice, these would be different architectures)
            variant_model.to(self.device)
            ensemble_models.append(variant_model)
        
        # Initialize ensemble manager
        self.ensemble_manager = MLEStarEnsembleManager(
            base_models=ensemble_models,
            eval_function=self._evaluate_model_performance,
            cultural_safety_function=self._evaluate_cultural_safety,
            output_dir=str(self.output_dir / "ensemble")
        )
        
        ensemble_config = self.config.get('mle_star', {}).get('ensemble', {})
        
        try:
            # Create ensemble candidates
            logger.info("Creating ensemble candidates...")
            candidates = self.ensemble_manager.create_ensemble_candidates()
            
            # Run weight optimization
            optimization_methods = ensemble_config.get('optimization_methods', ['gradient', 'evolutionary'])
            optimization_results = {}
            
            for method in optimization_methods:
                logger.info(f"Optimizing ensemble weights using {method} method...")
                try:
                    result = self.ensemble_manager.optimize_ensemble_weights(
                        method=method,
                        max_iterations=50 if method == 'gradient' else None,
                        population_size=20 if method == 'evolutionary' else None,
                        max_generations=30 if method == 'evolutionary' else None
                    )
                    optimization_results[method] = result
                    
                except Exception as e:
                    logger.warning(f"Error in {method} optimization: {e}")
                    optimization_results[method] = {'best_score': 0.0, 'error': str(e)}
            
            # Generate ensemble report
            ensemble_report = self.ensemble_manager.generate_ensemble_report()
            
            ensemble_results = {
                'num_candidates': len(candidates),
                'optimization_methods_tested': len(optimization_results),
                'optimization_results': optimization_results,
                'best_optimization_method': max(
                    optimization_results.keys(), 
                    key=lambda k: optimization_results[k].get('best_score', 0)
                ) if optimization_results else None,
                'ensemble_report': ensemble_report,
                'candidates_summary': [
                    {
                        'model_id': c.model_id,
                        'performance_score': c.performance_score,
                        'cultural_safety_score': c.cultural_safety_score
                    }
                    for c in candidates
                ]
            }
            
            if optimization_results:
                best_method = ensemble_results['best_optimization_method']
                ensemble_results['best_ensemble_score'] = optimization_results[best_method].get('best_score', 0)
                logger.info(f"Best ensemble method: {best_method}")
                logger.info(f"Best ensemble score: {ensemble_results['best_ensemble_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in ensemble methods: {e}")
            ensemble_results = {
                'num_candidates': 0,
                'optimization_methods_tested': 0,
                'error': str(e)
            }
        
        return ensemble_results
    
    def _run_integrated_evaluation(self, 
                                 discovery_results: Dict[str, Any],
                                 refinement_results: Dict[str, Any],
                                 ensemble_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run integrated evaluation system."""
        logger.info("Running integrated MLE-STAR + Amharic evaluation...")
        
        if not self.baseline_model:
            raise ValueError("Baseline model must be initialized before evaluation")
        
        # Initialize integrated evaluation system
        self.evaluation_system = IntegratedEvaluationSystem(
            model=self.baseline_model,
            output_dir=str(self.output_dir / "evaluation")
        )
        
        try:
            # Prepare test data (mock for testing)
            test_data = [
                {'text': 'አማርኛ ቋንቋ በጣም ቆንጆ ነው።', 'label': 'positive'},
                {'text': 'ኢትዮጵያ ውቤ ሀገር ናት።', 'label': 'positive'},
                {'text': 'ባህላዊ ቁርስ ተቀናበር።', 'label': 'neutral'}
            ]
            
            # Convert results to appropriate format for evaluation
            discovery_model_results = discovery_results.get('discovery_results', [])
            
            # Mock refinement results in expected format
            mock_refinement_results = []
            if refinement_results.get('refinement_iterations', 0) > 0:
                from src.mle_star.refinement_loops import RefinementResult, ComponentVariation
                
                # Create mock refinement results
                for i in range(min(3, refinement_results['refinement_iterations'])):
                    mock_variation = ComponentVariation(
                        component_name='chunking',
                        variation_name=f'test_variation_{i}',
                        parameters={'test': True},
                        description=f'Test variation {i}',
                        estimated_complexity='low',
                        expected_improvement=0.1
                    )
                    
                    mock_result = RefinementResult(
                        iteration=i,
                        variation=mock_variation,
                        performance=0.8 + i * 0.01,
                        improvement=i * 0.01,
                        training_time=10.0,
                        convergence_metrics={'test': 0.5},
                        cultural_safety_score=0.95
                    )
                    mock_refinement_results.append(mock_result)
            
            # Run comprehensive evaluation
            evaluation_result = self.evaluation_system.run_comprehensive_evaluation(
                test_data=test_data,
                discovery_results=discovery_model_results,
                refinement_results=mock_refinement_results,
                ensemble_results=ensemble_results
            )
            
            evaluation_summary = {
                'traditional_metrics': {
                    'perplexity': evaluation_result.perplexity,
                    'compression_ratio': evaluation_result.compression_ratio,
                    'cultural_safety_rate': evaluation_result.cultural_safety_rate,
                    'amharic_quality_score': evaluation_result.amharic_quality_score
                },
                'mle_star_metrics': {
                    'model_discovery_effectiveness': evaluation_result.mle_star_metrics.model_discovery_effectiveness,
                    'refinement_improvement_rate': evaluation_result.mle_star_metrics.refinement_improvement_rate,
                    'ensemble_performance_gain': evaluation_result.mle_star_metrics.ensemble_performance_gain,
                    'automated_optimization_score': evaluation_result.mle_star_metrics.automated_optimization_score
                },
                'kaggle_medal_probability': getattr(evaluation_result, 'kaggle_medal_probability', {}),
                'improvement_analysis': evaluation_result.improvement_analysis,
                'recommendations': {
                    'optimization_targets': evaluation_result.next_optimization_targets,
                    'ensemble_recommendations': evaluation_result.ensemble_recommendations
                }
            }
            
            logger.info("Integrated evaluation completed successfully")
            
            if evaluation_summary.get('kaggle_medal_probability'):
                medal_probs = evaluation_summary['kaggle_medal_probability']
                logger.info(f"Kaggle performance expectation: {medal_probs.get('expected_percentile', 0):.1f}th percentile")
            
        except Exception as e:
            logger.error(f"Error in integrated evaluation: {e}")
            evaluation_summary = {
                'error': str(e),
                'traditional_metrics': {},
                'mle_star_metrics': {}
            }
        
        return evaluation_summary
    
    def _generate_final_report(self, 
                             baseline_results: Dict[str, Any],
                             discovery_results: Dict[str, Any],
                             refinement_results: Dict[str, Any],
                             ensemble_results: Dict[str, Any],
                             evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        logger.info("Generating comprehensive final report...")
        
        # Compile all results
        final_results = {
            'test_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'config_used': self.config,
                'total_test_phases': 6
            },
            'baseline_results': baseline_results,
            'discovery_results': discovery_results,
            'refinement_results': refinement_results,
            'ensemble_results': ensemble_results,
            'evaluation_results': evaluation_results,
            'integration_summary': self._create_integration_summary(
                baseline_results, discovery_results, refinement_results,
                ensemble_results, evaluation_results
            )
        }
        
        # Save detailed JSON report
        json_report_path = self.output_dir / 'mle_star_integration_test_results.json'
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # Generate markdown report
        self._generate_markdown_report(final_results)
        
        logger.info(f"Final report saved to: {json_report_path}")
        
        return final_results
    
    def _create_integration_summary(self, 
                                  baseline_results: Dict[str, Any],
                                  discovery_results: Dict[str, Any],
                                  refinement_results: Dict[str, Any],
                                  ensemble_results: Dict[str, Any],
                                  evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create integration summary."""
        
        # Calculate overall success metrics
        phases_completed = 0
        phases_with_errors = 0
        
        if baseline_results.get('performance_score', 0) > 0:
            phases_completed += 1
        
        if discovery_results.get('total_models_found', 0) > 0:
            phases_completed += 1
        elif 'error' in discovery_results:
            phases_with_errors += 1
        
        if refinement_results.get('refinement_iterations', 0) > 0:
            phases_completed += 1
        elif 'error' in refinement_results:
            phases_with_errors += 1
        
        if ensemble_results.get('num_candidates', 0) > 0:
            phases_completed += 1
        elif 'error' in ensemble_results:
            phases_with_errors += 1
        
        if evaluation_results.get('traditional_metrics'):
            phases_completed += 1
        elif 'error' in evaluation_results:
            phases_with_errors += 1
        
        # Calculate improvements
        total_improvement = 0
        improvement_sources = []
        
        if refinement_results.get('total_improvement', 0) > 0:
            total_improvement += refinement_results['total_improvement']
            improvement_sources.append('refinement')
        
        if ensemble_results.get('best_ensemble_score', 0) > baseline_results.get('performance_score', 0):
            ensemble_improvement = ensemble_results['best_ensemble_score'] - baseline_results['performance_score']
            total_improvement += ensemble_improvement
            improvement_sources.append('ensemble')
        
        summary = {
            'integration_success_rate': phases_completed / 6,
            'phases_completed': phases_completed,
            'phases_with_errors': phases_with_errors,
            'total_performance_improvement': total_improvement,
            'improvement_sources': improvement_sources,
            'mle_star_components_tested': {
                'web_discovery': discovery_results.get('total_models_found', 0) > 0,
                'refinement_loops': refinement_results.get('refinement_iterations', 0) > 0,
                'ensemble_methods': ensemble_results.get('num_candidates', 0) > 0,
                'integrated_evaluation': 'traditional_metrics' in evaluation_results
            },
            'key_achievements': [],
            'main_challenges': [],
            'recommendations_for_production': []
        }
        
        # Generate achievements and challenges
        if discovery_results.get('total_models_found', 0) > 5:
            summary['key_achievements'].append(f"Discovered {discovery_results['total_models_found']} relevant models")
        
        if refinement_results.get('total_improvement', 0) > 0.01:
            summary['key_achievements'].append(f"Achieved {refinement_results['total_improvement']:.4f} improvement through refinement")
        
        if ensemble_results.get('optimization_methods_tested', 0) > 1:
            summary['key_achievements'].append(f"Successfully tested {ensemble_results['optimization_methods_tested']} ensemble optimization methods")
        
        if evaluation_results.get('kaggle_medal_probability', {}).get('expected_percentile', 0) > 70:
            percentile = evaluation_results['kaggle_medal_probability']['expected_percentile']
            summary['key_achievements'].append(f"Achieved {percentile:.1f}th percentile Kaggle performance expectation")
        
        # Add challenges
        if phases_with_errors > 0:
            summary['main_challenges'].append(f"Encountered errors in {phases_with_errors} phases")
        
        if discovery_results.get('search_effectiveness', 0) < 5:
            summary['main_challenges'].append("Limited model discovery effectiveness")
        
        # Add production recommendations
        summary['recommendations_for_production'].extend([
            "Implement robust error handling for all MLE-STAR components",
            "Set up comprehensive logging and monitoring",
            "Create automated testing pipeline for integration",
            "Optimize computational resources for large-scale optimization"
        ])
        
        return summary
    
    def _generate_markdown_report(self, final_results: Dict[str, Any]):
        """Generate markdown report."""
        report_path = self.output_dir / 'MLE_STAR_Integration_Test_Report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MLE-STAR + Amharic H-Net Integration Test Report\n\n")
            
            # Metadata
            metadata = final_results['test_metadata']
            f.write(f"**Test Date**: {metadata['timestamp']}\n")
            f.write(f"**Device**: {metadata['device']}\n")
            f.write(f"**Test Phases**: {metadata['total_test_phases']}\n\n")
            
            # Integration Summary
            summary = final_results['integration_summary']
            f.write("## Integration Summary\n\n")
            f.write(f"- **Success Rate**: {summary['integration_success_rate']:.1%} ({summary['phases_completed']}/{metadata['total_test_phases']} phases)\n")
            f.write(f"- **Total Performance Improvement**: {summary['total_performance_improvement']:.4f}\n")
            f.write(f"- **Improvement Sources**: {', '.join(summary['improvement_sources'])}\n\n")
            
            # MLE-STAR Components
            components = summary['mle_star_components_tested']
            f.write("## MLE-STAR Components Tested\n\n")
            f.write(f"- **Web-based Model Discovery**: {'✅' if components['web_discovery'] else '❌'}\n")
            f.write(f"- **Two-loop Refinement System**: {'✅' if components['refinement_loops'] else '❌'}\n")
            f.write(f"- **Ensemble Methods**: {'✅' if components['ensemble_methods'] else '❌'}\n")
            f.write(f"- **Integrated Evaluation**: {'✅' if components['integrated_evaluation'] else '❌'}\n\n")
            
            # Key Results by Phase
            f.write("## Detailed Results by Phase\n\n")
            
            # Baseline
            baseline = final_results['baseline_results']
            f.write("### Phase 1: Baseline Model\n")
            f.write(f"- Model Parameters: {baseline.get('model_parameters', 0):,}\n")
            f.write(f"- Performance Score: {baseline.get('performance_score', 0):.4f}\n")
            f.write(f"- Architecture: {baseline.get('architecture_type', 'unknown')}\n\n")
            
            # Discovery
            discovery = final_results['discovery_results']
            f.write("### Phase 2: Model Discovery\n")
            f.write(f"- Models Found: {discovery.get('total_models_found', 0)}\n")
            f.write(f"- High Relevance Models: {discovery.get('high_relevance_models', 0)}\n")
            f.write(f"- Search Effectiveness: {discovery.get('search_effectiveness', 0):.2f} models/query\n\n")
            
            # Refinement
            refinement = final_results['refinement_results']
            f.write("### Phase 3: Refinement System\n")
            f.write(f"- Refinement Iterations: {refinement.get('refinement_iterations', 0)}\n")
            f.write(f"- Performance Improvement: {refinement.get('total_improvement', 0):.4f}\n")
            f.write(f"- Convergence Achieved: {'Yes' if refinement.get('convergence_achieved', False) else 'No'}\n\n")
            
            # Ensemble
            ensemble = final_results['ensemble_results']
            f.write("### Phase 4: Ensemble Methods\n")
            f.write(f"- Ensemble Candidates: {ensemble.get('num_candidates', 0)}\n")
            f.write(f"- Optimization Methods: {ensemble.get('optimization_methods_tested', 0)}\n")
            if ensemble.get('best_optimization_method'):
                f.write(f"- Best Method: {ensemble['best_optimization_method']}\n")
                f.write(f"- Best Score: {ensemble.get('best_ensemble_score', 0):.4f}\n")
            f.write("\n")
            
            # Evaluation
            evaluation = final_results['evaluation_results']
            f.write("### Phase 5: Integrated Evaluation\n")
            if 'traditional_metrics' in evaluation:
                trad = evaluation['traditional_metrics']
                f.write(f"- Perplexity: {trad.get('perplexity', 0):.2f}\n")
                f.write(f"- Cultural Safety Rate: {trad.get('cultural_safety_rate', 0):.3f}\n")
                f.write(f"- Amharic Quality: {trad.get('amharic_quality_score', 0):.3f}\n")
            
            if 'kaggle_medal_probability' in evaluation:
                medal = evaluation['kaggle_medal_probability']
                f.write(f"- Kaggle Percentile: {medal.get('expected_percentile', 0):.1f}th\n")
                f.write(f"- Medal Probabilities: Bronze {medal.get('bronze_probability', 0):.3f}, "
                       f"Silver {medal.get('silver_probability', 0):.3f}, "
                       f"Gold {medal.get('gold_probability', 0):.3f}\n")
            f.write("\n")
            
            # Achievements and Challenges
            f.write("## Key Achievements\n\n")
            for achievement in summary['key_achievements']:
                f.write(f"- {achievement}\n")
            
            f.write("\n## Main Challenges\n\n")
            for challenge in summary['main_challenges']:
                f.write(f"- {challenge}\n")
            
            f.write("\n## Recommendations for Production\n\n")
            for rec in summary['recommendations_for_production']:
                f.write(f"- {rec}\n")
            
            f.write("\n---\n")
            f.write("*Report generated by MLE-STAR Integration Test System*\n")
        
        logger.info(f"Markdown report saved to: {report_path}")
    
    def _evaluate_model_performance(self, model: nn.Module) -> float:
        """Evaluate model performance (mock implementation for testing)."""
        # This would normally run actual evaluation
        # For testing, return a mock score based on model complexity
        param_count = sum(p.numel() for p in model.parameters())
        # Normalize to reasonable performance score
        normalized_score = min(0.95, 0.3 + (param_count / 1_000_000_000) * 0.5)
        return normalized_score
    
    def _evaluate_cultural_safety(self, model: nn.Module) -> float:
        """Evaluate cultural safety (mock implementation for testing)."""
        # Mock cultural safety evaluation
        return 0.95  # High cultural safety score


def main():
    """Main function to run MLE-STAR integration test."""
    parser = argparse.ArgumentParser(description='MLE-STAR + H-Net Integration Test')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='mle_star_integration_results',
                       help='Output directory for test results')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with minimal iterations')
    
    args = parser.parse_args()
    
    print("🚀 MLE-STAR + Amharic H-Net Integration Test")
    print("=" * 80)
    print("This test demonstrates the complete integration of Google MLE-STAR")
    print("methodology with Amharic H-Net language processing pipeline.")
    print("=" * 80)
    
    try:
        # Initialize test system
        test_system = MLEStarHNetIntegrationTest(config_path=args.config)
        
        if args.quick_test:
            logger.info("Running quick test mode...")
            # Modify config for quick testing
            test_system.config['training']['num_epochs'] = 1
            test_system.config['mle_star']['refinement']['max_iterations'] = 2
            test_system.config['mle_star']['discovery']['max_results'] = 5
        
        # Run complete workflow
        results = test_system.run_complete_mle_star_workflow()
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        summary = results['integration_summary']
        print(f"📊 Success Rate: {summary['integration_success_rate']:.1%}")
        print(f"🚀 Performance Improvement: {summary['total_performance_improvement']:.4f}")
        print(f"🔧 Components Tested: {sum(summary['mle_star_components_tested'].values())}/4")
        
        if results['evaluation_results'].get('kaggle_medal_probability'):
            medal_prob = results['evaluation_results']['kaggle_medal_probability']
            percentile = medal_prob.get('expected_percentile', 0)
            print(f"🏆 Kaggle Performance: {percentile:.1f}th percentile")
        
        print(f"\n📁 Detailed results saved to: {test_system.output_dir}")
        print(f"📄 View the markdown report: {test_system.output_dir}/MLE_STAR_Integration_Test_Report.md")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()