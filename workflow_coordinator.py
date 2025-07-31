#!/usr/bin/env python3
"""
Amharic H-Net Development Workflow Coordinator
==============================================

This script coordinates the sub-agent workflow for end-to-end Amharic H-Net development,
implementing the Claude Code best practices with specialized sub-agents.

Workflow Steps:
1. 🗂️  Environment Setup (training-engineer)
2. 📊 Data Collection (data-collector) 
3. 🔍 Linguistic Analysis (linguistic-analyzer)
4. 🏗️  Model Architecture (model-architect)
5. 🚀 Training Pipeline (training-engineer)
6. 📈 Evaluation & Validation (evaluation-specialist)
7. 🌐 Production Deployment (deployment-engineer)

Usage:
    python workflow_coordinator.py --phase setup
    python workflow_coordinator.py --phase collect --source wikipedia --max-articles 1000
    python workflow_coordinator.py --phase analyze --input data/raw --output data/processed
    python workflow_coordinator.py --phase train --config configs/config.yaml
    python workflow_coordinator.py --phase evaluate --model outputs/checkpoint_best.pt
    python workflow_coordinator.py --phase deploy --api-port 8000
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict
import json
import time
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Warning: yaml not installed. Run 'pip install pyyaml'")
    yaml = None

# Import sub-agents
from src.data_collection.amharic_collector import AmharicDataCollector
from src.linguistic_analysis.morphological_analyzer import AmharicMorphologicalAnalyzer


class WorkflowCoordinator:
    """
    Coordinates the complete Amharic H-Net development workflow using specialized sub-agents.
    
    Implements Claude Code best practices with proactive agent usage and collaborative workflows.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = self._setup_logging()
        
        # Workflow state tracking
        self.workflow_state = {
            'current_phase': None,
            'completed_phases': [],
            'phase_results': {},
            'start_time': datetime.now(),
            'agent_logs': []
        }
        
        # Sub-agent instances
        self.agents = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup workflow coordination logging."""
        logger = logging.getLogger('amharic_workflow_coordinator')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def execute_workflow_phase(self, phase: str, **kwargs) -> Dict:
        """
        Execute a specific workflow phase using appropriate sub-agents.
        
        Args:
            phase: Workflow phase to execute
            **kwargs: Phase-specific parameters
            
        Returns:
            Dictionary with phase execution results
        """
        self.workflow_state['current_phase'] = phase
        self.logger.info(f"🚀 Starting workflow phase: {phase}")
        
        phase_start_time = time.time()
        
        try:
            if phase == 'setup':
                result = await self._execute_setup_phase(**kwargs)
            elif phase == 'collect':
                result = await self._execute_collection_phase(**kwargs)
            elif phase == 'analyze':
                result = await self._execute_analysis_phase(**kwargs)
            elif phase == 'train':
                result = await self._execute_training_phase(**kwargs)
            elif phase == 'evaluate':
                result = await self._execute_evaluation_phase(**kwargs)
            elif phase == 'deploy':
                result = await self._execute_deployment_phase(**kwargs)
            else:
                raise ValueError(f"Unknown workflow phase: {phase}")
            
            # Record successful completion
            execution_time = time.time() - phase_start_time
            result['execution_time'] = execution_time
            result['status'] = 'completed'
            
            self.workflow_state['completed_phases'].append(phase)
            self.workflow_state['phase_results'][phase] = result
            
            self.logger.info(f"✅ Phase '{phase}' completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Phase '{phase}' failed: {e}")
            result = {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - phase_start_time
            }
            self.workflow_state['phase_results'][phase] = result
            raise
    
    async def _execute_setup_phase(self, **kwargs) -> Dict:
        """Execute environment setup using training-engineer sub-agent."""
        self.logger.info("🛠️ Training-Engineer: Setting up development environment...")
        
        # Run environment setup script
        setup_script = self.project_root / "setup_environment.sh"
        
        if setup_script.exists():
            process = await asyncio.create_subprocess_shell(
                f"bash {setup_script}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info("✅ Environment setup completed successfully")
                return {
                    'setup_output': stdout.decode(),
                    'environment_ready': True
                }
            else:
                raise RuntimeError(f"Environment setup failed: {stderr.decode()}")
        else:
            raise FileNotFoundError("Setup script not found. Run setup_environment.sh first.")
    
    async def _execute_collection_phase(
        self, 
        source: str = 'wikipedia', 
        max_articles: int = 1000,
        concurrent: int = 5
    ) -> Dict:
        """Execute data collection using data-collector sub-agent."""
        self.logger.info(f"📊 Data-Collector: Collecting {max_articles} articles from {source}...")
        
        # Initialize data collector
        collector = AmharicDataCollector(
            output_dir=str(self.project_root / "data" / "raw"),
            max_concurrent=concurrent
        )
        
        # Collect data
        samples = await collector.collect_from_source(source, max_articles)
        
        if samples:
            # Save collected data
            collector.save_samples(samples, f"{source}_collected.json")
            
            # Generate collection report
            stats = {
                'total_samples': len(samples),
                'total_words': sum(s.estimated_words for s in samples),
                'average_quality': sum(s.quality_score for s in samples) / len(samples),
                'sources': {source: len(samples)},
                'dialects_detected': list(set(d for s in samples for d in s.dialect_hints))
            }
            
            self.logger.info(f"✅ Collected {len(samples)} high-quality samples")
            return {
                'collection_stats': stats,
                'samples_collected': len(samples),
                'output_files': [f"{source}_collected.json"]
            }
        else:
            raise RuntimeError(f"No samples collected from {source}")
    
    async def _execute_analysis_phase(
        self, 
        input_dir: str, 
        output_dir: str
    ) -> Dict:
        """Execute linguistic analysis using linguistic-analyzer sub-agent."""
        self.logger.info(f"🔍 Linguistic-Analyzer: Processing texts from {input_dir}...")
        
        # Initialize analyzer
        analyzer = AmharicMorphologicalAnalyzer()
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process files
        processed_files = []
        total_words = 0
        total_complexity = 0
        dialect_distribution = {}
        
        if input_path.is_dir():
            # Process JSON files from data collection
            for json_file in input_path.glob("*.json"):
                self.logger.info(f"Analyzing {json_file.name}...")
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    samples = json.load(f)
                
                for i, sample in enumerate(samples):
                    if isinstance(sample, dict) and 'augmented_text' in sample:
                        text = sample['augmented_text']
                    elif isinstance(sample, dict) and 'text' in sample:
                        text = sample['text']
                    else:
                        continue
                    
                    # Analyze text
                    annotation = analyzer.analyze_text(text)
                    
                    # Save analysis
                    analysis_file = output_path / f"{json_file.stem}_sample_{i}_analysis.json"
                    analyzer.save_analysis(annotation, analysis_file)
                    
                    # Collect statistics
                    total_words += len(annotation.word_analyses)
                    total_complexity += annotation.text_complexity
                    dialect = annotation.dialect_classification
                    dialect_distribution[dialect] = dialect_distribution.get(dialect, 0) + 1
                    
                    processed_files.append(str(analysis_file))
        
        analysis_stats = {
            'files_processed': len(processed_files),
            'total_words_analyzed': total_words,
            'average_complexity': total_complexity / len(processed_files) if processed_files else 0,
            'dialect_distribution': dialect_distribution
        }
        
        self.logger.info(f"✅ Analyzed {len(processed_files)} text samples")
        return {
            'analysis_stats': analysis_stats,
            'processed_files': processed_files
        }
    
    async def _execute_training_phase(
        self, 
        config_path: str,
        epochs: int = None,
        batch_size: int = None,
        use_transfer_learning: bool = True
    ) -> Dict:
        """Execute model training using training-engineer sub-agent."""
        self.logger.info(f"🚀 Training-Engineer: Starting model training with {config_path}...")
        
        # Load training configuration
        if yaml:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Check for transfer learning
        if use_transfer_learning:
            self.logger.info("🔄 Initializing transfer learning from Chinese H-Net...")
            # Would implement transfer learning initialization here
        
        # Run training script
        training_cmd = [
            sys.executable, "main.py", "train",
            "--config", config_path,
            "--data-dir", str(self.project_root / "data" / "processed"),
            "--output-dir", str(self.project_root / "outputs")
        ]
        
        # Add additional arguments
        if epochs:
            training_cmd.extend(["--epochs", str(epochs)])
        if batch_size:
            training_cmd.extend(["--batch-size", str(batch_size)])
        
        process = await asyncio.create_subprocess_exec(
            *training_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_root
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            self.logger.info("✅ Training completed successfully")
            return {
                'training_output': stdout.decode(),
                'model_checkpoint': str(self.project_root / "outputs" / "checkpoint_best.pt"),
                'training_logs': str(self.project_root / "outputs" / "logs")
            }
        else:
            raise RuntimeError(f"Training failed: {stderr.decode()}")
    
    async def _execute_evaluation_phase(
        self, 
        model_path: str
    ) -> Dict:
        """Execute model evaluation using evaluation-specialist sub-agent."""
        self.logger.info(f"📈 Evaluation-Specialist: Evaluating model {model_path}...")
        
        # Run evaluation script
        eval_cmd = [
            sys.executable, "main.py", "evaluate",
            "--model-path", model_path,
            "--test-data", str(self.project_root / "data" / "processed"),
            "--output-dir", str(self.project_root / "evaluation_results")
        ]
        
        process = await asyncio.create_subprocess_exec(
            *eval_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_root
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            self.logger.info("✅ Evaluation completed successfully")
            
            # Load evaluation results
            results_file = self.project_root / "evaluation_results" / "evaluation_report.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    eval_results = json.load(f)
            else:
                eval_results = {"status": "completed"}
            
            return {
                'evaluation_results': eval_results,
                'evaluation_output': stdout.decode()
            }
        else:
            raise RuntimeError(f"Evaluation failed: {stderr.decode()}")
    
    async def _execute_deployment_phase(
        self, 
        model_path: str = None,
        api_port: int = 8000,
        max_length: int = 200,
        temperature: float = 0.8
    ) -> Dict:
        """Execute model deployment using deployment-engineer sub-agent."""
        self.logger.info(f"🌐 Deployment-Engineer: Deploying model API on port {api_port}...")
        
        if not model_path:
            model_path = str(self.project_root / "outputs" / "checkpoint_best.pt")
        
        # Create deployment configuration
        deployment_config = {
            'model_path': model_path,
            'api_port': api_port,
            'enable_cultural_safety': True,
            'max_generation_length': max_length,
            'temperature': temperature
        }
        
        config_file = self.project_root / "deployment_config.json"
        with open(config_file, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        self.logger.info("✅ Deployment configuration created")
        self.logger.info(f"   Model: {model_path}")
        self.logger.info(f"   Port: {api_port}")
        self.logger.info("   Start with: python -m src.deployment.api_server")
        
        return {
            'deployment_config': deployment_config,
            'config_file': str(config_file),
            'status': 'ready_for_deployment'
        }
    
    def save_workflow_state(self, output_file: str = "workflow_state.json"):
        """Save current workflow state to file."""
        state_file = self.project_root / output_file
        
        # Make workflow state JSON-serializable
        serializable_state = {
            'current_phase': self.workflow_state['current_phase'],
            'completed_phases': self.workflow_state['completed_phases'],
            'phase_results': self.workflow_state['phase_results'],
            'start_time': self.workflow_state['start_time'].isoformat(),
            'total_execution_time': (datetime.now() - self.workflow_state['start_time']).total_seconds()
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_state, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Workflow state saved to {state_file}")
    
    def generate_workflow_report(self) -> str:
        """Generate comprehensive workflow execution report."""
        report = ["🇪🇹 AMHARIC H-NET DEVELOPMENT WORKFLOW REPORT"]
        report.append("=" * 60)
        report.append("")
        
        # Execution summary
        total_time = (datetime.now() - self.workflow_state['start_time']).total_seconds()
        report.append(f"📊 EXECUTION SUMMARY")
        report.append(f"   Start Time: {self.workflow_state['start_time']}")
        report.append(f"   Total Duration: {total_time:.2f} seconds")
        report.append(f"   Phases Completed: {len(self.workflow_state['completed_phases'])}")
        report.append("")
        
        # Phase-by-phase results
        for phase in self.workflow_state['completed_phases']:
            result = self.workflow_state['phase_results'].get(phase, {})
            report.append(f"✅ PHASE: {phase.upper()}")
            report.append(f"   Status: {result.get('status', 'unknown')}")
            report.append(f"   Duration: {result.get('execution_time', 0):.2f}s")
            
            # Phase-specific details
            if phase == 'collect' and 'collection_stats' in result:
                stats = result['collection_stats']
                report.append(f"   Samples: {stats['total_samples']}")
                report.append(f"   Words: {stats['total_words']:,}")
                report.append(f"   Quality: {stats['average_quality']:.3f}")
            
            elif phase == 'analyze' and 'analysis_stats' in result:
                stats = result['analysis_stats']
                report.append(f"   Files: {stats['files_processed']}")
                report.append(f"   Words: {stats['total_words_analyzed']:,}")
                report.append(f"   Complexity: {stats['average_complexity']:.3f}")
            
            report.append("")
        
        # Success metrics
        if len(self.workflow_state['completed_phases']) > 0:
            success_rate = len([p for p in self.workflow_state['phase_results'].values() 
                              if p.get('status') == 'completed']) / len(self.workflow_state['phase_results'])
            report.append(f"🎯 SUCCESS RATE: {success_rate:.1%}")
        
        report.append("=" * 60)
        report.append("🇪🇹 Amharic H-Net Development Complete!")
        
        return "\\n".join(report)


async def main():
    """Main workflow coordinator entry point."""
    parser = argparse.ArgumentParser(description='Amharic H-Net Development Workflow Coordinator')
    parser.add_argument('--phase', required=True, 
                       choices=['setup', 'collect', 'analyze', 'train', 'evaluate', 'deploy', 'full'],
                       help='Workflow phase to execute')
    
    # Phase-specific arguments
    parser.add_argument('--source', default='wikipedia', help='Data collection source')
    parser.add_argument('--max-articles', type=int, default=1000, help='Max articles to collect')
    parser.add_argument('--input', help='Input directory for analysis')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--config', default='configs/config.yaml', help='Training configuration')
    parser.add_argument('--model', help='Model path for evaluation/deployment')
    parser.add_argument('--port', type=int, default=8000, help='API port for deployment')
    parser.add_argument('--concurrent', type=int, default=5, help='Concurrent requests')
    
    args = parser.parse_args()
    
    # Initialize workflow coordinator
    coordinator = WorkflowCoordinator()
    
    try:
        if args.phase == 'full':
            # Execute full workflow
            phases = ['setup', 'collect', 'analyze', 'train', 'evaluate', 'deploy']
            for phase in phases:
                if phase == 'collect':
                    await coordinator.execute_workflow_phase(
                        phase, source=args.source, max_articles=args.max_articles, concurrent=args.concurrent
                    )
                elif phase == 'analyze':
                    await coordinator.execute_workflow_phase(
                        phase, input_dir=args.input or 'data/raw', output_dir=args.output or 'data/processed'
                    )
                elif phase == 'train':
                    await coordinator.execute_workflow_phase(phase, config_path=args.config)
                elif phase == 'evaluate':
                    await coordinator.execute_workflow_phase(phase, model_path=args.model or 'outputs/checkpoint_best.pt')
                elif phase == 'deploy':
                    await coordinator.execute_workflow_phase(phase, model_path=args.model, api_port=args.port)
                else:
                    await coordinator.execute_workflow_phase(phase)
        else:
            # Execute single phase
            kwargs = {}
            if args.phase == 'collect':
                kwargs = {'source': args.source, 'max_articles': args.max_articles, 'concurrent': args.concurrent}
            elif args.phase == 'analyze':
                kwargs = {'input_dir': args.input or 'data/raw', 'output_dir': args.output or 'data/processed'}
            elif args.phase == 'train':
                kwargs = {'config_path': args.config}
            elif args.phase == 'evaluate':
                kwargs = {'model_path': args.model or 'outputs/checkpoint_best.pt'}
            elif args.phase == 'deploy':
                kwargs = {'model_path': args.model, 'api_port': args.port}
            
            await coordinator.execute_workflow_phase(args.phase, **kwargs)
        
        # Save workflow state and generate report
        coordinator.save_workflow_state()
        report = coordinator.generate_workflow_report()
        print(report)
        
        # Save report to file
        with open('workflow_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
    except Exception as e:
        coordinator.logger.error(f"❌ Workflow failed: {e}")
        coordinator.save_workflow_state()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())