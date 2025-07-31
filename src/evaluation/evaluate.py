import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse

from ..models.hnet_amharic import AmharicHNet
from ..preprocessing.prepare_amharic import AmharicPreprocessor
from ..safety.cultural_guardrails import AmharicCulturalGuardrails
from ..training.data_loader import create_data_loaders, load_data_from_directory


class AmharicHNetEvaluator:
    """
    Comprehensive evaluation system for Amharic H-Net model.
    """
    
    def __init__(
        self,
        model: AmharicHNet,
        preprocessor: AmharicPreprocessor,
        cultural_guardrails: AmharicCulturalGuardrails,
        device: torch.device
    ):
        self.model = model.to(device)
        self.preprocessor = preprocessor
        self.cultural_guardrails = cultural_guardrails
        self.device = device
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_perplexity(self, data_loader) -> float:
        """Calculate perplexity on validation data."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing perplexity"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                logits, _ = self.model(input_ids, target_ids)
                
                # Compute loss
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                # Count non-padding tokens
                valid_tokens = (target_ids != 0).sum().item()
                
                total_loss += loss.item()
                total_tokens += valid_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def evaluate_compression_ratio(self, data_loader) -> Dict[str, float]:
        """Evaluate dynamic chunking compression performance."""
        self.model.eval()
        compression_ratios = []
        boundary_stats = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating compression"):
                input_ids = batch['input_ids'].to(self.device)
                
                # Forward pass
                _, boundary_probs = self.model(input_ids)
                
                # Calculate compression ratios
                seq_lengths = (input_ids != 0).sum(dim=1).float()
                num_chunks = boundary_probs.sum(dim=1)
                
                batch_ratios = seq_lengths / (num_chunks + 1e-8)
                compression_ratios.extend(batch_ratios.cpu().numpy())
                
                # Boundary statistics
                boundary_stats.extend(boundary_probs.mean(dim=1).cpu().numpy())
        
        return {
            'mean_compression_ratio': np.mean(compression_ratios),
            'std_compression_ratio': np.std(compression_ratios),
            'min_compression_ratio': np.min(compression_ratios),
            'max_compression_ratio': np.max(compression_ratios),
            'mean_boundary_prob': np.mean(boundary_stats),
            'std_boundary_prob': np.std(boundary_stats)
        }
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        num_samples: int = 1
    ) -> List[str]:
        """Generate text with cultural safety checking."""
        self.model.eval()
        
        # Preprocess prompt
        cleaned_prompt = self.preprocessor.clean_text(prompt)
        prompt_bytes = self.preprocessor.extract_byte_sequences(cleaned_prompt, max_length=100)
        
        generated_texts = []
        
        for _ in range(num_samples):
            with torch.no_grad():
                # Convert prompt to tensor
                input_ids = torch.tensor([prompt_bytes], dtype=torch.long, device=self.device)
                generated = input_ids.clone()
                
                for step in range(max_length):
                    # Forward pass
                    logits, _ = self.model(generated)
                    
                    # Get next token logits
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to generated sequence
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                    
                    # Stop conditions
                    if next_token.item() == ord('።'):  # Amharic period
                        break
                    
                    if next_token.item() == 0:  # Padding token
                        break
                
                # Decode generated text
                generated_bytes = generated[0].cpu().numpy().tolist()
                generated_text = self.preprocessor.decode_byte_sequence(generated_bytes)
                
                # Remove original prompt
                if cleaned_prompt in generated_text:
                    generated_text = generated_text.replace(cleaned_prompt, "").strip()
                
                generated_texts.append(generated_text)
        
        return generated_texts
    
    def evaluate_cultural_safety(self, generated_texts: List[str]) -> Dict[str, float]:
        """Evaluate cultural safety of generated texts."""
        total_texts = len(generated_texts)
        if total_texts == 0:
            return {'safety_rate': 1.0, 'avg_violations': 0.0}
        
        safe_texts = 0
        total_violations = 0
        violation_types = defaultdict(int)
        
        for text in generated_texts:
            is_safe, violations = self.cultural_guardrails.check_cultural_safety(text)
            
            if is_safe:
                safe_texts += 1
            else:
                total_violations += len(violations)
                for violation in violations:
                    violation_types[violation.violation_type] += 1
        
        safety_metrics = {
            'safety_rate': safe_texts / total_texts,
            'avg_violations': total_violations / total_texts,
            'violation_types': dict(violation_types)
        }
        
        return safety_metrics
    
    def evaluate_amharic_quality(self, generated_texts: List[str]) -> Dict[str, float]:
        """Evaluate quality of Amharic text generation."""
        if not generated_texts:
            return {'amharic_ratio': 0.0, 'avg_length': 0.0}
        
        amharic_ratios = []
        text_lengths = []
        
        for text in generated_texts:
            if text.strip():
                ratio = self.preprocessor.get_amharic_ratio(text)
                amharic_ratios.append(ratio)
                text_lengths.append(len(text))
        
        return {
            'amharic_ratio': np.mean(amharic_ratios) if amharic_ratios else 0.0,
            'avg_length': np.mean(text_lengths) if text_lengths else 0.0,
            'min_length': np.min(text_lengths) if text_lengths else 0.0,
            'max_length': np.max(text_lengths) if text_lengths else 0.0
        }
    
    def comprehensive_evaluation(
        self,
        test_loader,
        test_prompts: List[str],
        output_dir: str = "evaluation_results"
    ) -> Dict:
        """Run comprehensive evaluation suite."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.logger.info("Starting comprehensive evaluation...")
        
        results = {}
        
        # 1. Perplexity evaluation
        self.logger.info("Evaluating perplexity...")
        perplexity = self.evaluate_perplexity(test_loader)
        results['perplexity'] = perplexity
        self.logger.info(f"Perplexity: {perplexity:.2f}")
        
        # 2. Compression evaluation
        self.logger.info("Evaluating compression...")
        compression_metrics = self.evaluate_compression_ratio(test_loader)
        results['compression'] = compression_metrics
        self.logger.info(f"Mean compression ratio: {compression_metrics['mean_compression_ratio']:.2f}")
        
        # 3. Text generation evaluation
        self.logger.info("Evaluating text generation...")
        all_generated_texts = []
        
        for i, prompt in enumerate(test_prompts):
            self.logger.info(f"Generating for prompt {i+1}/{len(test_prompts)}: {prompt[:30]}...")
            
            generated_texts = self.generate_text(
                prompt=prompt,
                max_length=100,
                temperature=0.8,
                num_samples=3
            )
            
            all_generated_texts.extend(generated_texts)
            
            # Save individual results
            with open(output_path / f"generation_{i:03d}.json", 'w', encoding='utf-8') as f:
                json.dump({
                    'prompt': prompt,
                    'generated_texts': generated_texts
                }, f, ensure_ascii=False, indent=2)
        
        # 4. Cultural safety evaluation
        self.logger.info("Evaluating cultural safety...")
        safety_metrics = self.evaluate_cultural_safety(all_generated_texts)
        results['cultural_safety'] = safety_metrics
        self.logger.info(f"Safety rate: {safety_metrics['safety_rate']:.2%}")
        
        # 5. Amharic quality evaluation
        self.logger.info("Evaluating Amharic quality...")
        quality_metrics = self.evaluate_amharic_quality(all_generated_texts)
        results['amharic_quality'] = quality_metrics
        self.logger.info(f"Amharic ratio: {quality_metrics['amharic_ratio']:.2%}")
        
        # Save comprehensive results
        with open(output_path / "evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Generate visualizations
        self._create_visualizations(results, output_path)
        
        self.logger.info("Evaluation completed!")
        return results
    
    def _create_visualizations(self, results: Dict, output_dir: Path):
        """Create evaluation visualizations."""
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Compression ratio histogram
            if 'compression' in results:
                plt.figure(figsize=(10, 6))
                plt.hist(
                    [results['compression']['mean_compression_ratio']], 
                    bins=30, 
                    alpha=0.7,
                    edgecolor='black'
                )
                plt.axvline(
                    results['compression']['mean_compression_ratio'], 
                    color='red', 
                    linestyle='--', 
                    label=f"Mean: {results['compression']['mean_compression_ratio']:.2f}"
                )
                plt.xlabel('Compression Ratio')
                plt.ylabel('Frequency')
                plt.title('Dynamic Chunking Compression Ratio Distribution')
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / 'compression_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Cultural safety metrics
            if 'cultural_safety' in results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Safety rate pie chart
                safety_rate = results['cultural_safety']['safety_rate']
                unsafe_rate = 1 - safety_rate
                ax1.pie(
                    [safety_rate, unsafe_rate],
                    labels=['Safe', 'Unsafe'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['lightgreen', 'lightcoral']
                )
                ax1.set_title('Cultural Safety Rate')
                
                # Violation types bar chart
                if 'violation_types' in results['cultural_safety']:
                    violation_types = results['cultural_safety']['violation_types']
                    if violation_types:
                        types = list(violation_types.keys())
                        counts = list(violation_types.values())
                        ax2.bar(types, counts)
                        ax2.set_xlabel('Violation Type')
                        ax2.set_ylabel('Count')
                        ax2.set_title('Cultural Safety Violation Types')
                        ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'cultural_safety_metrics.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Summary metrics
            fig, ax = plt.subplots(figsize=(12, 8))
            
            metrics_to_plot = []
            values_to_plot = []
            
            if 'perplexity' in results:
                metrics_to_plot.append('Perplexity')
                values_to_plot.append(results['perplexity'])
            
            if 'compression' in results:
                metrics_to_plot.append('Compression Ratio')
                values_to_plot.append(results['compression']['mean_compression_ratio'])
            
            if 'cultural_safety' in results:
                metrics_to_plot.append('Safety Rate (%)')
                values_to_plot.append(results['cultural_safety']['safety_rate'] * 100)
            
            if 'amharic_quality' in results:
                metrics_to_plot.append('Amharic Ratio (%)')
                values_to_plot.append(results['amharic_quality']['amharic_ratio'] * 100)
            
            if metrics_to_plot:
                bars = ax.bar(metrics_to_plot, values_to_plot, color=['skyblue', 'lightgreen', 'coral', 'gold'])
                ax.set_ylabel('Score')
                ax.set_title('Amharic H-Net Evaluation Summary')
                
                # Add value labels on bars
                for bar, value in zip(bars, values_to_plot):
                    height = bar.get_height()
                    ax.annotate(f'{value:.2f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            self.logger.warning(f"Could not create visualizations: {e}")


def load_test_prompts(prompts_file: str) -> List[str]:
    """Load test prompts from file."""
    if Path(prompts_file).exists():
        with open(prompts_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # Default test prompts
        return [
            "አማርኛ",
            "ቡና",
            "ኢትዮጵያ",
            "መስቀል",
            "ባህል",
            "ንጉሥ",
            "ገና",
            "ፋሲካ",
            "አዲስ አበባ",
            "ሃይማኖት"
        ]


def main():
    parser = argparse.ArgumentParser(description='Evaluate Amharic H-Net model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test-data', type=str, default='data/processed',
                       help='Path to test data directory')
    parser.add_argument('--prompts-file', type=str, default='test_prompts.txt',
                       help='File containing test prompts')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Initialize model with config from checkpoint
    model_config = checkpoint.get('config', {}).get('model', {})
    model = AmharicHNet(
        d_model=model_config.get('d_model', 768),
        n_encoder_layers=model_config.get('n_encoder_layers', 4),
        n_decoder_layers=model_config.get('n_decoder_layers', 4),
        n_main_layers=model_config.get('n_main_layers', 12),
        n_heads=model_config.get('n_heads', 12),
        compression_ratio=model_config.get('compression_ratio', 4.5)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Initialize components
    preprocessor = AmharicPreprocessor()
    cultural_guardrails = AmharicCulturalGuardrails()
    
    # Load test data
    logger.info("Loading test data...")
    _, test_texts = load_data_from_directory(args.test_data, train_ratio=0.0)
    
    if not test_texts:
        logger.warning("No test data found, using validation split")
        train_texts, test_texts = load_data_from_directory(args.test_data, train_ratio=0.8)
        test_texts = test_texts[:min(1000, len(test_texts))]  # Limit for evaluation
    
    # Create test data loader
    _, test_loader = create_data_loaders(
        train_texts=[],  # Empty for eval-only
        val_texts=test_texts,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        max_length=512,
        num_workers=4
    )
    
    # Load test prompts
    test_prompts = load_test_prompts(args.prompts_file)
    logger.info(f"Loaded {len(test_prompts)} test prompts")
    
    # Initialize evaluator
    evaluator = AmharicHNetEvaluator(
        model=model,
        preprocessor=preprocessor,
        cultural_guardrails=cultural_guardrails,
        device=device
    )
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(
        test_loader=test_loader,
        test_prompts=test_prompts,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Perplexity: {results.get('perplexity', 'N/A'):.2f}")
    print(f"Mean Compression Ratio: {results.get('compression', {}).get('mean_compression_ratio', 'N/A'):.2f}")
    print(f"Cultural Safety Rate: {results.get('cultural_safety', {}).get('safety_rate', 'N/A'):.2%}")
    print(f"Amharic Content Ratio: {results.get('amharic_quality', {}).get('amharic_ratio', 'N/A'):.2%}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()