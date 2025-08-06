#!/usr/bin/env python3
"""
Amharic H-Net v2 - Main CLI Entry Point

This script provides a unified command-line interface for training, evaluation,
preprocessing, and text generation with the Amharic H-Net model.

Usage:
    python main.py train --config configs/config.yaml --data-dir data/processed
    python main.py evaluate --model-path outputs/checkpoint_best.pt
    python main.py generate --model-path outputs/checkpoint_best.pt --prompt "አማርኛ"
    python main.py preprocess --input data/raw/corpus.txt --output data/processed
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import torch
import yaml
from typing import Dict, List

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.hnet_amharic import AmharicHNet
from src.preprocessing.prepare_amharic import AmharicPreprocessor, create_sample_data
from src.safety.cultural_guardrails import AmharicCulturalGuardrails
from src.training.train import main as train_main
from src.evaluation.evaluate import main as evaluate_main


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(model_path: str, device: torch.device) -> AmharicHNet:
    """Load trained model from checkpoint."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model config from checkpoint
    model_config = checkpoint.get('config', {}).get('model', {})
    
    # Initialize model
    model = AmharicHNet(
        d_model=model_config.get('d_model', 768),
        n_encoder_layers=model_config.get('n_encoder_layers', 4),
        n_decoder_layers=model_config.get('n_decoder_layers', 4),
        n_main_layers=model_config.get('n_main_layers', 12),
        n_heads=model_config.get('n_heads', 12),
        compression_ratio=model_config.get('compression_ratio', 4.5)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model


def cmd_train(args):
    """Execute training command."""
    print("🚀 Starting Amharic H-Net training...")
    
    # Set up arguments for training script
    sys.argv = [
        'train.py',
        '--config', args.config,
        '--data-dir', args.data_dir,
        '--output-dir', args.output_dir,
        '--device', args.device
    ]
    
    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    
    if args.wandb:
        sys.argv.append('--wandb')
    
    # Execute training
    train_main()


def cmd_evaluate(args):
    """Execute evaluation command."""
    print("📊 Starting Amharic H-Net evaluation...")
    
    # Set up arguments for evaluation script
    sys.argv = [
        'evaluate.py',
        '--model-path', args.model_path,
        '--test-data', args.test_data,
        '--output-dir', args.output_dir,
        '--device', args.device,
        '--batch-size', str(args.batch_size)
    ]
    
    if args.prompts_file:
        sys.argv.extend(['--prompts-file', args.prompts_file])
    
    # Execute evaluation
    evaluate_main()


def cmd_generate(args):
    """Execute text generation command."""
    print("✨ Generating Amharic text...")
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, device)
    model.eval()
    
    # Initialize components
    preprocessor = AmharicPreprocessor()
    cultural_guardrails = AmharicCulturalGuardrails()
    
    # Generate text
    print(f"Generating text for prompt: '{args.prompt}'")
    
    # Preprocess prompt
    cleaned_prompt = preprocessor.clean_text(args.prompt)
    prompt_bytes = preprocessor.extract_byte_sequences(cleaned_prompt, max_length=100)
    
    # Convert to tensor
    input_ids = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
    
    # Generate multiple samples
    generated_texts = []
    
    with torch.no_grad():
        for sample_idx in range(args.num_samples):
            print(f"Generating sample {sample_idx + 1}/{args.num_samples}...")
            
            # Use model's built-in generation method
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k
            )
            
            # Decode generated text
            generated_bytes = generated_ids[0].cpu().numpy().tolist()
            generated_text = preprocessor.decode_byte_sequence(generated_bytes)
            
            # Remove original prompt from output
            if cleaned_prompt in generated_text:
                generated_text = generated_text.replace(cleaned_prompt, "").strip()
            
            # Check cultural safety
            is_safe, violations = cultural_guardrails.check_cultural_safety(generated_text)
            
            generated_texts.append({
                'text': generated_text,
                'is_safe': is_safe,
                'violations': [v.context for v in violations] if violations else []
            })
    
    # Display results
    print("\n" + "="*60)
    print("GENERATION RESULTS")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"Cleaned prompt: {cleaned_prompt}")
    print()
    
    for i, result in enumerate(generated_texts, 1):
        print(f"Sample {i}:")
        print(f"Text: {result['text']}")
        print(f"Safe: {'✅' if result['is_safe'] else '❌'}")
        if result['violations']:
            print(f"Violations: {', '.join(result['violations'])}")
        print("-" * 40)
    
    # Save results if output file specified
    if args.output_file:
        import json
        output_data = {
            'prompt': args.prompt,
            'cleaned_prompt': cleaned_prompt,
            'generated_texts': generated_texts,
            'generation_params': {
                'max_length': args.max_length,
                'temperature': args.temperature,
                'top_k': args.top_k,
                'num_samples': args.num_samples
            }
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {args.output_file}")


def cmd_preprocess(args):
    """Execute preprocessing command."""
    print("🔄 Preprocessing Amharic text data...")
    
    # Initialize preprocessor
    preprocessor = AmharicPreprocessor()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.create_sample:
        # Create sample data
        sample_file = output_path / "sample_amharic.txt"
        create_sample_data(str(sample_file))
        print(f"Sample data created: {sample_file}")
        return
    
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Process the input file
    stats = preprocessor.create_training_dataset(
        input_file=args.input,
        output_dir=str(output_path),
        chunk_size=args.chunk_size
    )
    
    print("Preprocessing completed!")
    print(f"Original texts: {stats['original_count']}")
    print(f"Processed texts: {stats['processed_count']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Output directory: {output_path}")


def cmd_test(args):
    """Execute test command to verify installation."""
    print("🧪 Testing Amharic H-Net installation...")
    
    try:
        # Test imports
        print("✓ Testing imports...")
        from src.models.hnet_amharic import AmharicHNet
        from src.preprocessing.prepare_amharic import AmharicPreprocessor
        from src.safety.cultural_guardrails import AmharicCulturalGuardrails
        
        # Test model initialization
        print("✓ Testing model initialization...")
        model = AmharicHNet(d_model=256, n_main_layers=2, n_heads=8)
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test preprocessor
        print("✓ Testing preprocessor...")
        preprocessor = AmharicPreprocessor()
        test_text = "አማርኛ ቋንቋ ነው።"
        cleaned = preprocessor.clean_text(test_text)
        print(f"  Original: {test_text}")
        print(f"  Cleaned: {cleaned}")
        
        # Test cultural guardrails
        print("✓ Testing cultural guardrails...")
        guardrails = AmharicCulturalGuardrails()
        is_safe, violations = guardrails.check_cultural_safety(test_text)
        print(f"  Text safety: {'Safe' if is_safe else 'Unsafe'}")
        
        # Test device availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ PyTorch device: {device}")
        
        print("\n🎉 All tests passed! Amharic H-Net is ready to use.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Amharic H-Net v2 - Hierarchical Neural Network for Amharic Language Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --config configs/config.yaml --data-dir data/processed

  # Evaluate a trained model
  python main.py evaluate --model-path outputs/checkpoint_best.pt

  # Generate text
  python main.py generate --model-path outputs/checkpoint_best.pt --prompt "አማርኛ"

  # Preprocess data
  python main.py preprocess --input data/raw/corpus.txt --output data/processed

  # Test installation
  python main.py test
        """
    )
    
    # Global arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the Amharic H-Net model')
    train_parser.add_argument('--config', type=str, default='configs/config.yaml',
                             help='Path to configuration file')
    train_parser.add_argument('--data-dir', type=str, default='data/processed',
                             help='Path to training data directory')
    train_parser.add_argument('--output-dir', type=str, default='outputs',
                             help='Output directory for checkpoints and logs')
    train_parser.add_argument('--resume', type=str,
                             help='Path to checkpoint to resume from')
    train_parser.add_argument('--wandb', action='store_true',
                             help='Use Weights & Biases for logging')
    train_parser.set_defaults(func=cmd_train)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to trained model checkpoint')
    eval_parser.add_argument('--test-data', type=str, default='data/processed',
                            help='Path to test data directory')
    eval_parser.add_argument('--prompts-file', type=str,
                            help='File containing test prompts')
    eval_parser.add_argument('--output-dir', type=str, default='evaluation_results',
                            help='Output directory for results')
    eval_parser.add_argument('--batch-size', type=int, default=16,
                            help='Batch size for evaluation')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Generate command  
    gen_parser = subparsers.add_parser('generate', help='Generate text with the model')
    gen_parser.add_argument('--model-path', type=str, required=True,
                           help='Path to trained model checkpoint')
    gen_parser.add_argument('--prompt', type=str, required=True,
                           help='Text prompt for generation')
    gen_parser.add_argument('--max-length', type=int, default=200,
                           help='Maximum generation length')
    gen_parser.add_argument('--temperature', type=float, default=0.8,
                           help='Sampling temperature')
    gen_parser.add_argument('--top-k', type=int, default=50,
                           help='Top-k sampling parameter')
    gen_parser.add_argument('--num-samples', type=int, default=3,
                           help='Number of samples to generate')
    gen_parser.add_argument('--output-file', type=str,
                           help='File to save generation results')
    gen_parser.set_defaults(func=cmd_generate)
    
    # Preprocess command
    prep_parser = subparsers.add_parser('preprocess', help='Preprocess text data')
    prep_parser.add_argument('--input', type=str,
                            help='Input text file to preprocess')
    prep_parser.add_argument('--output', type=str, default='data/processed',
                            help='Output directory for processed data')
    prep_parser.add_argument('--chunk-size', type=int, default=1000,
                            help='Number of texts per chunk file')
    prep_parser.add_argument('--create-sample', action='store_true',
                            help='Create sample Amharic data for testing')
    prep_parser.set_defaults(func=cmd_preprocess)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test installation and setup')
    test_parser.set_defaults(func=cmd_test)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Show banner
    print("=" * 60)
    print("🇪🇹 Amharic H-Net v2 - Hierarchical Neural Network")
    print("   Advanced Language Modeling for Amharic")
    print("=" * 60)
    
    # Execute command
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Command failed: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()