# Amharic H-Net v2

🇪🇹 **Advanced Hierarchical Neural Network for Amharic Language Modeling**

Amharic H-Net v2 is a state-of-the-art language model specifically designed for Amharic text generation and understanding. It uses hierarchical neural networks with dynamic chunking to handle the morphological complexity of Amharic without traditional tokenization.

## ✨ Features

- **🔄 Dynamic Chunking**: No tokenization needed - works directly with byte sequences
- **🏗️ Hierarchical Architecture**: Multi-level processing for better morpheme understanding
- **🛡️ Cultural Safety**: Advanced guardrails with dialect-aware cultural protection
- **📊 Morpheme-Aware**: Deep Amharic morphological analysis with prefix/suffix detection
- **🌍 Multi-Dialect Support**: Handles Addis Ababa, Gojjam, and Eritrean variants
- **⚡ Efficient Training**: Optimized for both CPU and GPU training
- **🎯 Compression Control**: Configurable compression ratios for different use cases
- **🔤 Space-Free Processing**: Handles authentic Amharic text without artificial spaces
- **🧪 Comprehensive Testing**: Built-in morpheme and cultural safety validation

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd amharic-hnet-v2
   ```

2. **Run the setup script**:
   ```bash
   ./setup.sh
   ```

3. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

### Basic Usage

1. **Test the installation**:
   ```bash
   python main.py test
   ```

2. **Train a model** (using sample data):
   ```bash
   python main.py train --config configs/config.yaml --data-dir data/processed
   ```

3. **Generate text**:
   ```bash
   python main.py generate --model-path outputs/checkpoint_best.pt --prompt "አማርኛ"
   ```

4. **Evaluate the model**:
   ```bash
   python main.py evaluate --model-path outputs/checkpoint_best.pt
   ```

## 📁 Project Structure

```
amharic-hnet-v2/
├── src/                          # Source code
│   ├── models/                   # H-Net model implementation
│   │   └── hnet_amharic.py      # Main model architecture
│   ├── preprocessing/           # Text preprocessing utilities
│   │   └── prepare_amharic.py   # Amharic-specific preprocessing
│   ├── safety/                  # Cultural safety components
│   │   └── cultural_guardrails.py # Cultural safety checks
│   ├── training/                # Training infrastructure
│   │   ├── train.py            # Training script
│   │   └── data_loader.py      # Data loading utilities
│   └── evaluation/              # Evaluation tools
│       └── evaluate.py         # Comprehensive evaluation
├── configs/                     # Configuration files
│   └── config.yaml             # Main configuration
├── data/                       # Data directories
│   ├── raw/                    # Raw text files
│   ├── processed/              # Processed training data
│   └── morpheme_annotated/     # Morpheme annotations
├── outputs/                    # Model checkpoints and logs
├── evaluation_results/         # Evaluation outputs
├── main.py                     # CLI entry point
├── setup.sh                    # Setup script
└── requirements.txt            # Python dependencies
```

## 🔧 Configuration

The main configuration is in `configs/config.yaml`. Key parameters:

```yaml
model:
  d_model: 768              # Hidden dimension
  compression_ratio: 4.5    # Target compression ratio
  n_main_layers: 12        # Number of transformer layers

training:
  batch_size: 16           # Training batch size
  learning_rate: 1e-4      # Learning rate
  num_epochs: 10           # Training epochs

cultural_safety:
  enable_runtime_checking: true  # Enable cultural safety
  protect_sacred_terms: true     # Protect sacred terms
```

## 📊 Model Architecture

### H-Net Components

1. **Dynamic Chunker**: Identifies morpheme boundaries without tokenization
2. **Hierarchical Encoder**: Processes text at byte and chunk levels
3. **Main Transformer**: Standard transformer layers for language modeling
4. **Cultural Safety Layer**: Monitors and filters inappropriate content

### Key Innovations

- **Byte-level Processing**: Works directly with UTF-8 bytes
- **Morpheme Awareness**: Understands Amharic morphological patterns
- **Cultural Context**: Respects Amharic cultural and religious sensitivities
- **Dynamic Compression**: Adapts chunking to text complexity

## 🛡️ Cultural Safety

The model includes comprehensive cultural safety features:

- **Sacred Terms Protection**: Prevents inappropriate use of religious/cultural terms
- **Context-Aware Filtering**: Understands cultural context and sensitivities
- **Real-time Monitoring**: Checks generated content for cultural appropriateness
- **Violation Reporting**: Detailed feedback on cultural safety issues

### Protected Terms Examples
- `ቡና` (Coffee ceremony) - Protected as cultural ritual
- `መስቀል` (Cross) - Protected as religious symbol  
- `ቀዳማዊ` (Emperor) - Protected as historical title

## 📈 Training

### Data Preparation

1. **Prepare your data**:
   ```bash
   python main.py preprocess --input data/raw/your_corpus.txt --output data/processed
   ```

2. **Train the model**:
   ```bash
   python main.py train --config configs/config.yaml --data-dir data/processed
   ```

### Training Features

- **Mixed Precision**: Automatic mixed precision for faster training
- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Early Stopping**: Prevents overfitting
- **Tensorboard Logging**: Real-time training monitoring

## 🎯 Evaluation

The evaluation suite includes:

- **Perplexity**: Language modeling performance
- **Compression Ratio**: Dynamic chunking effectiveness  
- **Cultural Safety**: Safety violation detection
- **Amharic Quality**: Language-specific quality metrics

### Running Evaluation

```bash
python main.py evaluate \
  --model-path outputs/checkpoint_best.pt \
  --test-data data/processed \
  --output-dir evaluation_results
```

## 🔤 Text Generation

### Basic Generation

```bash
python main.py generate \
  --model-path outputs/checkpoint_best.pt \
  --prompt "አማርኛ ቋንቋ" \
  --max-length 200 \
  --temperature 0.8
```

### Advanced Parameters

- `--temperature`: Controls randomness (0.1-2.0)
- `--top-k`: Top-k sampling parameter
- `--top-p`: Nucleus sampling parameter
- `--num-samples`: Number of samples to generate

## 🔍 Development

### Adding New Features

1. **Model Components**: Add to `src/models/`
2. **Preprocessing**: Extend `src/preprocessing/`
3. **Safety Rules**: Update `src/safety/cultural_guardrails.py`
4. **Training Logic**: Modify `src/training/`

### Testing

```bash
# Test installation
python main.py test

# Test Amharic morpheme processing
python tests/test_morphemes.py

# Test preprocessing
python main.py preprocess --create-sample --output data/test

# Test with small config
python main.py train --config configs/config.yaml --data-dir data/test
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 8GB+ RAM (16GB+ recommended)
- 2GB+ disk space

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ethiopian AI community for cultural guidance
- Amharic language experts for linguistic insights
- Open source communities for foundational tools

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: This README and inline code comments

## 🗺️ Roadmap

- [ ] Multi-GPU distributed training
- [ ] ONNX model export
- [ ] REST API server
- [ ] Web interface
- [ ] Integration with other Ethiopian languages
- [ ] Mobile deployment support

---

**Made with ❤️ for the Amharic language community**