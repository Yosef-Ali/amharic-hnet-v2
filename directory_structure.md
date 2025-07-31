# Amharic H-Net v2 Directory Structure

```
amharic-hnet-v2/
├── README.md                          # Project overview and quick start
├── requirements.txt                   # Python dependencies with pinned versions
├── setup.sh                          # Environment setup script
├── .env                              # Environment variables
├── .gitignore                        # Git ignore patterns
├── pyproject.toml                    # Python project configuration
├── .pre-commit-config.yaml           # Pre-commit hooks configuration
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── models/                       # Model architectures
│   │   ├── __init__.py
│   │   ├── hnet_amharic.py          # Main H-Net implementation
│   │   └── transfer_learning.py     # Transfer learning utilities
│   ├── training/                     # Training modules
│   │   ├── __init__.py
│   │   ├── train.py                 # Main training script
│   │   ├── data_loader.py           # Data loading utilities
│   │   └── morpheme_masking.py      # Morpheme-aware masking
│   ├── preprocessing/                # Data preprocessing
│   │   ├── __init__.py
│   │   ├── prepare_amharic.py       # Amharic text preprocessing
│   │   └── data_augmentation.py     # Data augmentation techniques
│   ├── evaluation/                   # Evaluation modules
│   │   ├── __init__.py
│   │   ├── evaluate.py              # Evaluation scripts
│   │   └── amharic_metrics.py       # Amharic-specific metrics
│   └── safety/                       # Safety and cultural considerations
│       ├── __init__.py
│       └── cultural_guardrails.py   # Cultural sensitivity checks
│
├── configs/                          # Configuration files
│   ├── config.yaml                   # Main configuration
│   ├── training/                     # Training configurations
│   │   ├── base.yaml
│   │   ├── large_model.yaml
│   │   └── transfer_learning.yaml
│   ├── model/                        # Model configurations
│   │   ├── hnet_small.yaml
│   │   ├── hnet_base.yaml
│   │   └── hnet_large.yaml
│   └── data/                         # Data processing configurations
│       ├── preprocessing.yaml
│       └── augmentation.yaml
│
├── data/                             # Data directory
│   ├── raw/                          # Raw datasets
│   │   └── sample_amharic.txt
│   ├── processed/                    # Processed datasets
│   ├── morpheme_annotated/           # Morpheme-annotated data
│   └── evaluation/                   # Evaluation datasets
│
├── experiments/                      # Experiment tracking
│   ├── configs/                      # Experiment-specific configs
│   ├── logs/                         # Training logs
│   ├── checkpoints/                  # Model checkpoints
│   └── results/                      # Experiment results
│
├── outputs/                          # Training outputs
│   ├── models/                       # Trained models
│   ├── tensorboard/                  # TensorBoard logs
│   └── wandb/                        # Weights & Biases artifacts
│
├── scripts/                          # Utility scripts
│   ├── training/                     # Training scripts
│   │   ├── train_base_model.sh
│   │   ├── train_large_model.sh
│   │   └── resume_training.sh
│   ├── evaluation/                   # Evaluation scripts
│   │   ├── evaluate_model.sh
│   │   └── benchmark_performance.sh
│   └── preprocessing/                # Data preprocessing scripts
│       ├── prepare_dataset.sh
│       └── validate_data.sh
│
├── tests/                            # Test suite
│   ├── unit/                         # Unit tests
│   │   ├── test_models.py
│   │   ├── test_preprocessing.py
│   │   └── test_training.py
│   ├── integration/                  # Integration tests
│   │   ├── test_training_pipeline.py
│   │   └── test_evaluation_pipeline.py
│   └── conftest.py                   # Test configuration
│
├── notebooks/                        # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── model_analysis.ipynb
│   └── training_visualization.ipynb
│
├── models/                           # Saved models (legacy structure)
│
└── docs/                             # Documentation
    ├── setup_guide.md
    ├── training_guide.md
    ├── troubleshooting.md
    └── api_reference.md
```

## Directory Descriptions

### Core Directories

- **src/**: Main source code organized by functionality
- **configs/**: Configuration files for different training scenarios
- **data/**: All datasets in different processing stages
- **experiments/**: Experiment tracking and management
- **outputs/**: Training outputs and artifacts

### Development Directories

- **scripts/**: Automation scripts for common tasks
- **tests/**: Comprehensive test suite
- **notebooks/**: Analysis and visualization notebooks
- **docs/**: Project documentation

### Key Features

1. **Experiment Organization**: Separate directories for configs, logs, checkpoints, and results
2. **Configuration Management**: Hierarchical config structure for different scenarios
3. **Output Management**: Organized outputs with TensorBoard and W&B integration
4. **Testing Structure**: Separate unit and integration test directories
5. **Documentation**: Centralized documentation with guides and references