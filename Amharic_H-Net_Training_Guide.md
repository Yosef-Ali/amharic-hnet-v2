# Amharic H-Net Training Guide - Claude Code Orchestrator

## Project Overview
Training an Amharic language model using H-net architecture with smart agentic development workflows. This is the main orchestrator guide for Claude Code CLI to manage the entire project pipeline.

## Quick Start Commands
```bash
# Setup environment
./setup.sh && source venv/bin/activate

# Test installation
python main.py test

# Run full pipeline (use these Task commands in sequence)
```

## Key Technical Decisions

### Model Architecture
- **H-Net Base Configuration**: ~300M parameters (NOT GPT-3 Large/XL scale)
- **Hardware Target**: M2 8GB Mac (perfectly sufficient)
- **Transfer Learning**: Start from Chinese H-net weights (similar script complexity)
- **Memory Strategy**: Mixed precision (fp16) + gradient accumulation

### Why This Works on M2 8GB
- **Model size**: ~1-2GB (300M params)
- **Training overhead**: ~2-3GB
- **System memory**: ~2GB
- **Buffer**: 1-2GB remaining - very safe
- **Training time**: 1-2 weeks max with transfer learning

### Data Requirements
- **Quality over quantity**: Chinese model succeeded with only 46B tokens
- **Target**: 10-50M Amharic tokens initially
- **Amharic ratio**: >70% per document
- **Cultural safety**: >95% compliance required

## Agent Workflow Pipeline

Execute these Task commands in Claude Code CLI:

### 1. Data Collection
```python
Task(
    description="Collect Amharic corpus with cultural validation",
    prompt="amharic-corpus-collector: Collect 1000 Wikipedia articles with >70% Amharic ratio and cultural safety validation. Output to data/raw directory.",
    subagent_type="amharic-corpus-collector"
)
```
**Success Criteria**: >70% Amharic ratio, >95% cultural compliance, multi-dialect coverage

### 2. Linguistic Analysis
```python
Task(
    description="Process corpus for morphological analysis",
    prompt="amharic-linguistic-analyzer: Process corpus from data/raw for morpheme segmentation with >85% accuracy. Handle Ge'ez script complexity and cultural term protection.",
    subagent_type="amharic-linguistic-analyzer"
)
```
**Success Criteria**: >85% morphological accuracy, cultural term protection, confidence scoring

### 3. Architecture Design (Can Run Parallel with Step 2)
```python
Task(
    description="Design H-Net architecture for Amharic",
    prompt="h-net-model-architect: Design optimal H-Net architecture for Amharic morphological processing. Target 300M parameters for M2 8GB hardware with transfer learning from Chinese weights.",
    subagent_type="h-net-model-architect"
)
```
**Success Criteria**: Morpheme-aware components, M2-optimized architecture, transfer learning ready

### 4. Model Training
```python
Task(
    description="Train H-Net model with transfer learning",
    prompt="training-engineer: Train H-Net using processed data with transfer learning from Chinese weights. Optimize for M2 8GB with mixed precision and gradient accumulation.",
    subagent_type="training-engineer"
)
```
**Success Criteria**: Transfer learning active, cultural safety monitoring, >85% accuracy

### 5. Model Evaluation
```python
Task(
    description="Comprehensive model evaluation",
    prompt="evaluation-specialist: Evaluate model for >85% morphological accuracy and >95% cultural safety compliance. Include perplexity metrics and generation quality assessment.",
    subagent_type="evaluation-specialist"
)
```
**Success Criteria**: >85% morphological accuracy, >95% cultural safety, statistical significance

### 6. API Deployment
```python
Task(
    description="Deploy production API",
    prompt="deployment-engineer: Deploy validated model as production API with <200ms response time and real-time cultural safety monitoring.",
    subagent_type="deployment-engineer"
)
```
**Success Criteria**: <200ms response, monitoring active, production safety guardrails

## Training Monitoring - Key Metrics

### Technical Metrics
- **Perplexity**: Watch for steady decrease on Amharic validation set
- **Memory usage**: Should stay under 6GB during training
- **Training speed**: Expect faster convergence with transfer learning
- **Loss curves**: Both training and validation should decrease together

### Amharic-Specific Metrics
- **Morphological accuracy**: >85% segmentation precision
- **Cultural safety compliance**: >95% with zero critical violations
- **Generation quality**: Coherent Amharic text with proper Ge'ez script
- **Multi-dialect coverage**: Validation across Ethiopian/Eritrean variants

## Hardware Optimization for M2

### Memory Management
```python
# Use these settings in training config
batch_size = 4  # Start small, increase if memory allows
gradient_accumulation_steps = 8  # Simulate larger batches
fp16 = true  # Mixed precision - cuts memory in half
max_sequence_length = 512  # Good for Amharic text chunks
```

### Training Strategy
- **Progressive unfreezing**: Start with frozen base layers
- **Learning rate**: Conservative start (1e-5), use warmup
- **Checkpointing**: Save every few hundred steps
- **Early stopping**: Monitor validation perplexity

## Why This Approach Works

### Smart Scaling
- **Amharic morphology**: Rich word structure = efficient learning
- **Transfer learning**: Chinese script similarity gives head start
- **H-Net efficiency**: More memory-efficient than standard transformers
- **Quality data**: Your cultural safety pipeline ensures clean training data

### Agent Advantages
- **Domain expertise**: Each agent specialized for specific tasks
- **Quality compounding**: 85% → 89% → 92% → 95% → 98% → 99%
- **Cultural integration**: Safety embedded throughout, not afterthought
- **Parallel processing**: 3x faster development cycle

## Troubleshooting

### Memory Issues
- Reduce batch_size to 2
- Increase gradient_accumulation_steps
- Use sequence_length = 256

### Training Slow
- Verify MPS (Metal) acceleration active
- Check transfer learning weights loaded correctly
- Monitor GPU utilization

### Cultural Safety Violations
- Review protected terms in cultural_guardrails.py
- Validate input text Amharic ratio
- Check multi-dialect sensitivity settings

## Expected Timeline
- **Setup**: 1 day
- **Data collection**: 2-3 days
- **Preprocessing**: 1-2 days
- **Training**: 1-2 weeks
- **Evaluation**: 2-3 days
- **Deployment**: 1-2 days

**Total**: ~3-4 weeks for complete pipeline

## Success Indicators
- Model generates coherent Amharic text
- Maintains cultural sensitivity across contexts
- Runs efficiently on M2 hardware
- API responds under 200ms
- Evaluation metrics exceed targets

This guide serves as the main orchestrator for Claude Code CLI to manage the entire Amharic H-Net training pipeline efficiently and culturally responsibly.