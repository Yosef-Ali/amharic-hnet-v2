# Amharic H-Net Agent Workflow Examples

## Overview
This document demonstrates how to use Claude Code's native Task tool for coordinating specialized agents in Amharic H-Net development, following official best practices.

## Single Agent Usage

### Data Collection
```python
Task(
    description="Collect corpus",
    prompt="data-collector: Collect 500 high-quality Amharic Wikipedia articles with cultural safety validation",
    subagent_type="general-purpose"
)
```

### Linguistic Analysis
```python
Task(
    description="Analyze morphology",
    prompt="linguistic-analyzer: Process collected Wikipedia corpus for morpheme segmentation with >85% accuracy",
    subagent_type="general-purpose"
)
```

### Model Training
```python
Task(
    description="Train model",
    prompt="training-engineer: Train Amharic H-Net using processed corpus with transfer learning from Chinese weights",
    subagent_type="general-purpose"
)
```

## Agent Chaining Workflows

### Complete Development Pipeline
```python
# Step 1: Data Collection
Task(
    description="Collect data",
    prompt="data-collector: Collect 1000 Wikipedia articles with >70% Amharic ratio and cultural validation",
    subagent_type="general-purpose"
)

# Step 2: Linguistic Processing (after Step 1 completes)
Task(
    description="Process linguistics",
    prompt="linguistic-analyzer: Analyze collected corpus from data/raw, output processed data to data/processed with morphological annotations",
    subagent_type="general-purpose"
)

# Step 3: Architecture Design (parallel with Step 2)
Task(
    description="Design architecture",
    prompt="model-architect: Design optimal H-Net architecture for Amharic with morpheme-aware components",
    subagent_type="general-purpose"
)

# Step 4: Model Training (after Steps 2 & 3)
Task(
    description="Train model",
    prompt="training-engineer: Train H-Net using processed data from data/processed with architectural specifications",
    subagent_type="general-purpose"
)

# Step 5: Model Evaluation (after Step 4)
Task(
    description="Evaluate model",
    prompt="evaluation-specialist: Evaluate trained model for morphological accuracy >85% and cultural safety >95%",
    subagent_type="general-purpose"
)

# Step 6: Production Deployment (after Step 5)
Task(
    description="Deploy API",
    prompt="deployment-engineer: Deploy validated model as production API with <200ms response time and cultural monitoring",
    subagent_type="general-purpose"
)
```

## Parallel Processing Examples

### Multi-Source Data Collection
```python
# Collect from Wikipedia
Task(
    description="Collect Wikipedia",
    prompt="data-collector: Collect 500 Wikipedia articles with quality validation",
    subagent_type="general-purpose"
)

# Collect from BBC (parallel)
Task(
    description="Collect BBC",
    prompt="data-collector: Collect 300 BBC Amharic articles with cultural safety checks",
    subagent_type="general-purpose"
)
```

### Architecture and Data Processing
```python
# Design architecture
Task(
    description="Design architecture",
    prompt="model-architect: Create morpheme-aware H-Net architecture optimized for Amharic",
    subagent_type="general-purpose"
)

# Process linguistic data (parallel)
Task(
    description="Process data",
    prompt="linguistic-analyzer: Analyze corpus with morphological segmentation and cultural validation",
    subagent_type="general-purpose"
)
```

## Quality Gates and Validation

### Data Quality Validation
```python
Task(
    description="Validate data quality",
    prompt="data-collector: Verify corpus quality meets standards - >70% Amharic ratio, >95% cultural safety, diverse dialect coverage",
    subagent_type="general-purpose"
)
```

### Model Performance Validation
```python
Task(
    description="Validate performance",
    prompt="evaluation-specialist: Confirm model meets production standards - >85% morphological accuracy, >95% cultural safety, <200ms inference",
    subagent_type="general-purpose"
)
```

## Error Handling and Recovery

### Conditional Processing
```python
# Check if data collection succeeded before processing
Task(
    description="Check collection status",
    prompt="data-collector: Verify successful collection of minimum 500 samples before proceeding to analysis",
    subagent_type="general-purpose"
)

# Only proceed if quality thresholds met
Task(
    description="Conditional training",
    prompt="training-engineer: Only start training if processed data meets >85% morphological accuracy threshold",
    subagent_type="general-purpose"
)
```

## Best Practices

### 1. Single Responsibility
Each agent handles one specific task with clear constraints and success criteria.

### 2. Limited Tool Access
Agents only use tools necessary for their specific purpose (WebFetch for data-collector, Read/Write for processing).

### 3. Clear Handoffs
Each step outputs to specific directories that the next agent can consume.

### 4. Quality Validation
Every agent includes quality thresholds and validation criteria.

### 5. Cultural Safety
All agents incorporate cultural safety considerations appropriate to their domain.

## Directory Structure
```
data/
├── raw/           # Output from data-collector
├── processed/     # Output from linguistic-analyzer
models/
├── architectures/ # Output from model-architect
outputs/
├── checkpoints/   # Output from training-engineer
├── evaluations/   # Output from evaluation-specialist
deployment/        # Output from deployment-engineer
```

This approach eliminates complex coordination infrastructure while leveraging Claude Code's native capabilities for efficient, maintainable agent workflows.