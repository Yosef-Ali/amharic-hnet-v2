# Amharic H-Net Development Sub-Agents

## Available Specialized Agents

All agents are located in `.claude/agents/` directory with detailed specifications.

### 1. **data-collector** - Amharic Corpus Collection Specialist
- **File**: `.claude/agents/data-collector.md`
- **Expertise**: Web scraping, data validation, corpus quality assessment
- **Tools**: aiohttp, BeautifulSoup, pandas, asyncio
- **Tasks**: Wikipedia scraping, news corpus collection, cultural safety validation
- **Deliverables**: Clean JSON corpora with metadata and quality scores

### 2. **linguistic-analyzer** - Amharic Language Expert  
- **File**: `.claude/agents/linguistic-analyzer.md`
- **Expertise**: Morphological analysis, dialectal variations, cultural context
- **Tools**: morfessor, numpy, regex, unicodedata
- **Tasks**: Morpheme segmentation, POS tagging, dialect classification
- **Deliverables**: Linguistically annotated datasets with confidence scores

### 3. **model-architect** - H-Net Architecture Designer
- **File**: `.claude/agents/model-architect.md`
- **Expertise**: Neural architecture design, transfer learning, optimization
- **Tools**: torch.nn, transformers, einops, onnx
- **Tasks**: Architecture design, transfer learning strategy, optimization
- **Deliverables**: Optimized model architectures and implementation plans

### 4. **training-engineer** - Training Pipeline Specialist
- **File**: `.claude/agents/training-engineer.md`
- **Expertise**: Distributed training, hyperparameter optimization, monitoring
- **Tools**: accelerate, wandb, tensorboard, docker
- **Tasks**: Environment setup, training orchestration, performance monitoring
- **Deliverables**: Trained models, training logs, environment validation

### 5. **evaluation-specialist** - Amharic Model Assessment Expert
- **File**: `.claude/agents/evaluation-specialist.md`
- **Expertise**: Cultural safety evaluation, morphological accuracy, benchmarking
- **Tools**: sklearn, scipy, matplotlib, pandas
- **Tasks**: Comprehensive evaluation, cultural safety audit, human evaluation
- **Deliverables**: Evaluation reports, cultural safety certification

### 6. **deployment-engineer** - Production Deployment Expert
- **File**: `.claude/agents/deployment-engineer.md`
- **Expertise**: API development, cloud deployment, monitoring, scaling
- **Tools**: fastapi, docker, kubernetes, prometheus
- **Tasks**: API deployment, containerization, monitoring setup
- **Deliverables**: Production APIs, deployment configurations, monitoring dashboards

## Agent Collaboration Workflows

### Data Collection → Linguistic Analysis → Model Training
1. **data-collector** gathers raw Amharic texts
2. **linguistic-analyzer** processes and annotates the data
3. **training-engineer** uses processed data for model training

### Model Architecture → Training → Evaluation
1. **model-architect** designs optimal H-Net architecture
2. **training-engineer** implements training pipeline
3. **evaluation-specialist** validates model performance

### Training → Evaluation → Deployment
1. **training-engineer** produces trained model
2. **evaluation-specialist** validates safety and performance
3. **deployment-engineer** creates production deployment

## Usage Guidelines

- Use agents **proactively** based on task complexity
- Each agent has **specialized knowledge** and domain expertise
- Agents can **work collaboratively** on complex projects
- Always specify **clear requirements** when invoking agents
- **Chain agents** for end-to-end workflows

## Agent Invocation Examples

```bash
# Data collection workflow
/data-collector "Collect 10,000 high-quality Amharic news articles from BBC Amharic, EBC, and Fana Broadcasting"

# Linguistic analysis workflow  
/linguistic-analyzer "Process collected Amharic texts for morphological segmentation and cultural safety validation"

# Model architecture workflow
/model-architect "Design optimal H-Net architecture for Amharic with transfer learning from Chinese model"

# Training workflow
/training-engineer "Setup distributed training pipeline with morpheme-aware masking and cultural safety integration"

# Evaluation workflow
/evaluation-specialist "Conduct comprehensive evaluation including morphological accuracy and cultural safety assessment"

# Deployment workflow
/deployment-engineer "Create production API for Amharic H-Net with cultural safety monitoring and multi-dialect support"
```