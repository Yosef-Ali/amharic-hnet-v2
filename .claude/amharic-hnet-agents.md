# Amharic H-Net Development Sub-Agents

## Available Specialized Agents

### 1. **data-collector** - Amharic Corpus Collection Specialist
- **Expertise**: Web scraping, data validation, corpus quality assessment
- **Tools**: Beautiful Soup, Scrapy, data validation frameworks
- **Tasks**: Wikipedia scraping, news corpus collection, social media data gathering
- **Deliverables**: Clean, validated Amharic text corpora with metadata

### 2. **linguistic-analyzer** - Amharic Language Expert  
- **Expertise**: Morphological analysis, dialectal variations, cultural context
- **Tools**: Morpheme segmentation, POS tagging, cultural safety validation
- **Tasks**: Text preprocessing, morphological annotation, dialect classification
- **Deliverables**: Linguistically annotated datasets, morpheme boundaries

### 3. **model-architect** - H-Net Architecture Designer
- **Expertise**: Neural architecture design, transfer learning, optimization
- **Tools**: PyTorch, model optimization, architectural patterns
- **Tasks**: Model design, layer configuration, transfer learning setup
- **Deliverables**: Optimized model architectures, training configurations

### 4. **training-engineer** - Training Pipeline Specialist
- **Expertise**: Distributed training, hyperparameter optimization, monitoring
- **Tools**: Accelerate, Weights & Biases, TensorBoard, multi-GPU training
- **Tasks**: Training setup, monitoring, optimization, checkpointing
- **Deliverables**: Trained models, performance metrics, training logs

### 5. **evaluation-specialist** - Amharic Model Assessment Expert
- **Expertise**: Cultural safety evaluation, morphological accuracy, human evaluation
- **Tools**: Custom evaluation metrics, cultural safety frameworks
- **Tasks**: Model validation, safety assessment, performance benchmarking
- **Deliverables**: Comprehensive evaluation reports, safety certifications

### 6. **deployment-engineer** - Production Deployment Expert
- **Expertise**: API development, cloud deployment, monitoring, scaling
- **Tools**: FastAPI, Docker, Kubernetes, cloud platforms
- **Tasks**: API creation, containerization, deployment, monitoring
- **Deliverables**: Production-ready APIs, deployment infrastructure

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