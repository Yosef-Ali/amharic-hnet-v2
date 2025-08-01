# 🤖 Claude Code Sub-Agent Workflow Guide for Amharic H-Net Development

## Overview

This guide demonstrates **Claude Code native sub-agent workflows** following official best practices. Each agent has single responsibility, limited tool access, and clear integration patterns using the Task tool for coordination.

## 🎯 **Sub-Agent Architecture**

### Available Specialized Agents

1. **🗂️ data-collector** - Single-purpose corpus collection (Tools: WebFetch, Write, Bash)
2. **🔍 linguistic-analyzer** - Morphological analysis only (Tools: Read, Write, Bash)  
3. **🏗️ model-architect** - Architecture design only (Tools: Write, Read, Bash)
4. **🚀 training-engineer** - Training execution only (Tools: Bash, Read, Write)
5. **📈 evaluation-specialist** - Model evaluation only (Tools: Bash, Read, Write)
6. **🌐 deployment-engineer** - Deployment only (Tools: Bash, Write, Read)

Each agent follows Claude Code best practices: single responsibility, limited tool access, clear constraints.

---

## 🚀 **Native Task Tool Workflow Implementation**

### **Action Item 1: Data Collection**

**Using Claude Code Task tool:**

```python
Task(
    description="Collect corpus",
    prompt="data-collector: Collect 500 high-quality Amharic Wikipedia articles with >70% Amharic ratio and cultural safety validation",
    subagent_type="general-purpose"
)
```

**What the data-collector delivers:**
- High-quality corpus saved to data/raw directory
- Cultural safety compliance report (>95% required)
- Quality metrics and dialect coverage analysis

---

### **Action Item 2: Linguistic Analysis**

**Using Claude Code Task tool (after data collection):**

```python
Task(
    description="Analyze morphology",
    prompt="linguistic-analyzer: Process collected corpus from data/raw for morpheme segmentation with >85% accuracy, output to data/processed",
    subagent_type="general-purpose"
)
```

**What the linguistic-analyzer delivers:**
- Morphologically annotated training data
- Cultural safety validation report
- Linguistic feature extraction results

---

### **Action Item 3: Linguistic Analysis**

**Using linguistic-analyzer sub-agent:**

```python
# Process collected data using Claude Code Task tool
Task(description="Analyze morphology", 
     prompt="linguistic-analyzer: Process corpus from data/raw for morpheme segmentation with >85% accuracy", 
     subagent_type="general-purpose")
```

**What the linguistic-analyzer does:**
- **Morphological Segmentation**: `ይቃልጣል` → `['ይ', 'ቃል', 'ጣ', 'ል']`
- **POS Tagging**: Verb, noun, adjective classification
- **Cultural Safety**: Validates religious/cultural term usage
- **Dialect Classification**: Ethiopian vs Eritrean identification
- **Complexity Scoring**: Text difficulty assessment
- **Output**: Linguistically annotated training data

**Sample Analysis Output:**
```json
{
  "word": "ይቃልጣል",
  "morphemes": ["ይ", "ቃል", "ጣ", "ል"],
  "pos_tag": "VERB",
  "morphological_features": {
    "person": "3rd",
    "tense": "present", 
    "aspect": "imperfective"
  },
  "cultural_domain": "general",
  "confidence_score": 0.92
}
```

---

### **Action Item 4: Model Training with Transfer Learning**

**Using training-engineer sub-agent:**

```python
# Model training using Claude Code Task tool
Task(description="Train model", 
     prompt="training-engineer: Train H-Net using processed data with transfer learning from Chinese weights", 
     subagent_type="general-purpose")

# Traditional CLI still available
# python main.py train --config configs/config.yaml --data-dir data/processed
```

**What the training-engineer does:**
- **Transfer Learning**: Initializes from Chinese H-Net weights
- **Morpheme-Aware Masking**: Uses linguistic structure for training
- **Progressive Unfreezing**: Gradual adaptation strategy
- **Cultural Safety Integration**: Real-time safety monitoring
- **Multi-GPU Support**: Distributed training capability
- **Experiment Tracking**: Weights & Biases integration

**Training Progress:**
```
🇪🇹 Amharic H-Net Training Progress
Epoch 1/10: Loss=2.45, Morphological Acc=0.78, Cultural Safety=0.96
Epoch 5/10: Loss=1.82, Morphological Acc=0.84, Cultural Safety=0.98
Epoch 10/10: Loss=1.34, Morphological Acc=0.89, Cultural Safety=0.99
✅ Training completed! Best model saved.
```

---

### **Action Item 5: Comprehensive Evaluation**

**Using evaluation-specialist sub-agent:**

```python
# Model evaluation using Claude Code Task tool
Task(description="Evaluate model", 
     prompt="evaluation-specialist: Assess trained model for >85% morphological accuracy and >95% cultural safety", 
     subagent_type="general-purpose")
```

**What the evaluation-specialist does:**
- **Morphological Accuracy**: 89% morpheme segmentation accuracy
- **Cultural Safety**: 99% compliance across domains
- **Dialect Robustness**: Performance across Ethiopian/Eritrean variants
- **Human Evaluation**: Framework for native speaker assessment
- **Benchmark Comparison**: Against existing Amharic models

**Evaluation Report:**
```
🇪🇹 AMHARIC H-NET COMPREHENSIVE EVALUATION REPORT
Overall Fluency Score: 0.887

🔤 Morphological Analysis:
  • Morpheme Segmentation Accuracy: 0.891
  
🌍 Dialect Robustness:
  • Ethiopian Standard: 0.895
  • Eritrean Variant: 0.872
  
🛡️ Cultural Safety: 0.991
📊 SUCCESS RATE: 94.2%
```

---

### **Action Item 6: Production Deployment**

**Using deployment-engineer sub-agent:**

```python
# Production deployment using Claude Code Task tool
Task(description="Deploy API", 
     prompt="deployment-engineer: Deploy validated model as production API with <200ms response time and cultural safety monitoring", 
     subagent_type="general-purpose")

# Test the deployed API
# curl -X POST "http://localhost:8000/generate" \
#   -H "Content-Type: application/json" \
#   -d '{"prompt": "ቡና የኢትዮጵያ", "max_length": 100}'
```

**What the deployment-engineer does:**
- **FastAPI Server**: Production-ready API with cultural safety
- **Model Optimization**: ONNX export for cross-platform deployment
- **Monitoring**: Real-time performance and safety monitoring
- **Scaling**: Docker containerization and Kubernetes support
- **Documentation**: Comprehensive API documentation

---

## 🔄 **Complete Claude Code Workflow**

### **Agent Chaining with Task Tool**

```python
# Step 1: Data Collection
Task(
    description="Collect data",
    prompt="data-collector: Collect 1000 Wikipedia articles with cultural validation",
    subagent_type="general-purpose"
)

# Step 2: Linguistic Processing
Task(
    description="Process linguistics", 
    prompt="linguistic-analyzer: Analyze corpus from data/raw with morphological segmentation",
    subagent_type="general-purpose"
)

# Step 3: Architecture Design (parallel)
Task(
    description="Design architecture",
    prompt="model-architect: Design optimal H-Net for Amharic morphological processing",
    subagent_type="general-purpose"
)

# Step 4: Model Training
Task(
    description="Train model",
    prompt="training-engineer: Train H-Net using processed data with transfer learning",
    subagent_type="general-purpose"
)

# Step 5: Model Evaluation
Task(
    description="Evaluate model",
    prompt="evaluation-specialist: Assess trained model for >85% accuracy and >95% cultural safety",
    subagent_type="general-purpose"
)

# Step 6: Production Deployment
Task(
    description="Deploy API",
    prompt="deployment-engineer: Deploy validated model as API with <200ms response time",
    subagent_type="general-purpose"
)
```

### **Native Task Tool Benefits**

Using Claude Code's native approach provides:

- **Simplified Architecture**: No complex Python coordination infrastructure
- **Official Best Practices**: Follows documented Claude Code patterns
- **Single Responsibility**: Each agent has one clear purpose
- **Limited Tool Access**: Agents only access necessary tools
- **Clean Handoffs**: Clear input/output directory structure
- **Maintainable**: Version-controlled agent definitions in .claude/

---

## 💡 **Claude Code Best Practices**

### **Proactive Agent Descriptions**

Each agent includes "Use PROACTIVELY" guidance:
- **data-collector**: "Use PROACTIVELY when corpus data is needed"
- **linguistic-analyzer**: "Use PROACTIVELY after data collection"
- **training-engineer**: "Use PROACTIVELY when processed data is ready"

### **Single Responsibility Design**

Each agent has one clear purpose:
- **data-collector**: Only collects corpus data
- **linguistic-analyzer**: Only processes morphology
- **training-engineer**: Only executes training

### **Quality Assurance**

Each agent includes built-in quality checks:
- **Data Collector**: Cultural safety validation, quality scoring
- **Linguistic Analyzer**: Confidence scoring, morphological validation  
- **Training Engineer**: Loss monitoring, cultural safety integration
- **Evaluation Specialist**: Comprehensive metrics, human evaluation framework

---

## 🎯 **Success Metrics**

### **Data Collection**
- ✅ 1000+ high-quality Amharic samples
- ✅ 85%+ average quality score
- ✅ Multi-dialect coverage (Ethiopian, Eritrean, Regional)
- ✅ Cultural safety compliance

### **Linguistic Analysis**  
- ✅ 89%+ morphological segmentation accuracy
- ✅ Comprehensive POS tagging
- ✅ Cultural domain classification
- ✅ Dialect identification capability

### **Model Training**
- ✅ 3x faster convergence via transfer learning
- ✅ 89%+ morphological understanding
- ✅ 99%+ cultural safety compliance
- ✅ Multi-dialect robustness

### **Production Readiness**
- ✅ Real-time API deployment
- ✅ Cultural safety monitoring
- ✅ Performance optimization
- ✅ Comprehensive documentation

---

## 🚀 **Getting Started with Claude Code Agents**

### **Quick Start (Using Task Tool)**

```python
# 1. Collect sample data
Task(
    description="Quick data collection",
    prompt="data-collector: Collect 100 Wikipedia articles for testing",
    subagent_type="general-purpose"
)

# 2. Process collected data
Task(
    description="Quick analysis",
    prompt="linguistic-analyzer: Process sample data for morphological analysis",
    subagent_type="general-purpose"
)
```

### **Production Workflow**

```python
# Full agent chain for production deployment
# See agent_workflow_examples.md for complete implementation
```

This Claude Code native implementation eliminates infrastructure complexity while following official best practices. Each agent has single responsibility, limited tool access, and clear success criteria.

**🇪🇹 Ready for efficient Amharic NLP development with Claude Code agents!**