# Amharic H-Net v2 - Claude Code Native Sub-Agent System

## 🎯 Implementation Status - PRODUCTION READY

### ✅ Claude Code Native Implementation - COMPLETE
- **6 single-responsibility agents** following official best practices
- **Task tool coordination** using Claude Code's native approach
- **Limited tool access** for security and performance
- **Cultural safety integration** across all workflows

### ✅ Clean Architecture - SIMPLIFIED
```
.claude/
├── agents/                 # ✅ 6 Claude Code native agents
│   ├── data-collector.md
│   ├── linguistic-analyzer.md
│   ├── model-architect.md
│   ├── training-engineer.md
│   ├── evaluation-specialist.md
│   └── deployment-engineer.md
├── instructions.md         # ✅ Claude Code project instructions
├── README.md              # ✅ Agent system overview
└── settings.local.json    # ✅ Proper permissions

Root Files:
├── CLAUDE.md              # ✅ Claude Code project memory
├── README.md              # ✅ GitHub public documentation
└── agent_workflow_examples.md # ✅ Task tool usage examples
```

### ✅ Specialized Agents - ALL DEFINED
1. **data-collector** - Amharic corpus collection with cultural validation
2. **linguistic-analyzer** - Morphological analysis and dialect assessment  
3. **model-architect** - H-Net architecture optimization for Amharic
4. **training-engineer** - Training pipeline with transfer learning
5. **evaluation-specialist** - Cultural safety and performance assessment
6. **deployment-engineer** - Production API with monitoring

### ✅ Key Features - IMPLEMENTED
- **Morpheme-aware processing** for Ge'ez script
- **Cultural safety guardrails** with 99%+ compliance target
- **Multi-dialect support** (Ethiopian/Eritrean/Regional)
- **Transfer learning** from Chinese H-Net to Amharic
- **Production API** with <200ms response time target
- **Quality validation** with automated assessment

## 🚀 Usage Examples

### Claude Code Task Tool Usage
```python
# Data collection with cultural validation
Task(description="Collect corpus", 
     prompt="data-collector: Collect 1000 Wikipedia articles with >70% Amharic ratio and cultural validation", 
     subagent_type="general-purpose")

# Linguistic analysis with morpheme segmentation  
Task(description="Analyze morphology", 
     prompt="linguistic-analyzer: Process corpus from data/raw for morpheme segmentation with >85% accuracy", 
     subagent_type="general-purpose")

# Model training with transfer learning
Task(description="Train model", 
     prompt="training-engineer: Train H-Net using processed data with transfer learning from Chinese weights", 
     subagent_type="general-purpose")
```

### Complete Development Workflow Chain
```python
# Step 1: Data Collection
Task(description="Collect data", prompt="data-collector: Collect 1000 Wikipedia articles", subagent_type="general-purpose")

# Step 2: Linguistic Processing
Task(description="Process linguistics", prompt="linguistic-analyzer: Analyze collected corpus", subagent_type="general-purpose")

# Step 3: Model Training
Task(description="Train model", prompt="training-engineer: Train H-Net with processed data", subagent_type="general-purpose")

# Step 4: Evaluation
Task(description="Evaluate model", prompt="evaluation-specialist: Assess model performance", subagent_type="general-purpose")

# Step 5: Deployment
Task(description="Deploy API", prompt="deployment-engineer: Deploy as production API", subagent_type="general-purpose")
```

### Traditional CLI (Still Available)
```bash
# Direct model operations
python main.py train --config configs/config.yaml
python main.py generate --model-path outputs/best.pt --prompt "አማርኛ"
```

## 📊 Quality Standards - ENFORCED

### Data Quality
- **>70% Amharic character ratio** with cultural compliance validation
- **Multi-source collection** (Wikipedia, BBC sources)
- **Dialect coverage** across Ethiopian and Eritrean variants

### Morphological Accuracy  
- **>85% segmentation precision** with confidence scoring
- **Cultural term protection** for religious/historical contexts
- **Comprehensive linguistic feature extraction**

### Cultural Safety
- **>95% compliance** with zero critical violations
- **Religious context protection** (መስቀል, እግዚአብሔር, etc.)
- **Historical sensitivity** (ቀዳማዊ, ላሊበላ, etc.)
- **Multi-dialect awareness** protocols

### Production Performance
- **<200ms API response time** with monitoring
- **Single-responsibility agents** with clear success criteria
- **Comprehensive error handling** and logging
- **Native Claude Code integration**

## 🔄 Claude Code Native Workflows

### Task Tool Sequential Chain
```python
# Step-by-step execution using Claude Code Task tool
data-collector → linguistic-analyzer → training-engineer → evaluation-specialist → deployment-engineer
```

### Parallel Processing with Task Tool
```python
# Multiple concurrent tasks
Task(..., prompt="data-collector: Wikipedia articles", ...)
Task(..., prompt="data-collector: BBC articles", ...)
# Then combine results for linguistic analysis
```

### Quality Validation
- **Single responsibility** - each agent has one clear purpose
- **Limited tool access** - agents only use necessary tools for security
- **Explicit constraints** - success criteria defined for each agent
- **Cultural safety integration** - appropriate to each agent's domain

## 🎯 Quick Start Guide

### 1. Environment Setup
```bash
./setup.sh && source venv/bin/activate
python main.py test  # Verify installation
```

### 2. Agent Workflow Testing
```python
# Test data collection
Task(description="Test collection", 
     prompt="data-collector: Collect 50 articles for testing", 
     subagent_type="general-purpose")
```

### 3. Complete Development Cycle
See `CLAUDE.md` for detailed agent usage and `agent_workflow_examples.md` for complete examples.

## ✅ System Status - PRODUCTION READY

The Claude Code native sub-agent system is fully implemented with:
- ✅ **Single-responsibility agents** following official best practices
- ✅ **Clean architecture** - no complex Python coordination infrastructure
- ✅ **Task tool integration** - native Claude Code workflows
- ✅ **Cultural safety** - integrated throughout all agent workflows
- ✅ **Production ready** - tested and validated system
- ✅ **Team collaboration** - version-controlled agent definitions

**Ready for efficient Amharic H-Net development with Claude Code!** 🇪🇹