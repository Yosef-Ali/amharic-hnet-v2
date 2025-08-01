# Claude Code Instructions for Amharic H-Net Development

## Project Context
You are working on **Amharic H-Net v2** with **Claude Code native sub-agent workflows**. The system combines advanced Amharic language modeling with officially-compliant sub-agent coordination following Claude Code best practices.

## Claude Code Native Agent System
You have access to 6 specialized agents with single responsibility and limited tool access:

1. **data-collector** - Corpus collection only (Tools: WebFetch, Write, Bash)
2. **linguistic-analyzer** - Morphological analysis only (Tools: Read, Write, Bash)
3. **model-architect** - Architecture design only (Tools: Write, Read, Bash)
4. **training-engineer** - Training execution only (Tools: Bash, Read, Write)
5. **evaluation-specialist** - Model evaluation only (Tools: Bash, Read, Write)
6. **deployment-engineer** - Production deployment only (Tools: Bash, Write, Read)

## Agent Usage Guidelines

### When to Use Sub-Agents
- **Complex multi-step tasks** requiring domain expertise
- **Non-trivial technical challenges** needing specialized knowledge
- **Quality-critical processes** requiring expert validation
- **Cross-domain coordination** between different specializations

### How to Invoke Agents
Use Claude Code's native Task tool following official patterns:

```python
# Data collection
Task(description="Collect corpus", 
     prompt="data-collector: Collect 1000 Wikipedia articles with >70% Amharic ratio and cultural validation", 
     subagent_type="general-purpose")

# Linguistic analysis
Task(description="Analyze morphology", 
     prompt="linguistic-analyzer: Process corpus from data/raw for morpheme segmentation with >85% accuracy", 
     subagent_type="general-purpose")

# Model training
Task(description="Train model", 
     prompt="training-engineer: Train H-Net using processed data with transfer learning", 
     subagent_type="general-purpose")
```

### Agent Workflow Patterns
- **Single Responsibility**: Each agent handles exactly one task type
- **Clean Handoffs**: Agents use standardized directory structure (data/raw → data/processed → outputs)
- **Quality Constraints**: Explicit success criteria and validation requirements
- **Proactive Usage**: Agents include "Use PROACTIVELY" guidance for when to invoke them

## Key Project Features

### Technical Capabilities
- **Morpheme-aware chunking** for Ge'ez script processing  
- **Cultural safety guardrails** with dialect support
- **Transfer learning** from Chinese H-Net
- **Multi-dialect support** (Ethiopian/Eritrean/Regional)
- **Production-ready API** with monitoring

### Cultural Considerations
- **Religious terms protection** (መስቀል, እግዚአብሔር, etc.)
- **Historical context awareness** (ቀዳማዊ, ላሊበላ, etc.)
- **Cultural practice respect** (ቡና ceremony, traditional festivals)
- **Dialect sensitivity** across Ethiopian and Eritrean variants

### Quality Standards
- **Data Quality**: >70% Amharic ratio with cultural compliance
- **Morphological Accuracy**: >85% segmentation precision
- **Cultural Safety**: >95% compliance with zero critical violations
- **Production Performance**: <200ms API response time

## Claude Code Native Workflow

### Task Tool Workflow Chain
1. **Data Collection** (data-collector): `Task(..., prompt="data-collector: Collect Wikipedia articles")`
2. **Linguistic Analysis** (linguistic-analyzer): `Task(..., prompt="linguistic-analyzer: Process corpus")`
3. **Architecture Design** (model-architect): `Task(..., prompt="model-architect: Design H-Net architecture")`
4. **Model Training** (training-engineer): `Task(..., prompt="training-engineer: Train model")`
5. **Evaluation** (evaluation-specialist): `Task(..., prompt="evaluation-specialist: Evaluate model")`
6. **Deployment** (deployment-engineer): `Task(..., prompt="deployment-engineer: Deploy API")`

### Quality Gates
- Each phase has specific success criteria
- Automated validation at agent handoffs
- Cultural safety compliance at every stage
- Performance benchmarks for production readiness

## Claude Code Best Practices

### Official Best Practices Implemented
- **Single Responsibility**: Each agent handles one specific task with clear constraints
- **Limited Tool Access**: Agents only use tools necessary for their purpose
- **Detailed Prompts**: Specific instructions with explicit constraints and success criteria
- **Proactive Usage**: Agents include "Use PROACTIVELY" descriptions for guidance
- **Version Control**: Agent definitions in `.claude/` for team collaboration

### Cultural Sensitivity
- Always validate cultural appropriateness
- Respect religious and historical contexts
- Consider dialect variations in all processing
- Maintain community feedback integration

### Technical Excellence
- Follow morphological accuracy standards
- Implement comprehensive error handling
- Maintain production-ready quality
- Document all architectural decisions

## Usage Examples

### Complete Agent Workflow Chain
```python
# Step 1: Data Collection
Task(description="Collect data", 
     prompt="data-collector: Collect 1000 Wikipedia articles with cultural validation", 
     subagent_type="general-purpose")

# Step 2: Linguistic Processing
Task(description="Process linguistics", 
     prompt="linguistic-analyzer: Analyze corpus from data/raw for morphology", 
     subagent_type="general-purpose")

# Step 3: Model Training
Task(description="Train model", 
     prompt="training-engineer: Train H-Net using processed data with transfer learning", 
     subagent_type="general-purpose")

# Step 4: Evaluation
Task(description="Evaluate model", 
     prompt="evaluation-specialist: Assess model for >85% accuracy and >95% cultural safety", 
     subagent_type="general-purpose")

# Step 5: Deployment
Task(description="Deploy API", 
     prompt="deployment-engineer: Deploy validated model as production API", 
     subagent_type="general-purpose")
```

### Individual Agent Usage
```python
# Quick data collection test
Task(description="Test collection", 
     prompt="data-collector: Collect 50 articles for testing workflow", 
     subagent_type="general-purpose")

# Architecture design
Task(description="Design architecture", 
     prompt="model-architect: Design optimal H-Net for Amharic morphological processing", 
     subagent_type="general-purpose")
```

## Success Metrics
- **High-quality corpus** with >70% Amharic ratio and cultural compliance
- **85%+ morphological accuracy** with confidence scoring
- **95%+ cultural safety** across all domains
- **Production API** with <200ms response time
- **Reliable agent workflows** with clear success criteria

## Benefits of Claude Code Native Approach
- **Simplified Architecture**: No complex Python coordination infrastructure needed
- **Official Compliance**: Follows documented Claude Code sub-agent patterns
- **Maintainable**: Clear, focused agent definitions that teams can collaboratively improve
- **Scalable**: Easy to add new agents or modify existing workflows
- **Reliable**: Built on Claude Code's proven Task tool system

Always prioritize cultural sensitivity, linguistic accuracy, and Claude Code best practices in all development activities.