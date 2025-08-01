# Claude Code Configuration for Amharic H-Net v2

## Overview
This directory contains the **Claude Code native sub-agent system** following official best practices for Amharic H-Net development. Each agent has single responsibility, limited tool access, and clear Task tool integration.

## Directory Structure
```
.claude/
├── agents/                 # Individual agent specifications
│   ├── data-collector.md
│   ├── linguistic-analyzer.md
│   ├── model-architect.md
│   ├── training-engineer.md
│   ├── evaluation-specialist.md
│   └── deployment-engineer.md
├── prompts/               # Agent-specific workflow prompts
│   ├── data-collection.md
│   ├── linguistic-analysis.md
│   ├── model-architecture.md
│   ├── training-pipeline.md
│   ├── evaluation-assessment.md
│   └── deployment-production.md
├── workflows/             # Coordination and workflow definitions
│   ├── complete-development.md
│   └── agent-coordination.md
├── instructions.md        # Main Claude Code instructions
├── context.md            # Project context for Claude
├── amharic-hnet-agents.md # Agent registry and usage
└── README.md             # This file
```

## Key Features

### Claude Code Native Agents
Each agent follows official best practices with single responsibility and limited tool access:
- **data-collector**: Corpus collection only (Tools: WebFetch, Write, Bash)
- **linguistic-analyzer**: Morphological analysis only (Tools: Read, Write, Bash)
- **model-architect**: Architecture design only (Tools: Write, Read, Bash)
- **training-engineer**: Training execution only (Tools: Bash, Read, Write)
- **evaluation-specialist**: Model evaluation only (Tools: Bash, Read, Write)
- **deployment-engineer**: Production deployment only (Tools: Bash, Write, Read)

### Task Tool Coordination
- **Native integration**: Uses Claude Code's built-in Task tool for coordination
- **Clean handoffs**: Clear input/output directory structure
- **Quality constraints**: Explicit success criteria and validation requirements
- **Proactive guidance**: "Use PROACTIVELY" descriptions for each agent

### Cultural Safety Integration
- **Built-in cultural awareness** across all agents
- **Religious and historical context** protection
- **Multi-dialect sensitivity** (Ethiopian/Eritrean/Regional)
- **Community feedback integration** protocols

## Usage Examples

### Individual Agent Invocation
```python
# Data collection
Task(description="Collect corpus", 
     prompt="data-collector: Collect 1000 Wikipedia articles with >70% Amharic ratio and cultural validation", 
     subagent_type="general-purpose")

# Linguistic analysis
Task(description="Analyze morphology", 
     prompt="linguistic-analyzer: Process corpus from data/raw for morpheme segmentation", 
     subagent_type="general-purpose")
```

### Sequential Workflow Chain
```python
# Step 1: Data Collection
Task(description="Collect data", prompt="data-collector: Collect 500 Wikipedia articles", subagent_type="general-purpose")

# Step 2: Linguistic Processing
Task(description="Process linguistics", prompt="linguistic-analyzer: Analyze collected corpus", subagent_type="general-purpose")

# Step 3: Model Training
Task(description="Train model", prompt="training-engineer: Train H-Net with transfer learning", subagent_type="general-purpose")

# Step 4: Evaluation
Task(description="Evaluate model", prompt="evaluation-specialist: Assess model performance", subagent_type="general-purpose")

# Step 5: Deployment
Task(description="Deploy API", prompt="deployment-engineer: Deploy as production API", subagent_type="general-purpose")
```

### Parallel Processing
```python
# Multiple data sources (parallel)
Task(description="Collect Wikipedia", prompt="data-collector: Collect Wikipedia articles", subagent_type="general-purpose")
Task(description="Collect BBC", prompt="data-collector: Collect BBC Amharic articles", subagent_type="general-purpose")
```

## Quality Standards
- **Data Quality**: >70% Amharic ratio with cultural compliance
- **Morphological Accuracy**: >85% segmentation precision
- **Cultural Safety**: >95% compliance with zero critical violations
- **Production Performance**: <200ms API response time
- **Agent Reliability**: Single responsibility with clear success criteria

## Claude Code Best Practices Implemented
1. **Single Responsibility**: Each agent handles one specific task
2. **Limited Tool Access**: Agents only use necessary tools for security
3. **Detailed Prompts**: Specific instructions with constraints and examples
4. **Proactive Usage**: Agents include "Use PROACTIVELY" guidance
5. **Version Control**: Agent definitions in `.claude/` for team collaboration

## Benefits of Native Approach
- **Simplified Architecture**: No complex Python coordination infrastructure
- **Official Compliance**: Follows documented Claude Code patterns
- **Maintainable**: Clear, focused agent definitions
- **Scalable**: Easy to add new agents or modify existing ones
- **Reliable**: Built on Claude Code's proven Task tool system

This configuration provides production-ready Amharic H-Net development through Claude Code native sub-agent workflows.