# Amharic H-Net Development Context

## Project Overview
Advanced Hierarchical Neural Network specifically designed for Amharic language processing with cultural safety and morphological awareness.

## Sub-Agent Workflow
- **data-collector**: Corpus collection and validation
- **linguistic-analyzer**: Morphological analysis and cultural assessment  
- **model-architect**: Architecture design and optimization
- **training-engineer**: Training pipeline and environment setup
- **evaluation-specialist**: Performance and cultural safety evaluation
- **deployment-engineer**: Production API deployment

## Key Features
- Morpheme-aware dynamic chunking for Ge'ez script
- Cultural safety guardrails with dialect support
- Transfer learning from Chinese H-Net
- Multi-dialect support (Ethiopian/Eritrean/Regional)
- Production-ready API with monitoring

## Usage
```bash
# Quick start
python workflow_coordinator.py --phase full

# Individual phases  
python workflow_coordinator.py --phase collect --source wikipedia
python workflow_coordinator.py --phase train --config configs/config.yaml
```