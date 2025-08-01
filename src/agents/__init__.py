"""
Amharic H-Net - Claude Code Sub-Agent System

This directory previously contained Python agent infrastructure that has been
replaced with Claude Code native sub-agents defined in .claude/agents/

The new system uses Claude Code's Task tool for coordination following
official best practices with single-responsibility agents.

Available Claude Code Agents (in .claude/agents/):
- data-collector: Corpus collection and validation 
- linguistic-analyzer: Morphological analysis and cultural assessment
- model-architect: Architecture design and optimization
- training-engineer: Training pipeline and environment management
- evaluation-specialist: Performance and cultural safety evaluation
- deployment-engineer: Production API deployment and monitoring

See CLAUDE.md for complete usage instructions.
"""