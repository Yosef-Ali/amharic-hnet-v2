#!/usr/bin/env python3
"""
Cultural Safety Service for Production Deployment
===============================================

Production-ready cultural safety monitoring with comprehensive validation,
real-time monitoring, and detailed reporting.
"""

import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import logging

import structlog

# Import the cultural guardrails
import sys
sys.path.append('/Users/mekdesyared/amharic-hnet-v2/src')
from safety.cultural_guardrails import AmharicCulturalGuardrails, CulturalViolation

logger = structlog.get_logger(__name__)


@dataclass
class SafetyMetrics:
    """Cultural safety metrics for monitoring."""
    total_checks: int = 0
    violations_detected: int = 0
    false_positives: int = 0
    avg_check_time: float = 0.0
    severity_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.severity_counts is None:
            self.severity_counts = {"low": 0, "medium": 0, "high": 0}


class CulturalSafetyService:
    """
    Production cultural safety service with comprehensive monitoring.
    
    Features:
    - Real-time cultural safety validation
    - Performance monitoring and metrics
    - Detailed violation reporting
    - Cultural context awareness
    - Production-ready error handling
    """
    
    def __init__(self):
        self.guardrails = AmharicCulturalGuardrails()
        self.metrics = SafetyMetrics()
        self.violation_history: List[Dict] = []
        self.max_history_size = 1000
        
        # Enhanced cultural patterns for production
        self._initialize_enhanced_patterns()
        
        logger.info("Cultural safety service initialized")
    
    def _initialize_enhanced_patterns(self):
        """Initialize enhanced cultural safety patterns for production."""
        # Additional production-specific patterns
        self.production_patterns = {
            # Hate speech detection patterns
            "hate_indicators": [
                "hate", "kill", "destroy", "eliminate", "exterminate",
                "inferior", "superior", "worthless", "disgusting"
            ],
            
            # Bias detection patterns
            "bias_indicators": [
                "all", "always", "never", "typical", "characteristic",
                "inherent", "natural", "born", "genetic"
            ],
            
            # Stereotyping patterns
            "stereotype_indicators": [
                "lazy", "aggressive", "primitive", "backwards", "uncivilized",
                "savage", "barbaric", "ignorant", "stupid"
            ],
            
            # Religious sensitivity patterns
            "religious_sensitivity": [
                "fake", "false", "myth", "superstition", "primitive belief",
                "outdated", "irrelevant", "meaningless"
            ]
        }
        
        # Contextual safety rules for different domains
        self.domain_rules = {
            "education": {
                "required_respect": ["teacher", "student", "learning", "knowledge"],
                "forbidden_associations": ["stupid", "ignorant", "worthless"]
            },
            "healthcare": {
                "required_respect": ["doctor", "patient", "healing", "medicine"],
                "forbidden_associations": ["fake", "dangerous", "harmful"]
            },
            "business": {
                "required_respect": ["customer", "service", "quality", "trust"],
                "forbidden_associations": ["cheat", "scam", "fraud", "lie"]
            }
        }
    
    def check_input_safety(self, text: str) -> Tuple[bool, List[CulturalViolation]]:
        """
        Check input text for cultural safety violations.
        
        Args:
            text: Input text to validate
            
        Returns:
            Tuple of (is_safe, violations_list)
        """
        start_time = time.time()
        
        try:
            # Basic cultural safety check
            is_safe, violations = self.guardrails.check_cultural_safety(text)
            
            # Enhanced production checks
            additional_violations = self._check_production_patterns(text)
            violations.extend(additional_violations)
            
            # Update safety status
            is_safe = len(violations) == 0
            
            # Record metrics
            check_time = time.time() - start_time
            self._update_metrics(violations, check_time)
            
            # Log check results
            if violations:
                logger.warning("Cultural safety violations detected",
                             text_length=len(text),
                             violation_count=len(violations),
                             severity_levels=[v.severity for v in violations])
            else:
                logger.debug("Text passed cultural safety check",
                           text_length=len(text),
                           check_time=check_time)
            
            return is_safe, violations
            
        except Exception as e:
            logger.error("Cultural safety check failed", error=str(e))
            # Fail safe - reject if unable to validate
            return False, [CulturalViolation(
                term="system_error",
                violation_type="validation_error",
                context=f"Safety check failed: {str(e)}",
                severity="high"
            )]
    
    def _check_production_patterns(self, text: str) -> List[CulturalViolation]:
        """Check text against production-specific patterns."""
        violations = []
        text_lower = text.lower()
        
        # Check hate speech indicators
        for hate_word in self.production_patterns["hate_indicators"]:
            if hate_word in text_lower:
                violations.append(CulturalViolation(
                    term=hate_word,
                    violation_type="hate_speech",
                    context=f"Potential hate speech indicator: '{hate_word}'",
                    severity="high"
                ))
        
        # Check bias indicators in combination with cultural terms
        cultural_terms = list(self.guardrails.sacred_terms.keys())
        for cultural_term in cultural_terms:
            if cultural_term in text:
                for bias_word in self.production_patterns["bias_indicators"]:
                    if bias_word in text_lower:
                        # Check proximity
                        cultural_pos = text.find(cultural_term)
                        bias_pos = text_lower.find(bias_word)
                        
                        if abs(cultural_pos - bias_pos) <= 30:
                            violations.append(CulturalViolation(
                                term=cultural_term,
                                violation_type="cultural_bias",
                                context=f"Bias indicator '{bias_word}' near cultural term '{cultural_term}'",
                                severity="medium"
                            ))
        
        # Check stereotyping patterns
        for stereotype in self.production_patterns["stereotype_indicators"]:
            if stereotype in text_lower:
                violations.append(CulturalViolation(
                    term=stereotype,
                    violation_type="stereotyping",
                    context=f"Potential stereotyping language: '{stereotype}'",
                    severity="medium"
                ))
        
        return violations
    
    def validate_generation(self, generated_text: str, input_context: str = "") -> Tuple[bool, str]:
        """
        Validate generated text for cultural safety with detailed feedback.
        
        Args:
            generated_text: Generated text to validate
            input_context: Original input context
            
        Returns:
            Tuple of (is_valid, detailed_feedback)
        """
        start_time = time.time()
        
        try:
            # Check generated text
            is_safe, violations = self.check_input_safety(generated_text)
            
            # Additional context-aware checks
            context_violations = self._check_contextual_appropriateness(
                generated_text, input_context
            )
            violations.extend(context_violations)
            
            # Update final safety status
            is_safe = len(violations) == 0
            
            # Generate detailed feedback
            feedback = self._generate_detailed_feedback(violations, generated_text, input_context)
            
            # Record generation validation
            validation_time = time.time() - start_time
            self._record_validation(generated_text, input_context, violations, validation_time)
            
            return is_safe, feedback
            
        except Exception as e:
            logger.error("Generation validation failed", error=str(e))
            return False, f"Validation error: {str(e)}"
    
    def _check_contextual_appropriateness(self, generated_text: str, input_context: str) -> List[CulturalViolation]:
        """Check contextual appropriateness of generated text."""
        violations = []
        
        # Check for context consistency
        if input_context:
            # Detect domain from input context
            detected_domain = self._detect_domain(input_context)
            
            if detected_domain and detected_domain in self.domain_rules:
                rules = self.domain_rules[detected_domain]
                
                # Check forbidden associations in this domain
                for forbidden in rules.get("forbidden_associations", []):
                    if forbidden.lower() in generated_text.lower():
                        violations.append(CulturalViolation(
                            term=forbidden,
                            violation_type="contextual_inappropriateness",
                            context=f"Inappropriate for {detected_domain} domain: '{forbidden}'",
                            severity="medium"
                        ))
        
        # Check for tone consistency
        if self._is_formal_context(input_context) and self._contains_casual_language(generated_text):
            violations.append(CulturalViolation(
                term="tone_mismatch",
                violation_type="tone_inappropriateness",
                context="Casual language in formal context",
                severity="low"
            ))
        
        return violations
    
    def _detect_domain(self, text: str) -> Optional[str]:
        """Detect the domain/context of the text."""
        text_lower = text.lower()
        
        domain_keywords = {
            "education": ["school", "teacher", "student", "learn", "study", "education"],
            "healthcare": ["doctor", "patient", "medicine", "health", "hospital", "clinic"],
            "business": ["business", "company", "customer", "service", "product", "market"],
            "religious": ["church", "prayer", "god", "faith", "religion", "worship"],
            "government": ["government", "official", "policy", "law", "minister", "parliament"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain
        
        return None
    
    def _is_formal_context(self, text: str) -> bool:
        """Check if the context is formal."""
        formal_indicators = [
            "official", "formal", "government", "business", "academic",
            "professional", "legal", "medical", "educational"
        ]
        return any(indicator in text.lower() for indicator in formal_indicators)
    
    def _contains_casual_language(self, text: str) -> bool:
        """Check if text contains casual language."""
        casual_words = ["cool", "awesome", "fun", "party", "chill", "whatever", "stuff"]
        return any(word in text.lower() for word in casual_words)
    
    def _generate_detailed_feedback(self, violations: List[CulturalViolation], 
                                  generated_text: str, input_context: str) -> str:
        """Generate detailed feedback for violations."""
        if not violations:
            return "Generated text is culturally appropriate and safe."
        
        feedback_parts = ["Cultural safety assessment:"]
        
        # Group violations by severity
        severity_groups = {"high": [], "medium": [], "low": []}
        for violation in violations:
            severity_groups[violation.severity].append(violation)
        
        # Report by severity
        for severity in ["high", "medium", "low"]:
            if severity_groups[severity]:
                severity_emoji = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}
                feedback_parts.append(f"\n{severity_emoji[severity]} {severity.upper()} SEVERITY:")
                
                for violation in severity_groups[severity]:
                    feedback_parts.append(f"  • {violation.context}")
                    
                    # Add suggestions if available
                    suggestions = self.guardrails.suggest_alternatives(generated_text, [violation])
                    if suggestions:
                        for context, suggestion in suggestions.items():
                            feedback_parts.append(f"    Suggestion: {suggestion}")
        
        # Add general recommendations
        feedback_parts.append(f"\nGeneral recommendations:")
        feedback_parts.append("• Ensure respectful treatment of cultural and religious terms")
        feedback_parts.append("• Avoid associating sacred concepts with inappropriate themes")
        feedback_parts.append("• Maintain appropriate tone for the context")
        
        return "\n".join(feedback_parts)
    
    def _update_metrics(self, violations: List[CulturalViolation], check_time: float):
        """Update safety metrics."""
        self.metrics.total_checks += 1
        
        if violations:
            self.metrics.violations_detected += 1
            
            # Update severity counts
            for violation in violations:
                self.metrics.severity_counts[violation.severity] += 1
        
        # Update average check time
        total_time = self.metrics.avg_check_time * (self.metrics.total_checks - 1) + check_time
        self.metrics.avg_check_time = total_time / self.metrics.total_checks
    
    def _record_validation(self, generated_text: str, input_context: str, 
                          violations: List[CulturalViolation], validation_time: float):
        """Record validation for history and analysis."""
        record = {
            "timestamp": time.time(),
            "input_context": input_context[:100] + "..." if len(input_context) > 100 else input_context,
            "generated_text": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
            "violations": [asdict(v) for v in violations],
            "validation_time": validation_time,
            "is_safe": len(violations) == 0
        }
        
        self.violation_history.append(record)
        
        # Maintain history size
        if len(self.violation_history) > self.max_history_size:
            self.violation_history = self.violation_history[-self.max_history_size:]
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get comprehensive safety metrics."""
        total_checks = max(self.metrics.total_checks, 1)
        
        return {
            "total_checks": self.metrics.total_checks,
            "violations_detected": self.metrics.violations_detected,
            "violation_rate": self.metrics.violations_detected / total_checks,
            "false_positive_rate": self.metrics.false_positives / total_checks,
            "avg_check_time": self.metrics.avg_check_time,
            "severity_distribution": self.metrics.severity_counts,
            "performance": {
                "checks_per_second": 1 / max(self.metrics.avg_check_time, 0.001),
                "target_response_time_met": self.metrics.avg_check_time < 0.05,  # 50ms target
            },
            "recent_violations": len([r for r in self.violation_history[-100:] if not r["is_safe"]])
        }
    
    def get_violation_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get violation report for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_records = [r for r in self.violation_history if r["timestamp"] > cutoff_time]
        
        if not recent_records:
            return {"message": f"No violations recorded in the last {hours} hours"}
        
        total_recent = len(recent_records)
        violations_recent = len([r for r in recent_records if not r["is_safe"]])
        
        # Analyze violation patterns
        violation_types = {}
        for record in recent_records:
            if not record["is_safe"]:
                for violation in record["violations"]:
                    vtype = violation["violation_type"]
                    violation_types[vtype] = violation_types.get(vtype, 0) + 1
        
        return {
            "time_period_hours": hours,
            "total_validations": total_recent,
            "violations_detected": violations_recent,
            "violation_rate": violations_recent / max(total_recent, 1),
            "common_violation_types": dict(sorted(violation_types.items(), 
                                                key=lambda x: x[1], reverse=True)),
            "avg_validation_time": sum(r["validation_time"] for r in recent_records) / max(total_recent, 1)
        }
    
    def get_guidelines(self) -> Dict[str, Any]:
        """Get cultural safety guidelines and information."""
        return {
            "sacred_terms": {
                "description": "Terms requiring respectful treatment",
                "examples": list(self.guardrails.sacred_terms.keys())[:5],
                "count": len(self.guardrails.sacred_terms)
            },
            "sensitive_topics": {
                "categories": list(self.guardrails.sensitive_topics.keys()),
                "handling": "Neutral, respectful language required"
            },
            "violation_types": [
                "inappropriate_association",
                "sensitive_topic_mishandling", 
                "inappropriate_tone",
                "hate_speech",
                "cultural_bias",
                "stereotyping"
            ],
            "severity_levels": ["low", "medium", "high"],
            "positive_associations": {
                "description": "Encouraged cultural associations",
                "examples": {
                    "ቡና": self.guardrails.positive_associations.get("ቡና", [])[:3],
                    "መስቀል": self.guardrails.positive_associations.get("መስቀል", [])[:3]
                }
            },
            "performance_targets": {
                "check_time_ms": 50,
                "accuracy_target": "95%",
                "false_positive_rate": "<5%"
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform cultural safety service health check."""
        try:
            # Quick validation test
            test_text = "ሰላም እንዴት ነሽ?"
            start_time = time.time()
            
            is_safe, violations = self.check_input_safety(test_text)
            check_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "health_check_time": check_time,
                "test_passed": is_safe,
                "total_checks_performed": self.metrics.total_checks,
                "avg_check_time": self.metrics.avg_check_time
            }
            
        except Exception as e:
            logger.error("Cultural safety health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }