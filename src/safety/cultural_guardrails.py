import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class CulturalViolation:
    """Represents a cultural safety violation."""
    term: str
    violation_type: str
    context: str
    severity: str  # 'low', 'medium', 'high'


class AmharicCulturalGuardrails:
    """
    Cultural safety system for Amharic language generation.
    Prevents inappropriate associations with sacred, cultural, and sensitive terms.
    """
    
    def __init__(self):
        # Sacred and religious terms that require respectful treatment
        self.sacred_terms = {
            "ቡና": "የኢትዮጵያ ባህላዊ ማህበራዊ ሥርዓት",  # Coffee ceremony - social ritual
            "መስቀል": "ቅዱስ ምልክት",  # Cross - sacred symbol
            "ቀዳማዊ": "ታሪካዊ ንጉሥ",  # Kedamawi - historical king
            "ኢየሱስ": "ክርስቶስ",  # Jesus Christ
            "ማርያም": "ቅድስት ማርያም",  # Saint Mary
            "ገብርኤል": "መላክ ገብርኤል",  # Archangel Gabriel
            "ሚካኤል": "መላክ ሚካኤል",  # Archangel Michael
            "ገና": "የክርስቶስ ልደት",  # Christmas
            "ፋሲካ": "የትንሳኤ በዓል",  # Easter
            "እንጅብ": "ባህላዊ ስርዓት",  # Traditional bread/hospitality
            "በረከት": "ቅዱስ በረከት",  # Divine blessing
            "ሃይማኖት": "ቅዱስ ሃይማኖት",  # Religion/faith
            "አውሮፓ": "አለም አህጉር",  # Appropriate continent reference
            "ሸዋ": "ታሪካዊ ግዛት",  # Historical province
            "ንጉሥ": "ንጉሣዊ ማዕረግ",  # Royal title
            "እንደ": "አነጻጻሪ ቃል"  # Like/comparison word
        }
        
        # Terms that should never be associated with inappropriate concepts
        self.forbidden_associations = {
            "ቡና": [
                "addictive", "drug", "harmful", "dangerous", "toxic", "poison",
                "evil", "sin", "forbidden", "illegal", "bad", "negative"
            ],
            "መስቀል": [
                "decoration", "fashion", "jewelry", "ornament", "accessory",
                "style", "trend", "design", "art piece", "craft"
            ],
            "ቀዳማዊ": [
                "dictator", "tyrant", "oppressor", "brutal", "cruel", "harsh",
                "mean", "bad", "evil", "wrong"
            ],
            "ኢየሱስ": [
                "myth", "fake", "false", "story", "character", "fiction",
                "ordinary", "human", "man", "person"
            ],
            "ማርያም": [
                "woman", "girl", "female", "ordinary", "common", "regular"
            ],
            "ገና": [
                "party", "celebration", "fun", "entertainment", "commercial",
                "business", "shopping", "material"
            ],
            "ፋሲካ": [
                "party", "celebration", "fun", "entertainment", "holiday",
                "vacation", "break", "time off"
            ]
        }
        
        # Culturally sensitive topics requiring careful handling
        self.sensitive_topics = {
            "ethnicity": ["ኦሮሞ", "አማራ", "ትግሬ", "ሲዳማ", "ወላይታ", "ሶማሌ", "ኣፋር"],
            "politics": ["መንግሥት", "ፓርቲ", "ፖለቲካ", "ባለስልጣን", "ዲሞክራሲ"],
            "conflict": ["ግጭት", "ሰላም", "ጦርነት", "ሽር", "አለመግባባት"],
            "history": ["ደርግ", "ንጉሥ", "ተቃዋሚ", "አውሮፓ", "ቅኝ", "ኢጣሊያ"]
        }
        
        # Dialect-specific variations for cultural terms
        self.dialect_variations = {
            "addis_ababa": {
                "ቡና": "የኢትዮጵያ ባህላዊ ማህበራዊ ሥርዓት",
                "ሰላም": "ሰላም ይሁንልኝ"
            },
            "gojjam": {
                "ቡና": "የእግዚአብሔር ስጦታ",
                "ሰላም": "ሰላም ትሁንልኝ"
            },
            "eritrea": {
                "ቡና": "ዘኢትዮጵያ ባህላዊ ምግባር",
                "ሰላም": "ሰላም ዩሀነልካ"
            }
        }
        
        # Context-aware safety rules
        self.contextual_rules = {
            "religious": {
                "required_respect": ["ቡና", "መስቀል", "ገና", "ፋሲካ"],
                "forbidden_casual": ["fun", "party", "entertainment"]
            },
            "historical": {
                "required_accuracy": ["ቀዳማዊ", "ሃይለ ሥላሴ", "ላሊበላ"],
                "forbidden_modern": ["invention", "modern", "new"]
            }
        }
        
        # Positive cultural associations to encourage
        self.positive_associations = {
            "ቡና": [
                "ማህበራዊ", "ባህል", "ወግ", "አንድነት", "መግባባት", "ጓደኝነት",
                "እንግዳ ተቀባይነት", "ማህበረሰብ", "ወንድማማችነት"
            ],
            "መስቀል": [
                "ቅዱስ", "በረከት", "ሃይማኖት", "እምነት", "ፍቅር", "ሰላም",
                "ምህረት", "ታማኝነት", "ክብር"
            ],
            "ኢትዮጵያ": [
                "ታሪክ", "ባህል", "ወግ", "ቅርስ", "ነፃነት", "ኩራት",
                "ልዩነት", "አንድነት", "ውበት"
            ]
        }
        
        # Compile regex patterns for efficient matching
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient text matching."""
        self.sacred_pattern = re.compile(
            '|'.join(re.escape(term) for term in self.sacred_terms.keys()),
            re.IGNORECASE
        )
        
        # Create patterns for forbidden words in multiple languages
        forbidden_words = set()
        for word_list in self.forbidden_associations.values():
            forbidden_words.update(word_list)
        
        self.forbidden_pattern = re.compile(
            '|'.join(re.escape(word) for word in forbidden_words),
            re.IGNORECASE
        )
    
    def check_cultural_safety(self, text: str) -> Tuple[bool, List[CulturalViolation]]:
        """
        Check if text complies with Amharic cultural safety guidelines.
        
        Args:
            text: Input text to check
            
        Returns:
            Tuple of (is_safe: bool, violations: List[CulturalViolation])
        """
        violations = []
        
        # Check for sacred term violations
        sacred_violations = self._check_sacred_terms(text)
        violations.extend(sacred_violations)
        
        # Check for sensitive topic handling
        sensitive_violations = self._check_sensitive_topics(text)
        violations.extend(sensitive_violations)
        
        # Check for cultural appropriateness
        appropriateness_violations = self._check_cultural_appropriateness(text)
        violations.extend(appropriateness_violations)
        
        is_safe = len(violations) == 0
        return is_safe, violations
    
    def _check_sacred_terms(self, text: str) -> List[CulturalViolation]:
        """Check for inappropriate use of sacred terms."""
        violations = []
        text_lower = text.lower()
        
        for sacred_term, description in self.sacred_terms.items():
            if sacred_term in text:
                # Check if any forbidden associations appear near the sacred term
                forbidden_words = self.forbidden_associations.get(sacred_term, [])
                
                for forbidden in forbidden_words:
                    if forbidden.lower() in text_lower:
                        # Check proximity (within 50 characters)
                        sacred_pos = text.find(sacred_term)
                        forbidden_pos = text_lower.find(forbidden.lower())
                        
                        if abs(sacred_pos - forbidden_pos) <= 50:
                            violations.append(CulturalViolation(
                                term=sacred_term,
                                violation_type="inappropriate_association",
                                context=f"'{sacred_term}' associated with '{forbidden}'",
                                severity="high"
                            ))
        
        return violations
    
    def _check_sensitive_topics(self, text: str) -> List[CulturalViolation]:
        """Check handling of culturally sensitive topics."""
        violations = []
        
        for topic_category, terms in self.sensitive_topics.items():
            for term in terms:
                if term in text:
                    # Check for inflammatory language nearby
                    inflammatory_words = [
                        "hate", "bad", "evil", "wrong", "stupid", "inferior",
                        "superior", "better", "worse", "fight", "enemy"
                    ]
                    
                    text_lower = text.lower()
                    for inflammatory in inflammatory_words:
                        if inflammatory in text_lower:
                            term_pos = text.find(term)
                            inflammatory_pos = text_lower.find(inflammatory)
                            
                            if abs(term_pos - inflammatory_pos) <= 30:
                                violations.append(CulturalViolation(
                                    term=term,
                                    violation_type="sensitive_topic_mishandling",
                                    context=f"Sensitive term '{term}' near inflammatory '{inflammatory}'",
                                    severity="medium"
                                ))
        
        return violations
    
    def _check_cultural_appropriateness(self, text: str) -> List[CulturalViolation]:
        """Check general cultural appropriateness."""
        violations = []
        
        # Check for overly casual treatment of formal cultural concepts
        formal_terms = ["ንጉሥ", "መንግሥት", "ኪዳን", "በዓል", "ሥርዓት"]
        casual_words = ["fun", "party", "cool", "awesome", "whatever"]
        
        text_lower = text.lower()
        
        for formal_term in formal_terms:
            if formal_term in text:
                for casual in casual_words:
                    if casual in text_lower:
                        formal_pos = text.find(formal_term)
                        casual_pos = text_lower.find(casual)
                        
                        if abs(formal_pos - casual_pos) <= 40:
                            violations.append(CulturalViolation(
                                term=formal_term,
                                violation_type="inappropriate_tone",
                                context=f"Formal term '{formal_term}' treated casually with '{casual}'",
                                severity="low"
                            ))
        
        return violations
    
    def suggest_alternatives(self, text: str, violations: List[CulturalViolation]) -> Dict[str, str]:
        """
        Suggest culturally appropriate alternatives for problematic text.
        
        Args:
            text: Original text
            violations: List of violations found
            
        Returns:
            Dictionary mapping violation contexts to suggested alternatives
        """
        suggestions = {}
        
        for violation in violations:
            if violation.violation_type == "inappropriate_association":
                # Suggest positive associations instead
                term = violation.term
                if term in self.positive_associations:
                    positive_words = self.positive_associations[term]
                    suggestion = f"Consider associating '{term}' with positive concepts like: {', '.join(positive_words[:3])}"
                    suggestions[violation.context] = suggestion
            
            elif violation.violation_type == "sensitive_topic_mishandling":
                suggestions[violation.context] = "Use neutral, respectful language when discussing cultural groups and sensitive topics"
            
            elif violation.violation_type == "inappropriate_tone":
                suggestions[violation.context] = "Maintain respectful, formal tone when discussing cultural and religious concepts"
        
        return suggestions
    
    def get_cultural_context(self, term: str) -> str:
        """
        Get cultural context explanation for a term.
        
        Args:
            term: Amharic term to explain
            
        Returns:
            Cultural context explanation
        """
        if term in self.sacred_terms:
            return f"{term}: {self.sacred_terms[term]} - requires respectful treatment"
        
        # Check sensitive topics
        for category, terms in self.sensitive_topics.items():
            if term in terms:
                return f"{term}: {category} topic - handle with cultural sensitivity"
        
        return f"{term}: general term - follow standard cultural guidelines"
    
    def validate_generation(self, generated_text: str, input_context: str = "") -> Tuple[bool, str]:
        """
        Validate generated text for cultural safety.
        
        Args:
            generated_text: Text generated by the model
            input_context: Original input context
            
        Returns:
            Tuple of (is_valid: bool, feedback: str)
        """
        is_safe, violations = self.check_cultural_safety(generated_text)
        
        if is_safe:
            return True, "Text is culturally appropriate"
        
        feedback_parts = ["Cultural safety issues detected:"]
        
        for violation in violations:
            severity_emoji = {"low": "⚠️", "medium": "🔸", "high": "🚨"}
            emoji = severity_emoji.get(violation.severity, "⚠️")
            feedback_parts.append(f"{emoji} {violation.context}")
        
        suggestions = self.suggest_alternatives(generated_text, violations)
        if suggestions:
            feedback_parts.append("\nSuggestions:")
            for context, suggestion in suggestions.items():
                feedback_parts.append(f"• {suggestion}")
        
        return False, "\n".join(feedback_parts)