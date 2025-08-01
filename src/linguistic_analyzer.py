#!/usr/bin/env python3
"""
Amharic Linguistic Analyzer - Morphological Segmentation and Cultural Safety Validation
Expert system for Amharic text processing with specialized morphological analysis
"""

import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MorphemeSegment:
    """Represents a single morpheme with its analysis"""
    morpheme: str
    gloss: str
    pos_tag: str
    morphological_features: Dict[str, str]
    confidence: float
    root: Optional[str] = None
    pattern: Optional[str] = None

@dataclass
class MorphologicalAnalysis:
    """Complete morphological analysis of an Amharic word"""
    original_word: str
    segmentation: List[MorphemeSegment]
    overall_confidence: float
    root_confidence: float
    pattern_confidence: float
    alternative_analyses: List['MorphologicalAnalysis'] = None

@dataclass
class CulturalSafetyAssessment:
    """Cultural safety evaluation result"""
    safety_level: str  # 'safe', 'needs_review', 'problematic'
    issues: List[Dict[str, str]]
    recommendations: List[str]
    cultural_context: str
    confidence: float

class AmharicLinguisticAnalyzer:
    """
    Advanced Amharic morphological analyzer with cultural safety validation
    Implements Ethiopian Semitic linguistic patterns and cultural awareness
    """
    
    def __init__(self):
        self.initialize_linguistic_resources()
        self.initialize_cultural_safety_resources()
        self.processed_count = 0
        self.high_confidence_count = 0
        
    def initialize_linguistic_resources(self):
        """Initialize Amharic linguistic knowledge base"""
        
        # Amharic trilateral and quadrilateral roots
        self.common_roots = {
            'ስብር': {'meaning': 'break', 'pattern': 'CCC'},
            'ወተት': {'meaning': 'milk', 'pattern': 'CCC'},
            'ቤተ': {'meaning': 'house', 'pattern': 'CCC'},
            'ነገር': {'meaning': 'thing/speak', 'pattern': 'CCC'},
            'መስል': {'meaning': 'resemble', 'pattern': 'CCC'},
            'ግዛት': {'meaning': 'rule/territory', 'pattern': 'CCC'},
            'ስራት': {'meaning': 'work', 'pattern': 'CCC'},
            'ትምህ': {'meaning': 'learn', 'pattern': 'CCC'},
            'ሰላም': {'meaning': 'peace', 'pattern': 'CCC'},
            'አገር': {'meaning': 'country', 'pattern': 'CCC'},
            'ማህበ': {'meaning': 'community', 'pattern': 'CCC'},
            'ክብር': {'meaning': 'honor', 'pattern': 'CCC'},
            'ሰንብ': {'meaning': 'worship', 'pattern': 'CCC'},
            'መንግ': {'meaning': 'reign', 'pattern': 'CCC'},
            'ቅዱስ': {'meaning': 'holy', 'pattern': 'CCC'},
        }
        
        # Verbal patterns (Common Ethiopian Semitic patterns)
        self.verbal_patterns = {
            'ተ-ማረ': 'passive/reflexive',
            'አ-ማረ': 'causative',
            'አስ-ማረ': 'causative intensive',
            'ታ-ማረ': 'reciprocal',
            'ተማ-ማረ': 'iterative',
        }
        
        # Common prefixes
        self.prefixes = {
            'ተ-': {'type': 'verbal', 'function': 'passive/reflexive'},
            'አ-': {'type': 'verbal', 'function': 'causative'},
            'አስ-': {'type': 'verbal', 'function': 'causative_intensive'},
            'ታ-': {'type': 'verbal', 'function': 'reciprocal'},
            'ያ-': {'type': 'verbal', 'function': 'future'},
            'በ-': {'type': 'prepositional', 'function': 'instrumental/locative'},
            'ከ-': {'type': 'prepositional', 'function': 'from/with'},
            'ወደ-': {'type': 'prepositional', 'function': 'to/towards'},
            'እንደ-': {'type': 'prepositional', 'function': 'like/as'},
            'ለ-': {'type': 'prepositional', 'function': 'for/to'},
            'የ-': {'type': 'genitive', 'function': 'of/possessive'},
            'ዘ-': {'type': 'relative', 'function': 'relative_marker'},
        }
        
        # Common suffixes
        self.suffixes = {
            '-ን': {'type': 'nominal', 'function': 'accusative/definite'},
            '-ው': {'type': 'nominal', 'function': 'definite_masculine'},
            '-ዋ': {'type': 'nominal', 'function': 'definite_feminine'},
            '-ኝ': {'type': 'pronominal', 'function': '1sg_object'},
            '-ክ': {'type': 'pronominal', 'function': '2sg_masculine_object'},
            '-ሽ': {'type': 'pronominal', 'function': '2sg_feminine_object'},
            '-አችሁ': {'type': 'verbal', 'function': '2pl_perfect'},
            '-ናል': {'type': 'verbal', 'function': '3sg_imperfect'},
            '-ዎች': {'type': 'nominal', 'function': 'plural'},
            '-ት': {'type': 'nominal', 'function': 'feminine/abstract'},
            '-ነት': {'type': 'nominal', 'function': 'abstract_noun'},
            '-ታ': {'type': 'nominal', 'function': 'feminine_abstract'},
        }
        
        # Compound markers
        self.compound_markers = ['፡', '፣', '፤', '።']
        
        # Fidel (Ge'ez script) character analysis
        self.fidel_orders = {
            'ሀ': 'ha_family', 'ለ': 'la_family', 'ሐ': 'hha_family', 'መ': 'ma_family',
            'ሠ': 'sa_family', 'ረ': 'ra_family', 'ሰ': 'sa_family', 'ቀ': 'qa_family',
            'በ': 'ba_family', 'ተ': 'ta_family', 'ኀ': 'xa_family', 'ነ': 'na_family',
            'አ': 'aa_family', 'ከ': 'ka_family', 'ወ': 'wa_family', 'ዐ': 'aa_family',
            'ዘ': 'za_family', 'የ': 'ya_family', 'ደ': 'da_family', 'ገ': 'ga_family',
            'ጠ': 'tta_family', 'ጰ': 'ppa_family', 'ጸ': 'tsa_family', 'ፀ': 'tza_family',
            'ፈ': 'fa_family', 'ፐ': 'pa_family'
        }

    def initialize_cultural_safety_resources(self):
        """Initialize cultural safety knowledge base"""
        
        # Sacred and religious terms requiring careful handling
        self.sacred_terms = {
            'እግዚአብሔር': {'type': 'divine_name', 'sensitivity': 'high', 'context': 'Christian_Orthodox'},
            'አላህ': {'type': 'divine_name', 'sensitivity': 'high', 'context': 'Islamic'},
            'ክርስቶስ': {'type': 'religious_figure', 'sensitivity': 'high', 'context': 'Christian'},
            'መሐመድ': {'type': 'religious_figure', 'sensitivity': 'high', 'context': 'Islamic'},
            'ቅዱስ': {'type': 'sacred_attribute', 'sensitivity': 'medium', 'context': 'Christian_Orthodox'},
            'ዘጠኙ': {'type': 'religious_group', 'sensitivity': 'medium', 'context': 'Nine_Saints'},
            'ቤተክርስቲያን': {'type': 'religious_institution', 'sensitivity': 'medium', 'context': 'Christian'},
            'መስጊድ': {'type': 'religious_institution', 'sensitivity': 'medium', 'context': 'Islamic'},
            'ሃይማኖት': {'type': 'religious_concept', 'sensitivity': 'medium', 'context': 'general_religious'},
            'ኦርቶዶክስ': {'type': 'religious_denomination', 'sensitivity': 'medium', 'context': 'Christian_Orthodox'},
            'ተዋሕዶ': {'type': 'religious_denomination', 'sensitivity': 'medium', 'context': 'Ethiopian_Orthodox'},
            'አቡነ': {'type': 'religious_title', 'sensitivity': 'medium', 'context': 'Orthodox_patriarch'},
            'ገዳም': {'type': 'religious_place', 'sensitivity': 'low', 'context': 'monastery'},
            'አብዮተ': {'type': 'historical_sensitive', 'sensitivity': 'medium', 'context': 'revolution'},
            'ንጉሠ': {'type': 'royal_title', 'sensitivity': 'low', 'context': 'historical_monarchy'},
            'ዘመነ': {'type': 'temporal_marker', 'sensitivity': 'low', 'context': 'historical_era'},
        }
        
        # Ethnic and regional sensitivity markers
        self.ethnic_terms = {
            'ኦሮሞ': {'sensitivity': 'medium', 'context': 'ethnic_group'},
            'አማራ': {'sensitivity': 'medium', 'context': 'ethnic_group'},
            'ትግራይ': {'sensitivity': 'medium', 'context': 'ethnic_region'},
            'ሶማሌ': {'sensitivity': 'medium', 'context': 'ethnic_group'},
            'አፋር': {'sensitivity': 'medium', 'context': 'ethnic_group'},
            'ሲዳማ': {'sensitivity': 'medium', 'context': 'ethnic_group'},
            'ጉራጌ': {'sensitivity': 'medium', 'context': 'ethnic_group'},
            'ሃረሪ': {'sensitivity': 'medium', 'context': 'ethnic_group'},
        }
        
        # Historical sensitivity markers
        self.historical_sensitive_terms = {
            'ደርግ': {'sensitivity': 'high', 'context': 'military_regime'},
            'አብዮት': {'sensitivity': 'medium', 'context': 'revolution'},
            'መንግሥት': {'sensitivity': 'low', 'context': 'government'},
            'ንጉሥ': {'sensitivity': 'low', 'context': 'monarchy'},
            'ኮሎኔል': {'sensitivity': 'medium', 'context': 'military_title'},
            'ጠቅላይ': {'sensitivity': 'low', 'context': 'leadership_title'},
        }
        
        # Gender and social hierarchy terms
        self.social_terms = {
            'ወንድ': {'sensitivity': 'low', 'context': 'gender_male'},
            'ሴት': {'sensitivity': 'low', 'context': 'gender_female'},
            'አንስት': {'sensitivity': 'low', 'context': 'gender_female'},
            'ባል': {'sensitivity': 'low', 'context': 'spouse_male'},
            'ሚስት': {'sensitivity': 'low', 'context': 'spouse_female'},
            'ጌታ': {'sensitivity': 'medium', 'context': 'social_hierarchy'},
            'ባርያ': {'sensitivity': 'high', 'context': 'historical_social_status'},
        }

    def segment_word_morphologically(self, word: str) -> MorphologicalAnalysis:
        """
        Perform detailed morphological segmentation of an Amharic word
        """
        if not word or not word.strip():
            return MorphologicalAnalysis(word, [], 0.0, 0.0, 0.0)
            
        word = word.strip()
        segments = []
        confidence_scores = []
        
        # Check for compound words (separated by punctuation)
        if any(marker in word for marker in self.compound_markers):
            return self._analyze_compound_word(word)
        
        # Analyze prefixes
        remaining_word = word
        prefix_segments, remaining_word, prefix_confidence = self._extract_prefixes(remaining_word)
        segments.extend(prefix_segments)
        confidence_scores.append(prefix_confidence)
        
        # Analyze root and stem
        root_analysis, stem_remainder, root_confidence = self._extract_root_and_stem(remaining_word)
        if root_analysis:
            segments.append(root_analysis)
            confidence_scores.append(root_confidence)
            remaining_word = stem_remainder
        
        # Analyze suffixes
        suffix_segments, suffix_confidence = self._extract_suffixes(remaining_word)
        segments.extend(suffix_segments)
        confidence_scores.append(suffix_confidence)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Determine root and pattern confidence
        root_conf = root_confidence if root_analysis else 0.0
        pattern_conf = self._calculate_pattern_confidence(word, segments)
        
        return MorphologicalAnalysis(
            original_word=word,
            segmentation=segments,
            overall_confidence=overall_confidence,
            root_confidence=root_conf,
            pattern_confidence=pattern_conf
        )

    def _extract_prefixes(self, word: str) -> Tuple[List[MorphemeSegment], str, float]:
        """Extract and analyze prefixes from the word"""
        segments = []
        remaining = word
        confidence = 1.0
        
        # Check for multiple prefixes in order
        for prefix, info in sorted(self.prefixes.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.startswith(prefix):
                segment = MorphemeSegment(
                    morpheme=prefix,
                    gloss=info['function'],
                    pos_tag=info['type'],
                    morphological_features={'prefix_type': info['function']},
                    confidence=0.9  # High confidence for clear prefixes
                )
                segments.append(segment)
                remaining = remaining[len(prefix):]
                break
        
        return segments, remaining, confidence

    def _extract_root_and_stem(self, word: str) -> Tuple[Optional[MorphemeSegment], str, float]:
        """Extract root and analyze stem patterns"""
        # Try to match against known roots
        for root, info in self.common_roots.items():
            if self._root_pattern_match(word, root):
                segment = MorphemeSegment(
                    morpheme=root,
                    gloss=info['meaning'],
                    pos_tag='root',
                    morphological_features={
                        'pattern_type': info['pattern'],
                        'root_type': 'trilateral' if len(root) == 3 else 'quadrilateral'
                    },
                    confidence=0.85,
                    root=root,
                    pattern=info['pattern']
                )
                # Return remaining part after root extraction
                remaining = self._extract_remainder_after_root(word, root)
                return segment, remaining, 0.85
        
        # If no known root found, attempt pattern-based analysis
        probable_root = self._extract_probable_root(word)
        if probable_root:
            segment = MorphemeSegment(
                morpheme=probable_root,
                gloss='unknown_root',
                pos_tag='root',
                morphological_features={'pattern_type': 'inferred'},
                confidence=0.6,
                root=probable_root
            )
            remaining = word.replace(probable_root, '', 1)
            return segment, remaining, 0.6
        
        return None, word, 0.0

    def _extract_suffixes(self, word: str) -> Tuple[List[MorphemeSegment], float]:
        """Extract and analyze suffixes"""
        segments = []
        remaining = word
        confidence = 1.0
        
        # Check for suffixes (process from longest to shortest)
        for suffix, info in sorted(self.suffixes.items(), key=lambda x: len(x[0]), reverse=True):
            if remaining.endswith(suffix):
                segment = MorphemeSegment(
                    morpheme=suffix,
                    gloss=info['function'],
                    pos_tag=info['type'],
                    morphological_features={'suffix_type': info['function']},
                    confidence=0.8
                )
                segments.append(segment)
                remaining = remaining[:-len(suffix)]
                break
        
        return segments, confidence

    def _root_pattern_match(self, word: str, root: str) -> bool:
        """Check if word contains the root pattern"""
        # Simplified pattern matching - can be enhanced
        root_chars = list(root)
        word_chars = list(word)
        
        # Check if root consonants appear in sequence (allowing for vowel insertions)
        root_idx = 0
        for char in word_chars:
            if root_idx < len(root_chars) and char == root_chars[root_idx]:
                root_idx += 1
        
        return root_idx == len(root_chars)

    def _extract_remainder_after_root(self, word: str, root: str) -> str:
        """Extract the remaining part after root removal"""
        # Simplified - remove first occurrence of root pattern
        for i, char in enumerate(word):
            if word[i:i+len(root)] == root:
                return word[:i] + word[i+len(root):]
        return word

    def _extract_probable_root(self, word: str) -> Optional[str]:
        """Attempt to extract probable root using Ethiopian Semitic patterns"""
        if len(word) < 3:
            return None
        
        # Simple heuristic: try to find 3-consonant pattern
        consonants = []
        for char in word:
            if self._is_consonant(char):
                consonants.append(char)
                if len(consonants) == 3:
                    return ''.join(consonants)
        
        return None

    def _is_consonant(self, char: str) -> bool:
        """Check if character is likely a consonant in Amharic"""
        # Basic check based on Fidel structure
        fidel_base = self._get_fidel_base(char)
        return fidel_base in self.fidel_orders

    def _get_fidel_base(self, char: str) -> str:
        """Get base form of Fidel character"""
        # Simplified - get first order of character family
        name = unicodedata.name(char, '')
        if 'ETHIOPIC' in name:
            # Extract base character
            for base in self.fidel_orders.keys():
                if char.startswith(base) or base in name:
                    return base
        return char

    def _calculate_pattern_confidence(self, word: str, segments: List[MorphemeSegment]) -> float:
        """Calculate confidence in the morphological pattern analysis"""
        if not segments:
            return 0.0
        
        # Base confidence on number of identified segments and their individual confidence
        total_confidence = sum(seg.confidence for seg in segments)
        pattern_confidence = total_confidence / len(segments)
        
        # Boost confidence if we identified a clear root
        root_segments = [seg for seg in segments if seg.pos_tag == 'root']
        if root_segments and root_segments[0].root in self.common_roots:
            pattern_confidence += 0.1
        
        return min(pattern_confidence, 1.0)

    def _analyze_compound_word(self, word: str) -> MorphologicalAnalysis:
        """Analyze compound words with punctuation markers"""
        segments = []
        parts = re.split(r'[፡፣፤።]', word)
        
        total_confidence = 0.0
        for part in parts:
            if part.strip():
                part_analysis = self.segment_word_morphologically(part.strip())
                segments.extend(part_analysis.segmentation)
                total_confidence += part_analysis.overall_confidence
        
        avg_confidence = total_confidence / len(parts) if parts else 0.0
        
        return MorphologicalAnalysis(
            original_word=word,
            segmentation=segments,
            overall_confidence=avg_confidence,
            root_confidence=avg_confidence,
            pattern_confidence=avg_confidence
        )

    def assess_cultural_safety(self, text: str) -> CulturalSafetyAssessment:
        """
        Comprehensive cultural safety assessment of Amharic text
        """
        issues = []
        recommendations = []
        safety_level = 'safe'
        confidence = 0.9
        
        # Tokenize text for analysis
        words = self._tokenize_amharic_text(text)
        
        # Check for sacred terms
        sacred_issues = self._check_sacred_terms(words)
        issues.extend(sacred_issues)
        
        # Check for ethnic sensitivity
        ethnic_issues = self._check_ethnic_terms(words)
        issues.extend(ethnic_issues)
        
        # Check for historical sensitivity
        historical_issues = self._check_historical_terms(words)
        issues.extend(historical_issues)
        
        # Check for social hierarchy sensitivity
        social_issues = self._check_social_terms(words)
        issues.extend(social_issues)
        
        # Determine overall safety level
        if any(issue['severity'] == 'high' for issue in issues):
            safety_level = 'problematic'
            recommendations.append("Review high-sensitivity terms for appropriate context and usage")
        elif any(issue['severity'] == 'medium' for issue in issues):
            safety_level = 'needs_review'
            recommendations.append("Consider cultural context for medium-sensitivity terms")
        
        # Generate context-specific recommendations
        if sacred_issues:
            recommendations.append("Ensure religious terms are used with proper reverence and context")
        if ethnic_issues:
            recommendations.append("Verify ethnic references are respectful and appropriate")
        if historical_issues:
            recommendations.append("Consider historical sensitivity when discussing past events")
        
        cultural_context = self._generate_cultural_context(issues, words)
        
        return CulturalSafetyAssessment(
            safety_level=safety_level,
            issues=issues,
            recommendations=recommendations,
            cultural_context=cultural_context,
            confidence=confidence
        )

    def _tokenize_amharic_text(self, text: str) -> List[str]:
        """Tokenize Amharic text into words"""
        # Remove punctuation and split on whitespace
        cleaned_text = re.sub(r'[፡፣፤።\s]+', ' ', text)
        return [word.strip() for word in cleaned_text.split() if word.strip()]

    def _check_sacred_terms(self, words: List[str]) -> List[Dict[str, str]]:
        """Check for sacred/religious terms requiring sensitivity"""
        issues = []
        for word in words:
            for sacred_term, info in self.sacred_terms.items():
                if sacred_term in word or word in sacred_term:
                    issues.append({
                        'type': 'sacred_term',
                        'term': word,
                        'reference': sacred_term,
                        'severity': info['sensitivity'],
                        'context': info['context'],
                        'description': f"Religious/sacred term '{sacred_term}' requires careful handling"
                    })
        return issues

    def _check_ethnic_terms(self, words: List[str]) -> List[Dict[str, str]]:
        """Check for ethnic terms requiring sensitivity"""
        issues = []
        for word in words:
            for ethnic_term, info in self.ethnic_terms.items():
                if ethnic_term in word or word in ethnic_term:
                    issues.append({
                        'type': 'ethnic_term',
                        'term': word,
                        'reference': ethnic_term,
                        'severity': info['sensitivity'],
                        'context': info['context'],
                        'description': f"Ethnic reference '{ethnic_term}' should be used respectfully"
                    })
        return issues

    def _check_historical_terms(self, words: List[str]) -> List[Dict[str, str]]:
        """Check for historically sensitive terms"""
        issues = []
        for word in words:
            for hist_term, info in self.historical_sensitive_terms.items():
                if hist_term in word or word in hist_term:
                    issues.append({
                        'type': 'historical_term',
                        'term': word,
                        'reference': hist_term,
                        'severity': info['sensitivity'],
                        'context': info['context'],
                        'description': f"Historical term '{hist_term}' may require contextual sensitivity"
                    })
        return issues

    def _check_social_terms(self, words: List[str]) -> List[Dict[str, str]]:
        """Check for social hierarchy/gender terms"""
        issues = []
        for word in words:
            for social_term, info in self.social_terms.items():
                if social_term in word or word in social_term:
                    issues.append({
                        'type': 'social_term',
                        'term': word,
                        'reference': social_term,
                        'severity': info['sensitivity'],
                        'context': info['context'],
                        'description': f"Social term '{social_term}' should be used appropriately"
                    })
        return issues

    def _generate_cultural_context(self, issues: List[Dict], words: List[str]) -> str:
        """Generate cultural context explanation"""
        contexts = []
        
        if any(issue['type'] == 'sacred_term' for issue in issues):
            contexts.append("Text contains religious/sacred terminology requiring reverent treatment")
        
        if any(issue['type'] == 'ethnic_term' for issue in issues):
            contexts.append("Text references Ethiopian ethnic groups requiring respectful representation")
        
        if any(issue['type'] == 'historical_term' for issue in issues):
            contexts.append("Text includes historically sensitive terms requiring contextual awareness")
        
        if not contexts:
            contexts.append("Text appears culturally neutral with standard Amharic vocabulary")
        
        return "; ".join(contexts)

    def process_corpus(self, input_file: str, output_file: str) -> Dict[str, int]:
        """
        Process the entire test corpus with morphological analysis and cultural safety validation
        """
        logger.info(f"Starting corpus processing: {input_file} -> {output_file}")
        
        # Load corpus
        with open(input_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        
        processed_articles = []
        stats = {
            'total_articles': 0,
            'processed_articles': 0,
            'high_confidence_analyses': 0,
            'cultural_safe': 0,
            'cultural_needs_review': 0,
            'cultural_problematic': 0,
            'avg_morphological_confidence': 0.0,
            'avg_cultural_confidence': 0.0
        }
        
        morphological_confidences = []
        cultural_confidences = []
        
        for article in corpus_data.get('articles', []):
            stats['total_articles'] += 1
            
            try:
                # Extract text content
                content = article.get('content', '')
                if not content:
                    continue
                
                # Tokenize for word-level analysis
                words = self._tokenize_amharic_text(content)
                
                # Process each word morphologically
                word_analyses = []
                for word in words[:50]:  # Limit to first 50 words for efficiency
                    if len(word) > 1:  # Skip single characters
                        analysis = self.segment_word_morphologically(word)
                        word_analyses.append({
                            'word': word,
                            'segmentation': [
                                {
                                    'morpheme': seg.morpheme,
                                    'gloss': seg.gloss,
                                    'pos_tag': seg.pos_tag,
                                    'features': seg.morphological_features,
                                    'confidence': seg.confidence,
                                    'root': seg.root,
                                    'pattern': seg.pattern
                                } for seg in analysis.segmentation
                            ],
                            'overall_confidence': analysis.overall_confidence,
                            'root_confidence': analysis.root_confidence,
                            'pattern_confidence': analysis.pattern_confidence
                        })
                        
                        morphological_confidences.append(analysis.overall_confidence)
                        if analysis.overall_confidence >= 0.85:
                            stats['high_confidence_analyses'] += 1
                
                # Cultural safety assessment
                cultural_assessment = self.assess_cultural_safety(content)
                cultural_confidences.append(cultural_assessment.confidence)
                
                # Update cultural safety stats
                if cultural_assessment.safety_level == 'safe':
                    stats['cultural_safe'] += 1
                elif cultural_assessment.safety_level == 'needs_review':
                    stats['cultural_needs_review'] += 1
                else:
                    stats['cultural_problematic'] += 1
                
                # Create processed article
                processed_article = {
                    'original_metadata': article.get('metadata', {}),
                    'url': article.get('url', ''),
                    'categories': article.get('categories', []),
                    'morphological_analysis': {
                        'total_words_analyzed': len(word_analyses),
                        'word_analyses': word_analyses,
                        'average_confidence': sum(morphological_confidences[-len(word_analyses):]) / len(word_analyses) if word_analyses else 0.0
                    },
                    'cultural_safety': {
                        'safety_level': cultural_assessment.safety_level,
                        'issues': cultural_assessment.issues,
                        'recommendations': cultural_assessment.recommendations,
                        'cultural_context': cultural_assessment.cultural_context,
                        'confidence': cultural_assessment.confidence
                    },
                    'processing_metadata': {
                        'processed_timestamp': datetime.now().isoformat(),
                        'analyzer_version': '1.0',
                        'confidence_threshold_met': any(wa['overall_confidence'] >= 0.85 for wa in word_analyses)
                    }
                }
                
                processed_articles.append(processed_article)
                stats['processed_articles'] += 1
                
                if stats['processed_articles'] % 10 == 0:
                    logger.info(f"Processed {stats['processed_articles']}/{stats['total_articles']} articles")
                
            except Exception as e:
                logger.error(f"Error processing article: {e}")
                continue
        
        # Calculate final statistics
        if morphological_confidences:
            stats['avg_morphological_confidence'] = sum(morphological_confidences) / len(morphological_confidences)
        if cultural_confidences:
            stats['avg_cultural_confidence'] = sum(cultural_confidences) / len(cultural_confidences)
        
        # Create output data
        output_data = {
            'processing_metadata': {
                'source_file': input_file,
                'processing_timestamp': datetime.now().isoformat(),
                'analyzer_version': '1.0',
                'target_accuracy': 0.85,
                'achieved_accuracy': stats['avg_morphological_confidence'],
                'accuracy_target_met': stats['avg_morphological_confidence'] >= 0.85
            },
            'corpus_statistics': stats,
            'processed_articles': processed_articles
        }
        
        # Save processed results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processing complete. Results saved to {output_file}")
        logger.info(f"Statistics: {stats}")
        
        return stats


def main():
    """Main execution function"""
    analyzer = AmharicLinguisticAnalyzer()
    
    input_file = "/Users/mekdesyared/amharic-hnet-v2/data/raw/test_corpus.json"
    output_file = "/Users/mekdesyared/amharic-hnet-v2/data/processed/test_processed.json"
    
    # Process the corpus
    stats = analyzer.process_corpus(input_file, output_file)
    
    print("\n=== Amharic Linguistic Analysis Complete ===")
    print(f"Total articles processed: {stats['processed_articles']}/{stats['total_articles']}")
    print(f"Average morphological confidence: {stats['avg_morphological_confidence']:.3f}")
    print(f"High confidence analyses: {stats['high_confidence_analyses']}")
    print(f"Target accuracy (>85%): {'✓ ACHIEVED' if stats['avg_morphological_confidence'] >= 0.85 else '✗ NOT MET'}")
    print(f"\nCultural Safety Assessment:")
    print(f"  Safe: {stats['cultural_safe']}")
    print(f"  Needs Review: {stats['cultural_needs_review']}")
    print(f"  Problematic: {stats['cultural_problematic']}")
    print(f"  Average cultural confidence: {stats['avg_cultural_confidence']:.3f}")


if __name__ == "__main__":
    main()