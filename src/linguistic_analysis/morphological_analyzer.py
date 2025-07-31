#!/usr/bin/env python3
"""
Linguistic Analysis Sub-Agent for Amharic H-Net v2
==================================================

This module implements the linguistic-analyzer sub-agent specialized in
morphological processing, dialectal analysis, and linguistic validation
of Amharic text corpora for H-Net training.

Sub-Agent Capabilities:
1. Advanced morphological segmentation using rule-based and statistical methods
2. Dialect classification and variation analysis
3. Cultural context validation and safety assessment
4. Part-of-speech tagging with Amharic-specific tagsets
5. Morphological feature extraction (tense, person, number, case)
6. Text complexity analysis and readability assessment
7. Linguistic quality scoring for training data selection

Usage:
    python -m src.linguistic_analysis.morphological_analyzer --input data/raw --output data/processed
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, NamedTuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import defaultdict, Counter
import unicodedata
from enum import Enum
import pickle


class MorphemeType(Enum):
    """Types of morphemes in Amharic."""
    ROOT = "root"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    INFIX = "infix"
    COMPOUND = "compound"


class POSTag(Enum):
    """Part-of-speech tags for Amharic."""
    NOUN = "NOUN"
    VERB = "VERB"
    ADJ = "ADJ"
    ADV = "ADV"
    PRON = "PRON"
    DET = "DET"
    ADP = "ADP"
    CONJ = "CONJ"
    PART = "PART"
    INTJ = "INTJ"
    NUM = "NUM"
    PUNCT = "PUNCT"


@dataclass
class MorphologicalAnalysis:
    """Result of morphological analysis for a word."""
    word: str
    morphemes: List[str]
    morpheme_types: List[MorphemeType]
    pos_tag: POSTag
    morphological_features: Dict[str, str]
    confidence_score: float
    dialect_markers: List[str]
    cultural_domain: str


@dataclass
class LinguisticAnnotation:
    """Complete linguistic annotation for a text sample."""
    original_text: str
    sentences: List[str]
    word_analyses: List[MorphologicalAnalysis]
    text_complexity: float
    dialect_classification: str
    cultural_safety_score: float
    linguistic_quality_score: float
    readability_metrics: Dict[str, float]
    metadata: Dict[str, any]


class AmharicMorphologicalAnalyzer:
    """
    Advanced morphological analyzer for Amharic text processing.
    
    Implements rule-based morphological segmentation combined with
    statistical validation for high-accuracy morpheme boundary detection.
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Morphological patterns (enhanced from H-Net implementation)
        self.morphological_patterns = self._load_morphological_patterns()
        
        # Cultural domain classifiers
        self.cultural_classifiers = self._load_cultural_classifiers()
        
        # Dialect detection models
        self.dialect_detectors = self._load_dialect_detectors()
        
        # Statistical morpheme boundaries (would be trained)
        self.boundary_model = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for linguistic analysis sub-agent."""
        logger = logging.getLogger('amharic_linguistic_analyzer')
        if not logger.handlers:
            handler = logging.StreamHandler()  
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_morphological_patterns(self) -> Dict[str, any]:
        """Load comprehensive morphological patterns for Amharic."""
        return {
            'verb_patterns': {
                # Person markers
                'person_prefixes': {
                    '1st_sing': ['እ'],
                    '2nd_sing_masc': ['ት'],
                    '2nd_sing_fem': ['ት'],
                    '3rd_sing_masc': ['ይ', ''],
                    '3rd_sing_fem': ['ት'],
                    '1st_plur': ['እን'],
                    '2nd_plur': ['ት'],
                    '3rd_plur': ['ይ']
                },
                'person_suffixes': {
                    '1st_sing': ['ሁ', 'ኝ'],
                    '2nd_sing_masc': ['ህ', 'ክ'],
                    '2nd_sing_fem': ['ሽ', 'ሺ'],
                    '3rd_sing_masc': ['', 'ው'],
                    '3rd_sing_fem': ['', 'ች'],
                    '1st_plur': ['ን', 'ነን'],
                    '2nd_plur': ['ሁ', 'አችሁ'],
                    '3rd_plur': ['ው', 'ኡ']
                },
                # Tense/aspect markers
                'tense_prefixes': {
                    'past': ['', 'ተ'],
                    'present': ['ይ', 'ት', 'እ'],
                    'future': ['እ', 'ይ'],
                    'perfect': [''],
                    'imperative': ['']
                },
                'aspect_suffixes': {
                    'perfective': ['', 'አል'],
                    'imperfective': ['ላል', 'ታል'],
                    'progressive': ['ቶ', 'ታ']
                }
            },
            'noun_patterns': {
                # Case markers
                'case_markers': {
                    'nominative': '',
                    'accusative': 'ን',
                    'genitive': 'የ',
                    'dative': 'ለ',
                    'ablative': 'ከ',
                    'instrumental': 'በ'
                },
                # Number markers
                'number_markers': {
                    'singular': '',
                    'plural': ['ች', 'ኦች', 'ዎች', 'ዎች', 'አን']
                },
                # Definiteness markers
                'definiteness_markers': {
                    'indefinite': '',
                    'definite': ['ው', 'ዋ', 'ይቱ', 'ይቷ', 'ዎቹ']
                }
            },
            'adjective_patterns': {
                'comparative': ['ይበልጥ', 'በጣም'],
                'superlative': ['በጣም', 'ከሁሉ']
            },
            'compound_patterns': {
                'noun_noun': r'([ሀ-ፐ]+)([ሀ-ፐ]+)',
                'adj_noun': r'([ሀ-ፐ]+)([ሀ-ፐ]+)',
                'verb_noun': r'([ሀ-ፐ]+)([ሀ-ፐ]+)'
            }
        }
    
    def _load_cultural_classifiers(self) -> Dict[str, any]:
        """Load cultural domain classification patterns."""
        return {
            'religious_terms': {
                'christian': ['መስቀል', 'ኢየሱስ', 'ማርያም', 'ገብርኤል', 'ቤተክርስቲያን', 'ቅዳሴ', 'ፍስሐ'],
                'islamic': ['መስጊድ', 'አላህ', 'መሐመድ', 'ቁርዓን', 'ሶላህ', 'ሀጅ'],
                'traditional': ['አባ', 'አባት', 'ዘአር', 'ጣና', 'አዳር']
            },
            'cultural_practices': {
                'coffee_ceremony': ['ቡና', 'ጅቤና', 'መንበር', 'ሙከራ'],
                'festivals': ['ገና', 'ፋሲካ', 'መስቀል', 'ጥምቀት', 'አሸንዳ'],
                'traditional_food': ['እንጀራ', 'ዶሮ', 'ወጥ', 'ጠጅ', 'ተጅ'],
                'clothing': ['ሻማ', 'ቀሚስ', 'ነጠላ', 'ካባ']
            },
            'historical_references': {
                'emperors': ['ቀዳማዊ', 'ኃይለ', 'ሥላሴ', 'ምኒልክ', 'ተዎድሮስ', 'ዮሐንስ'],
                'battles': ['አድዋ', 'መቅዱላ', 'ሰገር', 'አንፋር'],
                'places': ['ላሊበላ', 'አክሱም', 'ሻሸመኔ', 'ሀረር']
            }
        }
    
    def _load_dialect_detectors(self) -> Dict[str, any]:
        """Load dialect detection patterns and markers."""
        return {
            'dialect_markers': {
                'eritrean': {
                    'lexical': ['ማይ', 'ቡን', 'ጽቡቅ', 'ክብሪ'],
                    'phonological': ['ሐ', 'ሠ', 'ዐ'],
                    'grammatical': ['እዩ', 'እያ', 'እዮም']
                },
                'gojjam': {
                    'lexical': ['እሱ', 'እሳ', 'እሸን', 'እሸም'],
                    'phonological': ['ጨ', 'ዥ'],
                    'grammatical': ['ነበረ', 'ነበረት']
                },
                'wollo': {
                    'lexical': ['ነበረ', 'ተዓደወ', 'ተመለሰ'],
                    'phonological': [],
                    'grammatical': []
                },
                'shewa': {
                    'lexical': ['ውሀ', 'ቡና', 'ጥሩ'],
                    'phonological': [],
                    'grammatical': ['ነው', 'ናት', 'ናቢው']
                }
            },
            'dialect_probabilities': {
                # Would be learned from corpus
                'default_prior': 0.25  # Equal probability for 4 main dialects
            }
        }
    
    def analyze_text(self, text: str) -> LinguisticAnnotation:
        """
        Perform comprehensive linguistic analysis of Amharic text.
        
        Args:
            text: Input Amharic text to analyze
            
        Returns:
            Complete linguistic annotation with morphological analysis
        """
        self.logger.info(f"🔍 Analyzing text: {text[:50]}...")
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Sentence segmentation
        sentences = self._segment_sentences(cleaned_text)
        
        # Word-level analysis
        word_analyses = []
        for sentence in sentences:
            words = self._tokenize_words(sentence)
            for word in words:
                if self._is_amharic_word(word):
                    analysis = self._analyze_word(word)
                    if analysis:
                        word_analyses.append(analysis)
        
        # Text-level analysis
        text_complexity = self._calculate_text_complexity(word_analyses)
        dialect_classification = self._classify_dialect(text, word_analyses)
        cultural_safety_score = self._assess_cultural_safety(text, word_analyses)
        linguistic_quality_score = self._calculate_linguistic_quality(word_analyses)
        readability_metrics = self._calculate_readability_metrics(text, sentences, word_analyses)
        
        return LinguisticAnnotation(
            original_text=text,
            sentences=sentences,
            word_analyses=word_analyses,
            text_complexity=text_complexity,
            dialect_classification=dialect_classification,
            cultural_safety_score=cultural_safety_score,
            linguistic_quality_score=linguistic_quality_score,
            readability_metrics=readability_metrics,
            metadata={
                'analysis_timestamp': str(np.datetime64('now')),
                'analyzer_version': '1.0.0',
                'total_words': len(word_analyses),
                'unique_morphemes': len(set(m for wa in word_analyses for m in wa.morphemes))
            }
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess and normalize Amharic text."""
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize Ethiopic punctuation
        text = re.sub(r'([።፣፤፥፦፧፨])', r' \\1 ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using Ethiopic punctuation."""
        # Split on Ethiopic sentence terminators
        sentences = re.split(r'[።፨]', text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _tokenize_words(self, sentence: str) -> List[str]:
        """Tokenize sentence into words."""
        # Split on whitespace and punctuation
        words = re.findall(r'[\u1200-\u137F]+|[^\u1200-\u137F\s]+', sentence)
        
        return [w.strip() for w in words if w.strip()]
    
    def _is_amharic_word(self, word: str) -> bool:
        """Check if word contains Amharic characters."""
        return bool(re.search(r'[\u1200-\u137F]', word))
    
    def _analyze_word(self, word: str) -> Optional[MorphologicalAnalysis]:
        """
        Perform morphological analysis of a single Amharic word.
        
        Args:
            word: Amharic word to analyze
            
        Returns:
            Morphological analysis or None if analysis fails
        """
        try:
            # Morpheme segmentation
            morphemes, morpheme_types = self._segment_morphemes(word)
            
            # POS tagging
            pos_tag = self._tag_pos(word, morphemes, morpheme_types)
            
            # Morphological feature extraction
            features = self._extract_morphological_features(word, morphemes, pos_tag)
            
            # Confidence scoring
            confidence = self._calculate_analysis_confidence(word, morphemes, pos_tag)
            
            # Dialect markers
            dialect_markers = self._detect_word_dialect_markers(word)
            
            # Cultural domain classification
            cultural_domain = self._classify_word_cultural_domain(word)
            
            return MorphologicalAnalysis(
                word=word,
                morphemes=morphemes,
                morpheme_types=morpheme_types,
                pos_tag=pos_tag,
                morphological_features=features,
                confidence_score=confidence,
                dialect_markers=dialect_markers,
                cultural_domain=cultural_domain
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze word '{word}': {e}")
            return None
    
    def _segment_morphemes(self, word: str) -> Tuple[List[str], List[MorphemeType]]:
        """
        Segment word into morphemes using rule-based approach.
        
        Args:
            word: Word to segment
            
        Returns:
            Tuple of (morphemes, morpheme_types)
        """
        morphemes = []
        morpheme_types = []
        
        remaining = word
        
        # Check for verb patterns
        if self._looks_like_verb(word):
            morphemes, morpheme_types = self._segment_verb(word)
        
        # Check for noun patterns
        elif self._looks_like_noun(word):
            morphemes, morpheme_types = self._segment_noun(word)
        
        # Check for compound patterns
        elif self._looks_like_compound(word):
            morphemes, morpheme_types = self._segment_compound(word)
        
        # Default: treat as single morpheme
        else:
            morphemes = [word]
            morpheme_types = [MorphemeType.ROOT]
        
        return morphemes, morpheme_types
    
    def _looks_like_verb(self, word: str) -> bool:
        """Check if word has verb-like characteristics."""
        verb_indicators = ['ይ', 'ት', 'እ', 'ን']  # Common verb prefixes
        return any(word.startswith(prefix) for prefix in verb_indicators)
    
    def _looks_like_noun(self, word: str) -> bool:
        """Check if word has noun-like characteristics."""
        noun_indicators = ['የ', 'ለ', 'በ', 'ከ']  # Common noun prefixes
        return any(word.startswith(prefix) for prefix in noun_indicators)
    
    def _looks_like_compound(self, word: str) -> bool:
        """Check if word looks like a compound."""
        return len(word) > 6  # Simple heuristic
    
    def _segment_verb(self, word: str) -> Tuple[List[str], List[MorphemeType]]:
        """Segment verb into morphemes."""
        morphemes = []
        types = []
        
        # Simplified verb segmentation
        patterns = self.morphological_patterns['verb_patterns']
        
        # Check for person prefixes
        for person, prefixes in patterns['person_prefixes'].items():
            for prefix in prefixes:
                if prefix and word.startswith(prefix):
                    morphemes.append(prefix)
                    types.append(MorphemeType.PREFIX)
                    word = word[len(prefix):]
                    break
        
        # Check for person suffixes
        for person, suffixes in patterns['person_suffixes'].items():
            for suffix in suffixes:
                if suffix and word.endswith(suffix):
                    # Save suffix for later
                    root = word[:-len(suffix)]
                    if root:
                        morphemes.append(root)
                        types.append(MorphemeType.ROOT)
                        morphemes.append(suffix)
                        types.append(MorphemeType.SUFFIX)
                        return morphemes, types
        
        # If no suffix found, treat remaining as root
        if word:
            morphemes.append(word)
            types.append(MorphemeType.ROOT)
        
        return morphemes, types
    
    def _segment_noun(self, word: str) -> Tuple[List[str], List[MorphemeType]]:
        """Segment noun into morphemes."""
        morphemes = []
        types = []
        
        patterns = self.morphological_patterns['noun_patterns']
        
        # Check for case markers (prefixes)
        for case, marker in patterns['case_markers'].items():
            if marker and word.startswith(marker):
                morphemes.append(marker)
                types.append(MorphemeType.PREFIX)
                word = word[len(marker):]
                break
        
        # Check for definiteness/number markers (suffixes)
        for marker_type in ['definiteness_markers', 'number_markers']:
            for category, markers in patterns[marker_type].items():
                if isinstance(markers, list):
                    for marker in markers:
                        if marker and word.endswith(marker):
                            root = word[:-len(marker)]
                            if root:
                                morphemes.append(root)
                                types.append(MorphemeType.ROOT)
                                morphemes.append(marker)
                                types.append(MorphemeType.SUFFIX)
                                return morphemes, types
        
        # Default: treat as root
        if word:
            morphemes.append(word)
            types.append(MorphemeType.ROOT)
        
        return morphemes, types
    
    def _segment_compound(self, word: str) -> Tuple[List[str], List[MorphemeType]]:
        """Segment compound word."""
        # Simplified compound segmentation
        # In practice, would use more sophisticated algorithms
        
        mid_point = len(word) // 2
        first_part = word[:mid_point]
        second_part = word[mid_point:]
        
        return [first_part, second_part], [MorphemeType.COMPOUND, MorphemeType.COMPOUND]
    
    def _tag_pos(self, word: str, morphemes: List[str], morpheme_types: List[MorphemeType]) -> POSTag:
        """Assign part-of-speech tag to word."""
        # Simplified POS tagging based on morphological patterns
        
        if self._looks_like_verb(word):
            return POSTag.VERB
        elif self._looks_like_noun(word):
            return POSTag.NOUN
        elif any(word.endswith(adj_suffix) for adj_suffix in ['ኛ', 'ታዊ', 'ሳዊ']):
            return POSTag.ADJ
        elif word in ['በጣም', 'ብዙ', 'ትንሽ']:
            return POSTag.ADV
        else:
            return POSTag.NOUN  # Default
    
    def _extract_morphological_features(
        self, 
        word: str, 
        morphemes: List[str], 
        pos_tag: POSTag
    ) -> Dict[str, str]:
        """Extract morphological features from word analysis."""
        features = {}
        
        if pos_tag == POSTag.VERB:
            # Extract verb features
            features.update(self._extract_verb_features(word, morphemes))
        elif pos_tag == POSTag.NOUN:
            # Extract noun features
            features.update(self._extract_noun_features(word, morphemes))
        
        return features
    
    def _extract_verb_features(self, word: str, morphemes: List[str]) -> Dict[str, str]:
        """Extract verb-specific morphological features."""
        features = {}
        
        # Simplified feature extraction
        if word.startswith('ይ'):
            features['person'] = '3rd'
            features['number'] = 'singular'
            features['tense'] = 'present'
        elif word.startswith('ት'):
            features['person'] = '2nd_or_3rd'
            features['tense'] = 'present'
        elif word.startswith('እ'):
            features['person'] = '1st'
            features['tense'] = 'present_or_future'
        
        return features
    
    def _extract_noun_features(self, word: str, morphemes: List[str]) -> Dict[str, str]:
        """Extract noun-specific morphological features."""
        features = {}
        
        # Check for number
        if any(word.endswith(pl) for pl in ['ች', 'ኦች', 'ዎች']):
            features['number'] = 'plural'
        else:
            features['number'] = 'singular'
        
        # Check for definiteness
        if any(word.endswith(def_marker) for def_marker in ['ው', 'ዋ', 'ይቱ']):
            features['definiteness'] = 'definite'
        else:
            features['definiteness'] = 'indefinite'
        
        return features
    
    def _calculate_analysis_confidence(
        self, 
        word: str, 
        morphemes: List[str], 
        pos_tag: POSTag
    ) -> float:
        """Calculate confidence score for morphological analysis."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for recognized patterns
        if len(morphemes) > 1:
            confidence += 0.2
        
        # Boost confidence for known POS patterns
        if pos_tag in [POSTag.VERB, POSTag.NOUN]:
            confidence += 0.2
        
        # Penalize very long words (likely compounds needing better segmentation)
        if len(word) > 10:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _detect_word_dialect_markers(self, word: str) -> List[str]:
        """Detect dialect markers in individual word."""
        markers = []
        
        for dialect, patterns in self.dialect_detectors['dialect_markers'].items():
            # Check lexical markers
            if word in patterns['lexical']:
                markers.append(f"{dialect}_lexical")
            
            # Check phonological markers
            if any(char in word for char in patterns['phonological']):
                markers.append(f"{dialect}_phonological")
        
        return markers if markers else ['standard']
    
    def _classify_word_cultural_domain(self, word: str) -> str:
        """Classify word into cultural domain."""
        for domain, categories in self.cultural_classifiers.items():
            for category, terms in categories.items():
                if word in terms:
                    return f"{domain}_{category}"
        
        return 'general'
    
    def _calculate_text_complexity(self, word_analyses: List[MorphologicalAnalysis]) -> float:
        """Calculate overall text complexity score."""
        if not word_analyses:
            return 0.0
        
        # Average morphemes per word
        avg_morphemes = np.mean([len(wa.morphemes) for wa in word_analyses])
        
        # Unique POS tag diversity
        pos_diversity = len(set(wa.pos_tag for wa in word_analyses)) / len(POSTag)
        
        # Morphological feature complexity
        feature_complexity = np.mean([
            len(wa.morphological_features) for wa in word_analyses
        ])
        
        # Combine metrics
        complexity = (0.4 * min(1.0, avg_morphemes / 3.0) +
                     0.3 * pos_diversity +
                     0.3 * min(1.0, feature_complexity / 5.0))
        
        return complexity
    
    def _classify_dialect(self, text: str, word_analyses: List[MorphologicalAnalysis]) -> str:
        """Classify text dialect based on linguistic markers."""
        dialect_scores = defaultdict(float)
        
        # Collect dialect markers from word analyses
        for analysis in word_analyses:
            for marker in analysis.dialect_markers:
                if '_' in marker:
                    dialect = marker.split('_')[0]
                    dialect_scores[dialect] += 1.0
        
        # Normalize scores
        total_markers = sum(dialect_scores.values())
        if total_markers == 0:
            return 'standard'
        
        # Return most likely dialect
        best_dialect = max(dialect_scores, key=dialect_scores.get)
        return best_dialect if dialect_scores[best_dialect] / total_markers > 0.3 else 'mixed'
    
    def _assess_cultural_safety(self, text: str, word_analyses: List[MorphologicalAnalysis]) -> float:
        """Assess cultural safety of text content."""
        safety_score = 1.0
        
        # Check for potentially problematic cultural usage
        for analysis in word_analyses:
            if 'religious' in analysis.cultural_domain:
                # Religious terms require careful context checking
                if self._is_inappropriate_religious_usage(analysis.word, text):
                    safety_score -= 0.2
            
            elif 'historical' in analysis.cultural_domain:
                # Historical references should be respectful
                if self._is_inappropriate_historical_usage(analysis.word, text):
                    safety_score -= 0.1
        
        return max(0.0, safety_score)
    
    def _is_inappropriate_religious_usage(self, word: str, context: str) -> bool:
        """Check if religious term is used inappropriately."""
        # Simplified check - in practice would be more sophisticated
        inappropriate_contexts = ['ወይ', 'ኣይ', 'ሣቅ']  # Mockery, jokes, etc.
        return any(inappropriate in context for inappropriate in inappropriate_contexts)
    
    def _is_inappropriate_historical_usage(self, word: str, context: str) -> bool:
        """Check if historical term is used inappropriately."""
        # Simplified check
        return False  # Would implement proper context analysis
    
    def _calculate_linguistic_quality(self, word_analyses: List[MorphologicalAnalysis]) -> float:
        """Calculate overall linguistic quality score."""
        if not word_analyses:
            return 0.0
        
        # Average analysis confidence
        avg_confidence = np.mean([wa.confidence_score for wa in word_analyses])
        
        # Morphological richness
        morpheme_diversity = len(set(m for wa in word_analyses for m in wa.morphemes))
        richness_score = min(1.0, morpheme_diversity / (len(word_analyses) * 0.5))
        
        # POS tag distribution (penalize if too skewed)
        pos_counts = Counter(wa.pos_tag for wa in word_analyses)
        pos_entropy = -sum((count/len(word_analyses)) * np.log2(count/len(word_analyses)) 
                          for count in pos_counts.values())
        max_entropy = np.log2(len(pos_counts))
        entropy_score = pos_entropy / max_entropy if max_entropy > 0 else 0
        
        # Combine scores
        quality = (0.4 * avg_confidence +
                  0.3 * richness_score +
                  0.3 * entropy_score)
        
        return quality
    
    def _calculate_readability_metrics(
        self, 
        text: str, 
        sentences: List[str], 
        word_analyses: List[MorphologicalAnalysis]
    ) -> Dict[str, float]:
        """Calculate readability metrics for Amharic text."""
        metrics = {}
        
        # Basic statistics
        total_words = len(word_analyses)
        total_sentences = len(sentences)
        total_syllables = sum(self._count_syllables(wa.word) for wa in word_analyses)
        
        if total_sentences > 0:
            metrics['avg_words_per_sentence'] = total_words / total_sentences
        else:
            metrics['avg_words_per_sentence'] = 0
        
        if total_words > 0:
            metrics['avg_syllables_per_word'] = total_syllables / total_words
            metrics['avg_morphemes_per_word'] = np.mean([len(wa.morphemes) for wa in word_analyses])
        else:
            metrics['avg_syllables_per_word'] = 0
            metrics['avg_morphemes_per_word'] = 0
        
        # Amharic-specific readability (simplified)
        if total_words > 0 and total_sentences > 0:
            # Adapted from Flesch Reading Ease formula for Amharic
            readability = (206.835 - 
                          1.015 * metrics['avg_words_per_sentence'] -
                          84.6 * metrics['avg_syllables_per_word'])
            metrics['amharic_readability_score'] = max(0, min(100, readability))
        else:
            metrics['amharic_readability_score'] = 0
        
        return metrics
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in Amharic word (simplified)."""
        # In Amharic, each Ge'ez character generally represents one syllable
        return len([c for c in word if '\u1200' <= c <= '\u137F'])
    
    def save_analysis(self, annotation: LinguisticAnnotation, output_path: Path):
        """Save linguistic annotation to file."""
        # Convert to serializable format
        annotation_dict = asdict(annotation)
        
        # Convert enums to strings
        for i, wa in enumerate(annotation_dict['word_analyses']):
            wa['pos_tag'] = wa['pos_tag'].value if hasattr(wa['pos_tag'], 'value') else str(wa['pos_tag'])
            wa['morpheme_types'] = [mt.value if hasattr(mt, 'value') else str(mt) for mt in wa['morpheme_types']]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_dict, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 Saved linguistic analysis to {output_path}")


def main():
    """Main function for linguistic analysis sub-agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Amharic Linguistic Analysis Sub-Agent')
    parser.add_argument('--input', required=True, help='Input text file or directory')
    parser.add_argument('--output', required=True, help='Output directory for analyses')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    
    args = parser.parse_args()
    
    analyzer = AmharicMorphologicalAnalyzer()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Process single file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        annotation = analyzer.analyze_text(text)
        output_file = output_path / f"{input_path.stem}_analysis.json"
        analyzer.save_analysis(annotation, output_file)
        
        print(f"✅ Analyzed {input_path.name}")
        print(f"   Words: {len(annotation.word_analyses)}")
        print(f"   Complexity: {annotation.text_complexity:.3f}")
        print(f"   Dialect: {annotation.dialect_classification}")
        print(f"   Quality: {annotation.linguistic_quality_score:.3f}")
    
    elif input_path.is_dir():
        # Process directory of files
        text_files = list(input_path.glob('*.txt'))
        
        for i, file_path in enumerate(text_files):
            print(f"Processing {i+1}/{len(text_files)}: {file_path.name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                annotation = analyzer.analyze_text(text)
                output_file = output_path / f"{file_path.stem}_analysis.json"
                analyzer.save_analysis(annotation, output_file)
                
            except Exception as e:
                print(f"❌ Failed to process {file_path.name}: {e}")
        
        print(f"🎉 Processed {len(text_files)} files")
    
    else:
        print(f"❌ Input path {input_path} not found")


if __name__ == "__main__":
    main()