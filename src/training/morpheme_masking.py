#!/usr/bin/env python3
"""
Morpheme-aware masking strategy for Amharic H-Net training.

This module implements advanced masking techniques that respect Amharic morphological 
boundaries, as identified in linguistic research. Instead of random token masking,
it masks entire morphological units like verb conjugations, noun inflections, and 
compound word components.

Key features:
- Syllabic boundary detection using Ge'ez script patterns
- Verb conjugation masking (e.g., mask entire "ይቃልጣል" → "he eats")  
- Noun inflection masking (e.g., mask "የኢትዮጵያ" → "of Ethiopia")
- Compound word component masking
- Morphological complexity-aware sampling
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass


@dataclass
class MorphologicalUnit:
    """Represents a morphological unit with its boundaries and type."""
    start: int
    end: int
    unit_type: str  # 'verb_conjugation', 'noun_inflection', 'compound', 'root'
    syllables: List[str]
    morphological_complexity: float  # 0.0 to 1.0


class AmharicMorphemeMasker:
    """Advanced morpheme-aware masking for Amharic text."""
    
    def __init__(self, mask_probability: float = 0.15, morpheme_unit_prob: float = 0.8):
        """
        Args:
            mask_probability: Overall probability of masking any unit
            morpheme_unit_prob: Probability of masking entire morphological units vs individual syllables
        """
        self.mask_probability = mask_probability
        self.morpheme_unit_prob = morpheme_unit_prob
        
        # Amharic morphological patterns (enhanced from chat insights)
        self.verb_patterns = {
            'past_tense': [r'ተ[\u1200-\u137f]+', r'[\u1200-\u137f]+አል', r'[\u1200-\u137f]+ኦች'],
            'present_tense': [r'ይ[\u1200-\u137f]+አል', r'ት[\u1200-\u137f]+ለች', r'እ[\u1200-\u137f]+ለን'],
            'future_tense': [r'እ[\u1200-\u137f]+ለሁ', r'ት[\u1200-\u137f]+ለሽ', r'ይ[\u1200-\u137f]+ል'],
            'imperatives': [r'[\u1200-\u137f]+!', r'[\u1200-\u137f]+ህ', r'[\u1200-\u137f]+ሽ']
        }
        
        self.noun_patterns = {
            'possessive': [r'የ[\u1200-\u137f]+', r'[\u1200-\u137f]+ዋ', r'[\u1200-\u137f]+ችን'],
            'plural': [r'[\u1200-\u137f]+ች', r'[\u1200-\u137f]+ኦች', r'[\u1200-\u137f]+ዎች'],
            'definite': [r'[\u1200-\u137f]+ው', r'[\u1200-\u137f]+ዋ', r'[\u1200-\u137f]+ይቱ']
        }
        
        self.compound_patterns = {
            'geographical': [r'[\u1200-\u137f]+አበባ', r'[\u1200-\u137f]+ሀይቅ', r'[\u1200-\u137f]+ወንዝ'],
            'cultural': [r'[\u1200-\u137f]+ቡና', r'[\u1200-\u137f]+በዓል', r'[\u1200-\u137f]+ባህል'],
            'temporal': [r'[\u1200-\u137f]+ወር', r'[\u1200-\u137f]+አመት', r'[\u1200-\u137f]+ቀን']
        }
        
        # Syllabic structure patterns for Ge'ez script
        self.fidel_ranges = {
            'basic': (0x1200, 0x1248),    # Basic Ethiopic
            'supplement': (0x1348, 0x137F), # Ethiopic Supplement  
            'extended': (0x2D80, 0x2DDF)    # Ethiopic Extended
        }
        
    def identify_morphological_units(self, text: str) -> List[MorphologicalUnit]:
        """
        Identify morphological units in Amharic text using pattern matching
        and syllabic analysis.
        
        Args:
            text (str): Input Amharic text
            
        Returns:
            List[MorphologicalUnit]: Detected morphological units
        """
        units = []
        
        # Process verb conjugations
        for verb_type, patterns in self.verb_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    complexity = self._calculate_morphological_complexity(match.group(), verb_type)
                    units.append(MorphologicalUnit(
                        start=match.start(),
                        end=match.end(),
                        unit_type=f'verb_{verb_type}',
                        syllables=self._extract_syllables(match.group()),
                        morphological_complexity=complexity
                    ))
        
        # Process noun inflections
        for noun_type, patterns in self.noun_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    complexity = self._calculate_morphological_complexity(match.group(), noun_type)
                    units.append(MorphologicalUnit(
                        start=match.start(),
                        end=match.end(),
                        unit_type=f'noun_{noun_type}',
                        syllables=self._extract_syllables(match.group()),
                        morphological_complexity=complexity
                    ))
        
        # Process compound words
        for compound_type, patterns in self.compound_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    complexity = self._calculate_morphological_complexity(match.group(), compound_type)
                    units.append(MorphologicalUnit(
                        start=match.start(),
                        end=match.end(),
                        unit_type=f'compound_{compound_type}',
                        syllables=self._extract_syllables(match.group()),
                        morphological_complexity=complexity
                    ))
        
        # Sort by position and remove overlaps
        units = sorted(units, key=lambda x: x.start)
        return self._remove_overlapping_units(units)
    
    def _extract_syllables(self, text: str) -> List[str]:
        """Extract individual Ge'ez syllables (Fidel characters) from text."""
        syllables = []
        for char in text:
            # Check if character is in Ethiopic Unicode ranges
            if any(start <= ord(char) <= end for start, end in self.fidel_ranges.values()):
                syllables.append(char)
        return syllables
    
    def _calculate_morphological_complexity(self, text: str, unit_type: str) -> float:
        """
        Calculate morphological complexity score based on:
        - Number of syllables
        - Presence of inflectional morphemes
        - Compound structure
        """
        syllables = self._extract_syllables(text)
        base_complexity = len(syllables) / 10.0  # Normalize by max expected length
        
        # Type-specific complexity adjustments
        type_multipliers = {
            'past_tense': 1.2,
            'present_tense': 1.0,  
            'future_tense': 1.1,
            'imperatives': 0.8,
            'possessive': 1.1,
            'plural': 0.9,
            'definite': 0.7,
            'geographical': 1.3,
            'cultural': 1.4,  # Higher complexity for cultural terms
            'temporal': 1.0
        }
        
        multiplier = type_multipliers.get(unit_type, 1.0)
        return min(1.0, base_complexity * multiplier)
    
    def _remove_overlapping_units(self, units: List[MorphologicalUnit]) -> List[MorphologicalUnit]:
        """Remove overlapping morphological units, keeping the most complex ones."""
        if not units:
            return units
            
        non_overlapping = [units[0]]
        
        for unit in units[1:]:
            # Check for overlap with last added unit
            last_unit = non_overlapping[-1]
            if unit.start >= last_unit.end:
                # No overlap
                non_overlapping.append(unit)
            elif unit.morphological_complexity > last_unit.morphological_complexity:
                # Current unit is more complex, replace the last one
                non_overlapping[-1] = unit
        
        return non_overlapping
    
    def apply_morpheme_aware_masking(
        self, 
        input_ids: torch.Tensor, 
        text_sequences: List[str],
        mask_token_id: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply morpheme-aware masking to input sequences.
        
        Args:
            input_ids: Tokenized input sequences [batch_size, seq_len]
            text_sequences: Original text for each sequence in batch
            mask_token_id: Token ID to use for masking
            
        Returns:
            Tuple of (masked_input_ids, mask_labels)
        """
        batch_size, seq_len = input_ids.shape
        masked_input = input_ids.clone()
        mask_labels = torch.full_like(input_ids, -100)  # -100 ignored in loss
        
        for batch_idx, text in enumerate(text_sequences):
            # Identify morphological units in this sequence
            morpho_units = self.identify_morphological_units(text)
            
            # Select units to mask based on probability and complexity
            units_to_mask = self._select_units_for_masking(morpho_units)
            
            # Apply masking to selected units
            for unit in units_to_mask:
                # Map text positions to token positions (simplified mapping)
                start_token = int(unit.start * seq_len / len(text))
                end_token = min(seq_len, int(unit.end * seq_len / len(text)) + 1)
                
                # Mask the entire morphological unit
                for pos in range(start_token, end_token):
                    if pos < seq_len:
                        mask_labels[batch_idx, pos] = masked_input[batch_idx, pos]
                        
                        # Apply different masking strategies
                        rand = np.random.random()
                        if rand < 0.8:
                            # 80% replace with mask token
                            masked_input[batch_idx, pos] = mask_token_id
                        elif rand < 0.9:
                            # 10% replace with random token
                            masked_input[batch_idx, pos] = np.random.randint(1, 256)
                        # 10% keep original (already handled)
        
        return masked_input, mask_labels
    
    def _select_units_for_masking(self, units: List[MorphologicalUnit]) -> List[MorphologicalUnit]:
        """
        Select morphological units for masking based on probability and complexity.
        Higher complexity units are more likely to be masked.
        """
        selected = []
        
        for unit in units:
            # Base probability adjusted by morphological complexity
            adjusted_prob = self.mask_probability * (0.5 + 0.5 * unit.morphological_complexity)
            
            if np.random.random() < adjusted_prob:
                selected.append(unit)
        
        return selected
    
    def get_masking_statistics(self, text_sequences: List[str]) -> Dict[str, float]:
        """Get statistics about morphological units in the corpus."""
        stats = {
            'total_units': 0,
            'verb_units': 0,
            'noun_units': 0,
            'compound_units': 0,
            'avg_complexity': 0.0,
            'avg_syllables_per_unit': 0.0
        }
        
        all_complexities = []
        all_syllable_counts = []
        
        for text in text_sequences:
            units = self.identify_morphological_units(text)
            stats['total_units'] += len(units)
            
            for unit in units:
                if 'verb' in unit.unit_type:
                    stats['verb_units'] += 1
                elif 'noun' in unit.unit_type:
                    stats['noun_units'] += 1
                elif 'compound' in unit.unit_type:
                    stats['compound_units'] += 1
                
                all_complexities.append(unit.morphological_complexity)
                all_syllable_counts.append(len(unit.syllables))
        
        if all_complexities:
            stats['avg_complexity'] = np.mean(all_complexities)
            stats['avg_syllables_per_unit'] = np.mean(all_syllable_counts)
        
        return stats


class MorphemeAwareLoss(nn.Module):
    """
    Loss function that weights morphological units more heavily.
    Based on insights that morphological boundaries are critical for Amharic understanding.
    """
    
    def __init__(self, morphological_weight: float = 2.0):
        super().__init__()
        self.morphological_weight = morphological_weight
        self.base_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        morphological_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute morpheme-aware loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            morphological_mask: Binary mask indicating morphological units [batch_size, seq_len]
        """
        batch_size, seq_len, vocab_size = predictions.shape
        
        # Compute base cross-entropy loss
        predictions_flat = predictions.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        loss_per_token = self.base_loss(predictions_flat, targets_flat)
        loss_per_token = loss_per_token.view(batch_size, seq_len)
        
        # Apply morphological weighting if mask is provided
        if morphological_mask is not None:
            # Higher weight for morphological units
            weights = torch.where(
                morphological_mask.bool(),
                torch.tensor(self.morphological_weight, device=predictions.device),
                torch.tensor(1.0, device=predictions.device)
            )
            loss_per_token = loss_per_token * weights
        
        # Return mean loss (ignoring -100 targets)
        valid_mask = targets != -100
        if valid_mask.sum() > 0:
            return loss_per_token[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)


def create_morpheme_aware_dataloader(
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 512,
    mask_probability: float = 0.15
):
    """
    Create a DataLoader with morpheme-aware masking for Amharic H-Net training.
    
    This is a factory function that integrates the morpheme masker with
    PyTorch DataLoader for seamless training integration.
    """
    masker = AmharicMorphemeMasker(mask_probability=mask_probability)
    
    # This would integrate with the existing data loading pipeline
    # Implementation details would depend on the existing data_loader.py structure
    
    return masker  # Placeholder - would return actual DataLoader


if __name__ == "__main__":
    # Example usage and testing
    masker = AmharicMorphemeMasker()
    
    test_texts = [
        "ይቃልጣል የኢትዮጵያ ተናገረ አድሳበባ",  # Complex morphological forms
        "ቡናዋ በገና በዓል ላይ",                     # Cultural and temporal compounds
        "እናመሻሰግላለን ይህንን መጽሐፍ"                # Future tense with complex verb
    ]
    
    for text in test_texts:
        print(f"\\nAnalyzing: {text}")
        units = masker.identify_morphological_units(text)
        for unit in units:
            print(f"  {unit.unit_type}: '{text[unit.start:unit.end]}' "
                  f"(complexity: {unit.morphological_complexity:.2f}, "
                  f"syllables: {len(unit.syllables)})")
    
    # Get corpus statistics
    stats = masker.get_masking_statistics(test_texts)
    print(f"\\nCorpus Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")