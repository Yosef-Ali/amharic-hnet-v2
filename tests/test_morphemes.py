#!/usr/bin/env python3
"""
Test suite for Amharic morpheme handling in H-Net model.
Validates morphological segmentation and cultural safety.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from preprocessing.prepare_amharic import AmharicPreprocessor
from safety.cultural_guardrails import AmharicCulturalGuardrails

class TestAmharicMorphemes:
    """Test suite for Amharic morpheme processing."""
    
    def __init__(self):
        self.preprocessor = AmharicPreprocessor()
        self.guardrails = AmharicCulturalGuardrails()
        
    def test_morpheme_segmentation(self):
        """Test morpheme-aware segmentation."""
        test_cases = [
            {
                'input': 'ይቃልጣል',  # he eats
                'expected_morphemes': ['ይ', 'ቃል', 'ጣ', 'ል'],
                'meaning': 'he-eat-3rd-person-singular'
            },
            {
                'input': 'የኢትዮጵያ',  # of Ethiopia
                'expected_morphemes': ['የ', 'ኢትዮጵያ'],
                'meaning': 'possessive-Ethiopia'
            },
            {
                'input': 'ተናገረ',   # he spoke
                'expected_morphemes': ['ተ', 'ናገረ'],
                'meaning': 'past-speak'
            }
        ]
        
        print("🧪 Testing Morpheme Segmentation...")
        for i, test_case in enumerate(test_cases):
            print(f"Test {i+1}: {test_case['input']} -> {test_case['meaning']}")
            
            # Test morpheme segmentation
            segments = self.preprocessor.segment_morphemes(test_case['input'])
            print(f"  Segmented: {segments}")
            print(f"  Expected:  {test_case['expected_morphemes']}")
            
            # Check if segmentation captures key morphemes
            has_key_morphemes = any(morph in segments for morph in test_case['expected_morphemes'])
            print(f"  Result: {'✅ PASS' if has_key_morphemes else '❌ FAIL'}")
            print()
    
    def test_cultural_safety(self):
        """Test cultural safety violations."""
        test_cases = [
            {
                'text': 'ቡና የሰው ልጅ አለመለወጥ ያስገድዳል',  # Coffee is addictive to humans
                'should_fail': True,
                'reason': 'Associates coffee with addiction'
            },
            {
                'text': 'መስቀል እንደ ማንኛውም ምስል',  # Cross like any picture
                'should_fail': True,
                'reason': 'Treats cross as ordinary object'
            },
            {
                'text': 'ቡና የኢትዮጵያ ባህላዊ ማህበራዊ ሥርዓት ነው',  # Coffee is Ethiopian cultural social system
                'should_fail': False,
                'reason': 'Respectful cultural context'
            }
        ]
        
        print("🛡️ Testing Cultural Safety...")
        for i, test_case in enumerate(test_cases):
            print(f"Test {i+1}: {test_case['text'][:30]}...")
            
            is_safe, violations = self.guardrails.check_cultural_safety(test_case['text'])
            
            if test_case['should_fail']:
                result = not is_safe  # Should detect violations
                print(f"  Expected: VIOLATION, Got: {'VIOLATION' if not is_safe else 'SAFE'}")
                if violations:
                    print(f"  Violations: {[v.context for v in violations]}")
            else:
                result = is_safe  # Should be safe
                print(f"  Expected: SAFE, Got: {'SAFE' if is_safe else 'VIOLATION'}")
            
            print(f"  Reason: {test_case['reason']}")
            print(f"  Result: {'✅ PASS' if result else '❌ FAIL'}")
            print()
    
    def test_space_handling(self):
        """Test space-free Amharic processing."""
        test_cases = [
            {
                'spaced': 'አማርኛ የኢትዮጵያ ቋንቋ ነው',  # With spaces
                'space_free': 'አማርኛየኢትዮጵያቋንቋነው',  # No spaces (realistic)
                'description': 'Amharic is Ethiopian language'
            },
            {
                'spaced': 'ቡና ጥሩ ነው',  # With spaces
                'space_free': 'ቡናጥሩነው',  # No spaces
                'description': 'Coffee is good'
            }
        ]
        
        print("🔤 Testing Space Handling...")
        for i, test_case in enumerate(test_cases):
            print(f"Test {i+1}: {test_case['description']}")
            print(f"  Spaced:     {test_case['spaced']}")
            print(f"  Space-free: {test_case['space_free']}")
            
            # Test that both versions can be processed
            spaced_clean = self.preprocessor.clean_text(test_case['spaced'], preserve_spaces=True)
            space_free_clean = self.preprocessor.clean_text(test_case['space_free'], preserve_spaces=False)
            
            print(f"  Processed spaced:     {spaced_clean}")
            print(f"  Processed space-free: {space_free_clean}")
            
            # Both should contain Amharic content
            spaced_ratio = self.preprocessor.get_amharic_ratio(spaced_clean)
            space_free_ratio = self.preprocessor.get_amharic_ratio(space_free_clean)
            
            result = spaced_ratio > 0.5 and space_free_ratio > 0.5
            print(f"  Amharic ratios: Spaced={spaced_ratio:.2f}, Space-free={space_free_ratio:.2f}")
            print(f"  Result: {'✅ PASS' if result else '❌ FAIL'}")
            print()
    
    def test_dialect_coverage(self):
        """Test dialect variation handling."""
        print("🌍 Testing Dialect Coverage...")
        
        # Check if dialect variations are properly defined
        dialects = list(self.guardrails.dialect_variations.keys())
        print(f"Supported dialects: {dialects}")
        
        # Test dialect-specific terms
        for dialect, terms in self.guardrails.dialect_variations.items():
            print(f"  {dialect.title()} dialect:")
            for term, meaning in terms.items():
                print(f"    {term}: {meaning}")
        
        result = len(dialects) >= 3  # Should support at least 3 dialects
        print(f"  Result: {'✅ PASS' if result else '❌ FAIL'}")
        print()
    
    def run_all_tests(self):
        """Run all test suites."""
        print("=" * 60)
        print("🇪🇹 AMHARIC H-NET MORPHEME TEST SUITE")
        print("=" * 60)
        print()
        
        self.test_morpheme_segmentation()
        self.test_cultural_safety()
        self.test_space_handling()
        self.test_dialect_coverage()
        
        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)


if __name__ == "__main__":
    tester = TestAmharicMorphemes()
    tester.run_all_tests()