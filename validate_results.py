#!/usr/bin/env python3
"""
Validation script for Amharic morphological analysis results
Ensures compliance with >85% accuracy target and cultural safety requirements
"""

import json

def validate_processing_results(processed_file: str):
    """Validate the processed results against requirements"""
    
    print("🔍 Validating Amharic Morphological Analysis Results")
    print("=" * 60)
    
    # Load processed data
    with open(processed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract key metrics
    metadata = data['processing_metadata']
    stats = data['corpus_statistics']
    
    print(f"📊 Processing Metadata:")
    print(f"   Target Accuracy: {metadata['target_accuracy'] * 100}%")
    print(f"   Achieved Accuracy: {metadata['achieved_accuracy'] * 100:.2f}%")
    print(f"   Target Met: {'✅ YES' if metadata['accuracy_target_met'] else '❌ NO'}")
    
    print(f"\n📈 Corpus Statistics:")
    print(f"   Total Articles: {stats['total_articles']}")
    print(f"   Successfully Processed: {stats['processed_articles']}")
    print(f"   Processing Success Rate: {stats['processed_articles']/stats['total_articles']*100:.1f}%")
    print(f"   High Confidence Analyses: {stats['high_confidence_analyses']}")
    print(f"   Average Morphological Confidence: {stats['avg_morphological_confidence']:.3f}")
    
    print(f"\n🛡️  Cultural Safety Assessment:")
    print(f"   Safe Articles: {stats['cultural_safe']} ({stats['cultural_safe']/stats['total_articles']*100:.1f}%)")
    print(f"   Need Review: {stats['cultural_needs_review']} ({stats['cultural_needs_review']/stats['total_articles']*100:.1f}%)")
    print(f"   Problematic: {stats['cultural_problematic']} ({stats['cultural_problematic']/stats['total_articles']*100:.1f}%)")
    print(f"   Cultural Confidence: {stats['avg_cultural_confidence']:.3f}")
    
    # Validation checks
    print(f"\n✅ Validation Results:")
    
    # Check accuracy target
    accuracy_met = metadata['achieved_accuracy'] >= 0.85
    print(f"   Accuracy Target (>85%): {'✅ PASSED' if accuracy_met else '❌ FAILED'}")
    
    # Check processing completeness
    complete_processing = stats['processed_articles'] == stats['total_articles']
    print(f"   Complete Processing: {'✅ PASSED' if complete_processing else '❌ FAILED'}")
    
    # Check cultural safety coverage
    cultural_assessed = (stats['cultural_safe'] + stats['cultural_needs_review'] + stats['cultural_problematic']) == stats['total_articles']
    print(f"   Cultural Safety Coverage: {'✅ PASSED' if cultural_assessed else '❌ FAILED'}")
    
    # Check confidence metrics
    high_confidence = stats['avg_morphological_confidence'] >= 0.9
    print(f"   High Confidence Analysis: {'✅ PASSED' if high_confidence else '⚠️  ACCEPTABLE'}")
    
    # Sample morphological analysis validation
    sample_articles = data['processed_articles'][:3]
    morphological_samples = []
    
    for article in sample_articles:
        if article['morphological_analysis']['word_analyses']:
            for analysis in article['morphological_analysis']['word_analyses'][:2]:
                morphological_samples.append({
                    'word': analysis['word'],
                    'confidence': analysis['overall_confidence'],
                    'segments': len(analysis['segmentation'])
                })
    
    print(f"\n🔬 Sample Morphological Analyses:")
    for i, sample in enumerate(morphological_samples[:5], 1):
        print(f"   {i}. '{sample['word']}' - Confidence: {sample['confidence']:.3f}, Segments: {sample['segments']}")
    
    # Cultural safety sample
    cultural_samples = []
    for article in sample_articles:
        cultural = article['cultural_safety']
        if cultural['issues']:
            cultural_samples.append({
                'safety_level': cultural['safety_level'],
                'issues_count': len(cultural['issues']),
                'recommendations_count': len(cultural['recommendations'])
            })
    
    print(f"\n🛡️  Sample Cultural Safety Assessments:")
    for i, sample in enumerate(cultural_samples[:3], 1):
        print(f"   {i}. Safety Level: {sample['safety_level']}, Issues: {sample['issues_count']}, Recommendations: {sample['recommendations_count']}")
    
    # Overall validation
    all_passed = accuracy_met and complete_processing and cultural_assessed
    
    print(f"\n🎯 Overall Validation:")
    print(f"   Status: {'✅ ALL REQUIREMENTS MET' if all_passed else '⚠️  SOME ISSUES DETECTED'}")
    print(f"   Ready for Production: {'YES' if all_passed and high_confidence else 'NEEDS REVIEW'}")
    
    return {
        'accuracy_met': accuracy_met,
        'complete_processing': complete_processing,
        'cultural_assessed': cultural_assessed,
        'high_confidence': high_confidence,
        'overall_passed': all_passed
    }

if __name__ == "__main__":
    processed_file = "/Users/mekdesyared/amharic-hnet-v2/data/processed/test_processed.json"
    results = validate_processing_results(processed_file)
    
    print(f"\n📋 Validation Summary:")
    for check, status in results.items():
        print(f"   {check}: {'✅' if status else '❌'}")