#!/usr/bin/env python3
"""
Test script for Amharic data collection with quality validation and cultural safety checks.
This script tests the agent system by collecting 5 Amharic Wikipedia articles.
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_collection.amharic_collector import AmharicDataCollector
from safety.cultural_guardrails import AmharicCulturalGuardrails


async def test_data_collection():
    """Test data collection workflow with cultural safety validation."""
    print("🔍 Testing Amharic Data Collection Agent System")
    print("=" * 50)
    
    # Initialize collector and cultural safety system
    collector = AmharicDataCollector(output_dir="data/raw", max_concurrent=3)
    cultural_guardrails = AmharicCulturalGuardrails()
    
    print("✅ Initialized data collector and cultural guardrails")
    
    # Collect 5 Wikipedia articles (try multiple batches if needed)
    print("\n📚 Collecting 5 Amharic Wikipedia articles...")
    samples = []
    attempts = 0
    max_attempts = 3
    
    while len(samples) < 5 and attempts < max_attempts:
        print(f"  Attempt {attempts + 1}/{max_attempts}...")
        batch_samples = await collector.collect_from_source('wikipedia', max_articles=10)
        samples.extend(batch_samples)
        attempts += 1
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_samples = []
        for sample in samples:
            if sample.title not in seen_titles:
                unique_samples.append(sample)
                seen_titles.add(sample.title)
        samples = unique_samples[:5]  # Keep only first 5
    
    if not samples:
        print("❌ No articles collected. Check internet connection or Wikipedia availability.")
        return
    
    print(f"✅ Successfully collected {len(samples)} articles")
    
    # Apply cultural safety checks
    print("\n🛡️ Applying cultural safety checks...")
    safe_samples = []
    cultural_violations = []
    
    for i, sample in enumerate(samples):
        print(f"\nChecking article {i+1}: {sample.title[:50]}...")
        
        # Check cultural safety
        is_safe, violations = cultural_guardrails.check_cultural_safety(sample.text)
        
        if is_safe:
            safe_samples.append(sample)
            print(f"  ✅ Culturally safe (Score: {sample.quality_score:.2f})")
        else:
            print(f"  ⚠️ Cultural violations found: {len(violations)}")
            for violation in violations[:2]:  # Show first 2 violations
                print(f"    - {violation.violation_type}: {violation.context}")
            cultural_violations.extend(violations)
            
            # Still include if violations are low severity
            if all(v.severity == 'low' for v in violations):
                safe_samples.append(sample)
                print(f"  ✅ Included (low severity violations only)")
    
    print(f"\n📊 Collection Results:")
    print(f"  Total articles collected: {len(samples)}")
    print(f"  Culturally safe articles: {len(safe_samples)}")
    print(f"  Cultural violations detected: {len(cultural_violations)}")
    
    # Generate detailed statistics
    if safe_samples:
        total_words = sum(s.estimated_words for s in safe_samples)
        avg_quality = sum(s.quality_score for s in safe_samples) / len(safe_samples)
        avg_amharic_ratio = sum(s.amharic_ratio for s in safe_samples) / len(safe_samples)
        
        print(f"\n📈 Quality Metrics:")
        print(f"  Total words: {total_words:,}")
        print(f"  Average quality score: {avg_quality:.3f}")
        print(f"  Average Amharic ratio: {avg_amharic_ratio:.3f}")
        
        # Cultural domain distribution
        domains = {}
        for sample in safe_samples:
            domain = sample.cultural_domain
            domains[domain] = domains.get(domain, 0) + 1
        
        print(f"\n🏛️ Cultural Domain Distribution:")
        for domain, count in domains.items():
            print(f"  {domain}: {count}")
        
        # Dialect coverage
        dialects = {}
        for sample in safe_samples:
            for dialect in sample.dialect_hints:
                dialects[dialect] = dialects.get(dialect, 0) + 1
        
        print(f"\n🗣️ Dialect Coverage:")
        for dialect, count in dialects.items():
            print(f"  {dialect}: {count}")
    
    # Save results
    if safe_samples:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_collection_{timestamp}.json"
        collector.save_samples(safe_samples, filename)
        
        # Create a detailed test report
        test_report = {
            "test_timestamp": datetime.now().isoformat(),
            "collection_stats": {
                "total_collected": len(samples),
                "culturally_safe": len(safe_samples),
                "total_violations": len(cultural_violations),
                "total_words": sum(s.estimated_words for s in safe_samples),
                "avg_quality_score": sum(s.quality_score for s in safe_samples) / len(safe_samples),
                "avg_amharic_ratio": sum(s.amharic_ratio for s in safe_samples) / len(safe_samples)
            },
            "cultural_violations": [
                {
                    "term": v.term,
                    "type": v.violation_type,
                    "severity": v.severity,
                    "context": v.context
                } for v in cultural_violations
            ],
            "domain_distribution": domains,
            "dialect_coverage": dialects,
            "sample_titles": [s.title for s in safe_samples],
            "test_status": "PASSED" if len(safe_samples) >= 3 else "PARTIAL"
        }
        
        report_path = Path("data/raw") / f"test_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"  Data: data/raw/{filename}")
        print(f"  Report: {report_path}")
        
        print(f"\n🎉 Test {'PASSED' if len(safe_samples) >= 3 else 'PARTIAL'}: Agent system successfully collected and validated Amharic text data!")
        
        # Show sample text from first article
        if safe_samples:
            first_sample = safe_samples[0]
            print(f"\n📝 Sample from '{first_sample.title}':")
            sample_text = first_sample.text[:200] + "..." if len(first_sample.text) > 200 else first_sample.text
            print(f"  {sample_text}")
            print(f"  Source: {first_sample.source}")
            print(f"  Quality Score: {first_sample.quality_score:.3f}")
            print(f"  Cultural Domain: {first_sample.cultural_domain}")
    
    else:
        print("\n❌ Test FAILED: No safe samples collected")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_data_collection())
    sys.exit(0 if success else 1)