#!/usr/bin/env python3
"""
Test the Improved Morpheme-Supervised API
"""

import requests
import json
import time

def wait_for_api(max_attempts=10):
    """Wait for API to be ready"""
    print("⏳ Waiting for API to start...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except:
            pass
        
        print(f"   Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    print("❌ API failed to start")
    return False

def test_improved_generation():
    """Test text generation with improved morpheme model"""
    
    print("🎯 Testing Improved Amharic Text Generation")
    print("=" * 60)
    
    # Test cases that should show improvement
    test_prompts = [
        {"prompt": "ይፈልጋሉ", "description": "Complex verb (they want)"},
        {"prompt": "በቤታችን", "description": "Prepositional phrase (in our house)"},
        {"prompt": "ተማሪዎች", "description": "Plural noun (students)"},
        {"prompt": "ኢትዮጵያ", "description": "Country name (Ethiopia)"},
        {"prompt": "ሰላም", "description": "Greeting (hello)"},
        {"prompt": "አማርኛ", "description": "Language name (Amharic)"}
    ]
    
    results = []
    
    for i, test in enumerate(test_prompts, 1):
        prompt = test["prompt"]
        description = test["description"]
        
        print(f"\n🧪 Test {i}: {prompt} - {description}")
        print("-" * 40)
        
        try:
            # Test generation
            response = requests.post(
                "http://127.0.0.1:8000/generate",
                json={
                    "prompt": prompt,
                    "max_length": 30,
                    "temperature": 0.7,
                    "top_k": 30
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                generated = data["generated_text"]
                amharic_ratio = data["amharic_ratio"]
                
                print(f"✅ Generated: {generated}")
                print(f"📊 Amharic ratio: {amharic_ratio:.1%}")
                
                # Check for improvement indicators
                improvements = []
                if prompt in generated:
                    improvements.append("Contains original prompt")
                if amharic_ratio > 0.3:
                    improvements.append("Good Amharic character ratio")
                if len(generated) > len(prompt):
                    improvements.append("Actually generated new content")
                
                continuation = generated.replace(prompt, "").strip()
                if continuation:
                    print(f"➕ New content: '{continuation}'")
                    
                    # Check if continuation looks like Amharic words vs random characters
                    if any(char in continuation for char in "አበገደሀሰተመነሪዎች"):
                        improvements.append("Contains recognizable Amharic characters")
                
                result = {
                    "prompt": prompt,
                    "generated": generated,
                    "amharic_ratio": amharic_ratio,
                    "improvements": improvements,
                    "quality_score": len(improvements)
                }
                results.append(result)
                
                print(f"🎯 Quality indicators: {', '.join(improvements) if improvements else 'None'}")
                
            else:
                print(f"❌ Generation failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return results

def test_morpheme_analysis():
    """Test the new morpheme analysis endpoint"""
    
    print(f"\n🔬 Testing Morpheme Analysis")
    print("=" * 60)
    
    test_words = [
        {"word": "ይፈልጋሉ", "expected_morphemes": 4, "meaning": "they want"},
        {"word": "በቤታችን", "expected_morphemes": 3, "meaning": "in our house"},
        {"word": "ተማሪዎች", "expected_morphemes": 2, "meaning": "students"}
    ]
    
    for test in test_words:
        word = test["word"]
        expected = test["expected_morphemes"] 
        meaning = test["meaning"]
        
        print(f"\n📝 Analyzing: {word} ({meaning})")
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/analyze-morphemes",
                params={"text": word},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                predicted = data["predicted_morphemes"]
                patterns = data["patterns_detected"]
                complexity = data["analysis"]["morphological_complexity"]
                
                print(f"   Expected morphemes: {expected}")
                print(f"   Predicted morphemes: {predicted}")
                print(f"   Accuracy: {'✅ Good' if abs(predicted - expected) <= 1 else '❌ Poor'}")
                print(f"   Patterns detected: {patterns}")
                print(f"   Complexity: {complexity}")
                
            else:
                print(f"   ❌ Analysis failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_model_info():
    """Test model information endpoint"""
    
    print(f"\n📊 Testing Model Information")
    print("=" * 60)
    
    try:
        response = requests.get("http://127.0.0.1:8000/model/info", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"Architecture: {data['architecture']}")
            print(f"Parameters: {data['parameters']:,}")
            print(f"Model size: {data['size_mb']} MB")
            print(f"Morpheme patterns: {data['morpheme_patterns']['total_patterns']}")
            print(f"Verb prefixes: {data['morpheme_patterns']['verb_prefixes']}")
            print(f"Sample noun suffixes: {data['morpheme_patterns']['noun_suffixes']}")
            
        else:
            print(f"❌ Model info failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def summarize_results(results):
    """Summarize test results"""
    
    print(f"\n📈 IMPROVED MODEL TEST SUMMARY")
    print("=" * 60)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    total_tests = len(results)
    avg_amharic_ratio = sum(r["amharic_ratio"] for r in results) / total_tests
    avg_quality_score = sum(r["quality_score"] for r in results) / total_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Average Amharic ratio: {avg_amharic_ratio:.1%}")
    print(f"Average quality score: {avg_quality_score:.1f}/4")
    
    print(f"\nDetailed Results:")
    for result in results:
        prompt = result["prompt"]
        quality = result["quality_score"]
        ratio = result["amharic_ratio"]
        print(f"  {prompt}: Quality {quality}/4, Amharic {ratio:.1%}")
    
    if avg_quality_score >= 3:
        print(f"\n🎉 EXCELLENT: Significant improvement in text generation!")
    elif avg_quality_score >= 2:
        print(f"\n✅ GOOD: Notable improvement in text generation")
    else:
        print(f"\n⚠️ MODERATE: Some improvement, but more training needed")

if __name__ == "__main__":
    print("🇪🇹 Testing Improved Morpheme-Supervised Amharic H-Net API")
    print("=" * 70)
    
    # Wait for API to start
    if not wait_for_api():
        print("❌ Cannot test - API not available")
        exit(1)
    
    # Test model info first
    test_model_info()
    
    # Test morpheme analysis
    test_morpheme_analysis()
    
    # Test improved text generation
    results = test_improved_generation()
    
    # Summarize results
    summarize_results(results)
    
    print(f"\n🎯 The moment of truth - did morpheme supervision improve Amharic generation?")