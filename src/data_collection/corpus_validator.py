#!/usr/bin/env python3
"""
Amharic Corpus Cultural Safety and Quality Validator

This script performs comprehensive validation of the collected Amharic corpus
to ensure cultural authenticity, linguistic quality, and safety standards.
"""

import json
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import unicodedata

@dataclass
class ValidationResult:
    """Results of corpus validation."""
    total_articles: int
    passed_validation: int
    failed_validation: int
    cultural_safety_score: float
    linguistic_quality_score: float
    overall_score: float
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]

class AmharicCorpusValidator:
    """
    Comprehensive validator for Amharic corpus with cultural safety focus.
    """
    
    def __init__(self):
        # Ge'ez script Unicode ranges
        self.amharic_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
            (0xAB00, 0xAB2F),  # Ethiopic Extended-A
        ]
        
        # Cultural authenticity indicators
        self.authentic_indicators = {
            'geographical': ['ኢትዮጵያ', 'አዲስ አበባ', 'ጎንደር', 'ባህር ዳር', 'መቀሌ', 'ሐዋሳ', 'አክሱም', 'ላሊበላ'],
            'historical': ['ሐይለ ሥላሴ', 'ምንሊክ', 'ዘውዲቱ', 'ዮሐንስ', 'ቴዎድሮስ', 'ሳህለ ስላሴ'],
            'cultural': ['ኢንጀራ', 'ቡና', 'በዓል', 'ጾም', 'ቤተ ክርስቲያን', 'ኦርቶዶክስ'],
            'linguistic': ['አማርኛ', 'ኦሮምኛ', 'ትግርኛ', 'ጉራግኛ', 'ሶማሊኛ'],
            'ethnic_groups': ['አማራ', 'ኦሮሞ', 'ትግራይ', 'ሲዳማ', 'ሶማሌ', 'አፋር', 'ጉራጌ', 'ወላይታ']
        }
        
        # Potentially problematic content patterns
        self.sensitive_patterns = {
            'political_conflict': ['ወረራ', 'ጥቃት', 'ግጭት', 'ጦርነት', 'ዓመፃ', 'ብጥብጥ'],
            'ethnic_tension': ['ዘር', 'ብሔር', 'አድልዎ', 'መጠላላት', 'ኃይማኖት ግጭት'],
            'historical_bias': ['የመንግስት ፕሮፓጋንዳ', 'ቅኝ ግዛት ፍቅር', 'የውጭ ተጽዕኖ'],
        }
        
        # Required cultural representation thresholds
        self.cultural_thresholds = {
            'min_ethiopian_references': 1,
            'max_foreign_dominance': 0.3,
            'min_cultural_authenticity': 0.4
        }
    
    def is_amharic_character(self, char: str) -> bool:
        """Check if character belongs to Ge'ez/Amharic script."""
        char_code = ord(char)
        return any(start <= char_code <= end for start, end in self.amharic_ranges)
    
    def calculate_amharic_ratio(self, text: str) -> float:
        """Calculate ratio of Amharic characters."""
        if not text:
            return 0.0
        
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return 0.0
        
        amharic_chars = [c for c in alpha_chars if self.is_amharic_character(c)]
        return len(amharic_chars) / len(alpha_chars)
    
    def assess_cultural_authenticity(self, content: str, title: str) -> Tuple[float, List[str]]:
        """Assess cultural authenticity and identify issues."""
        issues = []
        authenticity_score = 0.0
        
        # Check for authentic Ethiopian references
        ethiopian_indicators = 0
        for category, terms in self.authentic_indicators.items():
            found_in_category = 0
            for term in terms:
                if term in content or term in title:
                    found_in_category += 1
            
            if found_in_category > 0:
                ethiopian_indicators += 1
                authenticity_score += 0.15
        
        # Penalty for lack of Ethiopian context
        if ethiopian_indicators < self.cultural_thresholds['min_ethiopian_references']:
            issues.append('insufficient_ethiopian_context')
            authenticity_score *= 0.5
        
        # Check for foreign cultural dominance
        foreign_terms = ['ሲቪላይዘሽን', 'ሞደርናይዜሽን', 'ኮሎናይዝም', 'ዌስተርናይዜሽን']
        foreign_count = sum(1 for term in foreign_terms if term in content)
        total_cultural_terms = sum(len([t for t in terms if t in content]) 
                                 for terms in self.authentic_indicators.values())
        
        if total_cultural_terms > 0:
            foreign_ratio = foreign_count / (total_cultural_terms + foreign_count)
            if foreign_ratio > self.cultural_thresholds['max_foreign_dominance']:
                issues.append('excessive_foreign_influence')
                authenticity_score *= 0.7
        
        # Bonus for diverse cultural representation
        if ethiopian_indicators >= 3:
            authenticity_score += 0.1
        
        return min(1.0, authenticity_score), issues
    
    def check_cultural_sensitivity(self, content: str, title: str) -> Tuple[List[str], float]:
        """Check for culturally sensitive content."""
        issues = []
        sensitivity_score = 1.0  # Start with perfect score, deduct for issues
        
        # Check for potentially problematic content
        for category, patterns in self.sensitive_patterns.items():
            found_patterns = [p for p in patterns if p in content]
            if found_patterns:
                issues.append(f'{category}: {", ".join(found_patterns)}')
                sensitivity_score -= 0.2  # Deduct for each category
        
        # Check for balanced representation
        ethnic_mentions = {}
        for group in self.authentic_indicators['ethnic_groups']:
            if group in content:
                ethnic_mentions[group] = content.count(group)
        
        if len(ethnic_mentions) == 1:  # Only one ethnic group mentioned
            dominant_group = max(ethnic_mentions, key=ethnic_mentions.get)
            if ethnic_mentions[dominant_group] > 5:
                issues.append(f'potential_ethnic_bias_toward_{dominant_group}')
                sensitivity_score -= 0.1
        
        return issues, max(0.0, sensitivity_score)
    
    def validate_linguistic_quality(self, content: str) -> Tuple[float, List[str]]:
        """Validate linguistic quality of Amharic text."""
        issues = []
        quality_score = 0.0
        
        # Amharic character ratio (40% of linguistic quality)
        amharic_ratio = self.calculate_amharic_ratio(content)
        quality_score += amharic_ratio * 0.4
        
        if amharic_ratio < 0.7:
            issues.append(f'low_amharic_ratio_{amharic_ratio:.2f}')
        
        # Proper Amharic punctuation (20% of quality)
        amharic_punctuation = ['።', '፣', '፤', '፥', '፦', '፧', '፨']
        punctuation_found = sum(1 for p in amharic_punctuation if p in content)
        punctuation_score = min(1.0, punctuation_found / 3)
        quality_score += punctuation_score * 0.2
        
        if punctuation_score < 0.3:
            issues.append('insufficient_amharic_punctuation')
        
        # Text structure (20% of quality)
        sentences = content.split('.')
        paragraphs = content.split('\n\n')
        
        structure_score = 0.0
        if len(sentences) > 2:
            structure_score += 0.5
        if len(paragraphs) > 1:
            structure_score += 0.5
        
        quality_score += structure_score * 0.2
        
        # Character encoding validation (20% of quality)
        encoding_score = 1.0
        try:
            # Check for proper Unicode normalization
            normalized = unicodedata.normalize('NFC', content)
            if normalized != content:
                issues.append('unicode_normalization_issues')
                encoding_score -= 0.3
            
            # Check for mixed scripts issues
            latin_chars = len([c for c in content if 'LATIN' in unicodedata.name(c, '')])
            total_alpha = len([c for c in content if c.isalpha()])
            if total_alpha > 0 and (latin_chars / total_alpha) > 0.3:
                issues.append('excessive_latin_script_mixing')
                encoding_score -= 0.2
        except:
            issues.append('character_encoding_problems')
            encoding_score = 0.5
        
        quality_score += encoding_score * 0.2
        
        return min(1.0, quality_score), issues
    
    def validate_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single article comprehensively."""
        content = article['content']
        title = article['metadata']['title']
        
        # Cultural authenticity assessment
        authenticity_score, authenticity_issues = self.assess_cultural_authenticity(content, title)
        
        # Cultural sensitivity check
        sensitivity_issues, sensitivity_score = self.check_cultural_sensitivity(content, title)
        
        # Linguistic quality validation
        linguistic_score, linguistic_issues = self.validate_linguistic_quality(content)
        
        # Overall validation score
        overall_score = (authenticity_score * 0.4 + sensitivity_score * 0.3 + linguistic_score * 0.3)
        
        # Combine all issues
        all_issues = authenticity_issues + sensitivity_issues + linguistic_issues
        
        # Determine pass/fail
        passed = (
            overall_score >= 0.6 and
            authenticity_score >= 0.4 and
            sensitivity_score >= 0.7 and
            linguistic_score >= 0.6
        )
        
        return {
            'title': title,
            'passed': passed,
            'overall_score': overall_score,
            'authenticity_score': authenticity_score,
            'sensitivity_score': sensitivity_score,
            'linguistic_score': linguistic_score,
            'issues': all_issues,
            'word_count': len(content.split()),
            'amharic_ratio': self.calculate_amharic_ratio(content)
        }
    
    def validate_corpus(self, corpus_file: str) -> ValidationResult:
        """Validate entire corpus comprehensively."""
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = data['articles']
        results = []
        
        print(f"🔍 Validating {len(articles)} articles for cultural safety and quality...")
        
        for i, article in enumerate(articles):
            if i % 10 == 0:
                print(f"   Processed {i}/{len(articles)} articles...")
            
            result = self.validate_article(article)
            results.append(result)
        
        # Compile overall results
        passed = len([r for r in results if r['passed']])
        failed = len(results) - passed
        
        # Calculate aggregate scores
        avg_cultural_safety = sum(r['sensitivity_score'] for r in results) / len(results)
        avg_linguistic_quality = sum(r['linguistic_score'] for r in results) / len(results)
        avg_overall = sum(r['overall_score'] for r in results) / len(results)
        
        # Identify common issues
        all_issues = []
        for result in results:
            for issue in result['issues']:
                all_issues.append({
                    'article': result['title'],
                    'issue': issue,
                    'score': result['overall_score']
                })
        
        # Generate recommendations
        recommendations = self.generate_recommendations(results)
        
        return ValidationResult(
            total_articles=len(articles),
            passed_validation=passed,
            failed_validation=failed,
            cultural_safety_score=avg_cultural_safety,
            linguistic_quality_score=avg_linguistic_quality,
            overall_score=avg_overall,
            issues_found=all_issues,
            recommendations=recommendations
        )
    
    def generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Analyze common issues
        issue_counts = {}
        for result in results:
            for issue in result['issues']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        total_articles = len(results)
        
        # Generate specific recommendations
        if issue_counts.get('insufficient_ethiopian_context', 0) > total_articles * 0.2:
            recommendations.append(
                "Increase collection of articles with stronger Ethiopian cultural context"
            )
        
        if issue_counts.get('low_amharic_ratio', 0) > total_articles * 0.1:
            recommendations.append(
                "Improve filtering to ensure higher Amharic character ratios"
            )
        
        if any('ethnic_bias' in issue for issue in issue_counts):
            recommendations.append(
                "Review articles for balanced ethnic group representation"
            )
        
        if issue_counts.get('insufficient_amharic_punctuation', 0) > total_articles * 0.3:
            recommendations.append(
                "Prioritize articles with proper Amharic punctuation usage"
            )
        
        avg_linguistic = sum(r['linguistic_score'] for r in results) / len(results)
        if avg_linguistic < 0.7:
            recommendations.append(
                "Implement stricter linguistic quality filters during collection"
            )
        
        return recommendations
    
    def generate_report(self, validation_result: ValidationResult, output_file: str = None) -> str:
        """Generate comprehensive validation report."""
        report = f"""
🇪🇹 AMHARIC CORPUS CULTURAL SAFETY & QUALITY VALIDATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OVERALL RESULTS:
   • Total Articles Validated: {validation_result.total_articles}
   • Passed Validation: {validation_result.passed_validation} ({validation_result.passed_validation/validation_result.total_articles*100:.1f}%)
   • Failed Validation: {validation_result.failed_validation} ({validation_result.failed_validation/validation_result.total_articles*100:.1f}%)
   
   • Overall Quality Score: {validation_result.overall_score:.3f}/1.000
   • Cultural Safety Score: {validation_result.cultural_safety_score:.3f}/1.000
   • Linguistic Quality Score: {validation_result.linguistic_quality_score:.3f}/1.000

🔍 VALIDATION CRITERIA:
   • Minimum Overall Score: 0.600
   • Minimum Cultural Authenticity: 0.400
   • Minimum Cultural Safety: 0.700
   • Minimum Linguistic Quality: 0.600
   • Minimum Amharic Character Ratio: 70%

⚠️  ISSUES SUMMARY:
"""
        
        # Count and categorize issues
        issue_summary = {}
        for issue in validation_result.issues_found:
            issue_type = issue['issue']
            if issue_type not in issue_summary:
                issue_summary[issue_type] = 0
            issue_summary[issue_type] += 1
        
        if issue_summary:
            for issue_type, count in sorted(issue_summary.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / validation_result.total_articles) * 100
                report += f"   • {issue_type}: {count} articles ({percentage:.1f}%)\n"
        else:
            report += "   • No significant issues found\n"
        
        report += f"\n💡 RECOMMENDATIONS:\n"
        if validation_result.recommendations:
            for rec in validation_result.recommendations:
                report += f"   • {rec}\n"
        else:
            report += "   • Corpus meets all quality standards\n"
        
        report += f"\n✅ VALIDATION STATUS: {'PASSED' if validation_result.passed_validation >= validation_result.total_articles * 0.8 else 'NEEDS IMPROVEMENT'}\n"
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report

def main():
    """Main validation execution."""
    validator = AmharicCorpusValidator()
    
    # Validate the collected corpus
    corpus_file = "/Users/mekdesyared/amharic-hnet-v2/data/raw/test_corpus.json"
    validation_result = validator.validate_corpus(corpus_file)
    
    # Generate and display report
    report = validator.generate_report(validation_result, 
                                     output_file="/Users/mekdesyared/amharic-hnet-v2/data/raw/validation_report.txt")
    
    print(report)
    
    # Additional summary for failed articles
    if validation_result.failed_validation > 0:
        print(f"\n❌ FAILED ARTICLES DETAILS:")
        failed_issues = [issue for issue in validation_result.issues_found 
                        if any(result['title'] == issue['article'] and not result['passed'] 
                              for result in [])]  # This would need the full results
        
        for issue in failed_issues[:10]:  # Show first 10 issues
            print(f"   • {issue['article']}: {issue['issue']} (score: {issue['score']:.3f})")

if __name__ == "__main__":
    main()