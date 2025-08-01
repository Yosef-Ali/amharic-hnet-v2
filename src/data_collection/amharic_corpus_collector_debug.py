#!/usr/bin/env python3
"""
Amharic Wikipedia Corpus Collector - Debug Version

This is a more lenient version for debugging and testing the collection process.
"""

import json
import requests
import re
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from urllib.parse import quote
import unicodedata
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ArticleMetadata:
    """Metadata structure for collected articles."""
    title: str
    page_id: int
    revision_id: int
    timestamp: str
    category: str
    quality_score: float
    cultural_relevance: float
    amharic_ratio: float
    word_count: int
    contributors: int
    edit_count: int
    cultural_flags: List[str]

@dataclass
class QualityMetrics:
    """Quality assessment metrics for articles."""
    amharic_character_ratio: float
    text_length: int
    paragraph_count: int
    heading_count: int
    reference_count: int
    cultural_terms_count: int
    structure_score: float
    completeness_score: float

class AmharicCorpusCollectorDebug:
    """
    Debug version of Amharic Wikipedia article collector with relaxed thresholds.
    """
    
    def __init__(self, lenient_mode: bool = True):
        self.base_url = "https://am.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AmharicCorpusCollector/1.0 (Educational Research)'
        })
        
        # Ge'ez script Unicode ranges
        self.amharic_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
            (0xAB00, 0xAB2F),  # Ethiopic Extended-A
        ]
        
        # Cultural validation keywords
        self.cultural_keywords = {
            'ethiopian_places': ['አዲስ አበባ', 'ጎንደር', 'ባህር ዳር', 'መቀሌ', 'ሐዋሳ', 'ዲሬ ዳዋ', 'ጅማ', 'ደሴ'],
            'cultural_terms': ['ኢትዮጵያ', 'አማራ', 'ኦሮሞ', 'ትግራይ', 'ሲዳማ', 'ሶማሌ', 'አፋር', 'ጉራጌ'],
            'historical_terms': ['ንጉሥ', 'ንግሥት', 'ሐይለ ሥላሴ', 'ምንሊክ', 'ዘውዲቱ', 'አክሱም', 'ላሊበላ'],
            'religious_terms': ['ክርስቶስ', 'እግዚአብሔር', 'ቤተ ክርስቲያን', 'ኦርቶዶክስ', 'እስልምና', 'ሙስሊም'],
            'cultural_practices': ['ኢንጀራ', 'ወጥ', 'ቡና', 'ጌሽ', 'ዳቦ', 'ማሳ', 'በዓል', 'ጾም']
        }
        
        self.collected_articles = []
        
        # Adjust thresholds based on mode
        if lenient_mode:
            self.quality_threshold = 0.3  # Much more lenient
            self.amharic_ratio_threshold = 0.5  # Lower Amharic ratio requirement
            self.min_content_length = 200  # Shorter articles acceptable
        else:
            self.quality_threshold = 0.7
            self.amharic_ratio_threshold = 0.7
            self.min_content_length = 500
        
        self.debug_stats = {
            'total_processed': 0,
            'too_short': 0,
            'low_amharic_ratio': 0,
            'low_quality': 0,
            'collected': 0
        }
    
    def is_amharic_character(self, char: str) -> bool:
        """Check if a character belongs to Ge'ez/Amharic script."""
        char_code = ord(char)
        return any(start <= char_code <= end for start, end in self.amharic_ranges)
    
    def calculate_amharic_ratio(self, text: str) -> float:
        """Calculate the ratio of Amharic characters in the text."""
        if not text:
            return 0.0
        
        total_chars = len([c for c in text if c.isalpha()])
        if total_chars == 0:
            return 0.0
        
        amharic_chars = len([c for c in text if self.is_amharic_character(c)])
        return amharic_chars / total_chars
    
    def get_high_quality_articles(self) -> List[str]:
        """Get articles that are likely to be high quality."""
        # Focus on specific high-quality sources
        search_terms = [
            'ኢትዮጵያ',
            'አዲስ አበባ', 
            'ኢትዮጵያዊ ሀይማኖት',
            'የኢትዮጵያ ታሪክ',
            'አማርኛ',
            'ጎንደር',
            'አክሱም',
            'ሐይለ ሥላሴ',
            'ምንሊክ',
            'የኢትዮጵያ ባህል'
        ]
        
        all_titles = []
        
        for term in search_terms[:5]:  # Limit to avoid too many requests
            titles = self.search_articles(term, limit=10)
            all_titles.extend(titles)
            time.sleep(1)  # Rate limiting
        
        # Remove duplicates
        return list(set(all_titles))
    
    def search_articles(self, search_term: str, limit: int = 10) -> List[str]:
        """Search for articles using Wikipedia search API."""
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': search_term,
            'srnamespace': 0,
            'srlimit': limit,
            'srwhat': 'text'
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'search' in data['query']:
                return [result['title'] for result in data['query']['search']]
            return []
        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return []
    
    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch full article content and metadata."""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts|info|revisions|categories',
            'exintro': False,
            'explaintext': True,
            'exsectionformat': 'wiki',
            'inprop': 'url',
            'rvprop': 'timestamp|user|size',
            'rvlimit': 5,
            'cllimit': 50
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'pages' in data['query']:
                page_data = list(data['query']['pages'].values())[0]
                
                if 'missing' in page_data:
                    return None
                
                return {
                    'title': page_data.get('title', ''),
                    'pageid': page_data.get('pageid', 0),
                    'content': page_data.get('extract', ''),
                    'revisions': page_data.get('revisions', []),
                    'categories': page_data.get('categories', []),
                    'url': page_data.get('fullurl', '')
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching article content for '{title}': {e}")
            return None
    
    def assess_quality(self, content: str, metadata: Dict[str, Any]) -> QualityMetrics:
        """Assess the quality of an article based on multiple metrics."""
        if not content:
            return QualityMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Basic text metrics
        amharic_ratio = self.calculate_amharic_ratio(content)
        text_length = len(content)
        paragraphs = content.split('\n\n')
        paragraph_count = len([p for p in paragraphs if len(p.strip()) > 30])
        
        # Count headings
        heading_count = len(re.findall(r'^=+.*=+$', content, re.MULTILINE))
        
        # Count references
        reference_count = len(re.findall(r'\[\d+\]|\[አ\d+\]', content))
        
        # Count cultural terms
        cultural_terms_count = 0
        for category_terms in self.cultural_keywords.values():
            for term in category_terms:
                cultural_terms_count += content.count(term)
        
        # Calculate structure score (0-1) - more lenient
        structure_score = min(1.0, (heading_count * 0.15 + paragraph_count * 0.05))
        
        # Calculate completeness score (0-1) - more lenient
        completeness_score = min(1.0, text_length / 1000)
        
        return QualityMetrics(
            amharic_character_ratio=amharic_ratio,
            text_length=text_length,
            paragraph_count=paragraph_count,
            heading_count=heading_count,
            reference_count=reference_count,
            cultural_terms_count=cultural_terms_count,
            structure_score=structure_score,
            completeness_score=completeness_score
        )
    
    def validate_cultural_safety(self, content: str, title: str) -> Tuple[float, List[str]]:
        """Validate cultural appropriateness and authenticity."""
        flags = []
        relevance_score = 0.0
        
        # Check for Ethiopian cultural content
        ethiopian_terms = 0
        for category, terms in self.cultural_keywords.items():
            for term in terms:
                if term in content:
                    ethiopian_terms += 1
                    relevance_score += 0.15
        
        # Basic relevance if no specific terms found
        if ethiopian_terms == 0 and self.calculate_amharic_ratio(content) > 0.5:
            relevance_score = 0.4  # Base relevance for Amharic content
        
        # Normalize relevance score
        relevance_score = min(1.0, relevance_score)
        
        # Less strict flagging for debug mode
        if ethiopian_terms < 1:
            flags.append('low_cultural_relevance')
        
        return relevance_score, flags
    
    def calculate_overall_quality_score(self, metrics: QualityMetrics, cultural_relevance: float) -> float:
        """Calculate overall quality score - more lenient."""
        weights = {
            'amharic_ratio': 0.4,  # Higher weight on Amharic content
            'structure': 0.15,
            'completeness': 0.2,
            'cultural_relevance': 0.2,
            'references': 0.05
        }
        
        score = (
            metrics.amharic_character_ratio * weights['amharic_ratio'] +
            metrics.structure_score * weights['structure'] +
            metrics.completeness_score * weights['completeness'] +
            cultural_relevance * weights['cultural_relevance'] +
            min(1.0, metrics.reference_count / 5) * weights['references']
        )
        
        return min(1.0, score)
    
    def collect_articles(self, target_count: int = 50) -> List[Dict[str, Any]]:
        """Collect Amharic Wikipedia articles with debug information."""
        logger.info(f"Starting collection of {target_count} articles (debug mode)...")
        
        # Get candidate articles
        candidate_articles = self.get_high_quality_articles()
        logger.info(f"Found {len(candidate_articles)} candidate articles")
        
        collected = []
        
        for i, title in enumerate(candidate_articles):
            if len(collected) >= target_count:
                break
            
            self.debug_stats['total_processed'] += 1
            
            if i % 5 == 0:
                logger.info(f"Processed {i} articles, collected {len(collected)}")
                logger.info(f"Debug stats: {self.debug_stats}")
            
            # Fetch article content
            article_data = self.get_article_content(title)
            if not article_data:
                continue
            
            content = article_data['content']
            if len(content) < self.min_content_length:
                self.debug_stats['too_short'] += 1
                logger.debug(f"Skipping '{title}': too short ({len(content)} chars)")
                continue
            
            # Assess quality
            metrics = self.assess_quality(content, article_data)
            
            # Check Amharic ratio
            if metrics.amharic_character_ratio < self.amharic_ratio_threshold:
                self.debug_stats['low_amharic_ratio'] += 1
                logger.debug(f"Skipping '{title}': low Amharic ratio ({metrics.amharic_character_ratio:.2f})")
                continue
            
            # Validate cultural safety
            cultural_relevance, cultural_flags = self.validate_cultural_safety(content, title)
            
            # Calculate overall quality score
            quality_score = self.calculate_overall_quality_score(metrics, cultural_relevance)
            
            logger.info(f"Article '{title}': quality={quality_score:.3f}, amharic_ratio={metrics.amharic_character_ratio:.3f}")
            
            # Check quality threshold
            if quality_score < self.quality_threshold:
                self.debug_stats['low_quality'] += 1
                logger.debug(f"Skipping '{title}': low quality score ({quality_score:.3f})")
                continue
            
            # Article passed all filters
            self.debug_stats['collected'] += 1
            
            # Create metadata
            revisions = article_data.get('revisions', [])
            categories = article_data.get('categories', [])
            
            # Determine primary category
            primary_category = 'general'
            if categories:
                cat_name = categories[0].get('title', '').replace('Category:', '')
                if 'ታሪክ' in cat_name:
                    primary_category = 'history'
                elif 'ባህል' in cat_name:
                    primary_category = 'culture'
                elif 'ሳይንስ' in cat_name:
                    primary_category = 'science'
                elif 'ስፖርት' in cat_name:
                    primary_category = 'sports'
            
            metadata = ArticleMetadata(
                title=title,
                page_id=article_data['pageid'],
                revision_id=revisions[0].get('revid', 0) if revisions else 0,
                timestamp=revisions[0].get('timestamp', '') if revisions else '',
                category=primary_category,
                quality_score=quality_score,
                cultural_relevance=cultural_relevance,
                amharic_ratio=metrics.amharic_character_ratio,
                word_count=len(content.split()),
                contributors=len(set(rev.get('user', '') for rev in revisions)),
                edit_count=len(revisions),
                cultural_flags=cultural_flags
            )
            
            collected.append({
                'content': content,
                'metadata': asdict(metadata),
                'quality_metrics': asdict(metrics),
                'url': article_data.get('url', ''),
                'categories': [cat.get('title', '') for cat in categories]
            })
            
            logger.info(f"✅ Collected: '{title}' (quality: {quality_score:.3f})")
            
            # Rate limiting
            time.sleep(0.5)
        
        logger.info(f"Collection complete. Final stats: {self.debug_stats}")
        return collected
    
    def generate_quality_report(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not articles:
            return {
                'error': 'No articles to analyze',
                'debug_stats': self.debug_stats
            }
        
        # Calculate aggregate statistics
        quality_scores = [article['metadata']['quality_score'] for article in articles]
        amharic_ratios = [article['metadata']['amharic_ratio'] for article in articles]
        cultural_relevance = [article['metadata']['cultural_relevance'] for article in articles]
        word_counts = [article['metadata']['word_count'] for article in articles]
        
        # Category distribution
        categories = {}
        for article in articles:
            cat = article['metadata']['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        # Cultural flags analysis
        all_flags = []
        for article in articles:
            all_flags.extend(article['metadata']['cultural_flags'])
        
        flag_counts = {}
        for flag in all_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        return {
            'collection_summary': {
                'total_articles': len(articles),
                'collection_timestamp': datetime.now().isoformat(),
                'average_quality_score': sum(quality_scores) / len(quality_scores),
                'average_amharic_ratio': sum(amharic_ratios) / len(amharic_ratios),
                'average_cultural_relevance': sum(cultural_relevance) / len(cultural_relevance),
                'average_word_count': sum(word_counts) / len(word_counts)
            },
            'quality_distribution': {
                'high_quality': len([s for s in quality_scores if s >= 0.6]),
                'medium_quality': len([s for s in quality_scores if 0.4 <= s < 0.6]),
                'acceptable_quality': len([s for s in quality_scores if 0.3 <= s < 0.4])
            },
            'amharic_ratio_compliance': {
                'above_50_percent': len([r for r in amharic_ratios if r >= 0.5]),
                'above_70_percent': len([r for r in amharic_ratios if r >= 0.7]),
                'above_90_percent': len([r for r in amharic_ratios if r >= 0.9])
            },
            'category_distribution': categories,
            'cultural_safety_flags': flag_counts,
            'debug_stats': self.debug_stats,
            'validation_passed': all(
                article['metadata']['amharic_ratio'] >= 0.5 and
                article['metadata']['quality_score'] >= 0.3
                for article in articles
            )
        }
    
    def save_corpus(self, articles: List[Dict[str, Any]], filepath: str) -> None:
        """Save collected corpus with metadata and quality report."""
        quality_report = self.generate_quality_report(articles)
        
        corpus_data = {
            'corpus_metadata': {
                'version': '1.0-debug',
                'collection_date': datetime.now().isoformat(),
                'collector': 'AmharicCorpusCollectorDebug',
                'total_articles': len(articles),
                'quality_threshold': self.quality_threshold,
                'amharic_ratio_threshold': self.amharic_ratio_threshold,
                'min_content_length': self.min_content_length
            },
            'quality_report': quality_report,
            'articles': articles
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Corpus saved to {filepath}")

def main():
    """Main execution function."""
    collector = AmharicCorpusCollectorDebug(lenient_mode=True)
    
    # Collect articles
    articles = collector.collect_articles(target_count=50)
    
    if articles:
        # Save to specified location
        output_path = "/Users/mekdesyared/amharic-hnet-v2/data/raw/test_corpus.json"
        collector.save_corpus(articles, output_path)
        
        print(f"\n✅ Successfully collected {len(articles)} Amharic articles")
        print(f"📁 Saved to: {output_path}")
        
        # Print summary statistics
        quality_report = collector.generate_quality_report(articles)
        summary = quality_report['collection_summary']
        
        print(f"\n📊 Collection Summary:")
        print(f"   • Average Quality Score: {summary['average_quality_score']:.3f}")
        print(f"   • Average Amharic Ratio: {summary['average_amharic_ratio']:.3f}")
        print(f"   • Average Cultural Relevance: {summary['average_cultural_relevance']:.3f}")
        print(f"   • Average Word Count: {summary['average_word_count']:.0f}")
        print(f"   • Validation Passed: {quality_report['validation_passed']}")
        
        print(f"\n🔍 Debug Statistics:")
        debug_stats = quality_report['debug_stats']
        for key, value in debug_stats.items():
            print(f"   • {key}: {value}")
    else:
        print("❌ No articles met the quality criteria")
        print(f"Debug stats: {collector.debug_stats}")

if __name__ == "__main__":
    main()