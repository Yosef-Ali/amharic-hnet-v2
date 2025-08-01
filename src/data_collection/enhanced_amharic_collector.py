#!/usr/bin/env python3
"""
Enhanced Amharic Wikipedia Corpus Collector

This version includes multiple collection strategies to gather 50+ high-quality articles.
"""

import json
import requests
import re
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import random

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

class EnhancedAmharicCollector:
    """
    Enhanced Amharic Wikipedia article collector with multiple collection strategies.
    """
    
    def __init__(self):
        self.base_url = "https://am.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AmharicCorpusCollector/2.0 (Educational Research)'
        })
        
        # Ge'ez script Unicode ranges
        self.amharic_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
            (0xAB00, 0xAB2F),  # Ethiopic Extended-A
        ]
        
        # Expanded cultural keywords for better coverage
        self.cultural_keywords = {
            'ethiopian_places': ['አዲስ አበባ', 'ጎንደር', 'ባህር ዳር', 'መቀሌ', 'ሐዋሳ', 'ዲሬ ዳዋ', 'ጅማ', 'ደሴ', 'አርባ ምንጭ', 'መከላ'],
            'cultural_terms': ['ኢትዮጵያ', 'አማራ', 'ኦሮሞ', 'ትግራይ', 'ሲዳማ', 'ሶማሌ', 'አፋር', 'ጉራጌ', 'ወላይታ', 'ሃዲያ'],
            'historical_terms': ['ንጉሥ', 'ንግሥት', 'ሐይለ ሥላሴ', 'ምንሊክ', 'ዘውዲቱ', 'አክሱም', 'ላሊበላ', 'ሳህለ ስላሴ', 'ቴዎድሮስ'],
            'religious_terms': ['ክርስቶስ', 'እግዚአብሔር', 'ቤተ ክርስቲያን', 'ኦርቶዶክስ', 'እስልምና', 'ሙስሊም', 'መስጊድ', 'ሲናክሳር'],
            'cultural_practices': ['ኢንጀራ', 'ወጥ', 'ቡና', 'ጌሽ', 'ዳቦ', 'ማሳ', 'በዓል', 'ጾም', 'ኪዳን', 'ጋሻ'],
            'educational_terms': ['ዩኒቨርስቲ', 'ኮሌጅ', 'ትምህርት', 'ቤተ ሰላሳ', 'ፋይዳ', 'ክፍል', 'መጽሐፍ', 'መማሪያ'],
            'modern_terms': ['ቴክኖሎጂ', 'ኮምፒውተር', 'ኢንተርኔት', 'ፌስቡክ', 'ቴሌቪዥን', 'ሬዲዮ', 'መኪና', 'አውሮፕላን']
        }
        
        # Quality thresholds
        self.quality_threshold = 0.35
        self.amharic_ratio_threshold = 0.6
        self.min_content_length = 150
        
        self.collected_titles = set()  # Track collected titles to avoid duplicates
        self.debug_stats = {
            'total_processed': 0,
            'too_short': 0,
            'low_amharic_ratio': 0,
            'low_quality': 0,
            'duplicates': 0,
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
    
    def get_search_based_articles(self) -> List[str]:
        """Get articles using targeted search terms."""
        search_terms = [
            # Basic Ethiopia terms
            'ኢትዮጵያ', 'አዲስ አበባ', 'አማርኛ', 
            # Historical figures
            'ሐይለ ሥላሴ', 'ምንሊክ', 'ዘውዲቱ', 'ዮሐንስ', 'ቴዎድሮስ',
            # Places
            'ጎንደር', 'አክሱም', 'ላሊበላ', 'ባህር ዳር', 'መቀሌ',
            # Culture
            'ኢንጀራ', 'ቡና', 'በዓል', 'ጾም', 'ቤተ ክርስቲያን',
            # Modern
            'ዩኒቨርስቲ', 'ትምህርት', 'ሳይንስ', 'ቴክኖሎጂ',
            # Additional terms
            'ወንዝ', 'ተራራ', 'ሀይቅ', 'መንግሥት', 'ህዝብ'
        ]
        
        all_titles = []
        for term in search_terms:
            titles = self.search_articles(term, limit=8)
            all_titles.extend(titles)
            time.sleep(0.8)  # Rate limiting
        
        return list(set(all_titles))  # Remove duplicates
    
    def get_category_based_articles(self) -> List[str]:
        """Get articles from relevant categories."""
        categories = [
            'Category:ኢትዮጵያ',
            'Category:የኢትዮጵያ ታሪክ',
            'Category:የኢትዮጵያ ባህል',
            'Category:የኢትዮጵያ ከተሞች',
            'Category:የኢትዮጵያ ሰዎች',
            'Category:አማርኛ',
            'Category:ኢትዮጵያዊ ምግቦች',
            'Category:ኢትዮጵያዊ ፖለቲከኞች'
        ]
        
        all_titles = []
        for category in categories:
            titles = self.get_category_articles(category, limit=15)
            all_titles.extend(titles)
            time.sleep(1.0)  # Rate limiting
        
        return list(set(all_titles))
    
    def get_random_articles_filtered(self, count: int = 100) -> List[str]:
        """Get random articles and filter for likely quality candidates."""
        params = {
            'action': 'query',
            'format': 'json',
            'generator': 'random',
            'grnnamespace': 0,
            'grnlimit': count,
            'prop': 'info|extracts',
            'exintro': True,
            'explaintext': True,
            'exsentences': 2
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            good_titles = []
            if 'query' in data and 'pages' in data['query']:
                for page in data['query']['pages'].values():
                    title = page.get('title', '')
                    extract = page.get('extract', '')
                    
                    # Filter by basic quality indicators
                    if (len(extract) > 100 and 
                        self.calculate_amharic_ratio(extract) > 0.6 and
                        not title.startswith(('መምህር:', 'ክፍል:', 'ማህተም:', 'ተባባሪ:'))):
                        good_titles.append(title)
            
            return good_titles
        except Exception as e:
            logger.error(f"Error fetching random articles: {e}")
            return []
    
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
    
    def get_category_articles(self, category: str, limit: int = 20) -> List[str]:
        """Get articles from a specific category."""
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'categorymembers',
            'cmtitle': category,
            'cmnamespace': 0,
            'cmlimit': limit
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'categorymembers' in data['query']:
                return [page['title'] for page in data['query']['categorymembers']]
            return []
        except Exception as e:
            logger.error(f"Error fetching category articles: {e}")
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
    
    def assess_quality_enhanced(self, content: str, title: str) -> float:
        """Enhanced quality assessment."""
        if not content:
            return 0.0
        
        score = 0.0
        
        # Amharic ratio (40% of score)
        amharic_ratio = self.calculate_amharic_ratio(content)
        score += amharic_ratio * 0.4
        
        # Content length (20% of score)
        length_score = min(1.0, len(content) / 800)
        score += length_score * 0.2
        
        # Cultural relevance (25% of score)
        cultural_score = 0.0
        cultural_terms_found = 0
        for category_terms in self.cultural_keywords.values():
            for term in category_terms:
                if term in content or term in title:
                    cultural_terms_found += 1
                    cultural_score += 0.1
        
        cultural_score = min(1.0, cultural_score)
        score += cultural_score * 0.25
        
        # Structure indicators (15% of score)
        structure_score = 0.0
        if len(content.split('\n')) > 2:  # Multiple paragraphs
            structure_score += 0.3
        if '።' in content:  # Proper Amharic punctuation
            structure_score += 0.3
        if any(indicator in content for indicator in ['፣', '፤', '፥', '፦', '፧', '፨']):
            structure_score += 0.4
        
        structure_score = min(1.0, structure_score)
        score += structure_score * 0.15
        
        return min(1.0, score)
    
    def validate_cultural_safety_enhanced(self, content: str, title: str) -> Tuple[float, List[str]]:
        """Enhanced cultural safety validation."""
        flags = []
        relevance_score = 0.0
        
        # Check for Ethiopian cultural content
        cultural_indicators = 0
        for category, terms in self.cultural_keywords.items():
            category_found = False
            for term in terms:
                if term in content or term in title:
                    if not category_found:
                        cultural_indicators += 1
                        category_found = True
                    relevance_score += 0.15
        
        # Base relevance for Amharic content
        if cultural_indicators == 0 and self.calculate_amharic_ratio(content) > 0.7:
            relevance_score = 0.5
        
        # Bonus for diverse cultural representation
        if cultural_indicators >= 3:
            relevance_score += 0.2
        
        relevance_score = min(1.0, relevance_score)
        
        # Flag potential issues
        sensitive_terms = ['ወረራ', 'ጥቃት', 'ጦርነት', 'ግጭት']
        if any(term in content for term in sensitive_terms):
            flags.append('contains_conflict_content')
        
        if cultural_indicators < 1:
            flags.append('low_cultural_relevance')
        
        return relevance_score, flags
    
    def collect_articles(self, target_count: int = 50) -> List[Dict[str, Any]]:
        """Collect articles using multiple strategies."""
        logger.info(f"Starting enhanced collection of {target_count} articles...")
        
        # Collect candidate articles from multiple sources
        logger.info("Phase 1: Search-based collection...")
        search_candidates = self.get_search_based_articles()
        logger.info(f"Found {len(search_candidates)} search-based candidates")
        
        logger.info("Phase 2: Category-based collection...")
        category_candidates = self.get_category_based_articles()
        logger.info(f"Found {len(category_candidates)} category-based candidates")
        
        logger.info("Phase 3: Random filtered collection...")
        random_candidates = self.get_random_articles_filtered(150)
        logger.info(f"Found {len(random_candidates)} random candidates")
        
        # Combine and shuffle candidates
        all_candidates = search_candidates + category_candidates + random_candidates
        all_candidates = list(set(all_candidates))  # Remove duplicates
        random.shuffle(all_candidates)
        
        logger.info(f"Total unique candidates: {len(all_candidates)}")
        
        collected = []
        
        for i, title in enumerate(all_candidates):
            if len(collected) >= target_count:
                break
            
            self.debug_stats['total_processed'] += 1
            
            if i % 20 == 0:
                logger.info(f"Processed {i} articles, collected {len(collected)}")
            
            # Skip if already collected
            if title in self.collected_titles:
                self.debug_stats['duplicates'] += 1
                continue
            
            # Fetch article content
            article_data = self.get_article_content(title)
            if not article_data:
                continue
            
            content = article_data['content']
            if len(content) < self.min_content_length:
                self.debug_stats['too_short'] += 1
                continue
            
            # Check Amharic ratio
            amharic_ratio = self.calculate_amharic_ratio(content)
            if amharic_ratio < self.amharic_ratio_threshold:
                self.debug_stats['low_amharic_ratio'] += 1
                continue
            
            # Enhanced quality assessment
            quality_score = self.assess_quality_enhanced(content, title)
            if quality_score < self.quality_threshold:
                self.debug_stats['low_quality'] += 1
                continue
            
            # Cultural safety validation
            cultural_relevance, cultural_flags = self.validate_cultural_safety_enhanced(content, title)
            
            # Create article entry
            revisions = article_data.get('revisions', [])
            categories = article_data.get('categories', [])
            
            # Determine category
            primary_category = 'general'
            if categories:
                cat_name = categories[0].get('title', '').replace('Category:', '')
                if any(term in cat_name for term in ['ታሪክ', 'ሰው']):
                    primary_category = 'history'
                elif 'ባህል' in cat_name:
                    primary_category = 'culture'
                elif 'ሳይንስ' in cat_name:
                    primary_category = 'science'
                elif 'ስፖርት' in cat_name:
                    primary_category = 'sports'
                elif 'ከተማ' in cat_name:
                    primary_category = 'geography'
            
            metadata = ArticleMetadata(
                title=title,
                page_id=article_data['pageid'],
                revision_id=revisions[0].get('revid', 0) if revisions else 0,
                timestamp=revisions[0].get('timestamp', '') if revisions else '',
                category=primary_category,
                quality_score=quality_score,
                cultural_relevance=cultural_relevance,
                amharic_ratio=amharic_ratio,
                word_count=len(content.split()),
                contributors=len(set(rev.get('user', '') for rev in revisions)),
                edit_count=len(revisions),
                cultural_flags=cultural_flags
            )
            
            collected.append({
                'content': content,
                'metadata': asdict(metadata),
                'url': article_data.get('url', ''),
                'categories': [cat.get('title', '') for cat in categories]
            })
            
            self.collected_titles.add(title)
            self.debug_stats['collected'] += 1
            
            if len(collected) % 10 == 0:
                logger.info(f"✅ Collected {len(collected)} articles so far...")
            
            # Rate limiting
            time.sleep(0.6)
        
        logger.info(f"Collection complete! Collected {len(collected)} articles")
        logger.info(f"Final stats: {self.debug_stats}")
        return collected
    
    def generate_quality_report(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not articles:
            return {
                'error': 'No articles to analyze',
                'debug_stats': self.debug_stats
            }
        
        # Statistics
        quality_scores = [article['metadata']['quality_score'] for article in articles]
        amharic_ratios = [article['metadata']['amharic_ratio'] for article in articles]
        cultural_relevance = [article['metadata']['cultural_relevance'] for article in articles]
        word_counts = [article['metadata']['word_count'] for article in articles]
        
        # Category distribution
        categories = {}
        for article in articles:
            cat = article['metadata']['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'collection_summary': {
                'total_articles': len(articles),
                'collection_timestamp': datetime.now().isoformat(),
                'average_quality_score': sum(quality_scores) / len(quality_scores),
                'average_amharic_ratio': sum(amharic_ratios) / len(amharic_ratios),
                'average_cultural_relevance': sum(cultural_relevance) / len(cultural_relevance),
                'average_word_count': sum(word_counts) / len(word_counts),
                'min_quality_score': min(quality_scores),
                'max_quality_score': max(quality_scores)
            },
            'quality_distribution': {
                'high_quality': len([s for s in quality_scores if s >= 0.7]),
                'medium_quality': len([s for s in quality_scores if 0.5 <= s < 0.7]),
                'acceptable_quality': len([s for s in quality_scores if 0.35 <= s < 0.5])
            },
            'amharic_ratio_compliance': {
                'above_60_percent': len([r for r in amharic_ratios if r >= 0.6]),
                'above_70_percent': len([r for r in amharic_ratios if r >= 0.7]),
                'above_90_percent': len([r for r in amharic_ratios if r >= 0.9])
            },
            'category_distribution': categories,
            'debug_stats': self.debug_stats,
            'validation_passed': all(
                article['metadata']['amharic_ratio'] >= 0.6 and
                article['metadata']['quality_score'] >= 0.35
                for article in articles
            )
        }
    
    def save_corpus(self, articles: List[Dict[str, Any]], filepath: str) -> None:
        """Save collected corpus with metadata and quality report."""
        quality_report = self.generate_quality_report(articles)
        
        corpus_data = {
            'corpus_metadata': {
                'version': '2.0-enhanced',
                'collection_date': datetime.now().isoformat(),
                'collector': 'EnhancedAmharicCollector',
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
        
        logger.info(f"Enhanced corpus saved to {filepath}")

def main():
    """Main execution function."""
    collector = EnhancedAmharicCollector()
    
    # Collect articles
    articles = collector.collect_articles(target_count=50)
    
    if articles:
        # Save to specified location
        output_path = "/Users/mekdesyared/amharic-hnet-v2/data/raw/test_corpus.json"
        collector.save_corpus(articles, output_path)
        
        print(f"\n✅ Successfully collected {len(articles)} high-quality Amharic articles")
        print(f"📁 Saved to: {output_path}")
        
        # Print summary statistics
        quality_report = collector.generate_quality_report(articles)
        summary = quality_report['collection_summary']
        
        print(f"\n📊 Collection Summary:")
        print(f"   • Total Articles: {summary['total_articles']}")
        print(f"   • Average Quality Score: {summary['average_quality_score']:.3f}")
        print(f"   • Average Amharic Ratio: {summary['average_amharic_ratio']:.3f}")
        print(f"   • Average Cultural Relevance: {summary['average_cultural_relevance']:.3f}")
        print(f"   • Average Word Count: {summary['average_word_count']:.0f}")
        print(f"   • Quality Range: {summary['min_quality_score']:.3f} - {summary['max_quality_score']:.3f}")
        print(f"   • Validation Passed: {quality_report['validation_passed']}")
        
        print(f"\n📈 Quality Distribution:")
        dist = quality_report['quality_distribution']
        print(f"   • High Quality (≥0.7): {dist['high_quality']} articles")
        print(f"   • Medium Quality (0.5-0.7): {dist['medium_quality']} articles")
        print(f"   • Acceptable Quality (0.35-0.5): {dist['acceptable_quality']} articles")
        
        print(f"\n🇪🇹 Amharic Compliance:")
        compliance = quality_report['amharic_ratio_compliance']
        print(f"   • Above 60%: {compliance['above_60_percent']} articles")
        print(f"   • Above 70%: {compliance['above_70_percent']} articles")
        print(f"   • Above 90%: {compliance['above_90_percent']} articles")
        
        print(f"\n🏷️ Category Distribution:")
        for category, count in quality_report['category_distribution'].items():
            print(f"   • {category}: {count} articles")
        
    else:
        print("❌ No articles met the quality criteria")

if __name__ == "__main__":
    main()