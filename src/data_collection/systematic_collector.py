#!/usr/bin/env python3
"""
Systematic Amharic Wikipedia Corpus Collector

This final version systematically collects 50+ high-quality articles
using multiple strategies and progressive relaxation of thresholds.
"""

import json
import requests
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

class SystematicCollector:
    """
    Systematic Amharic Wikipedia article collector using progressive strategies.
    """
    
    def __init__(self):
        self.base_url = "https://am.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AmharicCorpusCollector/4.0 (Systematic Collection)'
        })
        
        # Amharic script ranges
        self.amharic_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
            (0xAB00, 0xAB2F),  # Ethiopic Extended-A
        ]
        
        # Multi-tier cultural keywords
        self.cultural_keywords = {
            'tier1_essential': {
                'terms': ['ኢትዮጵያ', 'አማርኛ', 'አዲስ አበባ', 'የኢትዮጵያ'],
                'weight': 3.0
            },
            'tier2_important': {
                'terms': ['ሐይለ ሥላሴ', 'ምንሊክ', 'ዘውዲቱ', 'አክሱም', 'ላሊበላ', 'ጎንደር'],
                'weight': 2.5
            },
            'tier3_cultural': {
                'terms': ['ኢንጀራ', 'ቡና', 'በዓል', 'ጾም', 'ቤተ ክርስቲያን', 'ኦርቶዶክስ'],
                'weight': 2.0
            },
            'tier4_general': {
                'terms': ['ባህል', 'ታሪክ', 'ሙዚቃ', 'ስፖርት', 'ትምህርት', 'ሳይንስ'],
                'weight': 1.5
            }
        }
        
        # Progressive quality strategies
        self.quality_strategies = [
            {
                'name': 'high_quality',
                'cultural_threshold': 0.8,
                'quality_threshold': 0.7,
                'amharic_threshold': 0.8,
                'min_length': 300
            },
            {
                'name': 'good_quality',
                'cultural_threshold': 0.6,
                'quality_threshold': 0.6,
                'amharic_threshold': 0.75,
                'min_length': 200
            },
            {
                'name': 'acceptable_quality',
                'cultural_threshold': 0.5,
                'quality_threshold': 0.5,
                'amharic_threshold': 0.7,
                'min_length': 150
            },
            {
                'name': 'minimal_quality',
                'cultural_threshold': 0.3,
                'quality_threshold': 0.4,
                'amharic_threshold': 0.65,
                'min_length': 120
            }
        ]
        
        # Avoid sensitive content
        self.avoid_terms = [
            'ወረራ', 'ጥቃት', 'ዓመፃ', 'ብጥብጥ', 'መጠላላት', 
            'ጦርነት', 'ግጭት', 'ወታደራዊ'
        ]
        
        self.collected_titles = set()
        self.quality_stats = {strategy['name']: 0 for strategy in self.quality_strategies}
        self.debug_stats = {
            'total_processed': 0,
            'too_short': 0,
            'low_amharic_ratio': 0,
            'low_cultural_relevance': 0,
            'contains_sensitive_content': 0,
            'low_quality': 0,
            'duplicates': 0,
            'collected': 0
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
    
    def get_comprehensive_search_terms(self) -> List[str]:
        """Get comprehensive list of search terms."""
        search_terms = [
            # Core Ethiopian terms
            'ኢትዮጵያ', 'የኢትዮጵያ', 'ኢትዮጵያዊ',
            'አማርኛ', 'የአማርኛ', 'አማርኛ ቋንቋ',
            
            # Historical figures
            'ሐይለ ሥላሴ', 'ምንሊክ', 'ዘውዲቱ', 'ዮሐንስ', 'ቴዎድሮስ',
            'ሣህለ ስላሴ', 'ዳዊት', 'ምንሊክ ሁለተኛ',
            
            # Geography
            'አዲስ አበባ', 'ጎንደር', 'አክሱም', 'ላሊበላ', 'ባህር ዳር',
            'መቀሌ', 'ሐዋሳ', 'ዲሬ ዳዋ', 'ጅማ', 'ሐረር',
            
            # Culture & Religion
            'ኢትዮጵያ ኦርቶዶክስ', 'ቤተ ክርስቲያን', 'መስጊድ',
            'ኢንጀራ', 'ቡና', 'በዓል', 'ጾም', 'ቅዳሴ',
            
            # Modern terms
            'ዩኒቨርስቲ', 'ትምህርት', 'ሙዚቃ', 'ስፖርት',
            'ብሮድካስት', 'ቴሌቪዥን', 'ሬዲዮ',
            
            # Literature & Language
            'ሥነ ፅሑፍ', 'ግዕዝ', 'ፊደል', 'መጽሐፍ',
            'ግጥም', 'ዘፈን', 'ልብ ወለድ',
            
            # Science & Education
            'ሳይንስ', 'መድሃኒት', 'ሐኪም', 'ህክምና',
            'ፍልስፍና', 'ሂሳብ', 'ኬሚስትሪ'
        ]
        return search_terms
    
    def get_comprehensive_categories(self) -> List[str]:
        """Get comprehensive list of categories."""
        return [
            'Category:ኢትዮጵያ',
            'Category:የኢትዮጵያ ታሪክ',
            'Category:የኢትዮጵያ ባህል',
            'Category:የኢትዮጵያ ከተሞች',
            'Category:የኢትዮጵያ ሰዎች',
            'Category:የኢትዮጵያ ነገሥታት',
            'Category:የኢትዮጵያ ፖለቲከኞች',
            'Category:አማርኛ',
            'Category:አማርኛ ሥነ ፅሑፍ',
            'Category:ኢትዮጵያዊ ምግቦች',
            'Category:ኢትዮጵያዊ ሙዚቃ',
            'Category:የኢትዮጵያ ዘፋኞች',
            'Category:ኢትዮጵያዊ በዓላት',
            'Category:አብያተ ክርስቲያናት',
            'Category:የኢትዮጵያ ዩኒቨርስቲዎች',
            'Category:ላሊበላ',
            'Category:አክሱም',
            'Category:ጎንደር',
            'Category:ሐረር'
        ]
    
    def search_articles(self, search_term: str, limit: int = 15) -> List[str]:
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
    
    def get_category_articles(self, category: str, limit: int = 25) -> List[str]:
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
    
    def assess_cultural_relevance(self, content: str, title: str) -> float:
        """Assess cultural relevance with multi-tier scoring."""
        relevance_score = 0.0
        
        # Check each tier
        for tier, data in self.cultural_keywords.items():
            tier_score = 0.0
            for term in data['terms']:
                if term in content or term in title:
                    tier_score = 1.0
                    break
            relevance_score += tier_score * data['weight']
        
        # Normalize score
        max_possible_score = sum(data['weight'] for data in self.cultural_keywords.values())
        normalized_score = relevance_score / max_possible_score
        
        # Bonus for multiple tiers
        tiers_found = sum(1 for tier, data in self.cultural_keywords.items()
                         if any(term in content or term in title for term in data['terms']))
        
        if tiers_found >= 3:
            normalized_score += 0.15
        elif tiers_found >= 2:
            normalized_score += 0.1
        
        return min(1.0, normalized_score)
    
    def is_content_safe(self, content: str, title: str) -> bool:
        """Check if content is safe (no sensitive terms)."""
        for avoid_term in self.avoid_terms:
            if avoid_term in content or avoid_term in title:
                return False
        return True
    
    def calculate_quality_score(self, content: str, title: str, cultural_relevance: float) -> float:
        """Calculate comprehensive quality score."""
        score = 0.0
        
        # Cultural relevance (35% of score)
        score += cultural_relevance * 0.35
        
        # Amharic ratio (30% of score)
        amharic_ratio = self.calculate_amharic_ratio(content)
        score += amharic_ratio * 0.3
        
        # Content quality (25% of score)
        content_score = 0.0
        if len(content) > 200:
            content_score += 0.3
        if '።' in content:
            content_score += 0.3
        if any(punct in content for punct in ['፣', '፤', '፥']):
            content_score += 0.2
        if len(content.split()) > 50:
            content_score += 0.2
        
        score += min(1.0, content_score) * 0.25
        
        # Title relevance (10% of score)
        title_score = 0.0
        if any(term in title for tier_data in self.cultural_keywords.values() 
               for term in tier_data['terms']):
            title_score = 1.0
        
        score += title_score * 0.1
        
        return min(1.0, score)
    
    def collect_candidates(self) -> List[str]:
        """Collect comprehensive list of candidate articles."""
        logger.info("Collecting candidate articles from multiple sources...")
        
        all_candidates = []
        
        # Search-based collection
        search_terms = self.get_comprehensive_search_terms()
        logger.info(f"Searching with {len(search_terms)} terms...")
        for term in search_terms:
            titles = self.search_articles(term, limit=10)
            all_candidates.extend(titles)
            time.sleep(0.8)  # Rate limiting
        
        # Category-based collection
        categories = self.get_comprehensive_categories()
        logger.info(f"Collecting from {len(categories)} categories...")
        for category in categories:
            titles = self.get_category_articles(category, limit=20)
            all_candidates.extend(titles)
            time.sleep(0.8)  # Rate limiting
        
        # Remove duplicates and shuffle
        unique_candidates = list(set(all_candidates))
        random.shuffle(unique_candidates)
        
        logger.info(f"Collected {len(unique_candidates)} unique candidate articles")
        return unique_candidates
    
    def collect_articles_systematically(self, target_count: int = 50) -> List[Dict[str, Any]]:
        """Collect articles using systematic progressive approach."""
        logger.info(f"Starting systematic collection of {target_count} articles...")
        
        # Collect all candidates
        candidates = self.collect_candidates()
        
        collected = []
        
        # Process candidates with progressive quality strategies
        for strategy in self.quality_strategies:
            if len(collected) >= target_count:
                break
            
            logger.info(f"Applying {strategy['name']} strategy...")
            strategy_collected = 0
            
            for title in candidates:
                if len(collected) >= target_count:
                    break
                
                if title in self.collected_titles:
                    self.debug_stats['duplicates'] += 1
                    continue
                
                self.debug_stats['total_processed'] += 1
                
                # Fetch content
                article_data = self.get_article_content(title)
                if not article_data:
                    continue
                
                content = article_data['content']
                
                # Apply strategy filters
                if len(content) < strategy['min_length']:
                    self.debug_stats['too_short'] += 1
                    continue
                
                amharic_ratio = self.calculate_amharic_ratio(content)
                if amharic_ratio < strategy['amharic_threshold']:
                    self.debug_stats['low_amharic_ratio'] += 1
                    continue
                
                cultural_relevance = self.assess_cultural_relevance(content, title)
                if cultural_relevance < strategy['cultural_threshold']:
                    self.debug_stats['low_cultural_relevance'] += 1
                    continue
                
                if not self.is_content_safe(content, title):
                    self.debug_stats['contains_sensitive_content'] += 1
                    continue
                
                quality_score = self.calculate_quality_score(content, title, cultural_relevance)
                if quality_score < strategy['quality_threshold']:
                    self.debug_stats['low_quality'] += 1
                    continue
                
                # Article passed all filters
                revisions = article_data.get('revisions', [])
                categories = article_data.get('categories', [])
                
                # Determine category
                primary_category = 'general'
                if categories:
                    cat_name = categories[0].get('title', '').replace('Category:', '')
                    if any(term in cat_name for term in ['ታሪክ', 'ሰው', 'ነገሥት']):
                        primary_category = 'history'
                    elif 'ባህል' in cat_name:
                        primary_category = 'culture'
                    elif 'ሙዚቃ' in cat_name:
                        primary_category = 'music'
                    elif 'ሳይንስ' in cat_name:
                        primary_category = 'science'
                    elif any(term in cat_name for term in ['ከተማ', 'ቦታ']):
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
                    cultural_flags=[]
                )
                
                collected.append({
                    'content': content,
                    'metadata': asdict(metadata),
                    'url': article_data.get('url', ''),
                    'categories': [cat.get('title', '') for cat in categories]
                })
                
                self.collected_titles.add(title)
                self.debug_stats['collected'] += 1
                self.quality_stats[strategy['name']] += 1
                strategy_collected += 1
                
                if strategy_collected % 5 == 0:
                    logger.info(f"  Collected {strategy_collected} articles with {strategy['name']} strategy")
                
                time.sleep(0.5)  # Rate limiting
            
            logger.info(f"Strategy '{strategy['name']}' collected {strategy_collected} articles")
        
        logger.info(f"Systematic collection complete! Total: {len(collected)} articles")
        logger.info(f"Quality breakdown: {self.quality_stats}")
        return collected
    
    def generate_quality_report(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not articles:
            return {'error': 'No articles to analyze', 'debug_stats': self.debug_stats}
        
        # Extract metrics
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
                'excellent': len([s for s in quality_scores if s >= 0.8]),
                'high_quality': len([s for s in quality_scores if 0.7 <= s < 0.8]),
                'good_quality': len([s for s in quality_scores if 0.6 <= s < 0.7]),
                'acceptable_quality': len([s for s in quality_scores if 0.5 <= s < 0.6]),
                'minimal_quality': len([s for s in quality_scores if s < 0.5])
            },
            'amharic_ratio_compliance': {
                'above_80_percent': len([r for r in amharic_ratios if r >= 0.8]),
                'above_70_percent': len([r for r in amharic_ratios if r >= 0.7]),
                'above_65_percent': len([r for r in amharic_ratios if r >= 0.65])
            },
            'cultural_relevance_distribution': {
                'high_relevance': len([r for r in cultural_relevance if r >= 0.7]),
                'moderate_relevance': len([r for r in cultural_relevance if 0.5 <= r < 0.7]),
                'basic_relevance': len([r for r in cultural_relevance if r < 0.5])
            },
            'category_distribution': categories,
            'quality_strategy_breakdown': self.quality_stats,
            'debug_stats': self.debug_stats,
            'validation_passed': all(
                article['metadata']['amharic_ratio'] >= 0.65 and
                article['metadata']['quality_score'] >= 0.4
                for article in articles
            )
        }
    
    def save_corpus(self, articles: List[Dict[str, Any]], filepath: str) -> None:
        """Save systematically collected corpus."""
        quality_report = self.generate_quality_report(articles)
        
        corpus_data = {
            'corpus_metadata': {
                'version': '4.0-systematic',
                'collection_date': datetime.now().isoformat(),
                'collector': 'SystematicCollector',
                'approach': 'Progressive Multi-Strategy Collection',
                'total_articles': len(articles),
                'quality_strategies': self.quality_strategies,
                'search_terms_count': len(self.get_comprehensive_search_terms()),
                'categories_count': len(self.get_comprehensive_categories())
            },
            'quality_report': quality_report,
            'articles': articles
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Systematic corpus saved to {filepath}")

def main():
    """Main execution function."""
    collector = SystematicCollector()
    
    # Collect articles systematically
    articles = collector.collect_articles_systematically(target_count=50)
    
    if articles:
        # Save to specified location
        output_path = "/Users/mekdesyared/amharic-hnet-v2/data/raw/test_corpus.json"
        collector.save_corpus(articles, output_path)
        
        print(f"\n🎯 Successfully collected {len(articles)} Amharic articles systematically")
        print(f"📁 Saved to: {output_path}")
        
        # Print detailed summary
        quality_report = collector.generate_quality_report(articles)
        summary = quality_report['collection_summary']
        
        print(f"\n📊 Systematic Collection Summary:")
        print(f"   • Total Articles: {summary['total_articles']}")
        print(f"   • Average Quality Score: {summary['average_quality_score']:.3f}")
        print(f"   • Average Cultural Relevance: {summary['average_cultural_relevance']:.3f}")
        print(f"   • Average Amharic Ratio: {summary['average_amharic_ratio']:.3f}")
        print(f"   • Average Word Count: {summary['average_word_count']:.0f}")
        print(f"   • Quality Range: {summary['min_quality_score']:.3f} - {summary['max_quality_score']:.3f}")
        
        print(f"\n🏆 Quality Distribution:")
        dist = quality_report['quality_distribution']
        print(f"   • Excellent (≥0.8): {dist['excellent']} articles")
        print(f"   • High Quality (0.7-0.8): {dist['high_quality']} articles")
        print(f"   • Good Quality (0.6-0.7): {dist['good_quality']} articles")
        print(f"   • Acceptable (0.5-0.6): {dist['acceptable_quality']} articles")
        print(f"   • Minimal (<0.5): {dist['minimal_quality']} articles")
        
        print(f"\n📝 Amharic Script Compliance:")
        amharic = quality_report['amharic_ratio_compliance']
        print(f"   • Above 80%: {amharic['above_80_percent']} articles")
        print(f"   • Above 70%: {amharic['above_70_percent']} articles")
        print(f"   • Above 65%: {amharic['above_65_percent']} articles")
        
        print(f"\n🇪🇹 Cultural Relevance:")
        cultural = quality_report['cultural_relevance_distribution']
        print(f"   • High Relevance (≥0.7): {cultural['high_relevance']} articles")
        print(f"   • Moderate Relevance (0.5-0.7): {cultural['moderate_relevance']} articles")
        print(f"   • Basic Relevance (<0.5): {cultural['basic_relevance']} articles")
        
        print(f"\n📈 Strategy Breakdown:")
        for strategy, count in collector.quality_stats.items():
            print(f"   • {strategy}: {count} articles")
        
        print(f"\n✅ Overall Validation: {'PASSED' if quality_report['validation_passed'] else 'NEEDS_IMPROVEMENT'}")
        
    else:
        print("❌ No articles collected")
        print(f"Debug stats: {collector.debug_stats}")

if __name__ == "__main__":
    main()