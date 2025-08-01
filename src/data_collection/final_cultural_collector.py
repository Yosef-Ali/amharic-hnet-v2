#!/usr/bin/env python3
"""
Final Cultural-Focused Amharic Corpus Collector

This version specifically targets culturally authentic and safe Amharic content
with improved filtering and validation mechanisms.
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

class FinalCulturalCollector:
    """
    Cultural-focused Amharic Wikipedia article collector with enhanced safety measures.
    """
    
    def __init__(self):
        self.base_url = "https://am.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AmharicCorpusCollector/3.0 (Cultural Focus)'
        })
        
        # Amharic script ranges
        self.amharic_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
            (0xAB00, 0xAB2F),  # Ethiopic Extended-A
        ]
        
        # High-priority cultural content
        self.priority_cultural_terms = {
            'core_ethiopian': {
                'terms': ['ኢትዮጵያ', 'የኢትዮጵያ', 'ኢትዮጵያዊ', 'ኢትዮጵያውያን'],
                'weight': 3.0
            },
            'amharic_language': {
                'terms': ['አማርኛ', 'የአማርኛ', 'አማርኛ ቋንቋ', 'አማርኛ ሥነ ፅሁፍ'],
                'weight': 2.5
            },
            'traditional_culture': {
                'terms': ['ኢንጀራ', 'ቡና', 'የቡና ስርአት', 'በዓል', 'ጾም', 'እንጉዳይ', 'ሽሮ', 'ወጥ'],
                'weight': 2.0
            },
            'religious_heritage': {
                'terms': ['ቤተ ክርስቲያን', 'ኢትዮጵያ ኦርቶዶክስ', 'መስጊድ', 'ሲናክሳር', 'ቅዳሴ', 'ጸሎት'],
                'weight': 2.0
            },
            'historical_figures': {
                'terms': ['ሐይለ ሥላሴ', 'ምንሊክ', 'ዘውዲቱ', 'ዮሐንስ', 'ቴዎድሮስ', 'ሣህለ ስላሴ'],
                'weight': 1.8
            },
            'geographical': {
                'terms': ['አዲስ አበባ', 'ጎንደር', 'አክሱም', 'ላሊበላ', 'ባህር ዳር', 'መቀሌ', 'ሐዋሳ'],
                'weight': 1.5
            }
        }
        
        # Terms to avoid (sensitive/problematic content)
        self.avoid_terms = [
            'ወረራ', 'ጥቃት', 'ዓመፃ', 'ብጥብጥ', 'መጠላላት', 'አድልዎ', 
            'ግጭት', 'ጦርነት', 'ዘር', 'ብሔር', 'ፖለቲካዊ ግጭት'
        ]
        
        # Quality thresholds (more stringent)
        self.quality_threshold = 0.6
        self.cultural_relevance_threshold = 0.7
        self.amharic_ratio_threshold = 0.75
        self.min_content_length = 200
        
        self.collected_titles = set()
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
    
    def get_culturally_focused_articles(self) -> List[str]:
        """Get articles specifically focused on Ethiopian culture."""
        # High-priority search terms for cultural content
        cultural_searches = [
            'ኢትዮጵያ ባህል',
            'የኢትዮጵያ ባህል',
            'አማርኛ ሥነ ፅሁፍ',
            'ኢትዮጵያዊ ምግብ',
            'ኢንጀራ',
            'የቡና ስርአት',
            'ኢትዮጵያ ኦርቶዶክስ',
            'የኢትዮጵያ ታሪክ', 
            'አክሱም',
            'ላሊበላ',
            'ሐይለ ሥላሴ',
            'አዲስ አበባ',
            'ኢትዮጵያዊ በዓላት',
            'የኢትዮጵያ ሙዚቃ',
            'ኢትዮጵያዊ ልብስ',
            'የኢትዮጵያ ፊደል'
        ]
        
        all_titles = []
        for term in cultural_searches:
            titles = self.search_articles(term, limit=12)
            all_titles.extend(titles)
            time.sleep(1.2)  # Respectful rate limiting
        
        return list(set(all_titles))
    
    def get_cultural_category_articles(self) -> List[str]:
        """Get articles from cultural categories."""
        cultural_categories = [
            'Category:የኢትዮጵያ ባህል',
            'Category:ኢትዮጵያዊ ምግቦች',
            'Category:የኢትዮጵያ ታሪክ',
            'Category:ኢትዮጵያዊ ሙዚቃ',
            'Category:የኢትዮጵያ ሥነ ጽሑፍ',
            'Category:ኢትዮጵያዊ በዓላት',
            'Category:አማርኛ ሥነ ፅሑፍ'
        ]
        
        all_titles = []
        for category in cultural_categories:
            titles = self.get_category_articles(category, limit=20)
            all_titles.extend(titles)
            time.sleep(1.0)
        
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
    
    def assess_cultural_relevance(self, content: str, title: str) -> float:
        """Assess cultural relevance with weighted scoring."""
        relevance_score = 0.0
        
        # Check for priority cultural terms with weights
        for category, data in self.priority_cultural_terms.items():
            category_score = 0.0
            for term in data['terms']:
                if term in content or term in title:
                    category_score = 1.0
                    break
            relevance_score += category_score * data['weight']
        
        # Normalize score
        max_possible_score = sum(data['weight'] for data in self.priority_cultural_terms.values())
        normalized_score = relevance_score / max_possible_score
        
        # Bonus for multiple cultural categories
        categories_found = 0
        for category, data in self.priority_cultural_terms.items():
            if any(term in content or term in title for term in data['terms']):
                categories_found += 1
        
        if categories_found >= 3:
            normalized_score += 0.2
        elif categories_found >= 2:
            normalized_score += 0.1
        
        return min(1.0, normalized_score)
    
    def check_content_safety(self, content: str, title: str) -> Tuple[bool, List[str]]:
        """Check if content is culturally safe and appropriate."""
        flags = []
        is_safe = True
        
        # Check for sensitive/problematic terms
        for avoid_term in self.avoid_terms:
            if avoid_term in content or avoid_term in title:
                flags.append(f'contains_sensitive_term_{avoid_term}')
                is_safe = False
        
        return is_safe, flags
    
    def calculate_quality_score(self, content: str, title: str, cultural_relevance: float) -> float:
        """Calculate comprehensive quality score."""
        score = 0.0
        
        # Cultural relevance (40% of score)
        score += cultural_relevance * 0.4
        
        # Amharic ratio (30% of score)
        amharic_ratio = self.calculate_amharic_ratio(content)
        score += amharic_ratio * 0.3
        
        # Content structure and completeness (20% of score)
        structure_score = 0.0
        if len(content) > 400:
            structure_score += 0.4
        if '።' in content:
            structure_score += 0.3
        if any(punct in content for punct in ['፣', '፤', '፥']):
            structure_score += 0.3
        
        score += min(1.0, structure_score) * 0.2
        
        # Title quality (10% of score)
        title_score = 0.0
        if any(term in title for terms in self.priority_cultural_terms.values() for term in terms['terms']):
            title_score = 1.0
        
        score += title_score * 0.1
        
        return min(1.0, score)
    
    def collect_articles(self, target_count: int = 50) -> List[Dict[str, Any]]:
        """Collect culturally focused, high-quality articles."""
        logger.info(f"Starting cultural-focused collection of {target_count} articles...")
        
        # Get culturally focused candidates
        logger.info("Phase 1: Cultural search-based collection...")
        search_candidates = self.get_culturally_focused_articles()
        logger.info(f"Found {len(search_candidates)} search-based candidates")
        
        logger.info("Phase 2: Cultural category-based collection...")
        category_candidates = self.get_cultural_category_articles()
        logger.info(f"Found {len(category_candidates)} category-based candidates")
        
        # Combine and prioritize
        all_candidates = search_candidates + category_candidates
        all_candidates = list(set(all_candidates))
        random.shuffle(all_candidates)
        
        logger.info(f"Total unique candidates: {len(all_candidates)}")
        
        collected = []
        
        for i, title in enumerate(all_candidates):
            if len(collected) >= target_count:
                break
            
            self.debug_stats['total_processed'] += 1
            
            if i % 15 == 0:
                logger.info(f"Processed {i} articles, collected {len(collected)}")
            
            # Skip duplicates
            if title in self.collected_titles:
                self.debug_stats['duplicates'] += 1
                continue
            
            # Fetch content
            article_data = self.get_article_content(title)
            if not article_data:
                continue
            
            content = article_data['content']
            
            # Length check
            if len(content) < self.min_content_length:
                self.debug_stats['too_short'] += 1
                continue
            
            # Amharic ratio check
            amharic_ratio = self.calculate_amharic_ratio(content)
            if amharic_ratio < self.amharic_ratio_threshold:
                self.debug_stats['low_amharic_ratio'] += 1
                continue
            
            # Cultural relevance check
            cultural_relevance = self.assess_cultural_relevance(content, title)
            if cultural_relevance < self.cultural_relevance_threshold:
                self.debug_stats['low_cultural_relevance'] += 1
                continue
            
            # Safety check
            is_safe, safety_flags = self.check_content_safety(content, title)
            if not is_safe:
                self.debug_stats['contains_sensitive_content'] += 1
                continue
            
            # Quality score
            quality_score = self.calculate_quality_score(content, title, cultural_relevance)
            if quality_score < self.quality_threshold:
                self.debug_stats['low_quality'] += 1
                continue
            
            # Article passed all checks - collect it
            revisions = article_data.get('revisions', [])
            categories = article_data.get('categories', [])
            
            # Determine category
            primary_category = 'culture'  # Default for cultural focus
            if categories:
                cat_name = categories[0].get('title', '').replace('Category:', '')
                if 'ታሪክ' in cat_name:
                    primary_category = 'history'
                elif 'ምግብ' in cat_name:
                    primary_category = 'food_culture'
                elif 'ሙዚቃ' in cat_name:
                    primary_category = 'music_culture'
                elif 'ሥነ ፅሑፍ' in cat_name:
                    primary_category = 'literature'
            
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
                cultural_flags=safety_flags
            )
            
            collected.append({
                'content': content,
                'metadata': asdict(metadata),
                'url': article_data.get('url', ''),
                'categories': [cat.get('title', '') for cat in categories]
            })
            
            self.collected_titles.add(title)
            self.debug_stats['collected'] += 1
            
            logger.info(f"✅ Collected: '{title}' (Q:{quality_score:.3f}, C:{cultural_relevance:.3f}, A:{amharic_ratio:.3f})")
            
            # Rate limiting
            time.sleep(0.8)
        
        logger.info(f"Collection complete! Collected {len(collected)} articles")
        logger.info(f"Final stats: {self.debug_stats}")
        return collected
    
    def generate_quality_report(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate quality report for collected articles."""
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
                'max_quality_score': max(quality_scores),
                'cultural_focus_validation': 'PASSED' if min(cultural_relevance) >= 0.7 else 'NEEDS_IMPROVEMENT'
            },
            'quality_distribution': {
                'high_quality': len([s for s in quality_scores if s >= 0.8]),
                'good_quality': len([s for s in quality_scores if 0.6 <= s < 0.8]),
                'acceptable_quality': len([s for s in quality_scores if 0.5 <= s < 0.6])
            },
            'cultural_compliance': {
                'high_cultural_relevance': len([r for r in cultural_relevance if r >= 0.8]),
                'good_cultural_relevance': len([r for r in cultural_relevance if 0.7 <= r < 0.8]),
                'minimum_cultural_relevance': len([r for r in cultural_relevance if 0.6 <= r < 0.7])
            },
            'amharic_ratio_compliance': {
                'above_75_percent': len([r for r in amharic_ratios if r >= 0.75]),
                'above_90_percent': len([r for r in amharic_ratios if r >= 0.9]),
                'perfect_amharic': len([r for r in amharic_ratios if r >= 0.99])
            },
            'category_distribution': categories,
            'debug_stats': self.debug_stats,
            'validation_passed': all(
                article['metadata']['amharic_ratio'] >= 0.75 and
                article['metadata']['quality_score'] >= 0.6 and
                article['metadata']['cultural_relevance'] >= 0.7
                for article in articles
            )
        }
    
    def save_corpus(self, articles: List[Dict[str, Any]], filepath: str) -> None:
        """Save culturally validated corpus."""
        quality_report = self.generate_quality_report(articles)
        
        corpus_data = {
            'corpus_metadata': {
                'version': '3.0-cultural-focus',
                'collection_date': datetime.now().isoformat(),
                'collector': 'FinalCulturalCollector',
                'focus': 'Ethiopian Cultural Authenticity',
                'total_articles': len(articles),
                'quality_threshold': self.quality_threshold,
                'cultural_relevance_threshold': self.cultural_relevance_threshold,
                'amharic_ratio_threshold': self.amharic_ratio_threshold,
                'min_content_length': self.min_content_length
            },
            'quality_report': quality_report,
            'articles': articles
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Cultural corpus saved to {filepath}")

def main():
    """Main execution function."""
    collector = FinalCulturalCollector()
    
    # Collect articles
    articles = collector.collect_articles(target_count=50)
    
    if articles:
        # Save to specified location
        output_path = "/Users/mekdesyared/amharic-hnet-v2/data/raw/test_corpus.json"
        collector.save_corpus(articles, output_path)
        
        print(f"\n🇪🇹 Successfully collected {len(articles)} culturally authentic Amharic articles")
        print(f"📁 Saved to: {output_path}")
        
        # Print detailed summary
        quality_report = collector.generate_quality_report(articles)
        summary = quality_report['collection_summary']
        
        print(f"\n📊 Cultural Collection Summary:")
        print(f"   • Total Articles: {summary['total_articles']}")
        print(f"   • Average Quality Score: {summary['average_quality_score']:.3f}")
        print(f"   • Average Cultural Relevance: {summary['average_cultural_relevance']:.3f}")
        print(f"   • Average Amharic Ratio: {summary['average_amharic_ratio']:.3f}")
        print(f"   • Average Word Count: {summary['average_word_count']:.0f}")
        print(f"   • Cultural Focus Validation: {summary['cultural_focus_validation']}")
        print(f"   • Overall Validation: {'PASSED' if quality_report['validation_passed'] else 'NEEDS_IMPROVEMENT'}")
        
        print(f"\n🏆 Quality Distribution:")
        dist = quality_report['quality_distribution']
        print(f"   • High Quality (≥0.8): {dist['high_quality']} articles")
        print(f"   • Good Quality (0.6-0.8): {dist['good_quality']} articles")
        print(f"   • Acceptable Quality (0.5-0.6): {dist['acceptable_quality']} articles")
        
        print(f"\n🇪🇹 Cultural Compliance:")
        cultural = quality_report['cultural_compliance']
        print(f"   • High Cultural Relevance (≥0.8): {cultural['high_cultural_relevance']} articles")
        print(f"   • Good Cultural Relevance (0.7-0.8): {cultural['good_cultural_relevance']} articles")
        
        print(f"\n📝 Amharic Script Compliance:")
        amharic = quality_report['amharic_ratio_compliance']
        print(f"   • Above 75%: {amharic['above_75_percent']} articles")
        print(f"   • Above 90%: {amharic['above_90_percent']} articles")
        print(f"   • Near Perfect (≥99%): {amharic['perfect_amharic']} articles")
        
    else:
        print("❌ No articles met the cultural authenticity criteria")
        print(f"Debug stats: {collector.debug_stats}")

if __name__ == "__main__":
    main()