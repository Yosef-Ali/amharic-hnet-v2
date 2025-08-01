#!/usr/bin/env python3
"""
Comprehensive Amharic Wikipedia Corpus Collector
Specialized for collecting >500 high-quality Amharic articles with cultural validation
Designed for H-Net training data enhancement
"""

import json
import time
import random
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveAmharicCollector:
    """
    Expert Amharic corpus collector with cultural validation and quality assurance
    """
    
    def __init__(self, target_count: int = 500, min_amharic_ratio: float = 0.70):
        self.target_count = target_count
        self.min_amharic_ratio = min_amharic_ratio
        
        # Setup HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Cultural safety terms and categories
        self.sacred_terms = {
            'እግዚአብሔር', 'መስቀል', 'ማርያም', 'ገብርኤል', 'ሚካኤል', 'እየሱስ', 'ክርስቶስ',
            'ቤተክርስቲያን', 'ጸሎት', 'ጾም', 'ኢየሱስ', 'መንግስተ', 'ሰማያት'
        }
        
        self.cultural_keywords = {
            'historical': [
                'ዓክሱም', 'ላሊበላ', 'ጎንደር', 'ሐረር', 'ባህርዳር', 'ምክትል', 'ንጉስ', 'ንግሥት',
                'ዓፄ', 'ዘመን', 'ታሪክ', 'ቅርስ', 'ቤተ መንግሥት', 'ዘመነ', 'መንግሥት'
            ],
            'cultural': [
                'ቡና', 'እንጀራ', 'ዶሮ', 'ወጥ', 'ሸሮ', 'ምስር', 'ዓሰመራ', 'አዲስ አበባ',
                'ባህል', 'ትውፊት', 'በዓል', 'ዓመተ', 'ገና', 'ፋሲካ', 'መስቀል', 'እርጎ'
            ],
            'literature': [
                'ግዕዝ', 'ሥነ ፅሁፍ', 'ግጥም', 'ሲራክ', 'ወርቅ', 'ግምብ', 'መጽሐፍ', 'ዋሳን',
                'መዝሙር', 'ንባብ', 'ድርሰት', 'ተረት', 'ፍቅር'
            ],
            'geography': [
                'ሸዋ', 'ትግራይ', 'አማራ', 'ኦሮሚያ', 'ደቡብ', 'ቤንሻንጉል', 'ጋምቤላ', 'ሐረርጌ',
                'አፋር', 'ሶማሊ', 'ሲዳማ', 'ወላይታ', 'ጉራጌ', 'ሀዲያ'
            ],
            'everyday': [
                'ቤተሰብ', 'ህወሐት', 'ጓደኛ', 'ወንድም', 'እህት', 'አባት', 'እናት', 'ወንድ', 'ሴት',
                'ልጅ', 'ተማሪ', 'መምህር', 'ዶክተር', 'ነርስ', 'ነጋዴ', 'ሰራተኛ'
            ]
        }
        
        # Statistics tracking
        self.collection_stats = {
            'total_attempted': 0,
            'successful_collections': 0,
            'cultural_violations': 0,
            'low_quality_rejected': 0,
            'amharic_ratio_failures': 0,
            'categories': {},
            'processing_time': 0
        }
        
        # Priority categories for targeted collection
        self.priority_categories = [
            'ኢትዮጵያ', 'ኤርትራ', 'ታሪክ', 'ባህል', 'ቋንቋ', 'ሥነ ፅሁፍ', 'ጂኦግራፊ',
            'ሃይማኖት', 'ፖለቲካ', 'ሳይንስ', 'ስፖርት', 'ኪነ ጥበብ', 'ሙዚቃ', 'ፊልም'
        ]

    def calculate_amharic_ratio(self, text: str) -> float:
        """Calculate the ratio of Amharic characters in text"""
        if not text:
            return 0.0
        
        # Ge'ez script Unicode range: U+1200-U+137F
        amharic_chars = len(re.findall(r'[\u1200-\u137F]', text))
        total_chars = len(re.sub(r'\s', '', text))  # Exclude whitespace
        
        return amharic_chars / max(total_chars, 1)

    def assess_cultural_safety(self, title: str, content: str) -> Dict[str, Any]:
        """Assess cultural safety and appropriateness of content"""
        safety_score = 100
        violations = []
        cultural_relevance = 0
        
        combined_text = f"{title} {content}".lower()
        
        # Check for sacred terms (require respectful context)
        for term in self.sacred_terms:
            if term.lower() in combined_text:
                # Simple context check - can be enhanced
                context_words = ['ትምህርት', 'ታሪክ', 'መጽሐፍ', 'ብሃይ', 'አምልኮ']
                if not any(ctx in combined_text for ctx in context_words):
                    violations.append(f"Sacred term '{term}' without appropriate context")
                    safety_score -= 10
                else:
                    cultural_relevance += 15  # Positive cultural content
        
        # Assess cultural relevance
        for category, keywords in self.cultural_keywords.items():
            matches = sum(1 for kw in keywords if kw.lower() in combined_text)
            cultural_relevance += matches * 5
            self.collection_stats['categories'][category] = self.collection_stats['categories'].get(category, 0) + matches
        
        # Check for potentially problematic content
        problematic_indicators = ['ውግዘት', 'ዘረኝነት', 'መንግሥት ተቃውሞ']
        for indicator in problematic_indicators:
            if indicator in combined_text:
                violations.append(f"Potentially problematic content: {indicator}")
                safety_score -= 20
        
        return {
            'safety_score': max(safety_score, 0),
            'violations': violations,
            'cultural_relevance': min(cultural_relevance, 100),
            'is_culturally_appropriate': safety_score >= 80 and cultural_relevance >= 30
        }

    def get_random_articles(self, count: int = 50) -> List[str]:
        """Get random Amharic Wikipedia articles"""
        try:
            url = "https://am.wikipedia.org/api/rest_v1/page/random/summary"
            articles = []
            
            for _ in range(count):
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if 'title' in data:
                            articles.append(data['title'])
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Error getting random article: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error in get_random_articles: {e}")
            return []

    def search_articles_by_category(self, category: str, limit: int = 100) -> List[str]:
        """Search for articles in specific categories"""
        try:
            # Use Wikipedia search API
            search_url = "https://am.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': category,
                'srlimit': limit,
                'srnamespace': 0
            }
            
            response = self.session.get(search_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                if 'query' in data and 'search' in data['query']:
                    for result in data['query']['search']:
                        articles.append(result['title'])
                
                return articles
            
        except Exception as e:
            logger.error(f"Error searching category {category}: {e}")
        
        return []

    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Get full content of a Wikipedia article"""
        try:
            # Get article content using Wikipedia API
            url = "https://am.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts|info|categories',
                'exintro': False,
                'explaintext': True,
                'inprop': 'url'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code != 200:
                return None
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page_data in pages.items():
                if page_id == '-1':  # Article not found
                    continue
                
                title = page_data.get('title', '')
                content = page_data.get('extract', '')
                url = page_data.get('fullurl', '')
                categories = [cat.get('title', '').replace('Category:', '') 
                            for cat in page_data.get('categories', [])]
                
                if not content or len(content.strip()) < 200:
                    return None
                
                return {
                    'title': title,
                    'content': content,
                    'url': url,
                    'categories': categories,
                    'word_count': len(content.split()),
                    'character_count': len(content)
                }
                
        except Exception as e:
            logger.error(f"Error getting article content for '{title}': {e}")
        
        return None

    def process_article(self, title: str) -> Optional[Dict[str, Any]]:
        """Process a single article with quality and cultural validation"""
        try:
            self.collection_stats['total_attempted'] += 1
            
            # Get article content
            article_data = self.get_article_content(title)
            if not article_data:
                return None
            
            content = article_data['content']
            
            # Calculate Amharic ratio
            amharic_ratio = self.calculate_amharic_ratio(content)
            if amharic_ratio < self.min_amharic_ratio:
                self.collection_stats['amharic_ratio_failures'] += 1
                logger.debug(f"Article '{title}' rejected: Amharic ratio {amharic_ratio:.2f} < {self.min_amharic_ratio}")
                return None
            
            # Assess cultural safety
            cultural_assessment = self.assess_cultural_safety(title, content)
            if not cultural_assessment['is_culturally_appropriate']:
                self.collection_stats['cultural_violations'] += 1
                logger.warning(f"Article '{title}' rejected: Cultural safety violations")
                return None
            
            # Quality checks
            if len(content.split()) < 50:  # Minimum word count
                self.collection_stats['low_quality_rejected'] += 1
                return None
            
            # Create processed article record
            processed_article = {
                'title': article_data['title'],
                'content': content,
                'url': article_data['url'],
                'categories': article_data['categories'],
                'metadata': {
                    'word_count': article_data['word_count'],
                    'character_count': article_data['character_count'],
                    'amharic_ratio': round(amharic_ratio, 3),
                    'cultural_safety_score': cultural_assessment['safety_score'],
                    'cultural_relevance_score': cultural_assessment['cultural_relevance'],
                    'collection_timestamp': datetime.now().isoformat(),
                    'quality_tier': self._assess_quality_tier(article_data, amharic_ratio, cultural_assessment)
                }
            }
            
            self.collection_stats['successful_collections'] += 1
            logger.info(f"Successfully processed: '{title}' (Amharic: {amharic_ratio:.2f}, Cultural: {cultural_assessment['cultural_relevance']})")
            
            return processed_article
            
        except Exception as e:
            logger.error(f"Error processing article '{title}': {e}")
            return None

    def _assess_quality_tier(self, article_data: Dict, amharic_ratio: float, cultural_assessment: Dict) -> str:
        """Assess quality tier of article"""
        score = 0
        
        # Length scoring
        if article_data['word_count'] > 500:
            score += 3
        elif article_data['word_count'] > 200:
            score += 2
        else:
            score += 1
        
        # Amharic ratio scoring
        if amharic_ratio > 0.9:
            score += 3
        elif amharic_ratio > 0.8:
            score += 2
        else:
            score += 1
        
        # Cultural relevance scoring
        if cultural_assessment['cultural_relevance'] > 70:
            score += 3
        elif cultural_assessment['cultural_relevance'] > 40:
            score += 2
        else:
            score += 1
        
        if score >= 8:
            return 'premium'
        elif score >= 6:
            return 'high'
        elif score >= 4:
            return 'medium'
        else:
            return 'basic'

    def collect_comprehensive_corpus(self) -> Dict[str, Any]:
        """Main collection method for comprehensive corpus"""
        logger.info(f"Starting comprehensive collection targeting {self.target_count} articles")
        start_time = time.time()
        
        collected_articles = []
        processed_titles = set()
        
        # Collection strategy: Mix of targeted and random articles
        collection_methods = [
            ('category_search', 0.6),  # 60% from targeted categories
            ('random_articles', 0.4)   # 40% random discovery
        ]
        
        for method, ratio in collection_methods:
            target_for_method = int(self.target_count * ratio)
            logger.info(f"Collecting {target_for_method} articles using {method}")
            
            if method == 'category_search':
                titles = []
                articles_per_category = target_for_method // len(self.priority_categories)
                
                for category in self.priority_categories:
                    category_titles = self.search_articles_by_category(category, articles_per_category + 10)
                    titles.extend(category_titles[:articles_per_category])
                    time.sleep(1)  # Rate limiting
                
            else:  # random_articles
                titles = self.get_random_articles(target_for_method * 2)  # Get extra for filtering
            
            # Process articles
            for title in titles:
                if len(collected_articles) >= self.target_count:
                    break
                
                if title in processed_titles:
                    continue
                
                processed_titles.add(title)
                article = self.process_article(title)
                
                if article:
                    collected_articles.append(article)
                    
                    # Progress reporting
                    if len(collected_articles) % 50 == 0:
                        logger.info(f"Progress: {len(collected_articles)}/{self.target_count} articles collected")
                
                time.sleep(0.3)  # Rate limiting
        
        # Final statistics
        self.collection_stats['processing_time'] = time.time() - start_time
        
        logger.info(f"Collection completed: {len(collected_articles)} articles collected")
        
        return {
            'articles': collected_articles,
            'collection_metadata': {
                'total_collected': len(collected_articles),
                'target_count': self.target_count,
                'collection_timestamp': datetime.now().isoformat(),
                'min_amharic_ratio': self.min_amharic_ratio,
                'quality_distribution': self._analyze_quality_distribution(collected_articles),
                'cultural_categories': dict(self.collection_stats['categories']),
                'statistics': self.collection_stats
            }
        }

    def _analyze_quality_distribution(self, articles: List[Dict]) -> Dict[str, int]:
        """Analyze quality distribution of collected articles"""
        distribution = {'premium': 0, 'high': 0, 'medium': 0, 'basic': 0}
        
        for article in articles:
            tier = article['metadata']['quality_tier']
            distribution[tier] += 1
        
        return distribution

    def save_corpus(self, corpus_data: Dict[str, Any], output_path: str) -> None:
        """Save collected corpus to JSON file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(corpus_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Corpus saved to: {output_file}")
            
            # Save statistics summary
            stats_file = output_file.with_suffix('.stats.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(corpus_data['collection_metadata'], f, ensure_ascii=False, indent=2)
            
            logger.info(f"Statistics saved to: {stats_file}")
            
        except Exception as e:
            logger.error(f"Error saving corpus: {e}")

def main():
    """Main execution function"""
    collector = ComprehensiveAmharicCollector(target_count=500, min_amharic_ratio=0.70)
    
    # Collect comprehensive corpus
    corpus_data = collector.collect_comprehensive_corpus()
    
    # Save corpus
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/mekdesyared/amharic-hnet-v2/data/raw/comprehensive_corpus_{timestamp}.json"
    
    collector.save_corpus(corpus_data, output_path)
    
    # Print summary
    metadata = corpus_data['collection_metadata']
    print("\n" + "="*60)
    print("COMPREHENSIVE AMHARIC CORPUS COLLECTION COMPLETE")
    print("="*60)
    print(f"Total Articles Collected: {metadata['total_collected']}")
    print(f"Target Achievement: {metadata['total_collected']}/{metadata['target_count']} ({100*metadata['total_collected']/metadata['target_count']:.1f}%)")
    print(f"Minimum Amharic Ratio: {metadata['min_amharic_ratio']:.1%}")
    print(f"Processing Time: {metadata['statistics']['processing_time']:.1f} seconds")
    print(f"Success Rate: {100*metadata['statistics']['successful_collections']/max(metadata['statistics']['total_attempted'], 1):.1f}%")
    
    print("\nQuality Distribution:")
    for tier, count in metadata['quality_distribution'].items():
        print(f"  {tier.capitalize()}: {count} articles")
    
    print("\nCultural Categories Covered:")
    for category, count in list(metadata['cultural_categories'].items())[:10]:
        print(f"  {category}: {count} references")
    
    print(f"\nCorpus saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()