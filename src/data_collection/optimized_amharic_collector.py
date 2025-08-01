#!/usr/bin/env python3
"""
Optimized Amharic Wikipedia Corpus Collector
Fast collection with improved cultural safety logic
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedAmharicCollector:
    """
    Optimized Amharic corpus collector with streamlined cultural validation
    """
    
    def __init__(self, target_count: int = 500, min_amharic_ratio: float = 0.70):
        self.target_count = target_count
        self.min_amharic_ratio = min_amharic_ratio
        
        # Setup HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(total=2, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Streamlined cultural keywords for positive matching
        self.positive_cultural_indicators = {
            'ኢትዮጵያ', 'ኤርትራ', 'አማርኛ', 'አዲስ አበባ', 'ሐረር', 'ጎንደር', 'ላሊበላ',
            'ዓክሱም', 'ባህል', 'ታሪክ', 'ቋንቋ', 'ስነ ፅሁፍ', 'ሙዚቃ', 'ፊልም',
            'ስፖርት', 'ሳይንስ', 'ጂኦግራፊ', 'ሃይማኖት', 'ክርስትና', 'እስልምና',
            'ቡና', 'እንጀራ', 'ወጥ', 'ቤተሰብ', 'ህብረተሰብ', 'ትምህርት'
        }
        
        # Simple negative filters (avoid obvious problematic content)
        self.avoid_terms = {
            'ውግዘት', 'ዘረኝነት', 'ፖለቲካዊ ተቃውሞ', 'ጦርነት መሪዎች'
        }
        
        # Statistics tracking
        self.stats = {
            'attempted': 0, 'collected': 0, 'rejected_amharic': 0, 
            'rejected_quality': 0, 'rejected_safety': 0
        }

    def calculate_amharic_ratio(self, text: str) -> float:
        """Calculate the ratio of Amharic characters in text"""
        if not text:
            return 0.0
        
        # Ge'ez script Unicode range: U+1200-U+137F
        amharic_chars = len(re.findall(r'[\u1200-\u137F]', text))
        total_chars = len(re.sub(r'\s', '', text))  # Exclude whitespace
        
        return amharic_chars / max(total_chars, 1)

    def is_culturally_appropriate(self, title: str, content: str) -> bool:
        """Simple but effective cultural appropriateness check"""
        combined_text = f"{title} {content}".lower()
        
        # Check for negative indicators
        for term in self.avoid_terms:
            if term.lower() in combined_text:
                return False
        
        # Check for positive cultural indicators
        cultural_score = sum(1 for term in self.positive_cultural_indicators 
                           if term.lower() in combined_text)
        
        # Accept if has cultural relevance or is general knowledge
        return cultural_score > 0 or len(content.split()) > 100

    def get_random_articles_batch(self, count: int = 100) -> List[str]:
        """Get batch of random Amharic Wikipedia articles efficiently"""
        try:
            # Use Wikipedia's random generator API for efficiency
            url = "https://am.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'generator': 'random',
                'grnnamespace': 0,
                'grnlimit': min(count, 50)  # API limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                if 'query' in data and 'pages' in data['query']:
                    for page_id, page_info in data['query']['pages'].items():
                        if 'title' in page_info:
                            articles.append(page_info['title'])
                
                return articles
            
        except Exception as e:
            logger.warning(f"Error getting random articles: {e}")
        
        return []

    def search_targeted_articles(self, query: str, limit: int = 50) -> List[str]:
        """Search for articles by topic"""
        try:
            url = "https://am.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': limit,
                'srnamespace': 0
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                if 'query' in data and 'search' in data['query']:
                    for result in data['query']['search']:
                        articles.append(result['title'])
                
                return articles
            
        except Exception as e:
            logger.warning(f"Error searching for '{query}': {e}")
        
        return []

    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Get article content efficiently"""
        try:
            url = "https://am.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts|info',
                'exintro': False,
                'explaintext': True,
                'inprop': 'url'
            }
            
            response = self.session.get(url, params=params, timeout=10)
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
                
                if not content or len(content.strip()) < 100:
                    return None
                
                return {
                    'title': title,
                    'content': content,
                    'url': url,
                    'word_count': len(content.split()),
                    'character_count': len(content)
                }
                
        except Exception as e:
            logger.debug(f"Error getting content for '{title}': {e}")
        
        return None

    def process_article(self, title: str) -> Optional[Dict[str, Any]]:
        """Process article with streamlined validation"""
        self.stats['attempted'] += 1
        
        # Get article content
        article_data = self.get_article_content(title)
        if not article_data:
            return None
        
        content = article_data['content']
        
        # Quick quality check
        if len(content.split()) < 30:
            self.stats['rejected_quality'] += 1
            return None
        
        # Calculate Amharic ratio
        amharic_ratio = self.calculate_amharic_ratio(content)
        if amharic_ratio < self.min_amharic_ratio:
            self.stats['rejected_amharic'] += 1
            return None
        
        # Cultural appropriateness check
        if not self.is_culturally_appropriate(title, content):
            self.stats['rejected_safety'] += 1
            return None
        
        # Create article record
        processed_article = {
            'title': article_data['title'],
            'content': content,
            'url': article_data['url'],
            'metadata': {
                'word_count': article_data['word_count'],
                'character_count': article_data['character_count'],
                'amharic_ratio': round(amharic_ratio, 3),
                'collection_timestamp': datetime.now().isoformat(),
                'quality_score': self._calculate_quality_score(article_data, amharic_ratio)
            }
        }
        
        self.stats['collected'] += 1
        
        if self.stats['collected'] % 25 == 0:
            logger.info(f"Progress: {self.stats['collected']}/{self.target_count} articles collected")
        
        return processed_article

    def _calculate_quality_score(self, article_data: Dict, amharic_ratio: float) -> float:
        """Calculate simple quality score"""
        score = 0.0
        
        # Length scoring (0-40 points)
        word_count = article_data['word_count']
        if word_count > 500:
            score += 40
        elif word_count > 200:
            score += 30
        elif word_count > 100:
            score += 20
        else:
            score += 10
        
        # Amharic ratio scoring (0-40 points)
        score += amharic_ratio * 40
        
        # Completeness scoring (0-20 points)
        if len(article_data['content']) > 1000:
            score += 20
        elif len(article_data['content']) > 500:
            score += 15
        else:
            score += 10
        
        return round(score, 1)

    def collect_articles(self) -> List[Dict[str, Any]]:
        """Main collection method - optimized for speed and quality"""
        logger.info(f"Starting optimized collection of {self.target_count} articles")
        start_time = time.time()
        
        collected_articles = []
        processed_titles = set()
        
        # Collection strategies with priorities
        strategies = [
            # Targeted searches for high-quality Ethiopian content
            ('targeted', ['ኢትዮጵያ ታሪክ', 'አማርኛ ስነ ፅሁፍ', 'ኢትዮጵያ ባህል', 'ኢትዮጵያ ጂኦግራፊ'], 0.4),
            # Random articles for diversity
            ('random', [], 0.6)
        ]
        
        for strategy_type, queries, ratio in strategies:
            target_for_strategy = int(self.target_count * ratio)
            
            if strategy_type == 'targeted':
                titles = []
                per_query = target_for_strategy // len(queries)
                
                for query in queries:
                    query_titles = self.search_targeted_articles(query, per_query * 2)
                    titles.extend(query_titles[:per_query])
                    time.sleep(0.5)  # Rate limiting
                    
            else:  # random
                titles = []
                batch_size = 50
                needed_batches = (target_for_strategy * 3) // batch_size  # Get extra for filtering
                
                for _ in range(needed_batches):
                    batch = self.get_random_articles_batch(batch_size)
                    titles.extend(batch)
                    time.sleep(0.3)
                    
                    if len(collected_articles) >= self.target_count:
                        break
            
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
                
                time.sleep(0.1)  # Minimal rate limiting
        
        processing_time = time.time() - start_time
        
        logger.info(f"Collection completed in {processing_time:.1f}s: {len(collected_articles)} articles")
        logger.info(f"Success rate: {100*self.stats['collected']/max(self.stats['attempted'], 1):.1f}%")
        
        return collected_articles

    def save_corpus(self, articles: List[Dict[str, Any]], output_path: str) -> None:
        """Save collected corpus"""
        corpus_data = {
            'articles': articles,
            'collection_metadata': {
                'total_collected': len(articles),
                'target_count': self.target_count,
                'min_amharic_ratio': self.min_amharic_ratio,
                'collection_timestamp': datetime.now().isoformat(),
                'statistics': self.stats,
                'average_quality_score': sum(a['metadata']['quality_score'] for a in articles) / len(articles) if articles else 0,
                'average_amharic_ratio': sum(a['metadata']['amharic_ratio'] for a in articles) / len(articles) if articles else 0
            }
        }
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(corpus_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Corpus saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving corpus: {e}")

def main():
    """Main execution function"""
    collector = OptimizedAmharicCollector(target_count=500, min_amharic_ratio=0.70)
    
    # Collect articles
    articles = collector.collect_articles()
    
    # Save corpus
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/mekdesyared/amharic-hnet-v2/data/raw/optimized_corpus_{timestamp}.json"
    
    collector.save_corpus(articles, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZED AMHARIC CORPUS COLLECTION COMPLETE")
    print("="*60)
    print(f"Articles Collected: {len(articles)}")
    print(f"Target Achievement: {len(articles)}/{collector.target_count} ({100*len(articles)/collector.target_count:.1f}%)")
    
    if articles:
        avg_quality = sum(a['metadata']['quality_score'] for a in articles) / len(articles)
        avg_amharic = sum(a['metadata']['amharic_ratio'] for a in articles) / len(articles)
        print(f"Average Quality Score: {avg_quality:.1f}/100")
        print(f"Average Amharic Ratio: {avg_amharic:.1%}")
    
    print(f"Statistics: {collector.stats}")
    print(f"Corpus saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()