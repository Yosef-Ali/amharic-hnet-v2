#!/usr/bin/env python3
"""
Production Amharic Corpus Collector
Reliable, efficient corpus collection for H-Net training
"""

import json
import time
import re
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionAmharicCollector:
    """Production-ready Amharic corpus collector"""
    
    def __init__(self, target_count: int = 500, min_amharic_ratio: float = 0.70):
        self.target_count = target_count
        self.min_amharic_ratio = min_amharic_ratio
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AmharicCorpusCollector/1.0'})
        
        # Track progress
        self.collected_articles = []
        self.processed_titles = set()
        self.stats = {
            'attempted': 0, 'collected': 0, 'rejected_amharic': 0,
            'rejected_quality': 0, 'rejected_duplicates': 0
        }
        
        # Predefined high-quality article titles from Amharic Wikipedia
        self.seed_articles = [
            'ኢትዮጵያ', 'አማርኛ', 'አዲስ አበባ', 'ሐረር', 'ጎንደር', 'ባህርዳር', 'ዓክሱም',
            'ላሊበላ', 'እንግሊዝኛ', 'ቡና', 'እንጀራ', 'የኢትዮጵያ ታሪክ', 'ክርስትና',
            'እስልምና', 'ቤተ ክርስቲያን', 'ደቡብ ኢትዮጵያ', 'ኦሮሚያ', 'አማራ ክልል',
            'ትግራይ', 'ኦሮምኛ', 'ትግርኛ', 'ወላይትኛ', 'ሙዚቃ', 'ስፖርት', 'ሳይንስ'
        ]

    def calculate_amharic_ratio(self, text: str) -> float:
        """Calculate Amharic character ratio"""
        if not text:
            return 0.0
        
        amharic_chars = len(re.findall(r'[\u1200-\u137F]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        return amharic_chars / max(total_chars, 1)

    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Get article content with error handling"""
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
            
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code != 200:
                return None
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page_data in pages.items():
                if page_id == '-1':
                    continue
                
                title = page_data.get('title', '')
                content = page_data.get('extract', '')
                url = page_data.get('fullurl', '')
                
                if not content or len(content.strip()) < 200:
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

    def get_random_articles(self, count: int = 50) -> List[str]:
        """Get random articles efficiently"""
        try:
            url = "https://am.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'generator': 'random',
                'grnnamespace': 0,
                'grnlimit': min(count, 50)
            }
            
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                if 'query' in data and 'pages' in data['query']:
                    for page_id, page_info in data['query']['pages'].items():
                        if 'title' in page_info:
                            articles.append(page_info['title'])
                
                return articles
            
        except Exception as e:
            logger.debug(f"Error getting random articles: {e}")
        
        return []

    def search_category_articles(self, category: str, limit: int = 30) -> List[str]:
        """Search articles in category"""
        try:
            url = "https://am.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': f'incategory:"{category}"',
                'srlimit': limit,
                'srnamespace': 0
            }
            
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                if 'query' in data and 'search' in data['query']:
                    for result in data['query']['search']:
                        articles.append(result['title'])
                
                return articles
            
        except Exception as e:
            logger.debug(f"Error searching category '{category}': {e}")
        
        return []

    def get_related_articles(self, title: str, limit: int = 10) -> List[str]:
        """Get articles related to given title"""
        try:
            url = "https://am.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': title,
                'srlimit': limit,
                'srnamespace': 0
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                if 'query' in data and 'search' in data['query']:
                    for result in data['query']['search']:
                        if result['title'] != title:  # Exclude the original
                            articles.append(result['title'])
                
                return articles[:limit]
            
        except Exception as e:
            logger.debug(f"Error getting related articles for '{title}': {e}")
        
        return []

    def process_article(self, title: str) -> Optional[Dict[str, Any]]:
        """Process single article"""
        if title in self.processed_titles:
            self.stats['rejected_duplicates'] += 1
            return None
        
        self.processed_titles.add(title)
        self.stats['attempted'] += 1
        
        # Get content
        article_data = self.get_article_content(title)
        if not article_data:
            return None
        
        content = article_data['content']
        
        # Quality checks
        if len(content.split()) < 40:
            self.stats['rejected_quality'] += 1
            return None
        
        # Amharic ratio check
        amharic_ratio = self.calculate_amharic_ratio(content)
        if amharic_ratio < self.min_amharic_ratio:
            self.stats['rejected_amharic'] += 1
            return None
        
        # Create processed article
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
        return processed_article

    def _calculate_quality_score(self, article_data: Dict, amharic_ratio: float) -> float:
        """Calculate quality score"""
        score = 0.0
        
        # Length scoring (0-40)
        word_count = article_data['word_count']
        if word_count > 500:
            score += 40
        elif word_count > 200:
            score += 30
        elif word_count > 100:
            score += 20
        else:
            score += 10
        
        # Amharic ratio scoring (0-40)
        score += amharic_ratio * 40
        
        # Content depth scoring (0-20)
        if len(article_data['content']) > 2000:
            score += 20
        elif len(article_data['content']) > 1000:
            score += 15
        else:
            score += 10
        
        return round(score, 1)

    def collect_articles(self) -> List[Dict[str, Any]]:
        """Main collection method"""
        logger.info(f"Starting production collection of {self.target_count} articles")
        start_time = time.time()
        
        # Collection strategies with weights
        collection_rounds = [
            ("seed_articles", self.seed_articles, 0.15),  # 15% from seed articles
            ("related_expansion", [], 0.25),  # 25% from related articles
            ("random_discovery", [], 0.60)    # 60% from random discovery
        ]
        
        for strategy, source_list, weight in collection_rounds:
            target_for_round = int(self.target_count * weight)
            logger.info(f"Collection round: {strategy} (target: {target_for_round})")
            
            if strategy == "seed_articles":
                # Process seed articles directly
                for title in source_list:
                    if len(self.collected_articles) >= target_for_round:
                        break
                    
                    article = self.process_article(title)
                    if article:
                        self.collected_articles.append(article)
                    
                    time.sleep(0.2)
            
            elif strategy == "related_expansion":
                # Get related articles from successful seed articles
                seed_titles = [a['title'] for a in self.collected_articles[:10]]  # Use first 10 successful
                
                for seed_title in seed_titles:
                    if len(self.collected_articles) >= target_for_round:
                        break
                    
                    related_titles = self.get_related_articles(seed_title, 5)
                    for title in related_titles:
                        if len(self.collected_articles) >= target_for_round:
                            break
                        
                        article = self.process_article(title)
                        if article:
                            self.collected_articles.append(article)
                        
                        time.sleep(0.2)
                    
                    time.sleep(0.5)
            
            else:  # random_discovery
                # Random article collection
                attempts = 0
                max_attempts = target_for_round * 3
                
                while len(self.collected_articles) < target_for_round and attempts < max_attempts:
                    random_titles = self.get_random_articles(30)
                    
                    for title in random_titles:
                        if len(self.collected_articles) >= target_for_round:
                            break
                        
                        article = self.process_article(title)
                        if article:
                            self.collected_articles.append(article)
                        
                        attempts += 1
                        time.sleep(0.1)
                    
                    # Progress reporting
                    if len(self.collected_articles) % 25 == 0:
                        logger.info(f"Progress: {len(self.collected_articles)}/{self.target_count} articles")
                    
                    time.sleep(0.3)
        
        processing_time = time.time() - start_time
        success_rate = 100 * self.stats['collected'] / max(self.stats['attempted'], 1)
        
        logger.info(f"Collection completed in {processing_time:.1f}s")
        logger.info(f"Collected: {len(self.collected_articles)} articles")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        return self.collected_articles

    def save_corpus(self, articles: List[Dict[str, Any]], output_path: str) -> None:
        """Save corpus with metadata"""
        corpus_data = {
            'articles': articles,
            'collection_metadata': {
                'total_collected': len(articles),
                'target_count': self.target_count,
                'min_amharic_ratio': self.min_amharic_ratio,
                'collection_timestamp': datetime.now().isoformat(),
                'collector_type': 'production_reliable',
                'statistics': self.stats
            }
        }
        
        if articles:
            corpus_data['collection_metadata'].update({
                'average_quality_score': sum(a['metadata']['quality_score'] for a in articles) / len(articles),
                'average_amharic_ratio': sum(a['metadata']['amharic_ratio'] for a in articles) / len(articles),
                'total_words': sum(a['metadata']['word_count'] for a in articles),
                'total_characters': sum(a['metadata']['character_count'] for a in articles)
            })
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(corpus_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Corpus saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving corpus: {e}")

def main():
    """Main execution"""
    # Create collector targeting 500 articles
    collector = ProductionAmharicCollector(target_count=500, min_amharic_ratio=0.70)
    
    # Collect articles
    articles = collector.collect_articles()
    
    # Save corpus
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/mekdesyared/amharic-hnet-v2/data/raw/production_corpus_{timestamp}.json"
    
    collector.save_corpus(articles, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("PRODUCTION AMHARIC CORPUS COLLECTION COMPLETE")
    print("="*60)
    print(f"Articles Collected: {len(articles)}")
    print(f"Target Achievement: {len(articles)}/{collector.target_count} ({100*len(articles)/collector.target_count:.1f}%)")
    
    if articles:
        metadata = articles[0]['metadata'] if articles else {}
        avg_quality = sum(a['metadata']['quality_score'] for a in articles) / len(articles)
        avg_amharic = sum(a['metadata']['amharic_ratio'] for a in articles) / len(articles)
        total_words = sum(a['metadata']['word_count'] for a in articles)
        
        print(f"Average Quality Score: {avg_quality:.1f}/100")
        print(f"Average Amharic Ratio: {avg_amharic:.1%}")
        print(f"Total Words: {total_words:,}")
    
    print(f"Collection Statistics: {collector.stats}")
    print(f"Corpus saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()