#!/usr/bin/env python3
"""
Fast Batch Amharic Corpus Collector
Optimized for speed with concurrent processing and batch operations
"""

import json
import time
import asyncio
import aiohttp
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastBatchCollector:
    """Fast batch collector using async operations"""
    
    def __init__(self, target_count: int = 500, min_amharic_ratio: float = 0.70):
        self.target_count = target_count
        self.min_amharic_ratio = min_amharic_ratio
        self.collected_articles = []
        self.processed_titles = set()
        
        # Statistics
        self.stats = {
            'attempted': 0, 'collected': 0, 'rejected_amharic': 0,
            'rejected_quality': 0, 'rejected_duplicates': 0
        }

    def calculate_amharic_ratio(self, text: str) -> float:
        """Calculate Amharic character ratio"""
        if not text:
            return 0.0
        
        amharic_chars = len(re.findall(r'[\u1200-\u137F]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        return amharic_chars / max(total_chars, 1)

    async def get_random_articles_batch_async(self, session: aiohttp.ClientSession, count: int = 50) -> List[str]:
        """Get batch of random articles asynchronously"""
        try:
            url = "https://am.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'generator': 'random',
                'grnnamespace': 0,
                'grnlimit': min(count, 50)
            }
            
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = []
                    
                    if 'query' in data and 'pages' in data['query']:
                        for page_id, page_info in data['query']['pages'].items():
                            if 'title' in page_info:
                                articles.append(page_info['title'])
                    
                    return articles
                    
        except Exception as e:
            logger.debug(f"Error getting random articles: {e}")
        
        return []

    async def get_article_content_async(self, session: aiohttp.ClientSession, title: str) -> Optional[Dict[str, Any]]:
        """Get article content asynchronously"""
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
            
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                pages = data.get('query', {}).get('pages', {})
                
                for page_id, page_data in pages.items():
                    if page_id == '-1':
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

    async def process_article_async(self, session: aiohttp.ClientSession, title: str) -> Optional[Dict[str, Any]]:
        """Process article asynchronously"""
        if title in self.processed_titles:
            self.stats['rejected_duplicates'] += 1
            return None
        
        self.processed_titles.add(title)
        self.stats['attempted'] += 1
        
        # Get article content
        article_data = await self.get_article_content_async(session, title)
        if not article_data:
            return None
        
        content = article_data['content']
        
        # Quality check
        if len(content.split()) < 30:
            self.stats['rejected_quality'] += 1
            return None
        
        # Amharic ratio check
        amharic_ratio = self.calculate_amharic_ratio(content)
        if amharic_ratio < self.min_amharic_ratio:
            self.stats['rejected_amharic'] += 1
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
        return processed_article

    def _calculate_quality_score(self, article_data: Dict, amharic_ratio: float) -> float:
        """Calculate quality score"""
        score = 0.0
        
        # Length scoring
        word_count = article_data['word_count']
        if word_count > 500:
            score += 40
        elif word_count > 200:
            score += 30
        else:
            score += 20
        
        # Amharic ratio scoring
        score += amharic_ratio * 40
        
        # Content length scoring
        if len(article_data['content']) > 1000:
            score += 20
        else:
            score += 10
        
        return round(score, 1)

    async def collect_batch_async(self, session: aiohttp.ClientSession, batch_size: int = 20) -> List[Dict[str, Any]]:
        """Collect a batch of articles asynchronously"""
        # Get titles
        titles = await self.get_random_articles_batch_async(session, batch_size * 2)
        
        # Process articles concurrently
        tasks = [self.process_article_async(session, title) for title in titles[:batch_size]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid articles
        articles = [result for result in results 
                   if isinstance(result, dict) and result is not None]
        
        return articles

    async def collect_all_async(self) -> List[Dict[str, Any]]:
        """Main async collection method"""
        logger.info(f"Starting fast batch collection of {self.target_count} articles")
        start_time = time.time()
        
        connector = aiohttp.TCPConnector(limit=10)  # Limit concurrent connections
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            batch_size = 20
            
            while len(self.collected_articles) < self.target_count:
                try:
                    # Collect batch
                    batch_articles = await self.collect_batch_async(session, batch_size)
                    self.collected_articles.extend(batch_articles)
                    
                    # Progress report
                    if len(self.collected_articles) % 50 == 0 or len(batch_articles) > 0:
                        logger.info(f"Progress: {len(self.collected_articles)}/{self.target_count} articles collected")
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                    # Safety check to avoid infinite loops
                    if self.stats['attempted'] > self.target_count * 5:
                        logger.warning("Too many attempts, stopping collection")
                        break
                        
                except Exception as e:
                    logger.error(f"Error in batch collection: {e}")
                    await asyncio.sleep(1)
        
        processing_time = time.time() - start_time
        logger.info(f"Fast collection completed in {processing_time:.1f}s: {len(self.collected_articles)} articles")
        
        return self.collected_articles

    def collect_sync(self) -> List[Dict[str, Any]]:
        """Synchronous wrapper for async collection"""
        return asyncio.run(self.collect_all_async())

    def save_corpus(self, articles: List[Dict[str, Any]], output_path: str) -> None:
        """Save collected corpus"""
        corpus_data = {
            'articles': articles,
            'collection_metadata': {
                'total_collected': len(articles),
                'target_count': self.target_count,
                'min_amharic_ratio': self.min_amharic_ratio,
                'collection_timestamp': datetime.now().isoformat(),
                'collector_type': 'fast_batch_async',
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
    collector = FastBatchCollector(target_count=500, min_amharic_ratio=0.70)
    
    # Collect articles
    articles = collector.collect_sync()
    
    # Save corpus
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/mekdesyared/amharic-hnet-v2/data/raw/fast_batch_corpus_{timestamp}.json"
    
    collector.save_corpus(articles, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("FAST BATCH AMHARIC CORPUS COLLECTION COMPLETE")
    print("="*60)
    print(f"Articles Collected: {len(articles)}")
    print(f"Target Achievement: {len(articles)}/{collector.target_count} ({100*len(articles)/collector.target_count:.1f}%)")
    
    if articles:
        avg_quality = sum(a['metadata']['quality_score'] for a in articles) / len(articles)
        avg_amharic = sum(a['metadata']['amharic_ratio'] for a in articles) / len(articles)
        print(f"Average Quality Score: {avg_quality:.1f}/100")
        print(f"Average Amharic Ratio: {avg_amharic:.1%}")
    
    print(f"Collection Statistics: {collector.stats}")
    print(f"Corpus saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()