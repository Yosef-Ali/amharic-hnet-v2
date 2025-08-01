#!/usr/bin/env python3
"""
Quick Completion Collector
Completes the corpus to 500+ articles by merging existing data and collecting additional articles
"""

import json
import time
import re
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickCompletionCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AmharicQuickCollector/1.0'})
        
        # High-success rate articles
        self.quick_articles = [
            'አማርኛ', 'ኢትዮጵያ', 'አዲስ አበባ', 'ቡና', 'እንጀራ', 'ክርስትና', 'እስልምና',
            'ሐረር', 'ጎንደር', 'ባህል', 'ታሪክ', 'ሙዚቃ', 'ወጥ', 'ዶሮ', 'ቤተሰብ',
            'ትምህርት', 'መምህር', 'ተማሪ', 'ዶክተር', 'ነርስ', 'ገበሬ', 'ነጋዴ',
            'ስዕል', 'ቅጥዓት', 'ሕግ', 'ፖለቲካ', 'ኢኮኖሚ', 'ማህበረሰብ', 'ቤተ ክርስቲያን',
            'መስጊድ', 'መምህር', 'ዩኒቨርስቲ', 'ኮሌጅ', 'ትምህርት ቤት', 'ሆስፒታል',
            'ወንዝ', 'ተራራ', 'ሐይቅ', 'ሜዳ', 'ጫካ', 'ዳዓል', 'ዛፍ', 'አበባ'
        ]

    def calculate_amharic_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        amharic_chars = len(re.findall(r'[\u1200-\u137F]', text))
        total_chars = len(re.sub(r'\s', '', text))
        return amharic_chars / max(total_chars, 1)

    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
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
            
            response = self.session.get(url, params=params, timeout=8)
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
                
                if content and len(content.strip()) > 100:
                    return {
                        'title': title,
                        'content': content,
                        'url': url,
                        'word_count': len(content.split()),
                        'character_count': len(content)
                    }
        except Exception as e:
            logger.debug(f"Error getting {title}: {e}")
        return None

    def get_random_articles(self, count: int = 30) -> List[str]:
        try:
            url = "https://am.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'generator': 'random',
                'grnnamespace': 0,
                'grnlimit': min(count, 50)
            }
            
            response = self.session.get(url, params=params, timeout=8)
            if response.status_code == 200:
                data = response.json()
                if 'query' in data and 'pages' in data['query']:
                    return [page_info.get('title', '') for page_info in data['query']['pages'].values() 
                           if page_info.get('title')]
        except Exception as e:
            logger.debug(f"Error getting random articles: {e}")
        return []

    def process_article(self, title: str) -> Optional[Dict[str, Any]]:
        article_data = self.get_article_content(title)
        if not article_data:
            return None
        
        content = article_data['content']
        
        # Quick quality checks
        if len(content.split()) < 25:
            return None
        
        # Amharic ratio check
        amharic_ratio = self.calculate_amharic_ratio(content)
        if amharic_ratio < 0.70:
            return None
        
        return {
            'title': article_data['title'],
            'content': content,
            'url': article_data['url'],
            'metadata': {
                'word_count': article_data['word_count'],
                'character_count': article_data['character_count'],
                'amharic_ratio': round(amharic_ratio, 3),
                'collection_timestamp': datetime.now().isoformat(),
                'quality_score': min(100, (amharic_ratio * 50) + (min(article_data['word_count'], 500) * 0.1))
            }
        }

    def load_existing_corpus(self) -> List[Dict[str, Any]]:
        """Load and merge existing corpus files"""
        data_dir = Path("/Users/mekdesyared/amharic-hnet-v2/data/raw")
        all_articles = []
        seen_titles = set()
        
        corpus_files = [
            'comprehensive_hnet_corpus.json',
            'hnet_demo_corpus.json', 
            'premium_hnet_demo.json',
            'test_corpus.json'
        ]
        
        for filename in corpus_files:
            file_path = data_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    articles = data.get('articles', []) if isinstance(data, dict) else data
                    
                    for article in articles:
                        if isinstance(article, dict) and 'title' in article:
                            title = article['title']
                            if title not in seen_titles:
                                seen_titles.add(title)
                                # Ensure proper structure
                                if 'metadata' not in article:
                                    article['metadata'] = {
                                        'word_count': len(article.get('content', '').split()),
                                        'character_count': len(article.get('content', '')),
                                        'amharic_ratio': self.calculate_amharic_ratio(article.get('content', '')),
                                        'collection_timestamp': datetime.now().isoformat(),
                                        'quality_score': 75.0
                                    }
                                all_articles.append(article)
                    
                    logger.info(f"Loaded {len(articles)} articles from {filename}")
                    
                except Exception as e:
                    logger.warning(f"Error loading {filename}: {e}")
        
        logger.info(f"Total existing articles loaded: {len(all_articles)}")
        return all_articles

    def collect_additional_articles(self, existing_count: int, target: int = 500) -> List[Dict[str, Any]]:
        """Collect additional articles to reach target"""
        needed = max(0, target - existing_count)
        logger.info(f"Need {needed} additional articles to reach {target}")
        
        if needed <= 0:
            return []
        
        articles = []
        processed = set()
        
        # Quick targeted collection
        logger.info("Collecting targeted articles...")
        for title in self.quick_articles:
            if len(articles) >= needed:
                break
            if title not in processed:
                processed.add(title)
                article = self.process_article(title)
                if article:
                    articles.append(article)
                    logger.info(f"Collected: {title} ({len(articles)}/{needed})")
                time.sleep(0.2)
        
        # Random collection to fill remaining
        logger.info("Collecting random articles...")
        attempts = 0
        max_attempts = needed * 3
        
        while len(articles) < needed and attempts < max_attempts:
            random_titles = self.get_random_articles(20)
            for title in random_titles:
                if len(articles) >= needed or attempts >= max_attempts:
                    break
                if title not in processed:
                    processed.add(title)
                    article = self.process_article(title)
                    if article:
                        articles.append(article)
                        if len(articles) % 10 == 0:
                            logger.info(f"Random articles: {len(articles)}/{needed}")
                    attempts += 1
                    time.sleep(0.1)
            time.sleep(0.2)
        
        logger.info(f"Collected {len(articles)} additional articles")
        return articles

    def create_final_corpus(self, target: int = 500) -> str:
        """Create final comprehensive corpus"""
        logger.info("Creating final comprehensive corpus...")
        
        # Load existing articles
        existing_articles = self.load_existing_corpus()
        
        # Collect additional if needed
        additional_articles = self.collect_additional_articles(len(existing_articles), target)
        
        # Combine all articles
        all_articles = existing_articles + additional_articles
        
        # Sort by quality score (if available)
        all_articles.sort(key=lambda x: x.get('metadata', {}).get('quality_score', 0), reverse=True)
        
        # Take top articles up to target
        final_articles = all_articles[:target]
        
        # Create final corpus
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/Users/mekdesyared/amharic-hnet-v2/data/raw/final_hnet_corpus_{timestamp}.json"
        
        corpus_data = {
            'articles': final_articles,
            'collection_metadata': {
                'total_articles': len(final_articles),
                'collection_timestamp': datetime.now().isoformat(),
                'collector': 'quick_completion_collector',
                'existing_articles_used': len(existing_articles),
                'new_articles_collected': len(additional_articles),
                'target_achieved': len(final_articles),
                'average_amharic_ratio': sum(a['metadata']['amharic_ratio'] for a in final_articles) / len(final_articles) if final_articles else 0,
                'average_quality_score': sum(a['metadata'].get('quality_score', 75) for a in final_articles) / len(final_articles) if final_articles else 0,
                'total_words': sum(a['metadata']['word_count'] for a in final_articles),
                'total_characters': sum(a['metadata']['character_count'] for a in final_articles)
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        return output_path

def main():
    collector = QuickCompletionCollector()
    output_path = collector.create_final_corpus(500)
    
    # Load and display summary
    with open(output_path, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    metadata = corpus_data['collection_metadata']
    
    print("\n" + "="*70)
    print("FINAL H-NET AMHARIC CORPUS COLLECTION COMPLETE")
    print("="*70)
    print(f"Total Articles: {metadata['total_articles']}")
    print(f"Existing Articles Used: {metadata['existing_articles_used']}")
    print(f"New Articles Collected: {metadata['new_articles_collected']}")
    print(f"Average Amharic Ratio: {metadata['average_amharic_ratio']:.1%}")
    print(f"Average Quality Score: {metadata['average_quality_score']:.1f}/100")
    print(f"Total Words: {metadata['total_words']:,}")
    print(f"Total Characters: {metadata['total_characters']:,}")
    print(f"Corpus saved to: {output_path}")
    print("="*70)
    print("\nCORPUS READY FOR H-NET TRAINING!")

if __name__ == "__main__":
    main()