#!/usr/bin/env python3
"""
Final H-Net Corpus Collector
Focused, efficient collection targeting 500+ high-quality Amharic articles
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

class FinalHNetCorpusCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AmharicHNetCollector/1.0'})
        
        # Target high-quality articles
        self.quality_articles = [
            # Core Ethiopian topics
            'ኢትዮጵያ', 'አዲስ አበባ', 'አማርኛ', 'ሐረር', 'ጎንደር', 'ባህርዳር', 'ዓክሱም', 'ላሊበላ',
            'ደሴ', 'ኣዋሳ', 'ምክትል', 'ጅማ', 'ሸዋ', 'ኦሮሚያ', 'አማራ', 'ትግራይ', 'ደቡብ',
            
            # Culture & Religion
            'ኢትዮጵያ ኦርቶዶክስ ተዋሕዶ', 'እስልምና', 'ክርስትና', 'ቡና', 'እንጀራ', 'ዶሮ ወጥ',
            'ባህል', 'ትውፊት', 'ገና', 'ፋሲካ', 'መስቀል', 'ኢርጎ', 'ወርቅ ሰዓት',
            
            # Languages
            'እንግሊዝኛ', 'አረብኛ', 'ኦሮምኛ', 'ትግርኛ', 'ወላይትኛ', 'ጉራጌኛ', 'ሐረርኛ',
            
            # History & People
            'የኢትዮጵያ ታሪክ', 'ዓፄ ምኒልክ', 'ዓፄ ሃይለ ሥላሴ', 'ዓፄ ቴዎድሮስ', 'ዓፄ ዮሃንስ',
            'ኣፄ ተክለ ሃይማኖት', 'ዳግማዊ ምኒልክ', 'አፄ ኮንስታንቲን', 'መኮንን እንዳልካቸው',
            
            # Literature & Arts  
            'ማኅተመ ሥላሴ ወልደ መስቀል', 'አፈወርቅ ተክሌ', 'ደመቀ ከበደ', 'ገብረ ክርስቶስ',
            'መባዓ ጽዮን', 'ስነ ጽሁፍ', 'ሙዚቃ', 'ስነ ውበት', 'ቅዱስ ዮሐንስ',
            
            # Science & Education
            'ሳይንስ', 'ጤንነት', 'ሕክምና', 'ትምህርት', 'ዩኒቨርስቲ', 'አዲስ አበባ ዩኒቨርስቲ',
            'ሃኪም', 'ነርስ', 'መምህር', 'ተማሪ', 'መዋለ ልጆች',
            
            # Geography & Nature
            'ኢትዮጵያ ጂኦግራፊ', 'ቀይ ባሕር', 'ኣባይ ወንዝ', 'ታና ሐይቅ', 'ሰሜን ተራራ',
            'ዳናኪል', 'ሪፍት ቫሊ', 'ሰሜን', 'ደቡብ', 'ምሥራቅ', 'ምእራብ',
            
            # Modern topics
            'ኢንተርኔት', 'ሞባይል', 'ኮምፒዩተር', 'ቴሌቪዥን', 'ሬዲዮ', 'ጋዜጣ', 'ወረት'
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
            
            response = self.session.get(url, params=params, timeout=10)
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
                
                if content and len(content.strip()) > 150:
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

    def get_random_articles(self, count: int = 50) -> List[str]:
        try:
            url = "https://am.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'generator': 'random',
                'grnnamespace': 0,
                'grnlimit': min(count, 50)
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'query' in data and 'pages' in data['query']:
                    return [page_info.get('title', '') for page_info in data['query']['pages'].values() 
                           if page_info.get('title')]
        except Exception as e:
            logger.debug(f"Error getting random articles: {e}")
        return []

    def search_articles(self, query: str, limit: int = 30) -> List[str]:
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
                if 'query' in data and 'search' in data['query']:
                    return [result['title'] for result in data['query']['search']]
        except Exception as e:
            logger.debug(f"Error searching {query}: {e}")
        return []

    def process_article(self, title: str) -> Optional[Dict[str, Any]]:
        article_data = self.get_article_content(title)
        if not article_data:
            return None
        
        content = article_data['content']
        
        # Quality checks
        if len(content.split()) < 30:
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

    def collect_corpus(self, target: int = 500) -> List[Dict[str, Any]]:
        logger.info(f"Starting targeted collection of {target} articles")
        articles = []
        processed = set()
        
        # 1. Process quality articles (40%)
        logger.info("Phase 1: Processing quality articles")
        quality_target = int(target * 0.4)
        for title in self.quality_articles:
            if len(articles) >= quality_target:
                break
            if title not in processed:
                processed.add(title)
                article = self.process_article(title)
                if article:
                    articles.append(article)
                    if len(articles) % 10 == 0:
                        logger.info(f"Quality articles: {len(articles)}/{quality_target}")
                time.sleep(0.3)
        
        # 2. Search expansion (30%)
        logger.info("Phase 2: Search expansion")
        search_target = int(target * 0.3)
        search_queries = ['ኢትዮጵያ', 'ባህል', 'ታሪክ', 'ቋንቋ', 'ሳይንስ', 'ሙዚቃ', 'ስፖርት']
        
        for query in search_queries:
            if len(articles) >= search_target:
                break
            search_results = self.search_articles(query, 20)
            for title in search_results:
                if len(articles) >= search_target:
                    break
                if title not in processed:
                    processed.add(title)
                    article = self.process_article(title)
                    if article:
                        articles.append(article)
                        if len(articles) % 25 == 0:
                            logger.info(f"Search articles: {len(articles)}/{search_target}")
                    time.sleep(0.2)
            time.sleep(0.5)
        
        # 3. Random discovery (30%)
        logger.info("Phase 3: Random discovery")
        while len(articles) < target:
            random_titles = self.get_random_articles(40)
            for title in random_titles:
                if len(articles) >= target:
                    break
                if title not in processed:
                    processed.add(title)
                    article = self.process_article(title)
                    if article:
                        articles.append(article)
                        if len(articles) % 25 == 0:
                            logger.info(f"Total articles: {len(articles)}/{target}")
                    time.sleep(0.1)
            time.sleep(0.3)
        
        logger.info(f"Collection complete: {len(articles)} articles")
        return articles[:target]  # Ensure we don't exceed target

    def save_corpus(self, articles: List[Dict[str, Any]]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/Users/mekdesyared/amharic-hnet-v2/data/raw/hnet_production_corpus_{timestamp}.json"
        
        corpus_data = {
            'articles': articles,
            'collection_metadata': {
                'total_articles': len(articles),
                'collection_timestamp': datetime.now().isoformat(),
                'collector': 'final_hnet_corpus_collector',
                'target_achieved': len(articles),
                'average_amharic_ratio': sum(a['metadata']['amharic_ratio'] for a in articles) / len(articles) if articles else 0,
                'average_quality_score': sum(a['metadata']['quality_score'] for a in articles) / len(articles) if articles else 0,
                'total_words': sum(a['metadata']['word_count'] for a in articles),
                'total_characters': sum(a['metadata']['character_count'] for a in articles)
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        return output_path

def main():
    collector = FinalHNetCorpusCollector()
    articles = collector.collect_corpus(500)
    output_path = collector.save_corpus(articles)
    
    # Summary
    if articles:
        avg_amharic = sum(a['metadata']['amharic_ratio'] for a in articles) / len(articles)
        avg_quality = sum(a['metadata']['quality_score'] for a in articles) / len(articles)
        total_words = sum(a['metadata']['word_count'] for a in articles)
        
        print("\n" + "="*60)
        print("H-NET PRODUCTION CORPUS COLLECTION COMPLETE")
        print("="*60)
        print(f"Articles Collected: {len(articles)}")
        print(f"Average Amharic Ratio: {avg_amharic:.1%}")
        print(f"Average Quality Score: {avg_quality:.1f}/100")
        print(f"Total Words: {total_words:,}")
        print(f"Corpus saved to: {output_path}")
        print("="*60)

if __name__ == "__main__":
    main()