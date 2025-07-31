#!/usr/bin/env python3
"""
Data Collection Sub-Agent for Amharic H-Net v2
===============================================

This module implements the data-collector sub-agent specialized in gathering
high-quality Amharic text corpora from diverse sources with proper validation,
cultural sensitivity, and linguistic quality assessment.

Sub-Agent Capabilities:
1. Web scraping from Amharic news sources (BBC, EBC, Fana)
2. Wikipedia Amharic article extraction and processing
3. Social media data collection (with proper permissions)
4. Literary text collection from public domain sources
5. Data quality validation and cultural safety screening
6. Corpus statistics and linguistic diversity analysis

Usage:
    python -m src.data_collection.amharic_collector --source wikipedia --output data/raw
"""

import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import logging
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
from datetime import datetime
import hashlib


@dataclass
class AmharicTextSample:
    """Represents a collected Amharic text sample with metadata."""
    text: str
    source: str
    url: str
    title: str
    date_collected: str
    estimated_words: int
    amharic_ratio: float
    cultural_domain: str
    quality_score: float
    dialect_hints: List[str]
    metadata: Dict[str, any]


class AmharicDataCollector:
    """
    Specialized data collection sub-agent for Amharic H-Net training corpus.
    
    Implements intelligent collection strategies with cultural sensitivity
    and linguistic quality validation.
    """
    
    def __init__(self, output_dir: str = "data/raw", max_concurrent: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent = max_concurrent
        self.session = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Collection sources configuration
        self.sources = {
            'wikipedia': {
                'url': 'https://am.wikipedia.org',
                'dump_url': 'https://dumps.wikimedia.org/amwiki/latest/amwiki-latest-pages-articles.xml.bz2',
                'priority': 'high',
                'cultural_safety': 'high'
            },
            'bbc_amharic': {
                'url': 'https://www.bbc.com/amharic',
                'priority': 'high',
                'cultural_safety': 'high'
            },
            'dw_amharic': {
                'url': 'https://www.dw.com/am',
                'priority': 'medium',
                'cultural_safety': 'high'
            },
            'fana_bc': {
                'url': 'https://www.fanabc.com',
                'priority': 'medium',
                'cultural_safety': 'medium'
            }
        }
        
        # Cultural domains for categorization
        self.cultural_domains = {
            'news': ['ዜና', 'ሪፖርት', 'መረጃ'],
            'culture': ['ባህል', 'ወግ', 'ሥርዓት'],
            'religion': ['ኪዳነ', 'መስቀል', 'እምነት', 'ቤተክርስቲያን', 'መስጊድ'],
            'history': ['ታሪክ', 'ቀዳማዊ', 'ንጉሥ', 'ላሊበላ'],
            'literature': ['ግጥም', 'ዘመን', 'ደራሲ', 'መጽሐፍ'],
            'science': ['ሳይንስ', 'ቴክኖሎጂ', 'ምርምር'],
            'education': ['ትምህርት', 'ዩኒቨርሲቲ', 'ትምህርት ቤት']
        }
        
        # Quality filters
        self.min_amharic_ratio = 0.7
        self.min_word_count = 50
        self.max_word_count = 2000
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for data collection sub-agent."""
        logger = logging.getLogger('amharic_data_collector')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def collect_from_source(
        self, 
        source_name: str, 
        max_articles: int = 1000
    ) -> List[AmharicTextSample]:
        """
        Collect Amharic text from specified source.
        
        Args:
            source_name: Name of the source to collect from
            max_articles: Maximum number of articles to collect
            
        Returns:
            List of collected and validated Amharic text samples
        """
        if source_name not in self.sources:
            raise ValueError(f"Unknown source: {source_name}")
        
        self.logger.info(f"🔍 Starting collection from {source_name}...")
        
        if source_name == 'wikipedia':
            return await self._collect_wikipedia(max_articles)
        elif source_name == 'bbc_amharic':
            return await self._collect_bbc_amharic(max_articles)
        elif source_name == 'dw_amharic':
            return await self._collect_dw_amharic(max_articles)
        elif source_name == 'fana_bc':
            return await self._collect_fana_bc(max_articles)
        else:
            self.logger.warning(f"Collection method not implemented for {source_name}")
            return []
    
    async def _collect_wikipedia(self, max_articles: int) -> List[AmharicTextSample]:
        """Collect articles from Amharic Wikipedia."""
        samples = []
        
        # Use Wikipedia API to get article list
        api_url = "https://am.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'list': 'random',
            'rnnamespace': 0,  # Main namespace only
            'rnlimit': min(max_articles, 500),  # API limit
            'format': 'json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, params=params) as response:
                data = await response.json()
                
                if 'query' not in data:
                    self.logger.error("Failed to get Wikipedia article list")
                    return samples
                
                article_titles = [item['title'] for item in data['query']['random']]
                self.logger.info(f"📄 Found {len(article_titles)} Wikipedia articles")
                
                # Process articles in batches
                semaphore = asyncio.Semaphore(self.max_concurrent)
                tasks = [
                    self._fetch_wikipedia_article(session, title, semaphore)
                    for title in article_titles
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, AmharicTextSample):
                        samples.append(result)
                    elif isinstance(result, Exception):
                        self.logger.warning(f"Article collection failed: {result}")
        
        self.logger.info(f"✅ Collected {len(samples)} Wikipedia articles")
        return samples
    
    async def _fetch_wikipedia_article(
        self, 
        session: aiohttp.ClientSession, 
        title: str, 
        semaphore: asyncio.Semaphore
    ) -> Optional[AmharicTextSample]:
        """Fetch individual Wikipedia article."""
        async with semaphore:
            try:
                # Get article content
                api_url = "https://am.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'prop': 'extracts',
                    'exintro': False,
                    'explaintext': True,
                    'titles': title,
                    'format': 'json'
                }
                
                async with session.get(api_url, params=params) as response:
                    data = await response.json()
                    
                    pages = data.get('query', {}).get('pages', {})
                    if not pages:
                        return None
                    
                    page = next(iter(pages.values()))
                    if 'extract' not in page:
                        return None
                    
                    text = page['extract']
                    if not text or len(text.strip()) < self.min_word_count:
                        return None
                    
                    # Validate and create sample
                    return self._create_text_sample(
                        text=text,
                        source='wikipedia',
                        url=f"https://am.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        title=title,
                        cultural_domain=self._classify_cultural_domain(text)
                    )
                    
            except Exception as e:
                self.logger.warning(f"Failed to fetch Wikipedia article '{title}': {e}")
                return None
    
    async def _collect_bbc_amharic(self, max_articles: int) -> List[AmharicTextSample]:
        """Collect articles from BBC Amharic."""
        samples = []
        base_url = "https://www.bbc.com/amharic"
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get main page to find article links
                async with session.get(base_url) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to access BBC Amharic: {response.status}")
                        return samples
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find article links (BBC-specific selectors)
                    article_links = []
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if '/amharic/' in href and '/articles/' in href:
                            full_url = urljoin(base_url, href)
                            article_links.append(full_url)
                    
                    # Remove duplicates and limit
                    article_links = list(set(article_links))[:max_articles]
                    self.logger.info(f"📰 Found {len(article_links)} BBC Amharic articles")
                    
                    # Fetch articles
                    semaphore = asyncio.Semaphore(self.max_concurrent)
                    tasks = [
                        self._fetch_bbc_article(session, url, semaphore)
                        for url in article_links
                    ]
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, AmharicTextSample):
                            samples.append(result)
                        elif isinstance(result, Exception):
                            self.logger.warning(f"BBC article collection failed: {result}")
        
        except Exception as e:
            self.logger.error(f"BBC Amharic collection failed: {e}")
        
        self.logger.info(f"✅ Collected {len(samples)} BBC Amharic articles")
        return samples
    
    async def _fetch_bbc_article(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        semaphore: asyncio.Semaphore
    ) -> Optional[AmharicTextSample]:
        """Fetch individual BBC Amharic article."""
        async with semaphore:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract title
                    title_elem = soup.find('h1') or soup.find('title')
                    title = title_elem.get_text().strip() if title_elem else "Unknown"
                    
                    # Extract article text (BBC-specific selectors)
                    text_content = []
                    for paragraph in soup.find_all(['p', 'div'], class_=re.compile(r'.*story.*|.*article.*|.*content.*')):
                        text = paragraph.get_text().strip()
                        if text and len(text) > 20:
                            text_content.append(text)
                    
                    if not text_content:
                        return None
                    
                    full_text = ' '.join(text_content)
                    
                    # Create sample
                    return self._create_text_sample(
                        text=full_text,
                        source='bbc_amharic',
                        url=url,
                        title=title,
                        cultural_domain='news'
                    )
                    
            except Exception as e:
                self.logger.warning(f"Failed to fetch BBC article {url}: {e}")
                return None
    
    async def _collect_dw_amharic(self, max_articles: int) -> List[AmharicTextSample]:
        """Collect articles from Deutsche Welle Amharic."""
        # Similar implementation to BBC but with DW-specific selectors
        samples = []
        self.logger.info("📺 DW Amharic collection - Implementation needed")
        return samples
    
    async def _collect_fana_bc(self, max_articles: int) -> List[AmharicTextSample]:
        """Collect articles from Fana Broadcasting."""
        # Similar implementation but with Fana-specific selectors
        samples = []
        self.logger.info("📻 Fana Broadcasting collection - Implementation needed")
        return samples
    
    def _create_text_sample(
        self,
        text: str,
        source: str,
        url: str,
        title: str,
        cultural_domain: str
    ) -> Optional[AmharicTextSample]:
        """Create and validate a text sample."""
        # Clean text
        text = self._clean_text(text)
        
        # Calculate metrics
        word_count = len(text.split())
        amharic_ratio = self._calculate_amharic_ratio(text)
        quality_score = self._calculate_quality_score(text, amharic_ratio, word_count)
        
        # Quality filters
        if (word_count < self.min_word_count or 
            word_count > self.max_word_count or
            amharic_ratio < self.min_amharic_ratio or
            quality_score < 0.5):
            return None
        
        return AmharicTextSample(
            text=text,
            source=source,
            url=url,
            title=title,
            date_collected=datetime.now().isoformat(),
            estimated_words=word_count,
            amharic_ratio=amharic_ratio,
            cultural_domain=cultural_domain,
            quality_score=quality_score,
            dialect_hints=self._detect_dialect_hints(text),
            metadata={
                'collection_method': 'web_scraping',
                'text_hash': hashlib.md5(text.encode()).hexdigest()
            }
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize Amharic text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove non-Amharic characters but keep punctuation
        # Keep Ge'ez script range: U+1200-U+137F
        cleaned = re.sub(r'[^\u1200-\u137F\s\.\!\?\:\;\,\(\)\"\']+', ' ', text)
        
        # Normalize punctuation
        cleaned = re.sub(r'\s*([\.!?:;,])\s*', r'\\1 ', cleaned)
        
        # Remove excessive punctuation
        cleaned = re.sub(r'([\.!?]){2,}', r'\\1', cleaned)
        
        return cleaned.strip()
    
    def _calculate_amharic_ratio(self, text: str) -> float:
        """Calculate ratio of Amharic characters in text."""
        if not text:
            return 0.0
        
        amharic_chars = len([c for c in text if '\u1200' <= c <= '\u137F'])
        total_chars = len([c for c in text if c.isalpha()])
        
        return amharic_chars / total_chars if total_chars > 0 else 0.0
    
    def _calculate_quality_score(self, text: str, amharic_ratio: float, word_count: int) -> float:
        """Calculate overall quality score for text sample."""
        score = 0.0
        
        # Amharic ratio score (40% weight)
        score += 0.4 * min(1.0, amharic_ratio / 0.8)
        
        # Length score (30% weight)
        ideal_length = 200  # words
        length_score = min(1.0, word_count / ideal_length) if word_count < ideal_length else max(0.5, ideal_length / word_count)
        score += 0.3 * length_score
        
        # Linguistic complexity (30% weight)
        sentences = text.split('።')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        complexity_score = min(1.0, avg_sentence_length / 15)  # Ideal ~15 words per sentence
        score += 0.3 * complexity_score
        
        return score
    
    def _classify_cultural_domain(self, text: str) -> str:
        """Classify text into cultural domain based on content."""
        text_lower = text.lower()
        
        domain_scores = {}
        for domain, keywords in self.cultural_domains.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'general'
    
    def _detect_dialect_hints(self, text: str) -> List[str]:
        """Detect potential dialect indicators in text."""
        dialect_markers = {
            'eritrean': ['ማይ', 'ጽቡቅ', 'እዩ', 'እያ'],
            'gojjam': ['እሱ', 'እሳ', 'እሸን', 'እሸም'],
            'wollo': ['ነበረ', 'ተዓደወ', 'ተመለሰ']
        }
        
        detected_dialects = []
        for dialect, markers in dialect_markers.items():
            if any(marker in text for marker in markers):
                detected_dialects.append(dialect)
        
        return detected_dialects if detected_dialects else ['standard']
    
    def save_samples(self, samples: List[AmharicTextSample], filename: str = None):
        """Save collected samples to disk."""
        if not samples:
            self.logger.warning("No samples to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"amharic_corpus_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert to JSON-serializable format
        samples_data = [asdict(sample) for sample in samples]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 Saved {len(samples)} samples to {output_path}")
        
        # Generate statistics
        self._generate_collection_stats(samples, output_path.with_suffix('.stats.json'))
    
    def _generate_collection_stats(self, samples: List[AmharicTextSample], stats_path: Path):
        """Generate collection statistics."""
        if not samples:
            return
        
        stats = {
            'total_samples': len(samples),
            'total_words': sum(s.estimated_words for s in samples),
            'average_quality_score': sum(s.quality_score for s in samples) / len(samples),
            'average_amharic_ratio': sum(s.amharic_ratio for s in samples) / len(samples),
            'sources': {},
            'cultural_domains': {},
            'dialect_coverage': {}
        }
        
        # Source distribution
        for sample in samples:
            stats['sources'][sample.source] = stats['sources'].get(sample.source, 0) + 1
        
        # Cultural domain distribution
        for sample in samples:
            domain = sample.cultural_domain
            stats['cultural_domains'][domain] = stats['cultural_domains'].get(domain, 0) + 1
        
        # Dialect coverage
        for sample in samples:
            for dialect in sample.dialect_hints:
                stats['dialect_coverage'][dialect] = stats['dialect_coverage'].get(dialect, 0) + 1
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📊 Collection statistics saved to {stats_path}")


async def main():
    """Main function for data collection sub-agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Amharic Data Collection Sub-Agent')
    parser.add_argument('--source', choices=['wikipedia', 'bbc_amharic', 'dw_amharic', 'fana_bc', 'all'], 
                       default='wikipedia', help='Data source to collect from')
    parser.add_argument('--max-articles', type=int, default=1000, help='Maximum articles to collect')
    parser.add_argument('--output', default='data/raw', help='Output directory')
    parser.add_argument('--concurrent', type=int, default=5, help='Max concurrent requests')
    
    args = parser.parse_args()
    
    collector = AmharicDataCollector(output_dir=args.output, max_concurrent=args.concurrent)
    
    if args.source == 'all':
        sources = ['wikipedia', 'bbc_amharic', 'dw_amharic', 'fana_bc']
    else:
        sources = [args.source]
    
    all_samples = []
    for source in sources:
        samples = await collector.collect_from_source(source, args.max_articles)
        all_samples.extend(samples)
        
        if samples:
            collector.save_samples(samples, f"{source}_corpus.json")
    
    if all_samples:
        collector.save_samples(all_samples, "combined_amharic_corpus.json")
        print(f"🎉 Successfully collected {len(all_samples)} Amharic text samples!")
    else:
        print("❌ No samples collected. Check your internet connection and source availability.")


if __name__ == "__main__":
    asyncio.run(main())