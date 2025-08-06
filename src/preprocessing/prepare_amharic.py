import re
import unicodedata
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path


class AmharicPreprocessor:
    """
    Specialized preprocessor for Amharic text that handles:
    - Unicode normalization
    - Script-specific cleaning
    - Morpheme-aware tokenization
    - Cultural context preservation
    """
    
    def __init__(self):
        # Amharic Unicode ranges
        self.amharic_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
            (0xAB00, 0xAB2F),  # Ethiopic Extended-A
        ]
        
        # Amharic punctuation marks
        self.amharic_punctuation = {
            '።': '.',    # Amharic full stop
            '፣': ',',    # Amharic comma
            '፤': ';',    # Amharic semicolon
            '፥': ':',    # Amharic colon
            '፦': ':',    # Amharic preface colon
            '፧': '?',    # Amharic question mark
            '፨': '!',    # Amharic exclamation mark
        }
        
        # Common Amharic prefixes and suffixes for morpheme awareness
        self.prefixes = [
            'የ', 'በ', 'ከ', 'ለ', 'ወ', 'እ', 'ም', 'ኣ', 'ት', 'ን', 'ኢ', 'አይ', 'አል', 'yä-', 'yämm-', 'መ', 'iyye-', 'እየ', 'ይ', 'ተ'
        ]
        
        self.suffixes = [
            'ኦች', 'ዎች', 'ዮች', 'ች', 'ኝ', 'ው', 'ሽ', 'ን', 'ተ', 'ና', 'ም', 'ህ', 'ሁ', 'ሻ', '-očč', '-wočč', '-yočč', '-at', '-an', '-u', '-wa', '-e', '-ye', '-h', '-sh', '-nät', '-ኛ', '-ተኛ', 'ኩ', 'ጣ', 'ል'
        ]
        
        # Cultural terms that should be preserved exactly
        self.cultural_terms = {
            'ቡና', 'መስቀል', 'ገና', 'ፋሲካ', 'እንጅብ', 'ንጉሥ', 'ንግሥት',
            'ቤተክርስትያን', 'መስጊድ', 'ሃይማኖት', 'ባህል', 'ወግ', 'ምግብ',
            'ኢትዮጵያ', 'አዲስ አበባ', 'አማራ', 'ኦሮሞ', 'ትግሬ', 'ሲዳማ'
        }
        
        # Common contractions and abbreviations
        self.contractions = {
            'ኣ/አ': 'አዲስ አበባ',
            'ኢ/ያ': 'ኢትዮጵያ',
            'ወ.ዘ.ተ': 'ወዘተ',
        }
        
        self.logger = logging.getLogger(__name__)
    
    def is_amharic_char(self, char: str) -> bool:
        """Check if a character is in Amharic Unicode ranges."""
        char_code = ord(char)
        return any(start <= char_code <= end for start, end in self.amharic_ranges)
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode representation of Amharic text.
        Handles various forms of the same character.
        """
        # Apply NFC normalization
        text = unicodedata.normalize('NFC', text)
        
        # Handle common character variations
        replacements = {
            'ሀ': 'ሃ',  # Normalize ha variations
            'ኸ': 'ኅ',  # Normalize kha variations
            'ዐ': 'ዓ',  # Normalize ain variations
            'ጸ': 'ፀ',  # Normalize tsa variations
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def clean_text(self, text: str, preserve_spaces: bool = False) -> str:
        """
        Clean and normalize Amharic text while preserving cultural context.
        """
        if not text:
            return ""
        
        # Normalize Unicode
        text = self.normalize_unicode(text)
        
        # Remove unnecessary whitespace but preserve meaningful spacing
        if not preserve_spaces:
            text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle mixed script text (preserve Latin for numbers, dates, etc.)
        # Keep numbers and basic Latin punctuation
        text = re.sub(r'[^\w\s\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F\u0030-\u0039\.\,\:\;\?\!\-\(\)]', '', text)
        
        # Normalize punctuation while preserving Amharic punctuation
        for amh_punct, latin_punct in self.amharic_punctuation.items():
            # Keep Amharic punctuation, don't replace
            pass
        
        # Handle contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def segment_morphemes(self, word: str) -> List[str]:
        """
        Basic morpheme segmentation for Amharic words.
        This is a simplified approach - real morphological analysis would be much more complex.
        """
        if len(word) <= 2:
            return [word]
        
        segments = []
        remaining = word
        
        # Iteratively check for prefixes
        while True:
            found_prefix = False
            for prefix in sorted(self.prefixes, key=len, reverse=True):
                if remaining.startswith(prefix) and len(remaining) > len(prefix):
                    segments.append(prefix)
                    remaining = remaining[len(prefix):]
                    found_prefix = True
                    break
            if not found_prefix:
                break

        # Iteratively check for suffixes
        temp_segments = []
        while True:
            found_suffix = False
            for suffix in sorted(self.suffixes, key=len, reverse=True):
                if remaining.endswith(suffix) and len(remaining) > len(suffix):
                    temp_segments.insert(0, suffix)
                    remaining = remaining[:-len(suffix)]
                    found_suffix = True
                    break
            if not found_suffix:
                break

        if remaining:
            segments.append(remaining)

        segments.extend(temp_segments)
        
        return [seg for seg in segments if seg]  # Remove empty segments
    
    def tokenize_morpheme_aware(self, text: str) -> List[str]:
        """
        Tokenize text with morpheme awareness.
        Preserves cultural terms as single tokens.
        """
        # First, protect cultural terms
        protected_terms = {}
        placeholder_id = 0
        
        for term in self.cultural_terms:
            if term in text:
                placeholder = f"__PROTECTED_{placeholder_id}__"
                protected_terms[placeholder] = term
                text = text.replace(term, placeholder)
                placeholder_id += 1
        
        # Split by whitespace and punctuation
        tokens = re.findall(r'\S+', text)
        
        # Process each token
        processed_tokens = []
        for token in tokens:
            # Restore protected terms
            if token in protected_terms:
                processed_tokens.append(protected_terms[token])
            elif self.is_amharic_word(token):
                # Apply morpheme segmentation to Amharic words
                morphemes = self.segment_morphemes(token)
                processed_tokens.extend(morphemes)
            else:
                # Keep non-Amharic tokens as-is (numbers, Latin text, etc.)
                processed_tokens.append(token)
        
        return processed_tokens
    
    def is_amharic_word(self, word: str) -> bool:
        """Check if a word is primarily Amharic."""
        if not word:
            return False
        
        amharic_chars = sum(1 for char in word if self.is_amharic_char(char))
        return amharic_chars / len(word) > 0.5
    
    def prepare_training_data(self, texts: List[str], min_length: int = 10) -> List[str]:
        """
        Prepare a list of texts for training.
        
        Args:
            texts: List of raw text strings
            min_length: Minimum length for text inclusion
            
        Returns:
            List of cleaned and processed texts
        """
        processed_texts = []
        
        for text in texts:
            try:
                # Clean the text
                cleaned = self.clean_text(text)
                
                # Skip very short texts
                if len(cleaned) < min_length:
                    continue
                
                # Skip texts with too little Amharic content
                amharic_ratio = self.get_amharic_ratio(cleaned)
                if amharic_ratio < 0.3:  # At least 30% Amharic
                    continue
                
                processed_texts.append(cleaned)
                
            except Exception as e:
                self.logger.warning(f"Error processing text: {e}")
                continue
        
        return processed_texts
    
    def get_amharic_ratio(self, text: str) -> float:
        """Calculate the ratio of Amharic characters in text."""
        if not text:
            return 0.0
        
        amharic_chars = sum(1 for char in text if self.is_amharic_char(char))
        return amharic_chars / len(text)
    
    def extract_byte_sequences(self, text: str, max_length: int = 512) -> List[int]:
        """
        Convert text to byte sequence for H-Net processing.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            List of byte values
        """
        # Encode text to UTF-8 bytes
        byte_sequence = list(text.encode('utf-8'))
        
        # Truncate if too long
        if len(byte_sequence) > max_length:
            byte_sequence = byte_sequence[:max_length]
        
        return byte_sequence
    
    def decode_byte_sequence(self, byte_sequence: List[int]) -> str:
        """
        Convert byte sequence back to text with robust UTF-8 handling.
        
        Args:
            byte_sequence: List of byte values
            
        Returns:
            Decoded text string
        """
        try:
            # Remove padding zeros and invalid byte values
            clean_bytes = []
            for b in byte_sequence:
                if isinstance(b, (int, float)) and 0 < b < 256:
                    clean_bytes.append(int(b))
                elif b == 0:
                    # Stop at padding/null bytes
                    break
            
            if not clean_bytes:
                return ""
            
            # Convert to bytes and decode with error handling
            byte_data = bytes(clean_bytes)
            
            # Try UTF-8 decoding with different error strategies
            try:
                # First try strict UTF-8
                text = byte_data.decode('utf-8')
            except UnicodeDecodeError:
                # Fall back to error replacement
                text = byte_data.decode('utf-8', errors='replace')
                # Remove replacement characters
                text = text.replace('\ufffd', '')
            
            # Clean up any remaining control characters except common ones
            cleaned_text = ""
            for char in text:
                char_code = ord(char)
                # Keep normal characters, Amharic script, whitespace, and basic punctuation
                if (char_code >= 32 or char in '\n\t\r') and char_code < 127 or (0x1200 <= char_code <= 0x137F) or (0x1380 <= char_code <= 0x139F):
                    cleaned_text += char
            
            return cleaned_text.strip()
            
        except Exception as e:
            self.logger.warning(f"Error decoding byte sequence: {e}")
            return ""
    
    def create_training_dataset(
        self, 
        input_file: str, 
        output_dir: str,
        chunk_size: int = 1000
    ) -> Dict[str, int]:
        """
        Create training dataset from raw text file.
        
        Args:
            input_file: Path to input text file
            output_dir: Directory to save processed data
            chunk_size: Number of texts per chunk file
            
        Returns:
            Statistics about processed data
        """
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_texts = f.readlines()
        
        # Process texts
        processed_texts = self.prepare_training_data(raw_texts)
        
        # Save in chunks
        chunk_files = []
        for i in range(0, len(processed_texts), chunk_size):
            chunk = processed_texts[i:i+chunk_size]
            chunk_file = output_path / f"chunk_{i//chunk_size:04d}.txt"
            
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for text in chunk:
                    f.write(text + '\n')
            
            chunk_files.append(str(chunk_file))
        
        # Save metadata
        metadata = {
            'total_texts': len(processed_texts),
            'total_chunks': len(chunk_files),
            'chunk_size': chunk_size,
            'chunk_files': chunk_files
        }
        
        with open(output_path / 'metadata.txt', 'w', encoding='utf-8') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        stats = {
            'original_count': len(raw_texts),
            'processed_count': len(processed_texts),
            'chunks_created': len(chunk_files)
        }
        
        return stats


def create_sample_data(output_file: str):
    """Create sample Amharic text for testing."""
    sample_texts = [
        "አማርኛ የኢትዮጵያ ሕዝብ ዋና ቋንቋ ነው።",
        "ቡና በኢትዮጵያ ባህል ውስጥ ልዩ ሚና አለው።",
        "መስቀል በኢትዮጵያ ኦርቶዶክስ ሃይማኖት ውስጥ ቅዱስ ምልክት ነው።",
        "ኢትዮጵያ ታሪካዊ ኩራት ያላት አገር ነች።",
        "ኢትዮጵያ ውስጥ ብዙ ብሔሮች እና ሕዝቦች ይኖራሉ።",
        "ገና የክርስቶስ ልደት የሚከበርበት ቅዱስ ጊዜ ነው።",
        "ፋሲካ የትንሳኤ በዓል በኢትዮጵያ ኦርቶዶክስ ሃይማኖት ውስጥ ዋና በዓል ነው።",
        "ንጉሥ ምኒልክ የኢትዮጵያ ታሪክ ውስጥ ታላቅ መሪ ነበር።",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ነች።",
        "ባህል እና ወግ የማንኛውም ሕዝብ መታወቂያ ናቸው።"
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
    
    print(f"Sample data created at: {output_file}")


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = AmharicPreprocessor()
    
    test_text = "በአማርኛ ቋንቋ ስለ ቡና እና መስቀል እያወራን ነው።"
    print(f"Original: {test_text}")
    
    cleaned = preprocessor.clean_text(test_text)
    print(f"Cleaned: {cleaned}")
    
    tokens = preprocessor.tokenize_morpheme_aware(cleaned)
    print(f"Tokens: {tokens}")
    
    byte_seq = preprocessor.extract_byte_sequences(cleaned)
    print(f"Byte sequence length: {len(byte_seq)}")
    
    decoded = preprocessor.decode_byte_sequence(byte_seq)
    print(f"Decoded: {decoded}")