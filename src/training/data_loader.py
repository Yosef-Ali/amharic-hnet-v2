import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from typing import List, Dict, Optional, Iterator, Tuple
import random
from pathlib import Path
import logging
import json

from ..preprocessing.prepare_amharic import AmharicPreprocessor


class AmharicDataset(Dataset):
    """
    Dataset class for Amharic text processing with H-Net.
    Handles byte-level sequences and dynamic chunking preparation.
    """
    
    def __init__(
        self, 
        texts: List[str], 
        preprocessor: AmharicPreprocessor,
        max_length: int = 512,
        min_length: int = 10,
        return_targets: bool = True
    ):
        self.texts = texts
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.min_length = min_length
        self.return_targets = return_targets
        
        # Filter and prepare texts
        self.processed_texts = self._prepare_texts()
        
        self.logger = logging.getLogger(__name__)
        
    def _prepare_texts(self) -> List[str]:
        """Prepare and filter texts for training."""
        processed = []
        
        for text in self.texts:
            # Clean and preprocess
            cleaned = self.preprocessor.clean_text(text)
            
            # Filter by length and Amharic content
            if (len(cleaned) >= self.min_length and 
                self.preprocessor.get_amharic_ratio(cleaned) >= 0.3):
                processed.append(cleaned)
        
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.processed_texts[idx]
        
        # Convert to byte sequence
        byte_sequence = self.preprocessor.extract_byte_sequences(
            text, self.max_length
        )
        
        # Pad or truncate to max_length
        if len(byte_sequence) > self.max_length:
            byte_sequence = byte_sequence[:self.max_length]
        else:
            # Pad with zeros
            byte_sequence = byte_sequence + [0] * (self.max_length - len(byte_sequence))
        
        input_ids = torch.tensor(byte_sequence, dtype=torch.long)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': (input_ids != 0).long(),  # Mask for padding
            'text': text
        }
        
        # For training, create shifted targets
        if self.return_targets:
            # Target is input shifted by one position
            target_ids = torch.cat([
                input_ids[1:], 
                torch.tensor([0], dtype=torch.long)  # Pad last position
            ])
            result['target_ids'] = target_ids
        
        return result


class AmharicStreamingDataset(IterableDataset):
    """
    Streaming dataset for large Amharic corpora.
    Loads data on-demand to handle datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        data_paths: List[str],
        preprocessor: AmharicPreprocessor,
        max_length: int = 512,
        buffer_size: int = 1000,
        shuffle_buffer: int = 10000
    ):
        self.data_paths = data_paths
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
        
        self.logger = logging.getLogger(__name__)
    
    def _read_texts(self) -> Iterator[str]:
        """Read texts from all data files."""
        for path in self.data_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line
            except Exception as e:
                self.logger.warning(f"Error reading {path}: {e}")
                continue
    
    def _shuffle_buffer(self, iterator: Iterator, buffer_size: int) -> Iterator:
        """Shuffle items using a buffer."""
        buffer = []
        
        for item in iterator:
            buffer.append(item)
            
            if len(buffer) >= buffer_size:
                random.shuffle(buffer)
                for i in range(buffer_size // 2):  # Yield half the buffer
                    yield buffer.pop()
        
        # Yield remaining items
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        texts_iter = self._read_texts()
        
        # Apply shuffling if requested
        if self.shuffle_buffer > 0:
            texts_iter = self._shuffle_buffer(texts_iter, self.shuffle_buffer)
        
        for text in texts_iter:
            try:
                # Preprocess text
                cleaned = self.preprocessor.clean_text(text)
                
                # Skip if not suitable
                if (len(cleaned) < 10 or 
                    self.preprocessor.get_amharic_ratio(cleaned) < 0.3):
                    continue
                
                # Convert to byte sequence
                byte_sequence = self.preprocessor.extract_byte_sequences(
                    cleaned, self.max_length
                )
                
                # Pad or truncate
                if len(byte_sequence) > self.max_length:
                    byte_sequence = byte_sequence[:self.max_length]
                else:
                    byte_sequence = byte_sequence + [0] * (self.max_length - len(byte_sequence))
                
                input_ids = torch.tensor(byte_sequence, dtype=torch.long)
                
                # Create targets (shifted input)
                target_ids = torch.cat([
                    input_ids[1:], 
                    torch.tensor([0], dtype=torch.long)
                ])
                
                yield {
                    'input_ids': input_ids,
                    'target_ids': target_ids,
                    'attention_mask': (input_ids != 0).long(),
                    'text': cleaned
                }
                
            except Exception as e:
                self.logger.warning(f"Error processing text: {e}")
                continue


class AmharicCollator:
    """
    Custom collator for batching Amharic data with padding and attention masks.
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch items with proper padding.
        """
        # Get maximum length in batch
        max_len = max(item['input_ids'].size(0) for item in batch)
        
        # Prepare batch tensors
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        target_ids = None
        if 'target_ids' in batch[0]:
            target_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        
        texts = []
        
        # Fill batch tensors
        for i, item in enumerate(batch):
            seq_len = item['input_ids'].size(0)
            input_ids[i, :seq_len] = item['input_ids']
            attention_mask[i, :seq_len] = item['attention_mask']
            
            if target_ids is not None:
                target_ids[i, :seq_len] = item['target_ids']
            
            texts.append(item['text'])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'texts': texts
        }
        
        if target_ids is not None:
            result['target_ids'] = target_ids
        
        return result


def create_data_loaders(
    train_texts: List[str],
    val_texts: List[str],
    preprocessor: AmharicPreprocessor,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_texts: Training texts
        val_texts: Validation texts
        preprocessor: Text preprocessor
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = AmharicDataset(
        texts=train_texts,
        preprocessor=preprocessor,
        max_length=max_length,
        return_targets=True
    )
    
    val_dataset = AmharicDataset(
        texts=val_texts,
        preprocessor=preprocessor,
        max_length=max_length,
        return_targets=True
    )
    
    # Create collator
    collator = AmharicCollator(pad_token_id=0)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_streaming_loader(
    data_paths: List[str],
    preprocessor: AmharicPreprocessor,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
    shuffle_buffer: int = 10000
) -> DataLoader:
    """
    Create streaming data loader for large datasets.
    
    Args:
        data_paths: Paths to data files
        preprocessor: Text preprocessor
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        shuffle_buffer: Size of shuffle buffer
        
    Returns:
        Streaming data loader
    """
    dataset = AmharicStreamingDataset(
        data_paths=data_paths,
        preprocessor=preprocessor,
        max_length=max_length,
        shuffle_buffer=shuffle_buffer
    )
    
    collator = AmharicCollator(pad_token_id=0)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )


def load_data_from_directory(
    data_dir: str,
    train_ratio: float = 0.9,
    max_files: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Load and split data from directory containing text files.
    
    Args:
        data_dir: Directory containing text files
        train_ratio: Ratio of data for training
        max_files: Maximum number of files to load
        
    Returns:
        Tuple of (train_texts, val_texts)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all text files
    text_files = list(data_path.glob("*.txt"))
    
    if max_files:
        text_files = text_files[:max_files]
    
    # Load all texts
    all_texts = []
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
                all_texts.extend(texts)
        except Exception as e:
            logging.warning(f"Error reading {file_path}: {e}")
            continue
    
    # Shuffle and split
    random.shuffle(all_texts)
    split_idx = int(len(all_texts) * train_ratio)
    
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    return train_texts, val_texts


def save_data_split(
    train_texts: List[str],
    val_texts: List[str],
    output_dir: str
):
    """
    Save train/validation split to files.
    
    Args:
        train_texts: Training texts
        val_texts: Validation texts
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save training data
    with open(output_path / "train.txt", 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(text + '\n')
    
    # Save validation data
    with open(output_path / "val.txt", 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(text + '\n')
    
    # Save metadata
    metadata = {
        'train_count': len(train_texts),
        'val_count': len(val_texts),
        'total_count': len(train_texts) + len(val_texts),
        'train_ratio': len(train_texts) / (len(train_texts) + len(val_texts))
    }
    
    with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data split saved to {output_dir}")
    print(f"Training samples: {metadata['train_count']}")
    print(f"Validation samples: {metadata['val_count']}")


if __name__ == "__main__":
    # Test the data loader
    from ..preprocessing.prepare_amharic import AmharicPreprocessor
    
    # Create sample data
    sample_texts = [
        "አማርኛ የኢትዮጵያ ሕዝብ ዋና ቋንቋ ነው።",
        "ቡና በኢትዮጵያ ባህል ውስጥ ልዩ ሚና አለው።",
        "መስቀል በኢትዮጵያ ኦርቶዶክስ ሃይማኖት ውስጥ ቅዱስ ምልክት ነው።",
        "ኢትዮጵያ ታሪካዊ ኩራት ያላት አገር ነች።"
    ]
    
    # Initialize preprocessor
    preprocessor = AmharicPreprocessor()
    
    # Create dataset
    dataset = AmharicDataset(
        texts=sample_texts,
        preprocessor=preprocessor,
        max_length=128
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test data loader
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=AmharicCollator()
    )
    
    for batch in loader:
        print(f"Batch input shape: {batch['input_ids'].shape}")
        print(f"Batch target shape: {batch['target_ids'].shape}")
        print(f"Batch texts: {batch['texts']}")
        break