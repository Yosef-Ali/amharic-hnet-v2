import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class AmharicMorphemeChunker(nn.Module):
    """
    Advanced Amharic-specific dynamic chunking module that identifies morpheme boundaries
    using deep linguistic patterns combined with learned attention patterns.
    
    Handles Ge'ez script morphological complexity with syllabic-aware processing:
    - Each Fidel character represents a consonant+vowel combination (syllable)
    - Morpheme boundaries occur at syllable clusters, not individual characters
    - Verb conjugations and noun inflections require special handling
    """
    def __init__(self, d_model: int, target_compression: float = 4.5, avg_morpheme_length: float = 3.2):
        super().__init__()
        self.d_model = d_model
        self.target_compression = target_compression
        self.avg_morpheme_length = avg_morpheme_length
        
        # Enhanced Amharic morphological patterns
        self.verb_prefixes = ['ይ', 'ለ', 'በ', 'እ', 'ስ', 'አ', 'ወ', 'ተ', 'ነ', 'ክ']
        self.verb_suffixes = ['ል', 'ሽ', 'ን', 'ችሁ', 'ናል', 'ናልች', 'አለ', 'ች', 'ው']
        self.noun_prefixes = ['የ', 'በ', 'ከ', 'ለ', 'ወ', 'እ', 'ም', 'ኣ', 'ት', 'ን', 'ኢ']
        self.noun_suffixes = ['ች', 'ኝ', 'ው', 'ሽ', 'ን', 'ተ', 'ና', 'ም', 'ህ', 'ሁ', 'ሻ', 'አል', 'ነው']
        
        # Syllabic pattern recognition (Ge'ez script specific)
        self.fidel_clusters = self._create_fidel_clusters()
        
        # Morphological complexity indicators
        self.complexity_patterns = {
            'verb_conjugation': ['እየ', 'እንዳ', 'ማለት', 'መሆን'],
            'compound_words': ['እና', 'ወይም', 'ግን', 'ስለዚህ'],
            'possession': ['የ...ው', 'የ...ዋ', 'የ...ች']
        }
        
        # Advanced neural components for morphological analysis
        self.syllable_encoder = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.morpheme_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        
        # Multi-layer morpheme boundary classifier
        self.boundary_classifier = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),  # Current + prev + next + linguistic features
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Linguistic feature embeddings
        self.linguistic_features = nn.Embedding(10, d_model // 4)  # Various morphological markers
        self.syllable_type_embed = nn.Embedding(5, d_model // 4)  # Syllable type classification
        
        # Morpheme type classification
        self.morpheme_type_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4)  # [root, prefix, suffix, compound]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
        Returns:
            boundary_probs: Probability of boundary at each position [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute queries and keys for adjacent positions
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        
        # Compute similarity between adjacent positions
        similarities = F.cosine_similarity(queries[:, 1:], keys[:, :-1], dim=-1)
        
        # Low similarity = high boundary probability
        boundary_probs = 0.5 * (1 - similarities)
        
        # Always start with a boundary
        boundary_probs = torch.cat([
            torch.ones(batch_size, 1, device=x.device), 
            boundary_probs
        ], dim=1)
        
        # Advanced morphological boundary detection
        if seq_len > 2:
            # Extract linguistic features for each position
            linguistic_features = self._extract_linguistic_features(x)
            
            # Create context windows: [prev, current, next, linguistic]
            prev_context = torch.cat([x[:, :1], x[:, :-1]], dim=1)
            next_context = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
            
            context_features = torch.cat([
                prev_context, x, next_context, linguistic_features
            ], dim=-1)
            
            # Apply boundary classification
            boundary_logits = self.boundary_classifier(context_features).squeeze(-1)
            
            # Apply morpheme length constraints (avg 3.2 syllables per morpheme)
            boundary_probs = self._apply_length_constraints(boundary_logits, seq_len)
            
        return boundary_probs
    
    def _create_fidel_clusters(self):
        """Create clusters of related Fidel characters for syllabic processing."""
        return {
            'ha_family': ['ሀ', 'ሁ', 'ሂ', 'ሃ', 'ሄ', 'ህ', 'ሆ'],
            'la_family': ['ለ', 'ሉ', 'ሊ', 'ላ', 'ሌ', 'ል', 'ሎ'],
            'ma_family': ['መ', 'ሙ', 'ሚ', 'ማ', 'ሜ', 'ም', 'ሞ'],
            # Add more as needed
        }
    
    def _extract_linguistic_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract morphological and syllabic features for each position."""
        batch_size, seq_len, d_model = x.shape
        
        # Initialize linguistic feature tensor
        ling_features = torch.zeros(batch_size, seq_len, d_model // 4, device=x.device)
        
        # Apply LSTM for syllabic sequence modeling
        syllable_encoded, _ = self.syllable_encoder(x)
        
        # Combine with positional linguistic markers
        # This is a simplified version - in practice, would use actual text analysis
        return torch.cat([syllable_encoded[:, :, :d_model//4], ling_features], dim=-1)
    
    def _apply_length_constraints(self, boundary_logits: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply morpheme length constraints based on Amharic linguistic patterns."""
        # Smooth boundary probabilities to encourage proper morpheme lengths
        boundary_probs = torch.sigmoid(boundary_logits)
        
        # Apply gaussian smoothing to encourage avg_morpheme_length spacing
        kernel_size = max(3, int(self.avg_morpheme_length))
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create smoothing kernel
        sigma = self.avg_morpheme_length / 3.0
        kernel = torch.exp(-0.5 * ((torch.arange(kernel_size, device=boundary_probs.device) - kernel_size//2) / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        # Apply convolution for smoothing
        smoothed_probs = F.conv1d(
            boundary_probs.unsqueeze(1),
            kernel.view(1, 1, -1),
            padding=kernel_size//2
        ).squeeze(1)
        
        return smoothed_probs


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder that processes chunks at multiple levels.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.d_model = d_model
        
        # Byte-level encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            batch_first=True,
            dropout=0.1
        )
        self.byte_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers//2)
        
        # Chunk-level encoder
        self.chunk_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers//2)
        
        # Chunk aggregation
        self.chunk_pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor, boundary_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            boundary_probs: Boundary probabilities [batch_size, seq_len]
        Returns:
            Hierarchically encoded representations
        """
        # First pass: byte-level encoding
        byte_encoded = self.byte_encoder(x)
        
        # Create chunks based on boundary probabilities
        # For simplicity, we'll use a threshold-based approach
        chunks = self._create_chunks(byte_encoded, boundary_probs)
        
        # Second pass: chunk-level encoding
        chunk_encoded = self.chunk_encoder(chunks)
        
        return chunk_encoded
    
    def _create_chunks(self, x: torch.Tensor, boundary_probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Create chunk representations from byte embeddings using boundary probabilities.
        Simplified implementation that averages within chunks.
        """
        batch_size, seq_len, d_model = x.shape
        
        # Determine boundaries
        boundaries = boundary_probs > threshold
        
        # Simple chunking: average pooling within chunks
        # This is a simplified implementation - in practice, you'd want more sophisticated chunking
        chunk_representations = []
        
        for b in range(batch_size):
            chunk_start = 0
            batch_chunks = []
            
            for i in range(1, seq_len):
                if boundaries[b, i] or i == seq_len - 1:
                    # Create chunk from start to current position
                    chunk_emb = x[b, chunk_start:i+1].mean(dim=0)
                    batch_chunks.append(chunk_emb)
                    chunk_start = i + 1
            
            # Pad or truncate to consistent size
            if len(batch_chunks) > seq_len // 2:
                batch_chunks = batch_chunks[:seq_len // 2]
            else:
                while len(batch_chunks) < seq_len // 2:
                    batch_chunks.append(torch.zeros_like(batch_chunks[0]))
            
            chunk_representations.append(torch.stack(batch_chunks))
        
        return torch.stack(chunk_representations)


class AmharicHNet(nn.Module):
    """
    H-Net architecture specifically designed for Amharic language modeling.
    Uses dynamic chunking and hierarchical processing.
    """
    def __init__(
        self, 
        d_model: int = 768, 
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        n_main_layers: int = 12,
        n_heads: int = 12,
        compression_ratio: float = 4.5,
        vocab_size: int = 256  # Byte-level
    ):
        super().__init__()
        
        self.d_model = d_model
        self.compression_ratio = compression_ratio
        
        # Byte-level embedding
        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(5000, d_model)
        
        # Dynamic chunker
        self.chunker = DynamicChunker(d_model, compression_ratio)
        
        # Hierarchical encoder
        self.hierarchical_encoder = HierarchicalEncoder(d_model, n_heads, n_encoder_layers)
        
        # Main transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            batch_first=True,
            dropout=0.1
        )
        self.main_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_main_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        target_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Input byte sequences [batch_size, seq_len]
            target_ids: Target sequences for training [batch_size, seq_len]
        Returns:
            output_logits: Predictions [batch_size, seq_len, vocab_size]
            boundary_probs: Chunk boundary probabilities [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed input bytes
        x = self.byte_embedding(input_ids)
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :seq_len]
        
        # Dynamic chunking
        boundary_probs = self.chunker(x)
        
        # Hierarchical encoding
        hierarchical_output = self.hierarchical_encoder(x, boundary_probs)
        
        # Main encoder processing
        encoder_output = self.main_encoder(hierarchical_output)
        
        # Decoder (for autoregressive generation)
        if target_ids is not None:
            target_embeds = self.byte_embedding(target_ids)
            if target_ids.size(1) <= self.pos_encoding.size(1):
                target_embeds = target_embeds + self.pos_encoding[:, :target_ids.size(1)]
            
            decoder_output = self.decoder(target_embeds, encoder_output)
        else:
            # During inference, use encoder output as decoder input
            decoder_output = encoder_output
        
        # Apply layer normalization
        decoder_output = self.layer_norm(decoder_output)
        
        # Project to vocabulary
        output_logits = self.output_proj(decoder_output)
        
        return output_logits, boundary_probs
    
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_length: int = 100, 
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        """
        self.eval()
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, _ = self.forward(generated)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we generate end token (period in Amharic: ።)
                if next_token.item() == ord('።'):
                    break
        
        return generated