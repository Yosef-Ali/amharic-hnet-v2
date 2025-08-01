# !IMPORTANT - H-NET THEORY CRITICAL REFERENCE

## 🚨 MANDATORY READING FOR ALL SUB-AGENTS 🚨

**THIS DOCUMENT MUST BE CONSULTED BEFORE ANY H-NET RELATED WORK**

---

## 🧠 TRUE H-NET ARCHITECTURE (ORIGINAL CHINESE IMPLEMENTATION)

**Source:** https://github.com/goombalab/hnet
**Research:** Completed analysis of dynamic chunking and hierarchical processing

### ⚠️ CRITICAL DISTINCTION: H-NET ≠ TRADITIONAL TRANSFORMER

**H-Net is NOT:**
- ❌ Token-by-token generation
- ❌ Fixed morpheme patterns
- ❌ Byte-level processing
- ❌ Traditional transformer with morpheme detection
- ❌ Sequential autoregressive generation

**H-Net IS:**
- ✅ Dynamic semantic chunking based on cosine similarity
- ✅ Hierarchical processing of discovered chunks  
- ✅ Chunk vocabulary creation through semantic boundaries
- ✅ Generation through hierarchically transformed representations
- ✅ Adaptive, context-aware segmentation

---

## 🔥 CORE H-NET COMPONENTS (MANDATORY IMPLEMENTATION)

### 1. DYNAMIC CHUNKING MECHANISM
```python
# FROM ORIGINAL: hnet/modules/dc.py
# Uses cosine similarity between consecutive hidden states
# Detects semantic boundaries dynamically
# Creates adaptive chunk vocabulary

def detect_boundaries(hidden_states):
    similarities = cosine_similarity(hidden_states[1:], hidden_states[:-1])
    boundary_probs = 0.5 * (1 - similarities)  # Low similarity = boundary
    return boundary_probs
```

### 2. HIERARCHICAL PROCESSING  
```python
# FROM ORIGINAL: ChunkLayer → process chunks → DeChunkLayer
# Transform (B, L, D) → (B, L, D) through hierarchical backbone
# NOT flat token processing

def hierarchical_transform(embeddings, chunks):
    # Process at chunk level, not token level
    chunk_representations = process_chunks(embeddings, chunks)
    return hierarchically_transform(chunk_representations)
```

### 3. GENERATION APPROACH
```python
# FROM ORIGINAL: mixer_seq.py
# Uses hierarchically transformed representations
# Applies LM head to transformed states → meaningful output
# NOT byte-level generation

def generate(input_ids):
    embeddings = embed(input_ids)
    hierarchical_output = hnet_backbone(embeddings)  # KEY: hierarchical transform
    logits = lm_head(hierarchical_output)
    return sample_from_logits(logits)
```

---

## 🎯 AMHARIC H-NET SPECIFIC REQUIREMENTS

### CURRENT PROBLEM DIAGNOSIS:
- **INPUT:** "ይፈልጋሉ" (they want)
- **CURRENT OUTPUT:** "ይፈልጋሉᐻo ᕙᕘὙᆐf҉ҙ" (meaningless bytes)
- **REQUIRED OUTPUT:** "ይፈልጋሉ ውሃ መጠጣት" (they want to drink water)

### ROOT CAUSE:
The current implementation is **NOT following H-Net theory**:
1. ❌ Using fixed morpheme patterns instead of dynamic chunking
2. ❌ Doing byte-level generation instead of hierarchical transformation
3. ❌ No chunk vocabulary creation for Amharic semantic units

### MANDATORY FIXES:

#### 1. REPLACE FIXED MORPHEME CHUNKER
```python
# WRONG (current):
class AmharicMorphemeChunker:
    self.verb_prefixes = ['ይ', 'ለ', 'በ', ...]  # Fixed patterns
    
# CORRECT (H-Net theory):
class DynamicSemanticChunker:
    def detect_chunks(self, hidden_states):
        return cosine_similarity_boundaries(hidden_states)  # Dynamic detection
```

#### 2. IMPLEMENT HIERARCHICAL PROCESSING
```python
# WRONG (current):
def forward(self, x):
    boundaries = self.chunker(x)
    return transformer_process(x)  # Still flat processing
    
# CORRECT (H-Net theory):
def forward(self, x):
    chunks = self.dynamic_chunker(x)
    hierarchical_repr = self.process_hierarchically(x, chunks)
    return hierarchical_repr
```

#### 3. FIX GENERATION MECHANISM
```python
# WRONG (current):
def generate(self):
    for step in range(max_length):
        next_byte = sample_byte()  # Byte-level generation
        
# CORRECT (H-Net theory):
def generate(self):
    hierarchical_output = self.hnet_transform(input_embeddings)
    meaningful_output = self.lm_head(hierarchical_output)
    return meaningful_output
```

---

## 🚨 MANDATORY SUB-AGENT COMPLIANCE

### FOR ALL SUB-AGENTS:

#### BEFORE ANY H-NET WORK:
1. ✅ **READ** this document completely
2. ✅ **UNDERSTAND** H-Net ≠ traditional transformer  
3. ✅ **VERIFY** approach follows dynamic chunking theory
4. ✅ **CONFIRM** no byte-level generation

#### FORBIDDEN APPROACHES:
- ❌ **NO** fixed morpheme patterns
- ❌ **NO** byte-by-byte generation
- ❌ **NO** traditional transformer modifications
- ❌ **NO** UTF-8 byte tricks
- ❌ **NO** autoregressive token sampling

#### REQUIRED APPROACHES:
- ✅ **YES** dynamic semantic chunking (cosine similarity)
- ✅ **YES** hierarchical chunk processing
- ✅ **YES** chunk vocabulary creation
- ✅ **YES** hierarchical transformation for generation
- ✅ **YES** meaningful Amharic output

#### VERIFICATION QUESTIONS:
1. Does this approach use dynamic chunking based on semantic similarity?
2. Does this create adaptive chunk vocabulary?
3. Does this process hierarchically, not token-by-token?
4. Will this generate meaningful Amharic words, not random bytes?

**IF ANY ANSWER IS "NO" → STOP AND REDESIGN**

---

## 🎯 SUCCESS CRITERIA

### GOAL:
```
INPUT:  "ይፈልጋሉ" (they want)
OUTPUT: "ይፈልጋሉ ውሃ መጠጣት" (they want to drink water)
```

### VALIDATION:
- ✅ Uses dynamic semantic chunking
- ✅ Creates Amharic chunk vocabulary
- ✅ Processes hierarchically
- ✅ Generates meaningful continuations
- ✅ NO random byte sequences

---

## 📚 REFERENCE IMPLEMENTATION

**Original H-Net:** https://github.com/goombalab/hnet
- `hnet/modules/dc.py` - Dynamic chunking mechanism
- `hnet/models/mixer_seq.py` - Generation approach
- Core principle: Transform (B, L, D) → (B, L, D) hierarchically

---

## ⚠️ FINAL WARNING

**ANY SUB-AGENT THAT IGNORES THIS DOCUMENT AND IMPLEMENTS:**
- Fixed morpheme patterns
- Byte-level generation  
- Traditional transformer approaches
- Non-hierarchical processing

**WILL FAIL TO ACHIEVE MEANINGFUL AMHARIC GENERATION**

**THIS IS NOT NEGOTIABLE - H-NET THEORY MUST BE FOLLOWED EXACTLY**

---

**Document Status:** CRITICAL - MANDATORY COMPLIANCE
**Last Updated:** Current Session  
**Authority:** Primary Agent based on original H-Net research