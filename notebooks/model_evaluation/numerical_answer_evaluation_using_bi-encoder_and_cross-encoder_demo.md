# Numerical Answer Evaluation using Bi-Encoder and Cross-Encoder

## UML Diagrams and Workflow Documentation

This document provides comprehensive UML diagrams (static class diagrams and sequence diagrams) with detailed explanations of the workflow present in the `numerical_answer_evaluation_using_bi-encoder_and_cross-encoder_demo.ipynb` notebook.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Static Class Diagrams](#2-static-class-diagrams)
   - [2.1 Data Models and Dataclasses](#21-data-models-and-dataclasses)
   - [2.2 Core Processing Classes](#22-core-processing-classes)
   - [2.3 Alternative Implementations](#23-alternative-implementations)
   - [2.4 Complete System Architecture](#24-complete-system-architecture)
3. [Sequence Diagrams](#3-sequence-diagrams)
   - [3.1 Main Evaluation Pipeline](#31-main-evaluation-pipeline)
   - [3.2 Number Extraction Flow](#32-number-extraction-flow)
   - [3.3 Two-Stage Semantic Matching](#33-two-stage-semantic-matching)
   - [3.4 Statistical Comparison Flow](#34-statistical-comparison-flow)
   - [3.5 Bi-Encoder Only Matching (Alternative)](#35-bi-encoder-only-matching-alternative)
4. [Component Descriptions](#4-component-descriptions)
5. [Data Flow Diagram](#5-data-flow-diagram)
6. [Bi-Encoder vs Cross-Encoder Comparison](#6-bi-encoder-vs-cross-encoder-comparison)
7. [Configuration Options](#7-configuration-options)

---

## 1. Architecture Overview

The Sentence Transformer-based Numerical Answer Evaluation system uses a **two-stage matching approach** combining the speed of Bi-Encoders with the accuracy of Cross-Encoders to evaluate numerical accuracy in model-generated answers.

### High-Level Architecture

```mermaid
flowchart TB
    subgraph Input["📥 Input"]
        Q[Question]
        MA[Model Answer]
        GT[Ground Truth]
    end
    
    subgraph Extraction["🔍 Number Extraction"]
        E[NumberContextExtractor]
    end
    
    subgraph Stage1["Stage 1: Bi-Encoder"]
        BE[Bi-Encoder<br/>all-MiniLM-L6-v2]
        EMB[Embeddings]
        SIM[Cosine Similarity<br/>Matrix]
    end
    
    subgraph Stage2["Stage 2: Cross-Encoder"]
        CE[Cross-Encoder<br/>ms-marco-MiniLM-L-6-v2]
        RERANK[Re-ranking]
    end
    
    subgraph Stats["📊 Statistical Analysis"]
        COMP[Statistical Comparator]
    end
    
    subgraph Output["📤 Output"]
        R[EvaluationResult]
        D[Decision: Accept/Marginal/Reject]
    end
    
    MA --> E
    GT --> E
    E --> BE
    BE --> EMB
    EMB --> SIM
    SIM --> CE
    CE --> RERANK
    RERANK --> COMP
    COMP --> R
    R --> D
```

### Key Concept: Two-Stage Retrieval and Re-ranking

```mermaid
flowchart LR
    subgraph BiEncoder["🔵 Bi-Encoder (Fast)"]
        direction TB
        B1[Encode all texts<br/>independently]
        B2[Compute similarity<br/>matrix]
        B3[Find top-k<br/>candidates]
    end
    
    subgraph CrossEncoder["🔴 Cross-Encoder (Accurate)"]
        direction TB
        C1[Take candidate<br/>pairs]
        C2[Score each pair<br/>directly]
        C3[Re-rank by<br/>score]
    end
    
    BiEncoder --> CrossEncoder
    
    style BiEncoder fill:#e3f2fd
    style CrossEncoder fill:#ffebee
```

---

## 2. Static Class Diagrams

### 2.1 Data Models and Dataclasses

These classes define the data structures used throughout the pipeline.

```mermaid
classDiagram
    class NumberWithContext {
        <<dataclass>>
        +float value
        +str original_text
        +str context_sentence
        +str context_before
        +str context_after
        +int start_pos
        +int end_pos
        +str number_type
        +Optional~ndarray~ embedding
        +get_context_text() str
    }
    
    class SemanticNumberMatch {
        <<dataclass>>
        +NumberWithContext model_number
        +NumberWithContext truth_number
        +float bi_encoder_score
        +float cross_encoder_score
        +float combined_score
        +str match_confidence
    }
    
    class AcceptanceDecision {
        <<Enum>>
        ACCEPT = "accept"
        MARGINAL = "marginal"
        REJECT = "reject"
    }
    
    class NumberPairComparison {
        <<dataclass>>
        +SemanticNumberMatch match
        +float model_value
        +float truth_value
        +float absolute_error
        +float relative_error
        +float percentage_error
        +float order_of_magnitude_diff
        +bool is_within_tolerance
        +AcceptanceDecision decision
    }
    
    class EvaluationResult {
        <<dataclass>>
        +str question
        +str model_answer
        +str ground_truth
        +List~NumberWithContext~ model_numbers
        +List~NumberWithContext~ truth_numbers
        +List~SemanticNumberMatch~ matches
        +List~NumberPairComparison~ comparisons
        +int total_model_numbers
        +int total_truth_numbers
        +int matched_pairs
        +int within_tolerance_count
        +float matching_coverage
        +float numerical_accuracy_score
        +float semantic_match_score
        +float overall_score
        +AcceptanceDecision overall_decision
    }
    
    SemanticNumberMatch "1" *-- "2" NumberWithContext : contains
    NumberPairComparison "1" *-- "1" SemanticNumberMatch : wraps
    NumberPairComparison "1" *-- "1" AcceptanceDecision : uses
    EvaluationResult "1" *-- "*" NumberWithContext : contains
    EvaluationResult "1" *-- "*" SemanticNumberMatch : contains
    EvaluationResult "1" *-- "*" NumberPairComparison : contains
    EvaluationResult "1" *-- "1" AcceptanceDecision : overall
```

#### Data Model Descriptions

| Class | Purpose | Key Fields |
|-------|---------|------------|
| **NumberWithContext** | A number extracted from text with its surrounding context for embedding | `value`, `context_sentence`, `embedding` |
| **SemanticNumberMatch** | A matched pair with bi-encoder and cross-encoder scores | `bi_encoder_score`, `cross_encoder_score`, `combined_score` |
| **AcceptanceDecision** | Enum for evaluation outcomes | `ACCEPT`, `MARGINAL`, `REJECT` |
| **NumberPairComparison** | Detailed statistical comparison of a matched pair | `relative_error`, `is_within_tolerance` |
| **EvaluationResult** | Complete evaluation result with all metrics | `overall_score`, `overall_decision` |

---

### 2.2 Core Processing Classes

These classes implement the main processing logic.

```mermaid
classDiagram
    class SentenceTransformer {
        <<External Library>>
        +encode(texts, convert_to_numpy) ndarray
        +get_sentence_embedding_dimension() int
    }
    
    class CrossEncoder {
        <<External Library>>
        +predict(pairs) ndarray
    }
    
    class NumberContextExtractor {
        <<Service Class>>
        -int context_window
        -Dict~str,str~ patterns
        -Dict~str,Pattern~ compiled_patterns
        +__init__(context_window)
        +extract(text) List~NumberWithContext~
        -_find_sentence_boundary(text, pos, direction) int
        -_parse_number(text, number_type) float
    }
    
    class SemanticNumberMatcher {
        <<Service Class>>
        -SentenceTransformer bi_encoder
        -CrossEncoder cross_encoder
        -float bi_encoder_weight
        -float cross_encoder_weight
        -float similarity_threshold
        -int top_k_candidates
        +__init__(bi_encoder, cross_encoder, weights, threshold, top_k)
        +match(model_numbers, truth_numbers) List~SemanticNumberMatch~
        -_compute_embeddings(numbers) ndarray
        -_compute_cross_encoder_scores(model, truth, pairs) Dict
        -_get_confidence_level(score) str
    }
    
    class SentenceTransformerNumericalEvaluator {
        <<Orchestrator>>
        -NumberContextExtractor extractor
        -SemanticNumberMatcher matcher
        -float relative_tolerance
        -float absolute_tolerance
        -float order_of_magnitude_tolerance
        -float marginal_multiplier
        -float accept_threshold
        -float marginal_threshold
        +__init__(bi_encoder, cross_encoder, tolerances, thresholds)
        +evaluate(question, model_answer, ground_truth) EvaluationResult
        -_compare_pair(match) NumberPairComparison
    }
    
    SemanticNumberMatcher --> SentenceTransformer : uses
    SemanticNumberMatcher --> CrossEncoder : uses
    SentenceTransformerNumericalEvaluator --> NumberContextExtractor : uses
    SentenceTransformerNumericalEvaluator --> SemanticNumberMatcher : uses
    
    NumberContextExtractor ..> NumberWithContext : produces
    SemanticNumberMatcher ..> SemanticNumberMatch : produces
    SentenceTransformerNumericalEvaluator ..> EvaluationResult : produces
```

#### Class Responsibilities

| Class | Responsibility | External Dependencies |
|-------|---------------|----------------------|
| **NumberContextExtractor** | Extract numbers with regex patterns and capture surrounding context | None (pure Python) |
| **SemanticNumberMatcher** | Two-stage matching using bi-encoder retrieval + cross-encoder re-ranking | SentenceTransformer, CrossEncoder |
| **SentenceTransformerNumericalEvaluator** | Orchestrate full evaluation pipeline with statistical comparison | Uses all above |

---

### 2.3 Alternative Implementations

The notebook also includes alternative matcher implementations for comparison.

```mermaid
classDiagram
    class SemanticNumberMatcher {
        <<Base Implementation>>
        +match(model_numbers, truth_numbers) List~SemanticNumberMatch~
    }
    
    class BiEncoderOnlyMatcher {
        <<Alternative>>
        +match(model_numbers, truth_numbers) List~SemanticNumberMatch~
    }
    
    class NumericalAnswerEvaluationPipeline {
        <<Production Wrapper>>
        -Dict config
        -SentenceTransformer bi_encoder
        -CrossEncoder cross_encoder
        -SentenceTransformerNumericalEvaluator evaluator
        +__init__(bi_encoder_model, cross_encoder_model, weights, tolerances)
        +evaluate_single(question, model_answer, ground_truth) Dict
        +evaluate_batch(data) DataFrame
        +get_config() Dict
    }
    
    SemanticNumberMatcher <|-- BiEncoderOnlyMatcher : extends
    NumericalAnswerEvaluationPipeline --> SentenceTransformerNumericalEvaluator : wraps
```

#### Implementation Comparison

| Implementation | Bi-Encoder | Cross-Encoder | Speed | Accuracy |
|---------------|------------|---------------|-------|----------|
| **SemanticNumberMatcher** | ✅ Used for retrieval | ✅ Used for re-ranking | Medium | High |
| **BiEncoderOnlyMatcher** | ✅ Used for matching | ❌ Skipped | Fast | Medium |

---

### 2.4 Complete System Architecture

```mermaid
classDiagram
    %% External Dependencies
    class SentenceTransformer {
        <<sentence-transformers>>
        +encode()
    }
    
    class CrossEncoder {
        <<sentence-transformers>>
        +predict()
    }
    
    class cosine_similarity {
        <<sklearn>>
        +compute(X, Y)
    }
    
    %% Data Classes
    class NumberWithContext {
        <<dataclass>>
        +value: float
        +context_sentence: str
        +embedding: ndarray
    }
    
    class SemanticNumberMatch {
        <<dataclass>>
        +bi_encoder_score: float
        +cross_encoder_score: float
    }
    
    class NumberPairComparison {
        <<dataclass>>
        +relative_error: float
        +decision: AcceptanceDecision
    }
    
    class EvaluationResult {
        <<dataclass>>
        +overall_score: float
        +overall_decision: AcceptanceDecision
    }
    
    %% Processing Classes
    class NumberContextExtractor {
        +extract(text)
    }
    
    class SemanticNumberMatcher {
        +match(model_nums, truth_nums)
    }
    
    class SentenceTransformerNumericalEvaluator {
        +evaluate(question, model, truth)
    }
    
    %% Production Pipeline
    class NumericalAnswerEvaluationPipeline {
        +evaluate_single()
        +evaluate_batch()
    }
    
    %% Relationships
    SemanticNumberMatcher --> SentenceTransformer
    SemanticNumberMatcher --> CrossEncoder
    SemanticNumberMatcher --> cosine_similarity
    
    NumberContextExtractor ..> NumberWithContext
    SemanticNumberMatcher ..> SemanticNumberMatch
    SentenceTransformerNumericalEvaluator ..> NumberPairComparison
    SentenceTransformerNumericalEvaluator ..> EvaluationResult
    
    SentenceTransformerNumericalEvaluator --> NumberContextExtractor
    SentenceTransformerNumericalEvaluator --> SemanticNumberMatcher
    
    NumericalAnswerEvaluationPipeline --> SentenceTransformerNumericalEvaluator
```

---

## 3. Sequence Diagrams

### 3.1 Main Evaluation Pipeline

This diagram shows the complete evaluation workflow from input to final decision.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Evaluator as SentenceTransformerNumericalEvaluator
    participant Extractor as NumberContextExtractor
    participant Matcher as SemanticNumberMatcher
    participant BiEnc as Bi-Encoder
    participant CrossEnc as Cross-Encoder
    
    User->>Evaluator: evaluate(question, model_answer, ground_truth)
    
    rect rgb(230, 245, 255)
        Note over Evaluator,Extractor: Step 1: Number Extraction
        Evaluator->>Extractor: extract(model_answer)
        Extractor-->>Evaluator: model_numbers[]
        
        Evaluator->>Extractor: extract(ground_truth)
        Extractor-->>Evaluator: truth_numbers[]
    end
    
    rect rgb(255, 245, 230)
        Note over Evaluator,CrossEnc: Step 2: Semantic Matching (Two-Stage)
        Evaluator->>Matcher: match(model_numbers, truth_numbers)
        
        Note over Matcher,BiEnc: Stage 1: Bi-Encoder Retrieval
        Matcher->>BiEnc: encode(model_contexts)
        BiEnc-->>Matcher: model_embeddings
        Matcher->>BiEnc: encode(truth_contexts)
        BiEnc-->>Matcher: truth_embeddings
        Matcher->>Matcher: cosine_similarity(model, truth)
        Matcher->>Matcher: Select top-k candidates per model number
        
        Note over Matcher,CrossEnc: Stage 2: Cross-Encoder Re-ranking
        Matcher->>CrossEnc: predict(candidate_pairs)
        CrossEnc-->>Matcher: pairwise_scores
        Matcher->>Matcher: Combine scores & greedy assignment
        Matcher-->>Evaluator: matches[]
    end
    
    rect rgb(230, 255, 230)
        Note over Evaluator: Step 3: Statistical Comparison
        loop For each match
            Evaluator->>Evaluator: _compare_pair(match)
            Note over Evaluator: Calculate errors:<br/>• Absolute error<br/>• Relative error<br/>• Order of magnitude
            Evaluator->>Evaluator: Check tolerance & assign decision
        end
    end
    
    rect rgb(255, 230, 255)
        Note over Evaluator: Step 4: Aggregate Scores & Decision
        Evaluator->>Evaluator: Calculate matching_coverage
        Evaluator->>Evaluator: Calculate numerical_accuracy_score
        Evaluator->>Evaluator: Calculate semantic_match_score
        Evaluator->>Evaluator: Compute overall_score (weighted)
        Evaluator->>Evaluator: Determine overall_decision
    end
    
    Evaluator-->>User: EvaluationResult
```

#### Pipeline Steps Explained

| Step | Component | Input | Output | Description |
|------|-----------|-------|--------|-------------|
| 1 | NumberContextExtractor | Text | NumberWithContext[] | Extract numbers with surrounding context |
| 2a | Bi-Encoder | Contexts | Embeddings | Encode text to dense vectors |
| 2b | Cosine Similarity | Embeddings | Similarity Matrix | Find candidate pairs |
| 2c | Cross-Encoder | Candidate Pairs | Pairwise Scores | Re-rank candidates |
| 3 | Comparator | Matches | Comparisons | Calculate error metrics |
| 4 | Decision Logic | All metrics | Decision | Determine accept/marginal/reject |

---

### 3.2 Number Extraction Flow

Detailed flow of how numbers are extracted from text with context.

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Extractor as NumberContextExtractor
    
    Caller->>Extractor: extract(text)
    
    Extractor->>Extractor: Initialize used_positions = {}
    
    loop For each number_type in priority order
        Note over Extractor: Types: scientific, percentage,<br/>currency, fraction, decimal_comma,<br/>integer_comma, decimal, integer
        
        Extractor->>Extractor: compiled_patterns[number_type].finditer(text)
        
        loop For each match
            alt Position not already used
                Extractor->>Extractor: _parse_number(original_text, number_type)
                
                rect rgb(240, 248, 255)
                    Note over Extractor: Extract Context
                    Extractor->>Extractor: context_start = max(0, start - context_window)
                    Extractor->>Extractor: context_end = min(len, end + context_window)
                    Extractor->>Extractor: _find_sentence_boundary(before)
                    Extractor->>Extractor: _find_sentence_boundary(after)
                end
                
                Extractor->>Extractor: Create NumberWithContext
                Extractor->>Extractor: Add position to used_positions
            else Position overlaps
                Extractor->>Extractor: Skip (avoid duplicate extraction)
            end
        end
    end
    
    Extractor->>Extractor: Sort by start_pos
    Extractor-->>Caller: List[NumberWithContext]
```

#### Number Parsing Logic

```mermaid
flowchart TD
    A[Input Text] --> B{Check for multipliers}
    B -->|trillion| C[multiplier = 1e12]
    B -->|billion| D[multiplier = 1e9]
    B -->|million| E[multiplier = 1e6]
    B -->|none| F[multiplier = 1]
    
    C --> G{Number Type}
    D --> G
    E --> G
    F --> G
    
    G -->|percentage| H[Remove %, convert]
    G -->|currency| I[Remove $€£, convert]
    G -->|fraction| J[Parse a/b, divide]
    G -->|comma format| K[Remove commas, convert]
    G -->|decimal/integer| L[Direct convert]
    
    H --> M[Apply multiplier]
    I --> M
    J --> M
    K --> M
    L --> M
    
    M --> N[Return float value]
```

---

### 3.3 Two-Stage Semantic Matching

The core matching algorithm using bi-encoder retrieval and cross-encoder re-ranking.

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Matcher as SemanticNumberMatcher
    participant BiEnc as SentenceTransformer
    participant CrossEnc as CrossEncoder
    participant SKLearn as cosine_similarity
    
    Caller->>Matcher: match(model_numbers, truth_numbers)
    
    alt Either list is empty
        Matcher-->>Caller: Empty list []
    else Both lists have numbers
        rect rgb(227, 242, 253)
            Note over Matcher,SKLearn: STAGE 1: Bi-Encoder Retrieval
            
            Matcher->>Matcher: Get context texts for model numbers
            Matcher->>BiEnc: encode(model_texts, convert_to_numpy=True)
            BiEnc-->>Matcher: model_embeddings [N x D]
            
            Matcher->>Matcher: Get context texts for truth numbers
            Matcher->>BiEnc: encode(truth_texts, convert_to_numpy=True)
            BiEnc-->>Matcher: truth_embeddings [M x D]
            
            Matcher->>Matcher: Store embeddings in NumberWithContext.embedding
            
            Matcher->>SKLearn: cosine_similarity(model_emb, truth_emb)
            SKLearn-->>Matcher: similarity_matrix [N x M]
            
            Note over Matcher: Select Candidates
            loop For each model_idx in range(N)
                Matcher->>Matcher: top_k = argsort(similarity[model_idx])[-k:]
                loop For each truth_idx in top_k
                    alt similarity >= threshold * 0.5
                        Matcher->>Matcher: Add (model_idx, truth_idx) to candidates
                    end
                end
            end
        end
        
        rect rgb(255, 235, 238)
            Note over Matcher,CrossEnc: STAGE 2: Cross-Encoder Re-ranking
            
            Matcher->>Matcher: Prepare text pairs for candidates
            loop For each (model_idx, truth_idx) in candidates
                Matcher->>Matcher: pairs.append([model_context, truth_context])
            end
            
            Matcher->>CrossEnc: predict(pairs)
            CrossEnc-->>Matcher: raw_scores[]
            
            Matcher->>Matcher: Normalize: sigmoid(raw_scores) → [0, 1]
            
            Note over Matcher: Compute Combined Scores
            loop For each candidate pair
                Matcher->>Matcher: bi_score = similarity_matrix[model_idx, truth_idx]
                Matcher->>Matcher: ce_score = normalized_scores[pair_idx]
                Matcher->>Matcher: combined = bi_weight * bi_score + ce_weight * ce_score
            end
        end
        
        rect rgb(232, 245, 233)
            Note over Matcher: Greedy Assignment
            Matcher->>Matcher: Sort pairs by combined_score DESC
            Matcher->>Matcher: Initialize used_truth = {}
            
            loop For each (model_idx, truth_idx, scores) in sorted_pairs
                alt truth_idx not used AND combined >= threshold AND model not matched
                    Matcher->>Matcher: Create SemanticNumberMatch
                    Matcher->>Matcher: Add to matches[]
                    Matcher->>Matcher: Mark truth_idx as used
                end
            end
        end
        
        Matcher-->>Caller: matches[]
    end
```

#### Combined Score Calculation

```mermaid
flowchart LR
    subgraph BiEncoder["Bi-Encoder Score"]
        BE[Cosine Similarity<br/>from embeddings]
    end
    
    subgraph CrossEncoder["Cross-Encoder Score"]
        CE[Sigmoid normalized<br/>pairwise score]
    end
    
    subgraph Combined["Combined Score"]
        FORMULA["combined = 0.3 × bi_score + 0.7 × ce_score"]
    end
    
    subgraph Confidence["Match Confidence"]
        HIGH["≥ 0.8 → High"]
        MED["≥ 0.6 → Medium"]
        LOW["< 0.6 → Low"]
    end
    
    BE --> FORMULA
    CE --> FORMULA
    FORMULA --> HIGH
    FORMULA --> MED
    FORMULA --> LOW
```

---

### 3.4 Statistical Comparison Flow

How numerical metrics are computed for matched pairs.

```mermaid
sequenceDiagram
    autonumber
    participant Evaluator as SentenceTransformerNumericalEvaluator
    
    Note over Evaluator: For each SemanticNumberMatch
    
    Evaluator->>Evaluator: Extract values
    Note over Evaluator: model_val = match.model_number.value<br/>truth_val = match.truth_number.value
    
    rect rgb(240, 240, 255)
        Note over Evaluator: Calculate Error Metrics
        
        Evaluator->>Evaluator: absolute_error = |model_val - truth_val|
        
        alt truth_val ≠ 0
            Evaluator->>Evaluator: relative_error = absolute_error / |truth_val|
        else truth_val = 0
            Evaluator->>Evaluator: relative_error = ∞ if model_val ≠ 0 else 0
        end
        
        Evaluator->>Evaluator: percentage_error = relative_error × 100
        
        alt model_val > 0 AND truth_val > 0
            Evaluator->>Evaluator: magnitude_diff = |log₁₀(model) - log₁₀(truth)|
        else either is 0
            Evaluator->>Evaluator: magnitude_diff = 0 or ∞
        end
    end
    
    rect rgb(240, 255, 240)
        Note over Evaluator: Check Tolerance
        
        alt relative_error ≤ relative_tolerance OR absolute_error ≤ absolute_tolerance
            alt magnitude_diff ≤ order_magnitude_tolerance
                Evaluator->>Evaluator: is_within_tolerance = True
                Evaluator->>Evaluator: decision = ACCEPT
            else
                Evaluator->>Evaluator: is_within_tolerance = False
            end
        else
            Evaluator->>Evaluator: is_within_tolerance = False
        end
    end
    
    rect rgb(255, 240, 240)
        Note over Evaluator: Determine Decision
        
        alt is_within_tolerance
            Evaluator->>Evaluator: decision = ACCEPT
        else relative_error ≤ tolerance × marginal_multiplier AND magnitude OK
            Evaluator->>Evaluator: decision = MARGINAL
        else
            Evaluator->>Evaluator: decision = REJECT
        end
    end
    
    Evaluator->>Evaluator: Create NumberPairComparison
```

#### Overall Score Calculation

```mermaid
flowchart TD
    subgraph Metrics["Individual Metrics"]
        MC[matching_coverage<br/>= matched / truth_total]
        NA[numerical_accuracy<br/>= within_tol / matched]
        SS[semantic_score<br/>= mean(combined_scores)]
    end
    
    subgraph Weights["Weighted Combination"]
        W["overall_score = 0.4 × coverage + 0.4 × accuracy + 0.2 × semantic"]
    end
    
    subgraph Decision["Final Decision"]
        D1{score ≥ 0.7 AND<br/>accuracy ≥ 0.7?}
        D2{score ≥ 0.5?}
        ACCEPT["✅ ACCEPT"]
        MARGINAL["⚠️ MARGINAL"]
        REJECT["❌ REJECT"]
    end
    
    MC --> W
    NA --> W
    SS --> W
    
    W --> D1
    D1 -->|Yes| ACCEPT
    D1 -->|No| D2
    D2 -->|Yes| MARGINAL
    D2 -->|No| REJECT
```

---

### 3.5 Bi-Encoder Only Matching (Alternative)

Simplified flow when skipping cross-encoder re-ranking.

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Matcher as BiEncoderOnlyMatcher
    participant BiEnc as SentenceTransformer
    participant SKLearn as cosine_similarity
    
    Caller->>Matcher: match(model_numbers, truth_numbers)
    
    alt Either list is empty
        Matcher-->>Caller: Empty list []
    else
        Matcher->>BiEnc: encode(model_contexts)
        BiEnc-->>Matcher: model_embeddings
        
        Matcher->>BiEnc: encode(truth_contexts)
        BiEnc-->>Matcher: truth_embeddings
        
        Matcher->>SKLearn: cosine_similarity(model_emb, truth_emb)
        SKLearn-->>Matcher: similarity_matrix
        
        Note over Matcher: Direct Greedy Matching<br/>(No Cross-Encoder)
        
        Matcher->>Matcher: Collect all pairs with score ≥ threshold
        Matcher->>Matcher: Sort by bi_encoder_score DESC
        
        loop For each (model_idx, truth_idx, score) in sorted_pairs
            alt neither index already used
                Matcher->>Matcher: Create SemanticNumberMatch
                Note over Matcher: cross_encoder_score = 0.0<br/>combined_score = bi_encoder_score
                Matcher->>Matcher: Add to matches[]
            end
        end
        
        Matcher-->>Caller: matches[]
    end
```

---

## 4. Component Descriptions

### 4.1 NumberContextExtractor

**Purpose**: Extract numerical values from text with rich surrounding context for semantic matching.

**Key Features**:
- Regex-based extraction with priority ordering
- Handles multiple formats: scientific, percentage, currency, fractions
- Captures sentence-level context for better embeddings
- Prevents duplicate extraction with position tracking

**Supported Number Types**:

| Type | Pattern Example | Parsed Value |
|------|-----------------|--------------|
| Scientific | `1.5e10` | 15,000,000,000 |
| Percentage | `15%` | 15.0 |
| Currency | `$2.5 billion` | 2,500,000,000 |
| Fraction | `3/4` | 0.75 |
| Decimal (comma) | `1,234.56` | 1234.56 |
| Integer (comma) | `1,234,567` | 1234567 |
| Decimal | `3.14159` | 3.14159 |
| Integer | `42` | 42 |

---

### 4.2 SemanticNumberMatcher

**Purpose**: Match numbers from model answer to ground truth using semantic similarity.

**Two-Stage Algorithm**:

```mermaid
flowchart TB
    subgraph Stage1["Stage 1: Bi-Encoder (Fast Retrieval)"]
        S1A[Encode all number contexts]
        S1B[Compute N×M similarity matrix]
        S1C[Select top-k candidates per query]
        S1A --> S1B --> S1C
    end
    
    subgraph Stage2["Stage 2: Cross-Encoder (Accurate Re-ranking)"]
        S2A[Score candidate pairs directly]
        S2B[Combine bi + cross scores]
        S2C[Greedy assignment]
        S2A --> S2B --> S2C
    end
    
    Stage1 --> Stage2
```

**Configuration Options**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bi_encoder_weight` | 0.3 | Weight for bi-encoder score |
| `cross_encoder_weight` | 0.7 | Weight for cross-encoder score |
| `similarity_threshold` | 0.5 | Minimum combined score for match |
| `top_k_candidates` | 3 | Candidates per query from bi-encoder |

---

### 4.3 SentenceTransformerNumericalEvaluator

**Purpose**: Orchestrate full evaluation pipeline with statistical comparison.

**Tolerance Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `relative_tolerance` | 0.10 | 10% relative error allowed |
| `absolute_tolerance` | 0.01 | For small numbers near zero |
| `order_of_magnitude_tolerance` | 0.5 | Max log10 difference |
| `marginal_multiplier` | 2.0 | Multiplier for marginal threshold |

**Score Composition**:

| Component | Weight | Description |
|-----------|--------|-------------|
| Matching Coverage | 40% | % of truth numbers matched |
| Numerical Accuracy | 40% | % of matches within tolerance |
| Semantic Score | 20% | Average combined similarity score |

---

### 4.4 NumericalAnswerEvaluationPipeline

**Purpose**: Production-ready wrapper with batch processing support.

**Features**:
- Configurable model selection
- Single and batch evaluation modes
- Structured JSON output
- Configuration management

---

## 5. Data Flow Diagram

```mermaid
flowchart LR
    subgraph Inputs
        MA["Model Answer<br/>(text)"]
        GT["Ground Truth<br/>(text)"]
    end
    
    subgraph Extraction["Number Extraction"]
        E1["Extract from<br/>Model Answer"]
        E2["Extract from<br/>Ground Truth"]
    end
    
    subgraph Numbers["Extracted Numbers"]
        MN["Model Numbers<br/>(with context)"]
        TN["Truth Numbers<br/>(with context)"]
    end
    
    subgraph BiEncoder["Bi-Encoder Stage"]
        EMB["Generate<br/>Embeddings"]
        SIM["Similarity<br/>Matrix"]
        CAND["Top-k<br/>Candidates"]
    end
    
    subgraph CrossEncoder["Cross-Encoder Stage"]
        PAIR["Pair<br/>Scoring"]
        RANK["Re-ranking"]
        MATCH["Final<br/>Matches"]
    end
    
    subgraph Comparison["Statistical Comparison"]
        STATS["Error<br/>Metrics"]
        TOL["Tolerance<br/>Check"]
    end
    
    subgraph Output["Final Output"]
        SCORES["Scores"]
        DEC["Decision"]
        RESULT["EvaluationResult"]
    end
    
    MA --> E1
    GT --> E2
    E1 --> MN
    E2 --> TN
    
    MN --> EMB
    TN --> EMB
    EMB --> SIM
    SIM --> CAND
    
    CAND --> PAIR
    PAIR --> RANK
    RANK --> MATCH
    
    MATCH --> STATS
    STATS --> TOL
    
    TOL --> SCORES
    SCORES --> DEC
    DEC --> RESULT
```

---

## 6. Bi-Encoder vs Cross-Encoder Comparison

```mermaid
flowchart TB
    subgraph BiEncoder["🔵 Bi-Encoder"]
        BE1["Input: Single text"]
        BE2["Output: Dense vector (embedding)"]
        BE3["Comparison: Cosine similarity"]
        BE4["Speed: ⚡ Fast"]
        BE5["Use: Candidate retrieval"]
        BE1 --> BE2 --> BE3
    end
    
    subgraph CrossEncoder["🔴 Cross-Encoder"]
        CE1["Input: Text pair"]
        CE2["Output: Scalar score"]
        CE3["Comparison: Direct"]
        CE4["Speed: 🐢 Slower"]
        CE5["Use: Accurate ranking"]
        CE1 --> CE2 --> CE3
    end
```

### Detailed Comparison

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|------------|---------------|
| **Architecture** | Encode texts separately | Encode pair together |
| **Output** | Vector embeddings | Relevance score |
| **Comparison** | Cosine similarity | Direct output |
| **Complexity** | O(N + M) encoding | O(N × M) pairs |
| **Speed** | Very fast | Slower |
| **Accuracy** | Good | Excellent |
| **Best For** | Candidate retrieval | Final ranking |
| **Model Used** | all-MiniLM-L6-v2 | ms-marco-MiniLM-L-6-v2 |

### Why Two Stages?

```mermaid
flowchart LR
    A["100 model nums ×<br/>100 truth nums<br/>= 10,000 pairs"] --> B["Bi-Encoder:<br/>200 encodings<br/>+ matrix multiply"]
    B --> C["Top-k candidates:<br/>~300 pairs"]
    C --> D["Cross-Encoder:<br/>300 pair scorings"]
    D --> E["Final matches:<br/>~50-100 pairs"]
    
    style A fill:#ffcdd2
    style B fill:#c8e6c9
    style C fill:#fff9c4
    style D fill:#bbdefb
    style E fill:#c8e6c9
```

**Without two stages**: 10,000 cross-encoder calls (slow)
**With two stages**: 200 bi-encoder calls + 300 cross-encoder calls (fast)

---

## 7. Configuration Options

### Model Selection Guide

```mermaid
flowchart TD
    START[Select Models] --> Q1{Speed vs Accuracy?}
    
    Q1 -->|Speed Priority| FAST["all-MiniLM-L6-v2 (bi)<br/>ms-marco-MiniLM-L-6-v2 (cross)"]
    Q1 -->|Accuracy Priority| ACC["all-mpnet-base-v2 (bi)<br/>stsb-roberta-large (cross)"]
    Q1 -->|Balanced| BAL["all-MiniLM-L6-v2 (bi)<br/>ms-marco-MiniLM-L-12-v2 (cross)"]
    
    FAST --> CONFIG1["bi_weight: 0.5<br/>ce_weight: 0.5"]
    ACC --> CONFIG2["bi_weight: 0.2<br/>ce_weight: 0.8"]
    BAL --> CONFIG3["bi_weight: 0.3<br/>ce_weight: 0.7"]
```

### Tolerance Recommendations by Domain

| Domain | relative_tolerance | absolute_tolerance | order_magnitude_tolerance |
|--------|-------------------|-------------------|--------------------------|
| Financial | 0.01 - 0.05 | 0.01 | 0.1 |
| Scientific | 0.005 - 0.02 | 0.001 | 0.1 |
| General Knowledge | 0.10 - 0.15 | 1.0 | 0.5 |
| Weather/Temperature | 0.05 | 1.0 - 2.0 | 0.3 |
| Population | 0.05 - 0.10 | 1000 | 0.5 |

### Weight Configuration Effects

```mermaid
xychart-beta
    title "Impact of Bi-Encoder Weight on Results"
    x-axis [0.0, 0.3, 0.5, 0.7, 1.0]
    y-axis "Score" 0 --> 1
    bar [0.72, 0.78, 0.76, 0.73, 0.68]
    line [0.72, 0.78, 0.76, 0.73, 0.68]
```

| Configuration | Bi-Encoder Weight | Cross-Encoder Weight | Best For |
|--------------|-------------------|---------------------|----------|
| Cross-Encoder Only | 0.0 | 1.0 | Maximum accuracy |
| Default | 0.3 | 0.7 | Balanced |
| Equal | 0.5 | 0.5 | General use |
| Bi-Encoder Heavy | 0.7 | 0.3 | Speed priority |
| Bi-Encoder Only | 1.0 | 0.0 | Maximum speed |

---

## Summary

The Sentence Transformer-based Numerical Answer Evaluation system provides a robust, fast, and deterministic approach to evaluating numerical accuracy:

### Key Components

1. **NumberContextExtractor**: Regex-based extraction with rich context capture
2. **SemanticNumberMatcher**: Two-stage bi-encoder + cross-encoder matching
3. **SentenceTransformerNumericalEvaluator**: Full pipeline with statistical comparison
4. **NumericalAnswerEvaluationPipeline**: Production-ready wrapper

### Pipeline Flow

```
Input → Extract Numbers → Bi-Encoder Retrieval → Cross-Encoder Re-ranking → Statistical Comparison → Decision
```

### Advantages

| Feature | Benefit |
|---------|---------|
| **Deterministic** | Same inputs always produce same outputs |
| **Fast** | Process hundreds of evaluations per second |
| **Local** | No API costs, no data privacy concerns |
| **Configurable** | Tune weights and thresholds per domain |
| **Accurate** | Two-stage matching captures semantic meaning |

### Comparison with LLM-as-Judge

| Aspect | Sentence Transformers | LLM-as-Judge |
|--------|----------------------|--------------|
| Speed | ⚡ Very Fast | 🐢 Slow |
| Cost | Free (local) | API costs |
| Consistency | Deterministic | Variable |
| Context Understanding | Pattern-based | Nuanced |
| Explainability | Scores only | Can explain reasoning |
| Offline Capable | ✅ Yes | ❌ No |


