# Numerical Answer Evaluation using Deterministic Heuristics

## UML Diagrams and Workflow Documentation

This document provides comprehensive UML diagrams (static class diagrams and sequence diagrams) with detailed explanations of the workflow present in the `numerical_answer_evaluation_using_deterministic_heuristics.ipynb` notebook.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Static Class Diagrams](#2-static-class-diagrams)
   - [2.1 Data Models and Dataclasses](#21-data-models-and-dataclasses)
   - [2.2 Core Processing Classes](#22-core-processing-classes)
   - [2.3 Complete System Architecture](#23-complete-system-architecture)
3. [Sequence Diagrams](#3-sequence-diagrams)
   - [3.1 Main Evaluation Pipeline](#31-main-evaluation-pipeline)
   - [3.2 Number Extraction Flow](#32-number-extraction-flow)
   - [3.3 Number Matching Flow](#33-number-matching-flow)
   - [3.4 Statistical Comparison Flow](#34-statistical-comparison-flow)
   - [3.5 Overall Decision Flow](#35-overall-decision-flow)
4. [Component Descriptions](#4-component-descriptions)
5. [Data Flow Diagram](#5-data-flow-diagram)
6. [Matching Strategies](#6-matching-strategies)
7. [Configuration and Thresholds](#7-configuration-and-thresholds)

---

## 1. Architecture Overview

The Deterministic Heuristics-based Numerical Answer Evaluation system uses **pure rule-based methods** (regex for extraction, keyword matching for pairing, and mathematical formulas for comparison) to evaluate numerical accuracy in model-generated answers without requiring any machine learning models or external APIs.

### High-Level Architecture

```mermaid
flowchart TB
    subgraph Input["📥 Input"]
        Q[Question]
        MA[Model Answer]
        GT[Ground Truth]
    end
    
    subgraph Extraction["🔍 Number Extraction"]
        E[NumberExtractor<br/>Regex-based]
    end
    
    subgraph Matching["🔗 Number Matching"]
        M[NumberMatcher<br/>Heuristic-based]
    end
    
    subgraph Comparison["📊 Statistical Comparison"]
        C[NumericalEvaluator<br/>Math formulas]
    end
    
    subgraph Output["📤 Output"]
        R[EvaluationResult]
        D[Decision: Accept/Marginal/Reject]
    end
    
    MA --> E
    GT --> E
    E --> M
    M --> C
    C --> R
    R --> D
```

### Key Characteristics

```mermaid
flowchart LR
    subgraph Characteristics["✨ Key Characteristics"]
        A["🔧 Deterministic<br/>Same input = Same output"]
        B["⚡ Fast<br/>No ML inference"]
        C["💰 Free<br/>No API costs"]
        D["🔒 Offline<br/>No network required"]
        E["📐 Rule-based<br/>Transparent logic"]
    end
```

---

## 2. Static Class Diagrams

### 2.1 Data Models and Dataclasses

These classes define the data structures used throughout the pipeline.

```mermaid
classDiagram
    class ExtractedNumber {
        <<dataclass>>
        +float value
        +str original_text
        +int start_pos
        +int end_pos
        +str context_before
        +str context_after
        +str number_type
    }
    
    class NumberPair {
        <<dataclass>>
        +ExtractedNumber model_number
        +ExtractedNumber truth_number
        +float match_score
        +str match_reason
    }
    
    class AcceptanceDecision {
        <<Enum>>
        ACCEPT = "accept"
        MARGINAL = "marginal"
        REJECT = "reject"
    }
    
    class PairComparison {
        <<dataclass>>
        +NumberPair pair
        +float absolute_error
        +float relative_error
        +float percentage_error
        +float order_of_magnitude_diff
        +bool is_within_tolerance
        +AcceptanceDecision decision
        +Dict details
    }
    
    class EvaluationResult {
        <<dataclass>>
        +str question
        +str model_answer
        +str ground_truth
        +List~PairComparison~ pair_comparisons
        +float overall_score
        +AcceptanceDecision overall_decision
        +Dict summary
    }
    
    NumberPair "1" *-- "2" ExtractedNumber : contains
    PairComparison "1" *-- "1" NumberPair : wraps
    PairComparison "1" *-- "1" AcceptanceDecision : uses
    EvaluationResult "1" *-- "*" PairComparison : contains
    EvaluationResult "1" *-- "1" AcceptanceDecision : overall
```

#### Data Model Descriptions

| Class | Purpose | Key Fields |
|-------|---------|------------|
| **ExtractedNumber** | A number extracted from text with position and context | `value`, `original_text`, `number_type` |
| **NumberPair** | A matched pair of numbers with match confidence | `model_number`, `truth_number`, `match_score` |
| **AcceptanceDecision** | Enum for evaluation outcomes | `ACCEPT`, `MARGINAL`, `REJECT` |
| **PairComparison** | Statistical comparison result for a pair | `relative_error`, `is_within_tolerance` |
| **EvaluationResult** | Complete evaluation with all comparisons | `overall_score`, `overall_decision` |

---

### 2.2 Core Processing Classes

These classes implement the main processing logic using deterministic heuristics.

```mermaid
classDiagram
    class NumberExtractor {
        <<Service Class>>
        -int context_window
        -Dict~str,str~ patterns
        -Dict~str,Pattern~ compiled_patterns
        +__init__(context_window)
        +extract(text) List~ExtractedNumber~
        -_parse_number(text, number_type) float
    }
    
    class NumberMatcher {
        <<Service Class>>
        -Dict~str,List~ indicator_words
        +__init__()
        +match(model_numbers, truth_numbers, strategy) List~NumberPair~
        -_match_by_position(model, truth) List~NumberPair~
        -_match_by_context(model, truth) List~NumberPair~
        -_match_hybrid(model, truth) List~NumberPair~
        -_get_context_keywords(num) set
        -_calculate_context_similarity(num1, num2) float
        -_get_semantic_category(num) Optional~str~
    }
    
    class NumericalEvaluator {
        <<Orchestrator>>
        -float relative_tolerance
        -float absolute_tolerance
        -float order_magnitude_tolerance
        -float acceptance_threshold
        -float marginal_threshold
        -NumberExtractor extractor
        -NumberMatcher matcher
        +__init__(tolerances, thresholds)
        +evaluate(question, model_answer, ground_truth) EvaluationResult
        +compare_pair(pair) PairComparison
        -_calculate_metrics(model_value, truth_value) Dict
        -_is_within_tolerance(metrics, truth_value) bool
        -_make_pair_decision(metrics, truth_value) AcceptanceDecision
    }
    
    NumericalEvaluator --> NumberExtractor : uses
    NumericalEvaluator --> NumberMatcher : uses
    
    NumberExtractor ..> ExtractedNumber : produces
    NumberMatcher ..> NumberPair : produces
    NumericalEvaluator ..> PairComparison : produces
    NumericalEvaluator ..> EvaluationResult : produces
```

#### Class Responsibilities

| Class | Responsibility | Method |
|-------|---------------|--------|
| **NumberExtractor** | Extract numbers from text using regex patterns | Pure regex matching |
| **NumberMatcher** | Match numbers using position, context, or hybrid strategy | Keyword overlap, Jaccard similarity |
| **NumericalEvaluator** | Orchestrate evaluation with statistical comparison | Mathematical formulas |

---

### 2.3 Complete System Architecture

```mermaid
classDiagram
    %% Enums
    class AcceptanceDecision {
        <<Enum>>
        ACCEPT
        MARGINAL
        REJECT
    }
    
    %% Data Classes
    class ExtractedNumber {
        <<dataclass>>
        +value: float
        +original_text: str
        +context_before: str
        +context_after: str
        +number_type: str
    }
    
    class NumberPair {
        <<dataclass>>
        +model_number: ExtractedNumber
        +truth_number: ExtractedNumber
        +match_score: float
        +match_reason: str
    }
    
    class PairComparison {
        <<dataclass>>
        +pair: NumberPair
        +relative_error: float
        +is_within_tolerance: bool
        +decision: AcceptanceDecision
    }
    
    class EvaluationResult {
        <<dataclass>>
        +overall_score: float
        +overall_decision: AcceptanceDecision
        +summary: Dict
    }
    
    %% Processing Classes
    class NumberExtractor {
        -patterns: Dict
        +extract(text)
    }
    
    class NumberMatcher {
        -indicator_words: Dict
        +match(model, truth, strategy)
    }
    
    class NumericalEvaluator {
        -tolerances
        +evaluate(question, model, truth)
        +compare_pair(pair)
    }
    
    %% Relationships
    NumericalEvaluator --> NumberExtractor
    NumericalEvaluator --> NumberMatcher
    
    NumberExtractor ..> ExtractedNumber
    NumberMatcher ..> NumberPair
    NumericalEvaluator ..> PairComparison
    NumericalEvaluator ..> EvaluationResult
    
    PairComparison --> AcceptanceDecision
    EvaluationResult --> AcceptanceDecision
```

---

## 3. Sequence Diagrams

### 3.1 Main Evaluation Pipeline

This diagram shows the complete evaluation workflow from input to final decision.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Evaluator as NumericalEvaluator
    participant Extractor as NumberExtractor
    participant Matcher as NumberMatcher
    
    User->>Evaluator: evaluate(question, model_answer, ground_truth)
    
    rect rgb(230, 245, 255)
        Note over Evaluator,Extractor: Step 1: Number Extraction
        Evaluator->>Extractor: extract(model_answer)
        Extractor->>Extractor: Apply regex patterns
        Extractor-->>Evaluator: model_numbers[]
        
        Evaluator->>Extractor: extract(ground_truth)
        Extractor->>Extractor: Apply regex patterns
        Extractor-->>Evaluator: truth_numbers[]
    end
    
    rect rgb(255, 245, 230)
        Note over Evaluator,Matcher: Step 2: Number Matching
        Evaluator->>Matcher: match(model_numbers, truth_numbers, "hybrid")
        Matcher->>Matcher: _match_by_context()
        Matcher->>Matcher: _match_by_position() for unmatched
        Matcher-->>Evaluator: pairs[]
    end
    
    rect rgb(230, 255, 230)
        Note over Evaluator: Step 3: Statistical Comparison
        loop For each pair
            Evaluator->>Evaluator: compare_pair(pair)
            Evaluator->>Evaluator: _calculate_metrics()
            Evaluator->>Evaluator: _is_within_tolerance()
            Evaluator->>Evaluator: _make_pair_decision()
        end
    end
    
    rect rgb(255, 230, 255)
        Note over Evaluator: Step 4: Overall Decision
        Evaluator->>Evaluator: Calculate accuracy_ratio
        Evaluator->>Evaluator: Calculate overall_score
        Evaluator->>Evaluator: Determine overall_decision
        Evaluator->>Evaluator: Build summary
    end
    
    Evaluator-->>User: EvaluationResult
```

#### Pipeline Steps Explained

| Step | Component | Input | Output | Method |
|------|-----------|-------|--------|--------|
| 1 | NumberExtractor | Text | ExtractedNumber[] | Regex pattern matching |
| 2 | NumberMatcher | Two number lists | NumberPair[] | Context similarity + position |
| 3 | NumericalEvaluator | NumberPair | PairComparison | Mathematical formulas |
| 4 | NumericalEvaluator | All comparisons | EvaluationResult | Threshold-based decision |

---

### 3.2 Number Extraction Flow

Detailed flow of how numbers are extracted from text using regex patterns.

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Extractor as NumberExtractor
    
    Caller->>Extractor: extract(text)
    
    Extractor->>Extractor: Initialize used_positions = set()
    
    loop For each number_type in priority order
        Note over Extractor: Priority Order:<br/>1. scientific<br/>2. percentage<br/>3. currency<br/>4. fraction<br/>5. decimal_comma<br/>6. integer_comma<br/>7. decimal<br/>8. integer
        
        Extractor->>Extractor: pattern = compiled_patterns[number_type]
        Extractor->>Extractor: pattern.finditer(text)
        
        loop For each regex match
            alt Position not overlapping
                Extractor->>Extractor: _parse_number(original_text, number_type)
                
                rect rgb(240, 248, 255)
                    Note over Extractor: Extract Context
                    Extractor->>Extractor: context_start = max(0, start - context_window)
                    Extractor->>Extractor: context_end = min(len, end + context_window)
                    Extractor->>Extractor: context_before = text[context_start:start]
                    Extractor->>Extractor: context_after = text[end:context_end]
                end
                
                Extractor->>Extractor: Create ExtractedNumber
                Extractor->>Extractor: used_positions.add((start, end))
            else Position overlaps
                Extractor->>Extractor: Skip (avoid duplicate)
            end
        end
    end
    
    Extractor->>Extractor: Sort by start_pos
    Extractor-->>Caller: List[ExtractedNumber]
```

#### Regex Patterns

```mermaid
flowchart TD
    subgraph Patterns["📐 Regex Patterns (Priority Order)"]
        P1["1️⃣ Scientific: [-+]?\d+\.?\d*[eE][-+]?\d+<br/>Example: 1.5e6, 3.2E-4"]
        P2["2️⃣ Percentage: [-+]?\d+\.?\d*\s*%<br/>Example: 15%, 23.4%"]
        P3["3️⃣ Currency: [$€£¥₹]\s*\d...<br/>Example: $100, €50.00"]
        P4["4️⃣ Fraction: \d+\s*/\s*\d+<br/>Example: 1/2, 3/4"]
        P5["5️⃣ Decimal w/comma: [-+]?\d{1,3}(?:,\d{3})*\.\d+<br/>Example: 1,234.56"]
        P6["6️⃣ Integer w/comma: [-+]?\d{1,3}(?:,\d{3})+<br/>Example: 1,234,567"]
        P7["7️⃣ Decimal: [-+]?\d+\.\d+<br/>Example: 3.14"]
        P8["8️⃣ Integer: [-+]?\d+<br/>Example: 42"]
        
        P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8
    end
```

#### Number Parsing Logic

```mermaid
flowchart TD
    A[Original Text] --> B{Number Type?}
    
    B -->|percentage| C["Remove %<br/>Remove commas<br/>Convert to float"]
    B -->|currency| D["Remove $€£¥₹<br/>Remove 'dollars/euros'<br/>Remove commas<br/>Convert to float"]
    B -->|fraction| E["Split by /<br/>Divide numerator/denominator"]
    B -->|comma format| F["Remove commas<br/>Convert to float"]
    B -->|scientific| G["Direct float conversion"]
    B -->|decimal/integer| H["Remove commas<br/>Convert to float"]
    
    C --> I[Return float value]
    D --> I
    E --> I
    F --> I
    G --> I
    H --> I
```

---

### 3.3 Number Matching Flow

How numbers from model answer and ground truth are matched using heuristics.

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Matcher as NumberMatcher
    
    Caller->>Matcher: match(model_numbers, truth_numbers, "hybrid")
    
    rect rgb(227, 242, 253)
        Note over Matcher: Phase 1: Context-Based Matching
        Matcher->>Matcher: _match_by_context()
        
        Matcher->>Matcher: Initialize used_truth = set()
        
        loop For each model_number
            Matcher->>Matcher: _get_semantic_category(model_num)
            Note over Matcher: Categories: revenue, profit,<br/>percentage, count, price, time, ratio
            
            loop For each truth_number (unused)
                Matcher->>Matcher: Calculate type_bonus (0.2 if same type)
                Matcher->>Matcher: _get_semantic_category(truth_num)
                Matcher->>Matcher: Calculate category_bonus (0.3 if same)
                Matcher->>Matcher: _calculate_context_similarity()
                Note over Matcher: Jaccard similarity of context keywords
                Matcher->>Matcher: total_score = 0.5*context + type + category
                Matcher->>Matcher: Track best match if score > threshold
            end
            
            alt Best match found (score > 0.2)
                Matcher->>Matcher: Create NumberPair with match_reason="context"
                Matcher->>Matcher: Mark truth_number as used
            end
        end
    end
    
    rect rgb(255, 243, 224)
        Note over Matcher: Phase 2: Position-Based Fallback
        Matcher->>Matcher: _match_by_position() for unmatched
        
        Matcher->>Matcher: Get unmatched model numbers
        Matcher->>Matcher: Get unmatched truth numbers
        Matcher->>Matcher: Pair by order (1st with 1st, etc.)
        Note over Matcher: match_reason="position"<br/>match_score=0.5 (medium confidence)
    end
    
    Matcher->>Matcher: Combine context_pairs + position_pairs
    Matcher-->>Caller: List[NumberPair]
```

#### Context Similarity Calculation

```mermaid
flowchart TD
    subgraph Input["Input Numbers"]
        N1["Number 1<br/>context_before + context_after"]
        N2["Number 2<br/>context_before + context_after"]
    end
    
    subgraph Tokenize["Tokenization"]
        T1["words1 = set(re.findall(r'\b\w+\b', context1))"]
        T2["words2 = set(re.findall(r'\b\w+\b', context2))"]
    end
    
    subgraph Calculate["Jaccard Similarity"]
        J["intersection = words1 ∩ words2<br/>union = words1 ∪ words2<br/>similarity = |intersection| / |union|"]
    end
    
    N1 --> T1
    N2 --> T2
    T1 --> J
    T2 --> J
    J --> R["Return similarity (0.0 - 1.0)"]
```

#### Semantic Category Detection

```mermaid
flowchart TD
    A[ExtractedNumber] --> B[Get context text]
    B --> C{Check indicator words}
    
    C -->|"revenue, sales, income"| D[Category: revenue]
    C -->|"profit, earnings, margin"| E[Category: profit]
    C -->|"percent, %, rate, growth"| F[Category: percentage]
    C -->|"total, number, count"| G[Category: count]
    C -->|"price, cost, $"| H[Category: price]
    C -->|"year, month, quarter"| I[Category: time]
    C -->|"ratio, proportion, fraction"| J[Category: ratio]
    C -->|"none match"| K{Check number_type}
    
    K -->|percentage| F
    K -->|currency| H
    K -->|fraction| J
    K -->|other| L[Category: None]
```

---

### 3.4 Statistical Comparison Flow

How numerical metrics are computed for matched pairs.

```mermaid
sequenceDiagram
    autonumber
    participant Evaluator as NumericalEvaluator
    
    Note over Evaluator: For each NumberPair
    
    Evaluator->>Evaluator: Extract values
    Note over Evaluator: model_val = pair.model_number.value<br/>truth_val = pair.truth_number.value
    
    Evaluator->>Evaluator: _calculate_metrics(model_val, truth_val)
    
    rect rgb(240, 240, 255)
        Note over Evaluator: Calculate Error Metrics
        
        Evaluator->>Evaluator: absolute_error = |model_val - truth_val|
        
        alt truth_val ≠ 0
            Evaluator->>Evaluator: relative_error = absolute_error / |truth_val|
        else truth_val = 0
            Evaluator->>Evaluator: relative_error = ∞ if model_val ≠ 0 else 0
        end
        
        Evaluator->>Evaluator: percentage_error = relative_error × 100
        
        alt model_val ≠ 0 AND truth_val ≠ 0
            Evaluator->>Evaluator: magnitude_diff = |log₁₀(|model|) - log₁₀(|truth|)|
        else either is 0
            Evaluator->>Evaluator: magnitude_diff = 0 if both 0 else ∞
        end
    end
    
    Evaluator->>Evaluator: _is_within_tolerance(metrics, truth_val)
    
    rect rgb(240, 255, 240)
        Note over Evaluator: Check Tolerance
        
        alt |truth_val| < 1.0
            Note over Evaluator: Use absolute tolerance for small numbers
            Evaluator->>Evaluator: within = absolute_error ≤ absolute_tolerance
        else |truth_val| ≥ 1.0
            Note over Evaluator: Use relative tolerance for larger numbers
            Evaluator->>Evaluator: within = relative_error ≤ relative_tolerance<br/>AND magnitude_diff ≤ order_magnitude_tolerance
        end
    end
    
    Evaluator->>Evaluator: _make_pair_decision(metrics, truth_val)
    
    rect rgb(255, 240, 240)
        Note over Evaluator: Make Decision
        
        alt is_within_tolerance
            Evaluator->>Evaluator: decision = ACCEPT
        else relative_error ≤ tolerance × 2
            Evaluator->>Evaluator: decision = MARGINAL
        else
            Evaluator->>Evaluator: decision = REJECT
        end
    end
    
    Evaluator->>Evaluator: Create PairComparison
```

#### Metrics Calculation Formulas

```mermaid
flowchart TB
    subgraph Formulas["📐 Error Metrics Formulas"]
        AE["Absolute Error<br/>|model - truth|"]
        RE["Relative Error<br/>|model - truth| / |truth|"]
        PE["Percentage Error<br/>relative_error × 100"]
        OM["Order of Magnitude<br/>|log₁₀(|model|) - log₁₀(|truth|)|"]
    end
    
    subgraph Examples["📊 Example: model=2.4B, truth=2.5B"]
        E1["AE = |2.4 - 2.5| = 0.1 billion"]
        E2["RE = 0.1 / 2.5 = 0.04 (4%)"]
        E3["PE = 0.04 × 100 = 4%"]
        E4["OM = |log(2.4) - log(2.5)| ≈ 0.018"]
    end
    
    AE --> E1
    RE --> E2
    PE --> E3
    OM --> E4
```

---

### 3.5 Overall Decision Flow

How the final evaluation decision is made.

```mermaid
sequenceDiagram
    autonumber
    participant Evaluator as NumericalEvaluator
    
    Note over Evaluator: After all pairs compared
    
    rect rgb(240, 248, 255)
        Note over Evaluator: Calculate Aggregate Statistics
        
        Evaluator->>Evaluator: accurate_count = count(is_within_tolerance=True)
        Evaluator->>Evaluator: accuracy_ratio = accurate_count / total_comparisons
        Evaluator->>Evaluator: avg_relative_error = mean(relative_errors)
        Evaluator->>Evaluator: avg_absolute_error = mean(absolute_errors)
    end
    
    rect rgb(255, 248, 240)
        Note over Evaluator: Determine Overall Decision
        
        alt accuracy_ratio ≥ acceptance_threshold (0.7)
            Evaluator->>Evaluator: overall_decision = ACCEPT
        else accuracy_ratio ≥ marginal_threshold (0.5)
            Evaluator->>Evaluator: overall_decision = MARGINAL
        else
            Evaluator->>Evaluator: overall_decision = REJECT
        end
    end
    
    rect rgb(240, 255, 240)
        Note over Evaluator: Calculate Overall Score
        
        loop For each comparison
            Evaluator->>Evaluator: score = 1.0 if ACCEPT, 0.5 if MARGINAL, 0.0 if REJECT
            Evaluator->>Evaluator: weighted_score = score × match_score
        end
        
        Evaluator->>Evaluator: overall_score = sum(weighted_scores) / sum(match_scores)
    end
    
    rect rgb(248, 240, 255)
        Note over Evaluator: Build Summary
        
        Evaluator->>Evaluator: summary = {<br/>  total_numbers_in_truth,<br/>  total_numbers_in_model,<br/>  matched_pairs,<br/>  accurate_pairs,<br/>  accuracy_ratio,<br/>  avg_relative_error,<br/>  decisions: {accept, marginal, reject}<br/>}
    end
    
    Evaluator->>Evaluator: Create EvaluationResult
```

#### Decision Thresholds

```mermaid
flowchart TD
    A[accuracy_ratio] --> B{≥ 0.7?}
    B -->|Yes| C["✅ ACCEPT<br/>70%+ numbers accurate"]
    B -->|No| D{≥ 0.5?}
    D -->|Yes| E["⚠️ MARGINAL<br/>50-70% numbers accurate"]
    D -->|No| F["❌ REJECT<br/><50% numbers accurate"]
    
    style C fill:#c8e6c9
    style E fill:#fff9c4
    style F fill:#ffcdd2
```

---

## 4. Component Descriptions

### 4.1 NumberExtractor

**Purpose**: Extract numerical values from text using regex patterns.

**Key Features**:
- Priority-ordered pattern matching to avoid overlaps
- Handles multiple formats: integers, decimals, percentages, scientific notation, currency, fractions
- Captures surrounding context for semantic matching
- Tracks positions to prevent duplicate extraction

**Supported Number Types**:

| Type | Regex Pattern | Example |
|------|--------------|---------|
| Scientific | `[-+]?\d+\.?\d*[eE][-+]?\d+` | `1.5e6`, `3.2E-4` |
| Percentage | `[-+]?\d+\.?\d*\s*%` | `15%`, `23.4%` |
| Currency | `[$€£¥₹]\s*\d...` | `$100`, `€50.00` |
| Fraction | `\d+\s*/\s*\d+` | `1/2`, `3/4` |
| Decimal (comma) | `[-+]?\d{1,3}(?:,\d{3})*\.\d+` | `1,234.56` |
| Integer (comma) | `[-+]?\d{1,3}(?:,\d{3})+` | `1,234,567` |
| Decimal | `[-+]?\d+\.\d+` | `3.14` |
| Integer | `[-+]?\d+` | `42` |

---

### 4.2 NumberMatcher

**Purpose**: Match numbers from model answer to ground truth using heuristic strategies.

**Matching Strategies**:

```mermaid
flowchart TD
    subgraph Strategies["🎯 Matching Strategies"]
        S1["Position-Based<br/>Match by order in text<br/>Score: 0.5 (medium)"]
        S2["Context-Based<br/>Match by semantic similarity<br/>Score: varies"]
        S3["Hybrid (Default)<br/>Context first, position fallback<br/>Best of both"]
    end
```

**Scoring Components**:

| Component | Weight | Description |
|-----------|--------|-------------|
| Context Similarity | 50% | Jaccard similarity of surrounding words |
| Type Bonus | +0.2 | Same number_type (e.g., percentage) |
| Category Bonus | +0.3 | Same semantic category (e.g., revenue) |

**Indicator Words for Categories**:

| Category | Keywords |
|----------|----------|
| Revenue | revenue, sales, income |
| Profit | profit, earnings, margin, net income |
| Percentage | percent, %, rate, growth, increase, decrease |
| Count | total, number, count, employees, users |
| Price | price, cost, value, $, dollar, euro |
| Time | year, month, quarter, q1, q2, q3, q4 |
| Ratio | ratio, proportion, fraction |

---

### 4.3 NumericalEvaluator

**Purpose**: Orchestrate full evaluation pipeline with statistical comparison.

**Configuration Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `relative_tolerance` | 0.10 | 10% relative error allowed |
| `absolute_tolerance` | 0.01 | For small numbers near zero |
| `order_magnitude_tolerance` | 1.0 | Max log10 difference |
| `acceptance_threshold` | 0.70 | 70% accurate for ACCEPT |
| `marginal_threshold` | 0.50 | 50% accurate for MARGINAL |

**Computed Metrics**:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Absolute Error | \|model - truth\| | Raw difference |
| Relative Error | \|model - truth\| / \|truth\| | Proportional difference |
| Percentage Error | Relative Error × 100 | Human-readable |
| Order of Magnitude | \|log₁₀(model) - log₁₀(truth)\| | Scale check |

---

## 5. Data Flow Diagram

```mermaid
flowchart LR
    subgraph Inputs
        MA["Model Answer<br/>(text)"]
        GT["Ground Truth<br/>(text)"]
    end
    
    subgraph Extraction["Number Extraction"]
        E1["Regex<br/>Matching"]
        E2["Context<br/>Capture"]
    end
    
    subgraph Numbers["Extracted Numbers"]
        MN["Model Numbers<br/>(with context)"]
        TN["Truth Numbers<br/>(with context)"]
    end
    
    subgraph Matching["Number Matching"]
        CTX["Context-Based<br/>Matching"]
        POS["Position-Based<br/>Fallback"]
    end
    
    subgraph Pairs["Matched Pairs"]
        MP["NumberPair[]<br/>(with scores)"]
    end
    
    subgraph Comparison["Statistical Comparison"]
        CALC["Calculate<br/>Errors"]
        TOL["Check<br/>Tolerance"]
        DEC["Make<br/>Decision"]
    end
    
    subgraph Output["Final Output"]
        COMP["PairComparison[]"]
        SCORE["Overall Score"]
        DECISION["Overall Decision"]
        RESULT["EvaluationResult"]
    end
    
    MA --> E1
    GT --> E1
    E1 --> E2
    E2 --> MN
    E2 --> TN
    
    MN --> CTX
    TN --> CTX
    CTX --> POS
    POS --> MP
    
    MP --> CALC
    CALC --> TOL
    TOL --> DEC
    DEC --> COMP
    
    COMP --> SCORE
    COMP --> DECISION
    SCORE --> RESULT
    DECISION --> RESULT
```

---

## 6. Matching Strategies

### Strategy Comparison

```mermaid
flowchart TB
    subgraph Position["📍 Position-Based"]
        P1["Pros:<br/>• Simple<br/>• Fast<br/>• Works for ordered data"]
        P2["Cons:<br/>• Ignores context<br/>• Breaks on reordering<br/>• Medium confidence only"]
    end
    
    subgraph Context["📝 Context-Based"]
        C1["Pros:<br/>• Semantic matching<br/>• Order-independent<br/>• Higher confidence"]
        C2["Cons:<br/>• Slower<br/>• Depends on context quality<br/>• May miss with poor context"]
    end
    
    subgraph Hybrid["🔀 Hybrid (Default)"]
        H1["Best of Both:<br/>• Context-first<br/>• Position fallback<br/>• Maximizes matches"]
    end
```

### When to Use Each Strategy

| Strategy | Best For | Avoid When |
|----------|----------|------------|
| **Position** | Structured data, tables, lists | Text is reordered |
| **Context** | Natural language, varied phrasing | Poor/minimal context |
| **Hybrid** | Most cases, general use | N/A (recommended default) |

---

## 7. Configuration and Thresholds

### Tolerance Recommendations by Domain

```mermaid
flowchart TD
    subgraph Financial["💰 Financial Data"]
        F["relative_tolerance: 0.01-0.05<br/>absolute_tolerance: 0.01<br/>Reason: Exact numbers critical"]
    end
    
    subgraph Scientific["🔬 Scientific Data"]
        S["relative_tolerance: 0.05<br/>absolute_tolerance: 0.001<br/>Reason: Precision matters"]
    end
    
    subgraph General["📊 General Facts"]
        G["relative_tolerance: 0.10<br/>absolute_tolerance: 0.01<br/>Reason: Approximations OK"]
    end
    
    subgraph Casual["💬 Casual/Estimates"]
        C["relative_tolerance: 0.20-0.30<br/>absolute_tolerance: 1.0<br/>Reason: Order of magnitude"]
    end
```

### Threshold Configuration Table

| Use Case | relative_tolerance | absolute_tolerance | acceptance_threshold |
|----------|-------------------|-------------------|---------------------|
| Financial Reporting | 1-2% | 0.01 | 90% |
| Scientific Calculations | 5% | 0.001 | 80% |
| General Fact-Checking | 10% | 0.01 | 70% |
| Casual Comparisons | 20-30% | 1.0 | 60% |

### Decision Matrix

```mermaid
flowchart LR
    subgraph Input["Accuracy Ratio"]
        I1["≥ 70%"]
        I2["50-69%"]
        I3["< 50%"]
    end
    
    subgraph Decision["Decision"]
        D1["✅ ACCEPT"]
        D2["⚠️ MARGINAL"]
        D3["❌ REJECT"]
    end
    
    I1 --> D1
    I2 --> D2
    I3 --> D3
    
    style D1 fill:#c8e6c9
    style D2 fill:#fff9c4
    style D3 fill:#ffcdd2
```

---

## Summary

The Deterministic Heuristics-based Numerical Answer Evaluation system provides a fast, transparent, and cost-free approach to evaluating numerical accuracy:

### Key Components

1. **NumberExtractor**: Regex-based extraction with position tracking
2. **NumberMatcher**: Heuristic matching using context similarity and position
3. **NumericalEvaluator**: Mathematical comparison with configurable tolerances

### Pipeline Flow

```
Input → Regex Extraction → Heuristic Matching → Statistical Comparison → Decision
```

### Advantages

| Feature | Benefit |
|---------|---------|
| **Deterministic** | Same inputs always produce same outputs |
| **Fast** | No ML inference, pure computation |
| **Free** | No API costs or subscriptions |
| **Offline** | Works without network connection |
| **Transparent** | Clear, auditable logic |
| **Configurable** | Tunable tolerances per domain |

### Comparison with Other Approaches

| Aspect | Deterministic Heuristics | Sentence Transformers | LLM-as-Judge |
|--------|-------------------------|----------------------|--------------|
| Speed | ⚡⚡⚡ Fastest | ⚡⚡ Fast | ⚡ Slow |
| Cost | Free | Free (local) | API costs |
| Accuracy | Good for structured | Better semantic | Best nuanced |
| Consistency | Perfect | High | Variable |
| Setup | Minimal | Model download | API key |
| Explainability | Full | Scores only | Can explain |

### When to Use

✅ **Recommended For**:
- High-throughput evaluation pipelines
- Structured numerical data (tables, lists)
- Cost-sensitive applications
- Offline/air-gapped environments
- Simple numerical comparisons

⚠️ **Limitations**:
- Limited semantic understanding
- Relies on context keywords
- May struggle with complex paraphrasing
- Position matching can break on reordering


