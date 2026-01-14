# Numerical Answer Evaluation using LLM-as-Judge

## UML Diagrams and Workflow Documentation

This document provides comprehensive UML diagrams (static class diagrams and sequence diagrams) with detailed explanations of the workflow present in the `numerical_answer_evaluation_using_llm-as-judge_demo.ipynb` notebook.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Static Class Diagrams](#2-static-class-diagrams)
   - [2.1 Pydantic Data Models](#21-pydantic-data-models)
   - [2.2 Core Processing Classes](#22-core-processing-classes)
   - [2.3 Result Dataclasses](#23-result-dataclasses)
   - [2.4 Complete System Architecture](#24-complete-system-architecture)
3. [Sequence Diagrams](#3-sequence-diagrams)
   - [3.1 Main Evaluation Pipeline](#31-main-evaluation-pipeline)
   - [3.2 Number Extraction Flow](#32-number-extraction-flow)
   - [3.3 Number Matching Flow](#33-number-matching-flow)
   - [3.4 Statistical Comparison Flow](#34-statistical-comparison-flow)
   - [3.5 LLM Judge Decision Flow](#35-llm-judge-decision-flow)
4. [Component Descriptions](#4-component-descriptions)
5. [Data Flow Diagram](#5-data-flow-diagram)
6. [Error Handling Strategy](#6-error-handling-strategy)

---

## 1. Architecture Overview

The LLM-as-Judge Numerical Answer Evaluation system is designed to evaluate the numerical accuracy of model-generated answers by comparing them to ground truth. The system leverages Large Language Models (LLMs) for semantic understanding while combining statistical metrics for quantitative assessment.

### High-Level Architecture

```mermaid
flowchart TB
    subgraph Input["📥 Input"]
        Q[Question]
        MA[Model Answer]
        GT[Ground Truth]
    end
    
    subgraph Pipeline["🔄 Evaluation Pipeline"]
        E[LLMNumberExtractor]
        M[LLMNumberMatcher]
        S[StatisticalComparator]
        J[LLMJudge]
    end
    
    subgraph Output["📤 Output"]
        R[FullEvaluationResult]
        D[Decision: Accept/Marginal/Reject]
    end
    
    Q --> E
    MA --> E
    GT --> E
    E --> M
    M --> S
    S --> J
    J --> R
    R --> D
```

---

## 2. Static Class Diagrams

### 2.1 Pydantic Data Models

These models define the structured data schemas used for LLM inputs and outputs.

```mermaid
classDiagram
    class ExtractedNumber {
        <<Pydantic BaseModel>>
        +float value
        +str original_text
        +str semantic_label
        +Optional~str~ unit
        +str context
    }
    
    class NumberExtractionResult {
        <<Pydantic BaseModel>>
        +List~ExtractedNumber~ numbers
        +str extraction_notes
    }
    
    class NumberPair {
        <<Pydantic BaseModel>>
        +ExtractedNumber model_number
        +ExtractedNumber truth_number
        +str match_confidence
        +str match_reasoning
    }
    
    class NumberMatchingResult {
        <<Pydantic BaseModel>>
        +List~NumberPair~ matched_pairs
        +List~ExtractedNumber~ unmatched_model_numbers
        +List~ExtractedNumber~ unmatched_truth_numbers
        +str matching_notes
    }
    
    class NumericalJudgment {
        <<Pydantic BaseModel>>
        +str decision
        +float overall_score
        +str reasoning
        +List~Dict~ pair_assessments
        +str recommendations
    }
    
    NumberExtractionResult "1" *-- "*" ExtractedNumber : contains
    NumberPair "1" *-- "2" ExtractedNumber : references
    NumberMatchingResult "1" *-- "*" NumberPair : contains
    NumberMatchingResult "1" *-- "*" ExtractedNumber : unmatched
```

#### Model Descriptions

| Model | Purpose | Key Fields |
|-------|---------|------------|
| **ExtractedNumber** | Represents a single number extracted from text | `value`, `semantic_label`, `context` |
| **NumberExtractionResult** | Contains all extracted numbers from a text | `numbers`, `extraction_notes` |
| **NumberPair** | A matched pair of numbers for comparison | `model_number`, `truth_number`, `match_confidence` |
| **NumberMatchingResult** | Result of the matching process | `matched_pairs`, `unmatched_*_numbers` |
| **NumericalJudgment** | Final judgment from the LLM judge | `decision`, `overall_score`, `reasoning` |

---

### 2.2 Core Processing Classes

These classes implement the main processing logic of the evaluation pipeline.

```mermaid
classDiagram
    class LLMNumberExtractor {
        <<Service Class>>
        -OpenAI client
        -str model
        -str EXTRACTION_PROMPT
        +__init__(client, model)
        +extract(text) NumberExtractionResult
        -_fallback_extraction(text) NumberExtractionResult
    }
    
    class LLMNumberMatcher {
        <<Service Class>>
        -OpenAI client
        -str model
        -str MATCHING_PROMPT
        +__init__(client, model)
        +match(model_numbers, truth_numbers) NumberMatchingResult
        -_format_numbers(numbers) str
        -_fallback_matching(model_numbers, truth_numbers) NumberMatchingResult
    }
    
    class StatisticalComparator {
        <<Service Class>>
        -float relative_tolerance
        -float absolute_tolerance
        -float order_magnitude_tolerance
        +__init__(relative_tolerance, absolute_tolerance, order_magnitude_tolerance)
        +compare_pair(pair) PairStatistics
        +is_within_tolerance(stats) bool
        +compare_all(matching_result) List~PairStatistics~
        +get_summary(stats_list) Dict
    }
    
    class LLMJudge {
        <<Service Class>>
        -OpenAI client
        -str model
        -str JUDGMENT_PROMPT
        +__init__(client, model)
        +judge(question, model_answer, ground_truth, matching_result, stats_list, summary, tolerance) NumericalJudgment
        -_format_pair_stats(matching_result, stats_list) str
        -_fallback_judgment(summary) NumericalJudgment
    }
    
    class LLMNumericalEvaluator {
        <<Orchestrator>>
        -OpenAI client
        -str model
        -LLMNumberExtractor extractor
        -LLMNumberMatcher matcher
        -StatisticalComparator comparator
        -LLMJudge judge
        -float relative_tolerance
        +__init__(client, model, relative_tolerance, absolute_tolerance)
        +evaluate(question, model_answer, ground_truth, verbose) FullEvaluationResult
    }
    
    LLMNumericalEvaluator "1" *-- "1" LLMNumberExtractor : uses
    LLMNumericalEvaluator "1" *-- "1" LLMNumberMatcher : uses
    LLMNumericalEvaluator "1" *-- "1" StatisticalComparator : uses
    LLMNumericalEvaluator "1" *-- "1" LLMJudge : uses
```

#### Class Responsibilities

| Class | Responsibility | LLM-Powered |
|-------|---------------|-------------|
| **LLMNumberExtractor** | Extract numbers with semantic labels from text | ✅ Yes |
| **LLMNumberMatcher** | Match corresponding numbers based on semantic meaning | ✅ Yes |
| **StatisticalComparator** | Compute error metrics for matched pairs | ❌ No |
| **LLMJudge** | Make final accept/marginal/reject decision | ✅ Yes |
| **LLMNumericalEvaluator** | Orchestrate the entire evaluation pipeline | N/A (Orchestrator) |

---

### 2.3 Result Dataclasses

These dataclasses store intermediate and final results.

```mermaid
classDiagram
    class PairStatistics {
        <<dataclass>>
        +float model_value
        +float truth_value
        +float absolute_error
        +float relative_error
        +float percentage_error
        +float order_of_magnitude_diff
        +bool is_exact_match
        +str semantic_label
        +str match_confidence
    }
    
    class FullEvaluationResult {
        <<dataclass>>
        +str question
        +str model_answer
        +str ground_truth
        +NumberExtractionResult model_numbers
        +NumberExtractionResult truth_numbers
        +NumberMatchingResult matching_result
        +List~PairStatistics~ pair_statistics
        +Dict summary_stats
        +NumericalJudgment judgment
    }
    
    FullEvaluationResult "1" *-- "2" NumberExtractionResult : contains
    FullEvaluationResult "1" *-- "1" NumberMatchingResult : contains
    FullEvaluationResult "1" *-- "*" PairStatistics : contains
    FullEvaluationResult "1" *-- "1" NumericalJudgment : contains
```

---

### 2.4 Complete System Architecture

```mermaid
classDiagram
    %% External Dependencies
    class OpenAI {
        <<External>>
        +beta.chat.completions.parse()
    }
    
    %% Pydantic Models
    class ExtractedNumber {
        <<Pydantic>>
        +float value
        +str semantic_label
        +str context
    }
    
    class NumberExtractionResult {
        <<Pydantic>>
        +List~ExtractedNumber~ numbers
    }
    
    class NumberPair {
        <<Pydantic>>
        +ExtractedNumber model_number
        +ExtractedNumber truth_number
    }
    
    class NumberMatchingResult {
        <<Pydantic>>
        +List~NumberPair~ matched_pairs
    }
    
    class NumericalJudgment {
        <<Pydantic>>
        +str decision
        +float overall_score
    }
    
    %% Service Classes
    class LLMNumberExtractor {
        +extract(text)
    }
    
    class LLMNumberMatcher {
        +match(model_nums, truth_nums)
    }
    
    class StatisticalComparator {
        +compare_pair(pair)
        +get_summary(stats)
    }
    
    class LLMJudge {
        +judge(...)
    }
    
    %% Orchestrator
    class LLMNumericalEvaluator {
        +evaluate(question, model_answer, ground_truth)
    }
    
    %% Result Classes
    class PairStatistics {
        <<dataclass>>
    }
    
    class FullEvaluationResult {
        <<dataclass>>
    }
    
    %% Relationships
    LLMNumberExtractor --> OpenAI : uses
    LLMNumberMatcher --> OpenAI : uses
    LLMJudge --> OpenAI : uses
    
    LLMNumberExtractor ..> NumberExtractionResult : produces
    LLMNumberMatcher ..> NumberMatchingResult : produces
    StatisticalComparator ..> PairStatistics : produces
    LLMJudge ..> NumericalJudgment : produces
    
    LLMNumericalEvaluator --> LLMNumberExtractor
    LLMNumericalEvaluator --> LLMNumberMatcher
    LLMNumericalEvaluator --> StatisticalComparator
    LLMNumericalEvaluator --> LLMJudge
    LLMNumericalEvaluator ..> FullEvaluationResult : produces
```

---

## 3. Sequence Diagrams

### 3.1 Main Evaluation Pipeline

This diagram shows the complete evaluation workflow from input to final judgment.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Evaluator as LLMNumericalEvaluator
    participant Extractor as LLMNumberExtractor
    participant Matcher as LLMNumberMatcher
    participant Comparator as StatisticalComparator
    participant Judge as LLMJudge
    participant LLM as OpenAI API
    
    User->>Evaluator: evaluate(question, model_answer, ground_truth)
    
    rect rgb(230, 245, 255)
        Note over Evaluator,LLM: Step 1: Number Extraction
        Evaluator->>Extractor: extract(model_answer)
        Extractor->>LLM: Parse request with EXTRACTION_PROMPT
        LLM-->>Extractor: NumberExtractionResult (model)
        Extractor-->>Evaluator: model_numbers
        
        Evaluator->>Extractor: extract(ground_truth)
        Extractor->>LLM: Parse request with EXTRACTION_PROMPT
        LLM-->>Extractor: NumberExtractionResult (truth)
        Extractor-->>Evaluator: truth_numbers
    end
    
    rect rgb(255, 245, 230)
        Note over Evaluator,LLM: Step 2: Number Matching
        Evaluator->>Matcher: match(model_numbers, truth_numbers)
        Matcher->>LLM: Parse request with MATCHING_PROMPT
        LLM-->>Matcher: NumberMatchingResult
        Matcher-->>Evaluator: matching_result
    end
    
    rect rgb(230, 255, 230)
        Note over Evaluator,Comparator: Step 3: Statistical Comparison
        Evaluator->>Comparator: compare_all(matching_result)
        loop For each matched pair
            Comparator->>Comparator: compare_pair(pair)
        end
        Comparator-->>Evaluator: pair_statistics[]
        Evaluator->>Comparator: get_summary(pair_statistics)
        Comparator-->>Evaluator: summary_stats
    end
    
    rect rgb(255, 230, 230)
        Note over Evaluator,LLM: Step 4: Final Judgment
        Evaluator->>Judge: judge(question, answers, matching_result, stats, summary)
        Judge->>Judge: _format_pair_stats()
        Judge->>LLM: Parse request with JUDGMENT_PROMPT
        LLM-->>Judge: NumericalJudgment
        Judge-->>Evaluator: judgment
    end
    
    Evaluator->>Evaluator: Create FullEvaluationResult
    Evaluator-->>User: FullEvaluationResult
```

#### Pipeline Steps Explained

| Step | Component | Input | Output | Description |
|------|-----------|-------|--------|-------------|
| 1a | LLMNumberExtractor | Model Answer | NumberExtractionResult | Extract numbers with semantic context from model answer |
| 1b | LLMNumberExtractor | Ground Truth | NumberExtractionResult | Extract numbers with semantic context from ground truth |
| 2 | LLMNumberMatcher | Two NumberExtractionResults | NumberMatchingResult | Match corresponding numbers semantically |
| 3 | StatisticalComparator | NumberMatchingResult | PairStatistics[] + Summary | Calculate error metrics |
| 4 | LLMJudge | All previous results | NumericalJudgment | Make final decision with reasoning |

---

### 3.2 Number Extraction Flow

Detailed flow of how numbers are extracted from text.

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Extractor as LLMNumberExtractor
    participant LLM as OpenAI API
    participant Fallback as Regex Fallback
    
    Caller->>Extractor: extract(text)
    
    Extractor->>Extractor: Format EXTRACTION_PROMPT with text
    
    alt LLM API Success
        Extractor->>LLM: beta.chat.completions.parse()
        Note over LLM: response_format=NumberExtractionResult
        Note over LLM: temperature=0.0
        LLM-->>Extractor: Parsed response
        Extractor->>Extractor: response.choices[0].message.parsed
        Extractor-->>Caller: NumberExtractionResult
    else LLM API Failure
        Extractor->>Fallback: _fallback_extraction(text)
        
        loop For each regex pattern
            Note over Fallback: Patterns: currency, percentage,<br/>large_number, decimal, integer
            Fallback->>Fallback: Find matches
            Fallback->>Fallback: Parse value & apply multiplier
            Fallback->>Fallback: Create ExtractedNumber
        end
        
        Fallback-->>Extractor: NumberExtractionResult
        Note over Extractor: extraction_notes="Fallback regex extraction used"
        Extractor-->>Caller: NumberExtractionResult
    end
```

#### Extraction Prompt Structure

The LLM is instructed to extract:
- **Value**: Numerical value in standard notation (e.g., "$2.5 billion" → 2,500,000,000)
- **Original Text**: The exact text representation
- **Semantic Label**: What the number represents (e.g., "revenue", "profit margin")
- **Unit**: Unit of measurement if applicable
- **Context**: Brief surrounding context

---

### 3.3 Number Matching Flow

How numbers from model answer and ground truth are matched semantically.

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Matcher as LLMNumberMatcher
    participant LLM as OpenAI API
    participant Fallback as Label Similarity Fallback
    
    Caller->>Matcher: match(model_numbers, truth_numbers)
    
    alt Empty Input Check
        Matcher->>Matcher: Check if either list is empty
        Matcher-->>Caller: NumberMatchingResult (empty, all unmatched)
    else Both Lists Have Numbers
        Matcher->>Matcher: _format_numbers(model_numbers)
        Matcher->>Matcher: _format_numbers(truth_numbers)
        
        alt LLM API Success
            Matcher->>LLM: beta.chat.completions.parse()
            Note over LLM: Semantic matching based on:<br/>1. Meaning (revenue ↔ sales)<br/>2. Context similarity<br/>3. Unit compatibility
            LLM-->>Matcher: NumberMatchingResult
            Matcher-->>Caller: NumberMatchingResult
        else LLM API Failure
            Matcher->>Fallback: _fallback_matching()
            
            loop For each model_number
                loop For each truth_number (unused)
                    Fallback->>Fallback: Calculate label word overlap
                    Fallback->>Fallback: Add unit match bonus (+0.3)
                    Fallback->>Fallback: Track best match if score > 0.2
                end
                Fallback->>Fallback: Create NumberPair or mark unmatched
            end
            
            Fallback-->>Matcher: NumberMatchingResult
            Matcher-->>Caller: NumberMatchingResult
        end
    end
```

#### Match Confidence Levels

| Confidence | Score Threshold | Description |
|------------|----------------|-------------|
| **High** | > 0.8 | Strong semantic match with same units |
| **Medium** | 0.5 - 0.8 | Good semantic match |
| **Low** | 0.2 - 0.5 | Weak match, primarily positional |

---

### 3.4 Statistical Comparison Flow

How statistical metrics are computed for each matched pair.

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Comparator as StatisticalComparator
    
    Caller->>Comparator: compare_all(matching_result)
    
    loop For each pair in matched_pairs
        Comparator->>Comparator: compare_pair(pair)
        
        Note over Comparator: Extract values:<br/>model_val = pair.model_number.value<br/>truth_val = pair.truth_number.value
        
        rect rgb(240, 240, 255)
            Note over Comparator: Calculate Metrics
            Comparator->>Comparator: absolute_error = |model_val - truth_val|
            
            alt truth_val ≠ 0
                Comparator->>Comparator: relative_error = abs_error / |truth_val|
            else truth_val = 0
                Comparator->>Comparator: relative_error = inf or 0
            end
            
            Comparator->>Comparator: percentage_error = relative_error × 100
            
            alt Both values > 0
                Comparator->>Comparator: magnitude_diff = |log10(model) - log10(truth)|
            else
                Comparator->>Comparator: magnitude_diff = 0 or inf
            end
            
            Comparator->>Comparator: is_exact = abs_error < 1e-9 or rel_error < 1e-9
        end
        
        Comparator->>Comparator: Create PairStatistics
    end
    
    Comparator-->>Caller: List[PairStatistics]
    
    Caller->>Comparator: get_summary(stats_list)
    
    rect rgb(255, 240, 240)
        Note over Comparator: Aggregate Summary
        Comparator->>Comparator: Count within_tolerance
        Comparator->>Comparator: Count exact_matches
        Comparator->>Comparator: Calculate accuracy_ratio
        Comparator->>Comparator: Calculate avg/max relative_error
        Comparator->>Comparator: Calculate avg absolute_error
    end
    
    Comparator-->>Caller: Summary Dict
```

#### Tolerance Check Logic

```mermaid
flowchart TD
    A[PairStatistics] --> B{|truth_value| < 1.0?}
    B -->|Yes| C{absolute_error ≤ absolute_tolerance?}
    C -->|Yes| D[✅ Within Tolerance]
    C -->|No| E[❌ Outside Tolerance]
    
    B -->|No| F{relative_error ≤ relative_tolerance?}
    F -->|No| E
    F -->|Yes| G{magnitude_diff ≤ magnitude_tolerance?}
    G -->|Yes| D
    G -->|No| E
```

---

### 3.5 LLM Judge Decision Flow

How the final judgment is made.

```mermaid
sequenceDiagram
    autonumber
    participant Caller
    participant Judge as LLMJudge
    participant LLM as OpenAI API
    participant Fallback as Rule-Based Fallback
    
    Caller->>Judge: judge(question, model_answer, ground_truth,<br/>matching_result, stats_list, summary, tolerance)
    
    Judge->>Judge: _format_pair_stats(matching_result, stats_list)
    Note over Judge: Format each pair:<br/>• Semantic label<br/>• Model vs Truth values<br/>• Relative error<br/>• Match confidence<br/>• Match reasoning
    
    Judge->>Judge: Build JUDGMENT_PROMPT
    Note over Judge: Include:<br/>- Question<br/>- Model Answer<br/>- Ground Truth<br/>- Pair Statistics<br/>- Summary Stats<br/>- Unmatched Counts
    
    alt LLM API Success
        Judge->>LLM: beta.chat.completions.parse()
        Note over LLM: response_format=NumericalJudgment
        
        LLM->>LLM: Analyze all evidence
        Note over LLM: Consider:<br/>• Error acceptability by data type<br/>• Compound error effects<br/>• Critical vs secondary numbers<br/>• Overall narrative correctness
        
        LLM-->>Judge: NumericalJudgment
        Judge-->>Caller: NumericalJudgment
    else LLM API Failure
        Judge->>Fallback: _fallback_judgment(summary)
        
        Fallback->>Fallback: Calculate accuracy_ratio
        
        alt accuracy_ratio ≥ 0.7
            Fallback->>Fallback: decision = "accept"
            Fallback->>Fallback: score = 0.8 + 0.2 × ratio
        else accuracy_ratio ≥ 0.5
            Fallback->>Fallback: decision = "marginal"
            Fallback->>Fallback: score = 0.4 + 0.4 × ratio
        else accuracy_ratio < 0.5
            Fallback->>Fallback: decision = "reject"
            Fallback->>Fallback: score = ratio × 0.5
        end
        
        Fallback-->>Judge: NumericalJudgment
        Judge-->>Caller: NumericalJudgment
    end
```

#### Decision Criteria

```mermaid
flowchart TD
    subgraph LLM_Decision["LLM-Based Decision (Primary)"]
        A[All Evidence] --> B[LLM Analysis]
        B --> C{Semantic Evaluation}
        C --> D[Consider error types]
        C --> E[Consider data domain]
        C --> F[Consider narrative impact]
        D --> G[NumericalJudgment]
        E --> G
        F --> G
    end
    
    subgraph Fallback_Decision["Rule-Based Fallback"]
        H[accuracy_ratio] --> I{ratio ≥ 0.7?}
        I -->|Yes| J[✅ ACCEPT<br/>score: 0.8-1.0]
        I -->|No| K{ratio ≥ 0.5?}
        K -->|Yes| L[⚠️ MARGINAL<br/>score: 0.4-0.7]
        K -->|No| M[❌ REJECT<br/>score: 0.0-0.25]
    end
```

---

## 4. Component Descriptions

### 4.1 LLMNumberExtractor

**Purpose**: Extract numerical values from text with semantic understanding.

**Key Features**:
- Uses structured output parsing with Pydantic models
- Converts various number formats to standard notation
- Captures semantic labels (what the number represents)
- Records surrounding context for better matching
- Has regex-based fallback for resilience

**Extracted Number Types**:
- Integers and decimals
- Percentages
- Currency amounts
- Large numbers with words (billion, million)
- Fractions
- Scientific notation

---

### 4.2 LLMNumberMatcher

**Purpose**: Match numbers from model answer to corresponding ground truth numbers.

**Matching Criteria**:
1. **Semantic Meaning**: "revenue" matches "sales", "profit margin" matches "margin"
2. **Context Similarity**: Numbers discussed in similar contexts
3. **Unit Compatibility**: Same or convertible units

**Output**:
- Matched pairs with confidence levels
- Unmatched numbers from both texts
- Matching reasoning for each pair

---

### 4.3 StatisticalComparator

**Purpose**: Compute quantitative error metrics for matched pairs.

**Computed Metrics**:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Absolute Error | \|model - truth\| | Raw difference |
| Relative Error | \|model - truth\| / \|truth\| | Proportional difference |
| Percentage Error | Relative Error × 100 | Human-readable proportion |
| Order of Magnitude | \|log₁₀(model) - log₁₀(truth)\| | Scale difference |

**Configurable Tolerances**:
- `relative_tolerance`: Default 10% (0.10)
- `absolute_tolerance`: Default 0.01 (for small numbers)
- `order_magnitude_tolerance`: Default 1.0 (within same order)

---

### 4.4 LLMJudge

**Purpose**: Make final judgment considering both statistical and semantic factors.

**Decision Options**:
| Decision | Meaning | Typical Conditions |
|----------|---------|-------------------|
| **accept** | Numerically accurate enough | Most pairs within tolerance |
| **marginal** | Borderline accuracy | Mixed results |
| **reject** | Too many or critical errors | Significant deviations |

**Evaluation Considerations**:
- Data type sensitivity (financial vs. general)
- Critical vs. secondary numbers
- Compound error effects
- Overall narrative correctness

---

### 4.5 LLMNumericalEvaluator

**Purpose**: Orchestrate the complete evaluation pipeline.

**Pipeline Steps**:
1. Extract numbers from model answer
2. Extract numbers from ground truth
3. Match corresponding numbers
4. Compute statistical metrics
5. Make final judgment

**Configuration**:
- `model`: LLM model to use (default: gpt-4o-mini)
- `relative_tolerance`: Error threshold (default: 10%)
- `absolute_tolerance`: Small number threshold (default: 0.01)

---

## 5. Data Flow Diagram

```mermaid
flowchart LR
    subgraph Inputs
        Q["Question"]
        MA["Model Answer"]
        GT["Ground Truth"]
    end
    
    subgraph Extraction["Number Extraction"]
        E1["Extract from<br/>Model Answer"]
        E2["Extract from<br/>Ground Truth"]
    end
    
    subgraph ExtractionResults["Extraction Results"]
        MN["Model Numbers<br/>(List of ExtractedNumber)"]
        TN["Truth Numbers<br/>(List of ExtractedNumber)"]
    end
    
    subgraph Matching["Number Matching"]
        M["Match Semantically"]
    end
    
    subgraph MatchResult["Match Result"]
        MP["Matched Pairs<br/>(List of NumberPair)"]
        UM["Unmatched Numbers"]
    end
    
    subgraph Stats["Statistical Analysis"]
        S["Compare Each Pair"]
        SUM["Aggregate Summary"]
    end
    
    subgraph StatsResult["Statistics"]
        PS["Pair Statistics<br/>(List of PairStatistics)"]
        SS["Summary Stats<br/>(Dict)"]
    end
    
    subgraph Judgment["Final Judgment"]
        J["LLM Judge"]
    end
    
    subgraph Output["Final Output"]
        NJ["NumericalJudgment<br/>• Decision<br/>• Score<br/>• Reasoning"]
        FER["FullEvaluationResult"]
    end
    
    MA --> E1
    GT --> E2
    E1 --> MN
    E2 --> TN
    MN --> M
    TN --> M
    M --> MP
    M --> UM
    MP --> S
    S --> PS
    PS --> SUM
    SUM --> SS
    
    Q --> J
    MA --> J
    GT --> J
    MP --> J
    PS --> J
    SS --> J
    UM --> J
    
    J --> NJ
    MN --> FER
    TN --> FER
    MP --> FER
    UM --> FER
    PS --> FER
    SS --> FER
    NJ --> FER
```

---

## 6. Error Handling Strategy

The system implements a multi-layer error handling approach:

```mermaid
flowchart TD
    subgraph Primary["Primary Path (LLM)"]
        P1[LLM API Call]
        P2[Structured Output Parsing]
        P3[Return Result]
    end
    
    subgraph Fallback["Fallback Path"]
        F1[Catch Exception]
        F2[Log Error]
        F3[Execute Fallback Logic]
        F4[Return Fallback Result]
    end
    
    P1 --> |Success| P2
    P2 --> |Success| P3
    P1 --> |Failure| F1
    P2 --> |Failure| F1
    F1 --> F2
    F2 --> F3
    F3 --> F4
```

### Fallback Mechanisms by Component

| Component | Primary Method | Fallback Method |
|-----------|---------------|-----------------|
| **LLMNumberExtractor** | LLM with structured output | Regex pattern matching |
| **LLMNumberMatcher** | LLM semantic matching | Label word overlap scoring |
| **LLMJudge** | LLM with full context | Accuracy ratio thresholds |

### Error Resilience Features

1. **Temperature = 0**: Ensures deterministic LLM outputs
2. **Structured Outputs**: Uses Pydantic models for type safety
3. **Graceful Degradation**: Falls back to rule-based methods on failure
4. **Empty Input Handling**: Handles edge cases where no numbers exist

---

## Summary

The LLM-as-Judge Numerical Answer Evaluation system provides a robust, multi-stage pipeline for evaluating numerical accuracy in generated text. By combining LLM-based semantic understanding with statistical metrics, it offers:

- **Semantic Awareness**: Understanding synonyms and context
- **Quantitative Rigor**: Precise error measurements
- **Explainability**: Detailed reasoning for decisions
- **Resilience**: Fallback mechanisms for reliability

The modular architecture allows for easy customization of tolerances, models, and evaluation criteria to suit different domains and use cases.


