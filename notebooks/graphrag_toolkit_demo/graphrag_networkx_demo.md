# GraphRAG NetworkX Demo — Architecture, Evaluation & Extension Guide

This document provides a comprehensive architectural analysis of the GraphRAG workflow implemented in `graphrag_networkx_demo.ipynb`, including UML diagrams, sequence flows, extension strategies, and evaluation metrics.

---

## 1. High-Level Pipeline Overview

The workflow processes raw text documents through an extraction–indexing–retrieval–generation pipeline that combines **vector similarity search** with **knowledge-graph traversal** for context-aware response generation.

```mermaid
flowchart TD
    DOCS["Raw Documents"] --> CHUNK["Chunking<br/>(RecursiveCharacterTextSplitter)"]
    CHUNK --> EXTRACT["Entity & Relationship<br/>Extraction (LLM)"]
    CHUNK --> EMBED["Embedding &<br/>Vector Store (Chroma)"]
    EXTRACT --> KG["Knowledge Graph<br/>(NetworkX DiGraph)"]

    Q["User Query"] --> VECT_SEARCH["Vector Similarity<br/>Search"]
    Q --> Q_EXTRACT["Query Entity<br/>Extraction"]
    EMBED --> VECT_SEARCH
    Q_EXTRACT --> GRAPH_TRAV["Graph Traversal<br/>(2-hop subgraph)"]
    KG --> GRAPH_TRAV

    VECT_SEARCH --> MERGE["Context Merge"]
    GRAPH_TRAV --> MERGE
    MERGE --> GEN["LLM Generation<br/>(ChatOpenAI)"]
    GEN --> RESP["Response"]
```

---

## 2. Static Class Diagram

```mermaid
classDiagram
    class GraphRAGConfig {
        +str llm_model
        +str embedding_model
        +float temperature
        +int max_tokens
        +int chunk_size
        +int chunk_overlap
        +int max_entities_per_chunk
        +int max_relationships_per_chunk
        +float min_entity_confidence
        +int top_k_documents
        +int top_k_graph_nodes
        +float similarity_threshold
        +str collection_name
        +str persist_directory
    }

    class Entity {
        +str name
        +str type
        +str description
        +Dict properties
        +float confidence
        +List~str~ source_chunks
    }

    class Relationship {
        +str source
        +str target
        +str type
        +str description
        +Dict properties
        +float confidence
        +List~str~ source_chunks
    }

    class KnowledgeGraph {
        +DiGraph graph
        +Dict~str, Entity~ entities
        +List~Relationship~ relationships
        +add_entity(Entity)
        +add_relationship(Relationship)
        +get_subgraph(str, int) DiGraph
        +get_entity_context(str) Dict
        +visualize(List, str, str) str
        -_get_node_color(str) str
    }

    class EntityRelationshipExtractor {
        -GraphRAGConfig config
        -ChatOpenAI llm
        -spacy.Language nlp
        +extract_entities_relationships(str) Tuple
        -_fallback_extraction(str) Tuple
    }

    class GraphRAGPipeline {
        -GraphRAGConfig config
        -KnowledgeGraph knowledge_graph
        -EntityRelationshipExtractor extractor
        -OpenAIEmbeddings embeddings
        -Chroma vector_store
        -RecursiveCharacterTextSplitter text_splitter
        -ChatOpenAI llm
        -Dict metrics
        +process_documents(List~str~, List~Dict~) Dict
        +retrieve(str) Dict
        +generate_response(str, Dict) str
        +query(str) Dict
        +get_query_metrics(Dict) Dict
        +visualize_graph(int) str
        +get_statistics() DataFrame
    }

    GraphRAGPipeline --> GraphRAGConfig : configured by
    GraphRAGPipeline --> KnowledgeGraph : owns
    GraphRAGPipeline --> EntityRelationshipExtractor : owns
    KnowledgeGraph --> Entity : stores
    KnowledgeGraph --> Relationship : stores
    EntityRelationshipExtractor --> GraphRAGConfig : configured by
    EntityRelationshipExtractor --> Entity : produces
    EntityRelationshipExtractor --> Relationship : produces
```

---

## 3. Document Ingestion Sequence

```mermaid
sequenceDiagram
    participant User
    participant Pipeline as GraphRAGPipeline
    participant Splitter as TextSplitter
    participant Extractor as EntityRelationshipExtractor
    participant LLM as ChatOpenAI
    participant KG as KnowledgeGraph
    participant VS as Chroma VectorStore

    User->>Pipeline: process_documents(texts, metadata)
    loop For each document
        Pipeline->>Splitter: split_text(text)
        Splitter-->>Pipeline: chunks[]
        loop For each chunk
            Pipeline->>Extractor: extract_entities_relationships(chunk)
            Extractor->>LLM: Structured extraction prompt
            LLM-->>Extractor: JSON {entities, relationships}
            alt LLM succeeds
                Extractor-->>Pipeline: (entities[], relationships[])
            else LLM fails
                Extractor->>Extractor: _fallback_extraction (spaCy NER)
                Extractor-->>Pipeline: (entities[], [])
            end
            loop For each entity
                Pipeline->>KG: add_entity(entity)
            end
            loop For each relationship
                Pipeline->>KG: add_relationship(relationship)
            end
        end
    end
    Pipeline->>VS: from_documents(all_chunks, embeddings)
    Pipeline-->>User: metrics dict
```

---

## 4. Query & Retrieval Sequence

```mermaid
sequenceDiagram
    participant User
    participant Pipeline as GraphRAGPipeline
    participant VS as Chroma VectorStore
    participant Extractor as EntityRelationshipExtractor
    participant KG as KnowledgeGraph
    participant LLM as ChatOpenAI

    User->>Pipeline: query(question)
    Pipeline->>Pipeline: retrieve(question)

    par Vector Search
        Pipeline->>VS: similarity_search_with_score(question, k)
        VS-->>Pipeline: (doc, score)[]
    and Entity Extraction from Query
        Pipeline->>Extractor: extract_entities_relationships(question)
        Extractor-->>Pipeline: query_entities[]
    end

    loop For each query entity found in KG
        Pipeline->>KG: get_entity_context(entity_name)
        KG-->>Pipeline: {entity, neighbors, predecessors, degree, pagerank}
        Pipeline->>KG: get_subgraph(entity_name, depth=2)
        KG-->>Pipeline: related_entities set
    end

    Pipeline->>Pipeline: generate_response(question, retrieval_results)
    Pipeline->>LLM: Generation prompt with vector + graph context
    LLM-->>Pipeline: response text
    Pipeline-->>User: {query, response, retrieval_results, metrics}
```

---

## 5. Entity Extraction — LLM with spaCy Fallback

```mermaid
flowchart TD
    INPUT["Text Chunk"] --> LLM_CALL["LLM Structured Extraction<br/>(ChatPromptTemplate → JSON)"]
    LLM_CALL --> PARSE{"JSON parse<br/>succeeds?"}
    PARSE -- Yes --> FILTER["Filter entities by<br/>min_entity_confidence"]
    PARSE -- No --> FALLBACK["spaCy NER Fallback<br/>(en_core_web_sm)"]
    FALLBACK --> MAP["Map spaCy labels<br/>to Entity types"]
    MAP --> LIMIT["Limit to<br/>max_entities_per_chunk"]
    FILTER --> VALIDATE["Validate relationships:<br/>both endpoints must<br/>exist in entity set"]
    VALIDATE --> LIMIT_REL["Limit to<br/>max_relationships_per_chunk"]
    LIMIT --> OUT["Return (entities, relationships)"]
    LIMIT_REL --> OUT
```

---

## 6. Hybrid Retrieval Strategy

The retrieval step combines two complementary signals:

```mermaid
flowchart LR
    subgraph VectorPath["Vector Retrieval Path"]
        direction TB
        V1["Embed query"] --> V2["Cosine similarity<br/>search in Chroma"]
        V2 --> V3["Top-k document<br/>chunks with scores"]
    end

    subgraph GraphPath["Graph Retrieval Path"]
        direction TB
        G1["Extract entities<br/>from query (LLM)"] --> G2["Look up entities<br/>in KnowledgeGraph"]
        G2 --> G3["get_entity_context:<br/>neighbors, predecessors,<br/>degree, PageRank"]
        G3 --> G4["get_subgraph:<br/>2-hop neighborhood"]
    end

    VectorPath --> MERGE["Merge into<br/>unified context"]
    GraphPath --> MERGE
    MERGE --> GEN["LLM Generation"]
```

**Vector path** excels at surface-level semantic similarity.
**Graph path** adds structural context — how entities relate, their centrality (PageRank), and multi-hop connections the vector search would miss entirely.

---

## 7. Knowledge Graph Visualization Flow

```mermaid
flowchart TD
    VIZ_CALL["visualize_graph(max_nodes)"] --> CHECK{"num nodes ><br/>max_nodes?"}
    CHECK -- Yes --> PR["Compute PageRank"] --> TOP["Select top-k<br/>by PageRank"]
    CHECK -- No --> ALL["Use full graph"]
    TOP --> PYVIS["KnowledgeGraph.visualize()"]
    ALL --> PYVIS
    PYVIS --> NET["Build pyvis Network:<br/>color-coded nodes,<br/>labeled edges"]
    NET --> HTML["generate_html()"]
    HTML --> IFRAME["Render via<br/>iframe srcdoc"]
```

---

## 8. Extensions for Latency & Performance

### 8.1 Extraction Parallelism

The current implementation processes chunks sequentially. Since each LLM extraction call is independent, chunks can be processed concurrently.

```mermaid
flowchart LR
    subgraph Current["Current: Sequential"]
        direction TB
        C1["Chunk 1"] --> C2["Chunk 2"] --> C3["Chunk 3"] --> C4["Chunk N"]
    end

    subgraph Proposed["Proposed: Concurrent"]
        direction TB
        P1["Chunk 1"]
        P2["Chunk 2"]
        P3["Chunk 3"]
        P4["Chunk N"]
    end

    Proposed --> MERGE["Merge entities<br/>+ relationships"]
```

Implementation: use `asyncio.gather` with the async OpenAI client, or `concurrent.futures.ThreadPoolExecutor` with rate-limit-aware batching. Expected speedup: **3–8× on ingestion** for typical corpus sizes.

### 8.2 Caching & Incremental Updates

| Technique | Latency Impact | Implementation |
|-----------|---------------|----------------|
| **Extraction cache** (hash chunk → cached entities) | Eliminates redundant LLM calls on re-runs | Content-hash keyed dict or SQLite |
| **Incremental graph updates** | Only process new/changed documents | Track processed chunk IDs in metadata |
| **Embedding cache** | Skip re-embedding unchanged chunks | ChromaDB already handles this if collection persists |
| **LLM response cache** | Avoid identical extraction calls | `langchain` `SQLiteCache` or Redis |

### 8.3 Graph Backend Upgrade

```mermaid
flowchart LR
    NX["NetworkX<br/>(in-memory)"] -->|"< 10k nodes"| OK["Sufficient"]
    NX -->|"> 10k nodes"| NEO["Migrate to<br/>Neo4j / FalkorDB"]
    NEO --> CYPHER["Cypher queries<br/>replace Python traversal"]
    CYPHER --> PERF["Sub-ms graph lookups<br/>at scale"]
```

NetworkX is single-threaded and memory-bound. For production graphs (>10k nodes), a dedicated graph database delivers orders-of-magnitude faster traversals and supports concurrent access.

### 8.4 Embedding Model Optimization

- Replace OpenAI API embeddings with a local model (e.g. `all-MiniLM-L6-v2`) to eliminate network round-trips during retrieval.
- Use quantized ONNX models for ~3× faster inference on CPU.
- Batch embed all chunks in one call instead of per-document.

---

## 9. Extensions for Robustness & Repeatability

### 9.1 Deterministic Extraction

LLM outputs are inherently stochastic. To improve repeatability:

| Strategy | Effect |
|----------|--------|
| **temperature=0** | Greedy decoding — near-deterministic output |
| **Seed parameter** (OpenAI `seed` field) | Reproducible sampling for a given prompt |
| **Structured output** (`response_format={"type": "json_object"}`) | Guarantees valid JSON, eliminates parse failures |
| **Schema validation** (Pydantic model) | Reject malformed extractions before they enter the graph |

### 9.2 Entity Resolution & Deduplication

The current pipeline treats entity names as exact-match keys. "Apple Inc.", "Apple", and "AAPL" would create three separate nodes.

```mermaid
flowchart TD
    RAW["Raw Entities"] --> NORM["Normalize:<br/>lowercase, strip,<br/>expand abbreviations"]
    NORM --> EMBED["Embed entity names"]
    EMBED --> CLUSTER["Cosine-similarity<br/>clustering<br/>(threshold > 0.9)"]
    CLUSTER --> CANON["Canonical name<br/>selection<br/>(most frequent form)"]
    CANON --> MERGE_NODES["Merge duplicate<br/>graph nodes"]
```

### 9.3 Confidence-Weighted Graph

Instead of binary edge presence, use the extraction confidence scores as edge weights. During retrieval, weight graph-context contributions by confidence. This down-ranks hallucinated edges and strengthens well-supported connections.

### 9.4 Multi-Pass Validation

Add a second LLM pass to validate extracted relationships:
1. Present the LLM with each `(source, relationship, target)` triple and the original text.
2. Ask: *"Is this relationship supported by the text? Yes/No + evidence."*
3. Remove triples that fail validation.

This trades ingestion latency for significantly higher graph quality.

---

## 10. Evaluation Metrics

### 10.1 Extraction Quality

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Entity Precision** | Fraction of extracted entities that are correct | Manual annotation on sample chunks |
| **Entity Recall** | Fraction of true entities that were found | Compare against gold-standard annotations |
| **Relation Precision** | Fraction of extracted relations that are correct | Manual review of (src, rel, tgt) triples |
| **Relation Recall** | Fraction of true relations that were found | Gold-standard comparison |
| **Entity F1** | Harmonic mean of entity precision and recall | Standard F1 formula |
| **Relation F1** | Harmonic mean of relation precision and recall | Standard F1 formula |

### 10.2 Retrieval Quality

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Recall@k** | Fraction of relevant chunks in top-k results | Requires relevance labels for queries |
| **MRR (Mean Reciprocal Rank)** | Position of first relevant result | 1/rank of first relevant hit, averaged |
| **Graph Hit Rate** | How often graph traversal finds query entities | Count of entity matches / total queries |
| **Context Coverage** | Are all facts needed to answer in the context? | Manual or LLM-as-judge annotation |

### 10.3 Generation Quality

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Faithfulness** | Does the answer only use facts from context? | LLM-as-judge or RAGAS faithfulness score |
| **Answer Relevancy** | Does the answer address the question? | LLM-as-judge or RAGAS relevancy |
| **Groundedness** | Can every claim be traced to a source? | Claim-level attribution check |
| **BLEU / ROUGE** | N-gram overlap with reference answer | Automated, requires reference answers |
| **BERTScore** | Semantic similarity to reference answer | Embedding-based, more robust than n-gram |

### 10.4 Graph Quality

| Metric | What It Measures | Current Implementation |
|--------|-----------------|----------------------|
| **Graph Density** | Connectivity of the graph | `nx.density()` — already reported |
| **Weakly Connected Components** | Fragmentation | `nx.number_weakly_connected_components()` — already reported |
| **Average Degree** | Typical connectivity per node | Already reported |
| **PageRank Entropy** | How evenly distributed centrality is | `scipy.stats.entropy(pagerank_values)` |
| **Edge Confidence Distribution** | Quality of extracted relationships | Histogram of confidence scores |

### 10.5 End-to-End Evaluation Framework

```mermaid
flowchart TD
    QA_SET["Evaluation QA Set<br/>(question + gold answer)"] --> RUN["Run pipeline.query()"]
    RUN --> EXTRACT_EVAL["Extraction Metrics<br/>(P / R / F1)"]
    RUN --> RETR_EVAL["Retrieval Metrics<br/>(Recall@k, MRR)"]
    RUN --> GEN_EVAL["Generation Metrics<br/>(Faithfulness, Relevancy,<br/>BERTScore)"]
    EXTRACT_EVAL --> REPORT["Evaluation Report"]
    RETR_EVAL --> REPORT
    GEN_EVAL --> REPORT
    REPORT --> COMPARE["Compare across<br/>config variations"]
```

Use the [RAGAS](https://docs.ragas.io/) framework for automated RAG evaluation: it computes faithfulness, answer relevancy, and context precision/recall without requiring gold answers for every question.

---

## 11. Summary

The `graphrag_networkx_demo` notebook implements a functional GraphRAG system with four main components — document processing, LLM-based entity/relationship extraction, hybrid vector+graph retrieval, and context-aware generation. The architecture is clean and extensible but currently optimized for clarity over production readiness. The key levers for improvement are:

- **Latency**: parallelized extraction, local embeddings, caching, graph DB backend
- **Robustness**: deterministic LLM settings, entity deduplication, multi-pass validation, schema enforcement
- **Evaluation**: extraction F1, retrieval recall, generation faithfulness via RAGAS, graph quality metrics
