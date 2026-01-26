# Policy Network (pi_net) Visualization - Mermaid Diagrams

## Network Overview

**Type**: Policy Network (π-network)  
**Purpose**: Typically used for action selection in Reinforcement Learning  
**Architecture**: 3-layer fully connected network with Tanh activations

---

## Complete Architecture Flow

```mermaid
graph TB
    subgraph Input
        A["Input State/Observation<br/>🎮 Shape: (batch, obs_dim)<br/>📊 State features"]
    end
    
    subgraph "Layer 1: First Hidden Layer"
        B["Linear Layer<br/>⚡ obs_dim → 64<br/>📦 Parameters: obs_dim × 64 + 64<br/>W₁: (obs_dim×64), b₁: (64)"]
    end
    
    subgraph "Layer 2: Activation"
        C["Tanh Activation<br/>🎯 f(x) = tanh(x)<br/>Range: [-1, 1]<br/>📊 Output: (batch, 64)"]
    end
    
    subgraph "Layer 3: Second Hidden Layer"
        D["Linear Layer<br/>⚡ 64 → 64<br/>📦 Parameters: 4,160<br/>W₂: (64×64), b₂: (64)"]
    end
    
    subgraph "Layer 4: Activation"
        E["Tanh Activation<br/>🎯 f(x) = tanh(x)<br/>Range: [-1, 1]<br/>📊 Output: (batch, 64)"]
    end
    
    subgraph "Layer 5: Output Layer"
        F["Linear Layer<br/>⚡ 64 → act_dim<br/>📦 Parameters: 64 × act_dim + act_dim<br/>W₃: (64×act_dim), b₃: (act_dim)"]
    end
    
    subgraph Output
        G["Action Logits/Values<br/>🎲 Shape: (batch, act_dim)<br/>Used for action selection"]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style B fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style C fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style D fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style E fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style F fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style G fill:#fff3e0,stroke:#f57c00,stroke-width:3px
```

---

## Tensor Shape Transformations

```mermaid
graph LR
    subgraph "Batch Processing"
        S1["(batch, obs_dim)<br/>Input State"]
        S2["(batch, 64)<br/>After Linear 1"]
        S3["(batch, 64)<br/>After Tanh"]
        S4["(batch, 64)<br/>After Linear 2"]
        S5["(batch, 64)<br/>After Tanh"]
        S6["(batch, act_dim)<br/>Action Output"]
    end
    
    S1 -->|"Linear(obs_dim→64)"| S2
    S2 -->|"Tanh"| S3
    S3 -->|"Linear(64→64)"| S4
    S4 -->|"Tanh"| S5
    S5 -->|"Linear(64→act_dim)"| S6
    
    style S1 fill:#bbdefb
    style S2 fill:#d1c4e9
    style S3 fill:#c8e6c9
    style S4 fill:#d1c4e9
    style S5 fill:#c8e6c9
    style S6 fill:#ffccbc
```

---

## Network Layer Stack

```mermaid
flowchart TD
    Start([Start: Observation/State])
    
    Layer0["Layer 0: Input<br/>────────────<br/>Type: State Vector<br/>Dimensions: obs_dim<br/>Examples: position, velocity, etc."]
    
    Layer1["Layer 1: Linear<br/>────────────<br/>Type: Fully Connected<br/>Input → Hidden: obs_dim → 64<br/>Weights: obs_dim × 64<br/>Biases: 64"]
    
    Layer2["Layer 2: Tanh<br/>────────────<br/>Type: Activation<br/>Function: (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)<br/>Range: [-1, 1]<br/>Smooth & Bounded"]
    
    Layer3["Layer 3: Linear<br/>────────────<br/>Type: Fully Connected<br/>Hidden → Hidden: 64 → 64<br/>Weights: 4,096<br/>Biases: 64"]
    
    Layer4["Layer 4: Tanh<br/>────────────<br/>Type: Activation<br/>Function: (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)<br/>Range: [-1, 1]<br/>Smooth & Bounded"]
    
    Layer5["Layer 5: Linear<br/>────────────<br/>Type: Fully Connected<br/>Hidden → Output: 64 → act_dim<br/>Weights: 64 × act_dim<br/>Biases: act_dim"]
    
    End([Output: Action Logits])
    
    Start --> Layer0
    Layer0 --> Layer1
    Layer1 --> Layer2
    Layer2 --> Layer3
    Layer3 --> Layer4
    Layer4 --> Layer5
    Layer5 --> End
    
    style Start fill:#4caf50,color:#fff
    style End fill:#ff9800,color:#fff
    style Layer1 fill:#f8bbd0
    style Layer2 fill:#c8e6c9
    style Layer3 fill:#f8bbd0
    style Layer4 fill:#c8e6c9
    style Layer5 fill:#f8bbd0
```

---

## Detailed Network Architecture

```mermaid
graph TB
    subgraph "INPUT LAYER"
        I1["Observation Vector<br/>obs_dim features<br/>(e.g., state variables)"]
    end
    
    subgraph "HIDDEN LAYER 1"
        direction TB
        H11["Neuron 1<br/>w₁ᵀx + b₁"]
        H12["Neuron 2<br/>w₂ᵀx + b₂"]
        H13["Neuron 3<br/>w₃ᵀx + b₃"]
        H1dot["...<br/>(59 more)"]
        H164["Neuron 64<br/>w₆₄ᵀx + b₆₄"]
    end
    
    subgraph "TANH ACTIVATION 1"
        T11["h₁ = tanh(z₁)"]
        T12["h₂ = tanh(z₂)"]
        T1dot["..."]
        T164["h₆₄ = tanh(z₆₄)"]
    end
    
    subgraph "HIDDEN LAYER 2"
        H21["Neuron 1"]
        H22["Neuron 2"]
        H2dot["...<br/>(60 more)"]
        H264["Neuron 64"]
    end
    
    subgraph "TANH ACTIVATION 2"
        T21["a₁ = tanh(z'₁)"]
        T22["a₂ = tanh(z'₂)"]
        T2dot["..."]
        T264["a₆₄ = tanh(z'₆₄)"]
    end
    
    subgraph "OUTPUT LAYER"
        O1["Action 1<br/>logit₁"]
        O2["Action 2<br/>logit₂"]
        Odot["...<br/>(act_dim-2 more)"]
        Oact["Action act_dim<br/>logit_act_dim"]
    end
    
    I1 --> H11 & H12 & H13 & H1dot & H164
    
    H11 --> T11
    H12 --> T12
    H13 --> T1dot
    H164 --> T164
    
    T11 & T12 & T1dot & T164 --> H21 & H22 & H2dot & H264
    
    H21 --> T21
    H22 --> T22
    H2dot --> T2dot
    H264 --> T264
    
    T21 & T22 & T2dot & T264 --> O1 & O2 & Odot & Oact
    
    style I1 fill:#e3f2fd
    style H11 fill:#f8bbd0
    style H12 fill:#f8bbd0
    style T11 fill:#c8e6c9
    style T12 fill:#c8e6c9
    style H21 fill:#f8bbd0
    style H22 fill:#f8bbd0
    style T21 fill:#c8e6c9
    style T22 fill:#c8e6c9
    style O1 fill:#ffccbc
    style O2 fill:#ffccbc
```

---

## Parameter Distribution

```mermaid
graph TD
    subgraph "Total Parameters"
        Total["Total = 64×obs_dim + 64 + 64×64 + 64 + 64×act_dim + act_dim"]
    end
    
    subgraph "Layer 1 Parameters"
        L1W["Weights W₁<br/>Shape: (64, obs_dim)<br/>Count: 64 × obs_dim"]
        L1B["Biases b₁<br/>Shape: (64,)<br/>Count: 64"]
    end
    
    subgraph "Layer 2 Parameters"
        L2W["Weights W₂<br/>Shape: (64, 64)<br/>Count: 4,096"]
        L2B["Biases b₂<br/>Shape: (64,)<br/>Count: 64"]
    end
    
    subgraph "Layer 3 Parameters"
        L3W["Weights W₃<br/>Shape: (act_dim, 64)<br/>Count: 64 × act_dim"]
        L3B["Biases b₃<br/>Shape: (act_dim,)<br/>Count: act_dim"]
    end
    
    Total --> L1W & L1B
    Total --> L2W & L2B
    Total --> L3W & L3B
    
    style Total fill:#ffeb3b,stroke:#f57f17,stroke-width:3px
    style L1W fill:#f8bbd0
    style L1B fill:#f8bbd0
    style L2W fill:#ce93d8
    style L2B fill:#ce93d8
    style L3W fill:#f8bbd0
    style L3B fill:#f8bbd0
```

---

## Computation Flow Sequence

```mermaid
sequenceDiagram
    participant Obs as Observation<br/>(obs_dim)
    participant L1 as Linear 1
    participant T1 as Tanh 1
    participant L2 as Linear 2
    participant T2 as Tanh 2
    participant L3 as Linear 3
    participant Act as Action<br/>(act_dim)
    
    Obs->>L1: Input state
    Note over L1: z₁ = W₁x + b₁<br/>obs_dim → 64
    L1->>T1: (batch, 64)
    Note over T1: h₁ = tanh(z₁)<br/>Range: [-1, 1]
    T1->>L2: (batch, 64)
    Note over L2: z₂ = W₂h₁ + b₂<br/>64 → 64
    L2->>T2: (batch, 64)
    Note over T2: h₂ = tanh(z₂)<br/>Range: [-1, 1]
    T2->>L3: (batch, 64)
    Note over L3: logits = W₃h₂ + b₃<br/>64 → act_dim
    L3->>Act: Action logits
```

---

## Tanh Activation Function

```mermaid
graph LR
    subgraph "Tanh Properties"
        Input["Input x<br/>Range: (-∞, +∞)"]
        Formula["Formula:<br/>tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)"]
        Output["Output<br/>Range: (-1, +1)"]
        
        Props["Properties:<br/>✓ Zero-centered<br/>✓ Smooth gradient<br/>✓ Bounded output<br/>✓ Symmetric"]
    end
    
    Input --> Formula
    Formula --> Output
    Output --> Props
    
    style Input fill:#bbdefb
    style Formula fill:#c8e6c9
    style Output fill:#bbdefb
    style Props fill:#fff9c4
```

---

## Forward Pass Computation Graph

```mermaid
graph TB
    subgraph "Input"
        X["x ∈ ℝ^obs_dim<br/>(Observation)"]
    end
    
    subgraph "First Linear Transform"
        W1["W₁ ∈ ℝ^(64×obs_dim)"]
        b1["b₁ ∈ ℝ^64"]
        z1["z₁ = W₁ᵀx + b₁"]
    end
    
    subgraph "First Activation"
        h1["h₁ = tanh(z₁)<br/>∈ [-1, 1]^64"]
    end
    
    subgraph "Second Linear Transform"
        W2["W₂ ∈ ℝ^(64×64)"]
        b2["b₂ ∈ ℝ^64"]
        z2["z₂ = W₂ᵀh₁ + b₂"]
    end
    
    subgraph "Second Activation"
        h2["h₂ = tanh(z₂)<br/>∈ [-1, 1]^64"]
    end
    
    subgraph "Output Transform"
        W3["W₃ ∈ ℝ^(act_dim×64)"]
        b3["b₃ ∈ ℝ^act_dim"]
        y["y = W₃ᵀh₂ + b₃<br/>∈ ℝ^act_dim"]
    end
    
    X --> z1
    W1 --> z1
    b1 --> z1
    z1 --> h1
    h1 --> z2
    W2 --> z2
    b2 --> z2
    z2 --> h2
    h2 --> y
    W3 --> y
    b3 --> y
    
    style X fill:#e3f2fd
    style W1 fill:#ffebee
    style W2 fill:#ffebee
    style W3 fill:#ffebee
    style b1 fill:#ffe0b2
    style b2 fill:#ffe0b2
    style b3 fill:#ffe0b2
    style z1 fill:#f3e5f5
    style z2 fill:#f3e5f5
    style h1 fill:#e8f5e9
    style h2 fill:#e8f5e9
    style y fill:#fff3e0
```

---

## Network Characteristics Mind Map

```mermaid
mindmap
  root((π-Network<br/>pi_net))
    Architecture
      Type: Feedforward
      Layers: 5 total
      Hidden: 2 layers
      Depth: Deep
    Parameters
      Layer 1: 64×obs_dim + 64
      Layer 2: 4,096 + 64
      Layer 3: 64×act_dim + act_dim
      Trainable: 100%
    Activations
      Function: Tanh
      Count: 2
      Range: minus 1 to 1
      Properties: Smooth, Bounded
    Input/Output
      Input: obs_dim features
      Output: act_dim actions
      Batch: Variable
    Use Cases
      RL Policy Network
      Continuous Control
      Action Selection
      Actor in Actor-Critic
```

---

## Example Configurations

### Configuration 1: CartPole
```mermaid
graph LR
    Input["Input: 4 features<br/>(position, velocity,<br/>angle, angular velocity)"]
    Hidden1["64 neurons"]
    Hidden2["64 neurons"]
    Output["Output: 2 actions<br/>(left, right)"]
    
    Input -->|"Linear(4→64)<br/>+ Tanh"| Hidden1
    Hidden1 -->|"Linear(64→64)<br/>+ Tanh"| Hidden2
    Hidden2 -->|"Linear(64→2)"| Output
    
    style Input fill:#e3f2fd
    style Hidden1 fill:#f8bbd0
    style Hidden2 fill:#ce93d8
    style Output fill:#ffccbc
```

**Parameters**: 4×64 + 64 + 64×64 + 64 + 64×2 + 2 = **4,546 parameters**

### Configuration 2: LunarLander
```mermaid
graph LR
    Input["Input: 8 features<br/>(position, velocity,<br/>angle, etc.)"]
    Hidden1["64 neurons"]
    Hidden2["64 neurons"]
    Output["Output: 4 actions<br/>(do nothing, left,<br/>main, right)"]
    
    Input -->|"Linear(8→64)<br/>+ Tanh"| Hidden1
    Hidden1 -->|"Linear(64→64)<br/>+ Tanh"| Hidden2
    Hidden2 -->|"Linear(64→4)"| Output
    
    style Input fill:#e3f2fd
    style Hidden1 fill:#f8bbd0
    style Hidden2 fill:#ce93d8
    style Output fill:#ffccbc
```

**Parameters**: 8×64 + 64 + 64×64 + 64 + 64×4 + 4 = **4,996 parameters**

---

## Usage in Reinforcement Learning

```mermaid
graph TD
    subgraph "Policy Network in RL"
        State["Current State<br/>(observation)"]
        PiNet["π-Network<br/>(pi_net)"]
        Logits["Action Logits<br/>(unnormalized)"]
        
        Dist["Probability Distribution<br/>(e.g., Categorical/Normal)"]
        
        Sample["Sample Action<br/>(stochastic)"]
        Greedy["Argmax Action<br/>(deterministic)"]
        
        Env["Environment<br/>Execution"]
    end
    
    State --> PiNet
    PiNet --> Logits
    
    Logits --> Dist
    
    Dist --> Sample
    Dist --> Greedy
    
    Sample --> Env
    Greedy --> Env
    
    Env --> Reward["Reward"]
    Env --> NextState["Next State"]
    
    style State fill:#e3f2fd
    style PiNet fill:#f8bbd0
    style Logits fill:#fff9c4
    style Dist fill:#c8e6c9
    style Sample fill:#ffccbc
    style Greedy fill:#ffccbc
    style Env fill:#ce93d8
```

---

## Layer Summary Table

| Layer | Type | Input Shape | Output Shape | Parameters | Activation |
|-------|------|-------------|--------------|------------|------------|
| 0 | Input | - | (batch, obs_dim) | 0 | - |
| 1 | Linear | (batch, obs_dim) | (batch, 64) | 64×obs_dim + 64 | - |
| 2 | Tanh | (batch, 64) | (batch, 64) | 0 | tanh(x) |
| 3 | Linear | (batch, 64) | (batch, 64) | 4,096 + 64 = 4,160 | - |
| 4 | Tanh | (batch, 64) | (batch, 64) | 0 | tanh(x) |
| 5 | Linear | (batch, 64) | (batch, act_dim) | 64×act_dim + act_dim | - |

**Total Parameters**: **64×obs_dim + 64 + 4,160 + 64×act_dim + act_dim**

---

## Mathematical Formulation

### Complete Forward Pass

```
Input: x ∈ ℝ^obs_dim

Layer 1: z₁ = W₁x + b₁,  where W₁ ∈ ℝ^(64×obs_dim), b₁ ∈ ℝ^64
Layer 2: h₁ = tanh(z₁) ∈ [-1, 1]^64
Layer 3: z₂ = W₂h₁ + b₂, where W₂ ∈ ℝ^(64×64), b₂ ∈ ℝ^64
Layer 4: h₂ = tanh(z₂) ∈ [-1, 1]^64
Layer 5: y = W₃h₂ + b₃,  where W₃ ∈ ℝ^(act_dim×64), b₃ ∈ ℝ^act_dim

Output: y ∈ ℝ^act_dim (action logits)
```

### Tanh Function

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ) = (e²ˣ - 1) / (e²ˣ + 1)

Properties:
- tanh(0) = 0
- tanh(-x) = -tanh(x)  (odd function)
- lim_{x→∞} tanh(x) = 1
- lim_{x→-∞} tanh(x) = -1
- d/dx tanh(x) = 1 - tanh²(x)
```

---

## Network Advantages for RL

✅ **Tanh Activation**: Zero-centered, better gradient flow than sigmoid  
✅ **Bounded Output**: Activations in [-1, 1] prevent explosion  
✅ **Two Hidden Layers**: Good capacity without over-parameterization  
✅ **64 Units**: Standard size balancing expressiveness and efficiency  
✅ **No Final Activation**: Allows flexible output interpretation  

---

## Common RL Applications

| Algorithm | Use of pi_net | Output Processing |
|-----------|---------------|-------------------|
| **PPO** | Policy network | Softmax for discrete / Gaussian for continuous |
| **A2C/A3C** | Actor network | Softmax for action probabilities |
| **TRPO** | Policy network | KL-constrained updates |
| **SAC** | Stochastic policy | Squashed Gaussian (tanh) |
| **REINFORCE** | Policy network | Softmax for discrete actions |

---

## Network Visualization Key

🎮 Input/Output related  
⚡ Linear transformation  
🎯 Activation function  
📦 Parameter storage  
📊 Shape information  
🎲 Stochastic/probability elements  

---

**Generated for PyTorch nn.Sequential Policy Network**  
*Commonly used in Reinforcement Learning algorithms*
