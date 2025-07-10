flowchart TD
    A[📥 Input: test.txt] --> B[🧠 test3.py<br/>Memory-Enhanced Analysis]
    
    %% Three Memory Systems
    B --> M1[🧠 Progressive Questioning System<br/>Conversation history + follow-ups]
    B --> M2[🧠 Document Memory System<br/>Findings database + cross-references]
    B --> M3[🧠 Iterative Deep-Dive Analyzer<br/>Multi-iteration analysis]
    
    %% File Processing
    B --> B1[📄 Load text file]
    B1 --> B2[📏 Split into 5 equal sections]
    
    %% Memory-Enhanced Processing
    B2 --> C1[🔄 Section 1 with Memory Context<br/>Extract entities + store findings]
    C1 --> M2
    C1 --> C2[🔄 Section 2 with Enhanced Context<br/>Use previous findings + memory]
    C2 --> M2
    C2 --> C3[🔄 Section 3 with Full Context<br/>Cross-references + patterns]
    C3 --> M2
    C3 --> C4[🔄 Section 4 with Deep Context<br/>Entity tracking + insights]
    C4 --> M2
    C4 --> C5[🔄 Section 5 with Complete Context<br/>Full memory integration]
    
    %% Memory Features
    M1 --> D1[💡 Generate Follow-up Questions<br/>Based on findings + context]
    M2 --> D2[🔗 Cross-Section Insights<br/>Multi-section entity tracking]
    M3 --> D3[📈 Iterative Refinement<br/>Progressive analysis depth]
    
    %% Output Generation
    C5 --> E1[📊 Memory Statistics<br/>Findings + entities + conversation]
    D1 --> E2[📋 Enhanced Results<br/>with memory insights]
    D2 --> E2
    D3 --> E2
    E1 --> E2
    
    E2 --> F[📤 Output: memory_enhanced_analysis.json<br/>Complete analysis + memory state + suggestions]
    
    %% Styling
    classDef input fill:#e3f2fd,color:#000000
    classDef script fill:#f3e5f5,color:#000000
    classDef memory fill:#e8eaf6,color:#000000
    classDef process fill:#fff3e0,color:#000000
    classDef analysis fill:#fce4ec,color:#000000
    classDef insights fill:#f3e5f5,color:#000000
    classDef output fill:#e8f5e8,color:#000000
    
    class A input
    class B script
    class M1,M2,M3 memory
    class B1,B2 process
    class C1,C2,C3,C4,C5 analysis
    class D1,D2,D3,E1 insights
    class E2,F output