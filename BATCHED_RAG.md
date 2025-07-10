flowchart TD
    A[📥 Input: test_extracted.json] --> B[🧠 create_embeddings.py<br/>Vector Embeddings Creator]
    
    %% Embedding Creation Phase
    B --> B1[📄 Load JSON Document<br/>pages + tables + metadata]
    B1 --> B2[🔧 Document Chunker<br/>chunk_size=512, overlap=50]
    
    %% Chunking Process
    B2 --> C1[📝 Process Text Content<br/>Split into overlapping chunks]
    B2 --> C2[📊 Process Table Content<br/>Format tables as readable text]
    B2 --> C3[📋 Process Metadata<br/>Create metadata chunks]
    
    %% Chunk Enhancement
    C1 --> D1[🏷️ Add Metadata Tags<br/>page numbers + types + indices]
    C2 --> D1
    C3 --> D1
    
    %% Embedding Generation
    D1 --> E1[🔐 Azure OpenAI Authentication]
    E1 --> E2[🤖 Generate Embeddings<br/>text-embedding-ada-002<br/>1536 dimensions, batch processing]
    
    %% Vector Storage
    E2 --> F1[💾 Create Vector Store<br/>vectors + metadata + texts]
    F1 --> F2[📚 Save to local_vector_store.pkl<br/>+ ChromaDB + FAISS options]
    F2 --> F3[📊 Generate chunk_analysis.json<br/>statistics + metadata]
    
    %% RAG Retrieval Phase
    F2 --> G[🧠 batched_rag.py<br/>Enhanced Batched RAG System]
    
    %% User Query Input
    H[❓ User Question] --> G
    
    %% RAG Initialization
    G --> G1[🔐 Azure OpenAI Auth<br/>Embedding + Chat models]
    G1 --> G2[📚 Load Vector Store<br/>from local_vector_store.pkl]
    
    %% Multi-Pass Search Strategy
    G2 --> I1[🔍 Multi-Pass Search Pipeline]
    I1 --> I2[🎯 Pass 1: Semantic Similarity<br/>Generate query embedding + cosine similarity]
    I1 --> I3[🔤 Pass 2: Keyword Search<br/>Extract pharmaceutical keywords]
    I1 --> I4[🔗 Pass 3: Related Concepts<br/>Generate related queries + broader search]
    
    %% Search Enhancement
    I2 --> J1[📊 Enhanced Similarity Calculation<br/>Base + content type + keyword bonuses]
    I3 --> J1
    I4 --> J1
    
    %% Adaptive Batching
    J1 --> K1[📦 Adaptive Batch Creation<br/>Group by relevance + complexity]
    K1 --> K2[🎯 High Relevance Batch: 6 chunks<br/>Direct answers + high similarity]
    K1 --> K3[📊 Quantitative Batch: 8-12 chunks<br/>Tables + numerical data]
    K1 --> K4[📝 Contextual Batch: Variable size<br/>Supporting information]
    
    %% Enhanced AI Processing
    K2 --> L1[🧠 Pharmaceutical Expert Analysis<br/>Regulatory framework + evidence-based]
    K3 --> L2[🧠 Quantitative Data Analysis<br/>Statistical validation + patterns]
    K4 --> L3[🧠 Contextual Analysis<br/>Compliance + safety signals]
    
    %% AI Enhancement Features
    M1[⚡ RAG Enhancement Features:<br/>- Multi-pass retrieval strategy<br/>- Pharmaceutical domain expertise<br/>- Regulatory analysis framework<br/>- Adaptive batching by complexity<br/>- Evidence-based conclusions<br/>- Cross-reference validation<br/>- Safety signal detection] -.-> I1
    M1 -.-> L1
    M1 -.-> L2
    M1 -.-> L3
    
    %% Results Synthesis
    L1 --> N1[📋 Comprehensive Results Compilation<br/>All batch responses + coverage analysis]
    L2 --> N1
    L3 --> N1
    N1 --> N2[📊 Analysis Metadata<br/>Pages covered + similarity ranges + chunk types]
    N2 --> N3[💡 Generate Follow-up Questions<br/>Context-aware pharmaceutical insights]
    
    %% Final Output
    N3 --> O[📤 Output: enhanced_pharma_analysis.json<br/>Complete RAG analysis + batch details + follow-ups]
    
    %% Pipeline Connection
    P1[🔄 RAG Pipeline Flow:<br/>Embeddings → Storage → Retrieval → Analysis] -.-> F2
    P1 -.-> G2
    P1 -.-> I1
    P1 -.-> N1
    
    %% Styling
    classDef input fill:#e3f2fd,color:#000000
    classDef embedding fill:#f3e5f5,color:#000000
    classDef process fill:#fff3e0,color:#000000
    classDef chunking fill:#f8bbd9,color:#000000
    classDef storage fill:#e1f5fe,color:#000000
    classDef rag fill:#f3e5f5,color:#000000
    classDef search fill:#fff8e1,color:#000000
    classDef batch fill:#fce4ec,color:#000000
    classDef analysis fill:#e8f5e8,color:#000000
    classDef enhancement fill:#f1f8e9,color:#000000
    classDef output fill:#c8e6c9,color:#000000
    classDef pipeline fill:#f9fbe7,color:#000000
    
    class A,H input
    class B embedding
    class B1,B2 process
    class C1,C2,C3,D1 chunking
    class E1,E2,F1,F2,F3 storage
    class G,G1,G2 rag
    class I1,I2,I3,I4,J1 search
    class K1,K2,K3,K4 batch
    class L1,L2,L3,N1,N2,N3 analysis
    class M1 enhancement
    class O output
    class P1 pipeline