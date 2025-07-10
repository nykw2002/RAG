flowchart TD
    A[📥 Input: test_extracted.json] --> B[🧠 test.py<br/>Memory-Based Direct Processing]
    
    %% Initialization
    B --> B1[🔐 Azure OpenAI Authentication<br/>with GPT-o3-mini option]
    B1 --> B2[📄 Load JSON document<br/>123 pages pharmaceutical data]
    
    %% Document Processing
    B2 --> B3[📏 Split document optimally<br/>o3-mini: 4 parts<br/>GPT-4o: 3 parts]
    B3 --> B4[📋 Convert each part to<br/>structured text format]
    
    %% Analysis Steps
    B4 --> C1[🧠 Step 1: Document Structure Analysis<br/>AI creates systematic search strategy]
    C1 --> C2[🔄 Step 2: Part-by-Part Processing<br/>Enhanced systematic reasoning]
    
    %% Part Processing Loop
    C2 --> D1[📄 Process Part 1<br/>with search strategy context]
    D1 --> D2[📄 Process Part 2<br/>with previous findings]
    D2 --> D3[📄 Process Part 3<br/>with accumulated context]
    D3 --> D4[📄 Process Part 4<br/>with full context]
    
    %% Enhanced Features
    E1[⚡ Enhanced Features:<br/>- GPT-o3-mini reasoning<br/>- Systematic analysis<br/>- Evidence tracking<br/>- Confidence scoring] -.-> C1
    E1 -.-> C2
    
    %% Final Synthesis
    D4 --> F1[🔗 Step 3: Final Synthesis<br/>Comprehensive consolidation]
    F1 --> F2[✅ Enhanced Results<br/>with verification & validation]
    
    %% Output
    F2 --> G[📤 Output: enhanced_reasoning_analysis.json<br/>Complete findings + strategy + metadata]
    
    %% Styling
    classDef input fill:#e3f2fd,color:#000000
    classDef script fill:#f3e5f5,color:#000000
    classDef process fill:#fff3e0,color:#000000
    classDef analysis fill:#fce4ec,color:#000000
    classDef output fill:#e8f5e8,color:#000000
    classDef features fill:#f1f8e9,color:#000000
    
    class A input
    class B script
    class B1,B2,B3,B4 process
    class C1,C2,D1,D2,D3,D4,F1,F2 analysis
    class G output
    class E1 features