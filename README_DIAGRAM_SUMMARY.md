flowchart TD
    %% Input Files
    A[📄 Input Document<br/>test.txt/test.xml] --> B[🤖 summary1.py]
    
    %% Summary1 Process
    B --> B1[Split into 5 sections]
    B1 --> B2[AI analyzes each section]
    B2 --> B3[Create summaries]
    B3 --> C[📋 doc_summary.json]
    
    %% Summary2 Process  
    C --> D[🧠 summary2.py]
    A --> D
    D --> D1[Phase 1: AI reads JSON]
    D1 --> D2[AI selects top 3 sections]
    D2 --> D3[Extract section numbers]
    D3 --> D4[Phase 2: Get sections from doc]
    D4 --> D5[AI deep research per section]
    D5 --> E[🔍 deep_research.json]
    
    %% Summary3 Process
    E --> F[🎯 summary3.py]
    F --> F1[Load all findings]
    F1 --> F2[AI synthesis with math rules]
    F2 --> F3[Add all sections: A+B+C=Total]
    F3 --> F4[Create final answer]
    F4 --> G[✅ final_answer.json]
    
    %% Key Features
    H[🔐 Authentication<br/>OAuth2 + Token Refresh]
    I[⚠️ Error Handling<br/>Retries + Rate Limiting]
    J[🔧 Features<br/>Progress + Validation]
    
    %% Connect features
    B -.-> H
    D -.-> H
    F -.-> H
    
    B -.-> I
    D -.-> I
    F -.-> I
    
    B -.-> J
    D -.-> J
    F -.-> J
    
    %% Styling with black text
    classDef inputFile fill:#e1f5fe,color:#000000
    classDef script fill:#f3e5f5,color:#000000
    classDef output fill:#e8f5e8,color:#000000
    classDef process fill:#fff3e0,color:#000000
    classDef system fill:#f1f8e9,color:#000000
    
    class A inputFile
    class B,D,F script
    class C,E,G output
    class B1,B2,B3,D1,D2,D3,D4,D5,F1,F2,F3,F4 process
    class H,I,J system