flowchart TD
    A[📥 Input: test.txt] --> B[🧠 test2.py<br/>Simple File Split Analysis]
    
    %% Simple Processing
    B --> B1[📄 Load text file content]
    B1 --> B2[📏 Split into exactly 5 equal sections<br/>by character count]
    
    %% Section Processing
    B2 --> C1[🧠 Section 1 Analysis<br/>Simple AI instruction]
    B2 --> C2[🧠 Section 2 Analysis<br/>Simple AI instruction]
    B2 --> C3[🧠 Section 3 Analysis<br/>Simple AI instruction]
    B2 --> C4[🧠 Section 4 Analysis<br/>Simple AI instruction]
    B2 --> C5[🧠 Section 5 Analysis<br/>Simple AI instruction]
    
    %% Simple Features
    D1[⚡ Simple Features:<br/>- Basic AI instructions<br/>- Look closely at data<br/>- Focus on tabular data<br/>- No complex processing] -.-> C1
    D1 -.-> C2
    D1 -.-> C3
    D1 -.-> C4
    D1 -.-> C5
    
    %% Delays
    C1 --> E1[⏱️ Wait 2 seconds]
    E1 --> C2
    C2 --> E2[⏱️ Wait 2 seconds]
    E2 --> C3
    C3 --> E3[⏱️ Wait 2 seconds]
    E3 --> C4
    C4 --> E4[⏱️ Wait 2 seconds]
    E4 --> C5
    
    %% Output
    C5 --> F[📤 Output: simple_analysis.json<br/>5 section responses + metadata]
    
    %% Styling
    classDef input fill:#e3f2fd,color:#000000
    classDef script fill:#f3e5f5,color:#000000
    classDef process fill:#fff3e0,color:#000000
    classDef analysis fill:#fce4ec,color:#000000
    classDef output fill:#e8f5e8,color:#000000
    classDef features fill:#f1f8e9,color:#000000
    classDef delay fill:#fff8e1,color:#000000
    
    class A input
    class B script
    class B1,B2 process
    class C1,C2,C3,C4,C5 analysis
    class F output
    class D1 features
    class E1,E2,E3,E4 delay