# RAG Document Analysis Pipeline

A sophisticated three-stage document analysis system using Azure OpenAI for intelligent document processing and research synthesis.

## Overview

This pipeline processes large documents through three stages:
1. **Summary Generation** (`summary1.py`) - Creates structured summaries of document sections
2. **Deep Research** (`summary2.py`) - AI-driven section selection and deep analysis  
3. **Research Synthesis** (`summary3.py`) - Combines findings into comprehensive final answers

## Files

### Core Pipeline Scripts
- `summary1.py` - Document summarization and section analysis
- `summary2.py` - AI-driven section selection and deep research
- `summary3.py` - Research synthesis and final answer generation

### Support Scripts
- `create_embeddings.py` - Vector embedding generation
- `batched_rag.py` - Batch processing for RAG operations
- `extract.py` - Document text extraction utilities
- `pdf_to_json.py` - PDF to JSON conversion
- `test.py`, `test2.py`, `test3.py` - Various testing scripts

### Configuration
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (not tracked)
- `.gitignore` - Git ignore patterns

## Workflow

```
Input Document → summary1.py → doc_summary.json
                      ↓
                 summary2.py → deep_research.json  
                      ↓
                 summary3.py → final_answer.json
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/nykw2002/RAG.git
cd RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:
```
PING_FED_URL=your_auth_url
KGW_CLIENT_ID=your_client_id
KGW_CLIENT_SECRET=your_client_secret
KGW_ENDPOINT=your_azure_endpoint
AOAI_API_VERSION=your_api_version
CHAT_MODEL_DEPLOYMENT_NAME=your_model_name
```

## Usage

### Stage 1: Document Summarization
```bash
python summary1.py
```
- Input: Document file (e.g., `test.txt`)
- Output: `doc_summary.json`

### Stage 2: Deep Research
```bash
python summary2.py
```
- Input: `doc_summary.json` + original document
- Output: `deep_research.json`

### Stage 3: Research Synthesis
```bash
python summary3.py
```
- Input: `deep_research.json`
- Output: `final_answer.json`

## Features

- **AI-Driven Intelligence**: Uses Azure OpenAI for intelligent section selection
- **Adaptive Investigation**: Dynamic research strategy based on findings
- **Comprehensive Synthesis**: Mathematical accuracy in combining results
- **Evidence Tracking**: Maintains traceability throughout the pipeline
- **Error Handling**: Robust error handling and retry mechanisms
- **Flexible Configuration**: Easy customization of parameters

## Requirements

- Python 3.7+
- Azure OpenAI access
- Required Python packages (see `requirements.txt`)

## License

[Add your license here]
