#!/usr/bin/env python3
"""
RAG Embeddings Creator - Updated to use OpenAI's text-embedding-ada-002
Processes the extracted PDF JSON and creates vector embeddings for RAG system
"""

import json
import os
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
import requests
import time

# Vector storage options
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

class DocumentChunker:
    """Intelligently chunks document content for optimal RAG performance"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, max_chunk_size: int = None) -> List[str]:
        """Split text into overlapping chunks"""
        if max_chunk_size is None:
            max_chunk_size = self.chunk_size
            
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_json_data(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process JSON data and create chunks with metadata"""
        chunks = []
        
        # Process metadata
        if json_data.get('metadata'):
            metadata_text = f"Document metadata: {json.dumps(json_data['metadata'], indent=2)}"
            chunks.extend(self._create_chunks_with_metadata(
                metadata_text, 
                {
                    'type': 'metadata',
                    'source': json_data.get('source_file', 'unknown'),
                    'page_number': 0
                }
            ))
        
        # Process each page
        for page in json_data.get('pages', []):
            page_num = page.get('page_number', 0)
            
            # Process page text
            page_text = page.get('text', '').strip()
            if page_text:
                chunks.extend(self._create_chunks_with_metadata(
                    page_text,
                    {
                        'type': 'page_text',
                        'page_number': page_num,
                        'source': json_data.get('source_file', 'unknown'),
                        'page_info': page.get('page_info', {})
                    }
                ))
            
            # Process tables
            for table_idx, table in enumerate(page.get('tables', [])):
                table_text = self._format_table_text(table)
                if table_text:
                    chunks.extend(self._create_chunks_with_metadata(
                        table_text,
                        {
                            'type': 'table',
                            'page_number': page_num,
                            'table_index': table_idx,
                            'source': json_data.get('source_file', 'unknown'),
                            'table_info': {
                                'row_count': table.get('row_count', 0),
                                'column_count': table.get('column_count', 0)
                            }
                        }
                    ))
        
        return chunks
    
    def _create_chunks_with_metadata(self, text: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from text with associated metadata"""
        text_chunks = self.chunk_text(text)
        chunks = []
        
        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():  # Only add non-empty chunks
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(text_chunks),
                    'character_count': len(chunk_text),
                    'word_count': len(chunk_text.split())
                })
                
                chunks.append({
                    'text': chunk_text.strip(),
                    'metadata': chunk_metadata
                })
        
        return chunks
    
    def _format_table_text(self, table: Dict[str, Any]) -> str:
        """Convert table data to readable text format"""
        if not table.get('rows'):
            return ""
        
        rows = table['rows']
        formatted_rows = []
        
        for row in rows:
            if row and any(cell for cell in row if cell is not None):
                clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
                formatted_rows.append(" | ".join(clean_row))
        
        if formatted_rows:
            table_text = f"Table with {table.get('row_count', len(rows))} rows and {table.get('column_count', 0)} columns:\n"
            table_text += "\n".join(formatted_rows)
            return table_text
        
        return ""

class AzureOpenAIAuth:
    """Handle Azure OpenAI authentication"""
    
    def __init__(self):
        self.ping_fed_url = os.getenv('PING_FED_URL')
        self.kgw_client_id = os.getenv('KGW_CLIENT_ID')
        self.kgw_client_secret = os.getenv('KGW_CLIENT_SECRET')
        self.access_token = None
        self.token_expires_at = None
    
    def get_access_token(self) -> str:
        """Get or refresh access token"""
        if self.access_token and self.token_expires_at and time.time() < self.token_expires_at:
            return self.access_token
        
        # Request new token
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.kgw_client_id,
            'client_secret': self.kgw_client_secret
            # Removed the scope parameter that was causing the error
        }
        
        response = requests.post(self.ping_fed_url, headers=headers, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            # Set expiration time (subtract 5 minutes for buffer)
            self.token_expires_at = time.time() + token_data.get('expires_in', 3600) - 300
            return self.access_token
        else:
            raise Exception(f"Failed to get access token: {response.status_code} - {response.text}")

class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI's text-embedding-ada-002"""
    
    def __init__(self):
        self.endpoint = os.getenv('KGW_ENDPOINT')
        self.api_version = os.getenv('AOAI_API_VERSION')
        self.deployment_name = os.getenv('EMBEDDING_MODEL_DEPLOYMENT_NAME')
        self.auth = AzureOpenAIAuth()
        
        if not all([self.endpoint, self.api_version, self.deployment_name]):
            raise ValueError("Missing required Azure OpenAI configuration")
        
        print(f"✅ OpenAI Embedding Generator initialized")
        print(f"   - Endpoint: {self.endpoint}")
        print(f"   - Deployment: {self.deployment_name}")
        print(f"   - API Version: {self.api_version}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 50) -> np.ndarray:
        """Generate embeddings for a list of texts in batches"""
        all_embeddings = []
        
        # Reduce batch size to avoid rate limits
        print(f"🔄 Generating embeddings for {len(texts)} texts in batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"   Processing batch {i//batch_size + 1}/{(len(texts) - 1)//batch_size + 1}")
            
            batch_embeddings = self._generate_batch_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
            # More conservative rate limiting - wait longer between batches
            if i + batch_size < len(texts):
                print(f"   Waiting 3 seconds before next batch...")
                time.sleep(3)
        
        return np.array(all_embeddings)
    
    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        access_token = self.auth.get_access_token()
        
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/embeddings?api-version={self.api_version}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        payload = {
            'input': texts,
            'model': self.deployment_name
        }
        
        max_retries = 5  # Increased retries
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    return [item['embedding'] for item in result['data']]
                elif response.status_code == 429:
                    # Rate limit hit, wait progressively longer
                    wait_time = (2 ** attempt) + 5  # Add base 5 seconds
                    print(f"   Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"API request failed: {response.status_code} - {response.text}")
            
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Request failed after {max_retries} attempts: {e}")
                wait_time = (2 ** attempt) + 2
                print(f"   Request error, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        raise Exception(f"Failed to generate embeddings after {max_retries} attempts")

class VectorStore:
    """Store and manage vector embeddings"""
    
    def __init__(self, store_type: str = "local", collection_name: str = "pdf_rag"):
        self.store_type = store_type
        self.collection_name = collection_name
        self.vectors = []
        self.metadata = []
        self.texts = []
        
        if store_type == "chroma" and CHROMA_AVAILABLE:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        elif store_type == "faiss" and FAISS_AVAILABLE:
            self.faiss_index = None
        
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents with embeddings to the vector store"""
        texts = [chunk['text'] for chunk in chunks]
        metadata = [chunk['metadata'] for chunk in chunks]
        
        if self.store_type == "local":
            self.vectors.extend(embeddings)
            self.metadata.extend(metadata)
            self.texts.extend(texts)
            
        elif self.store_type == "chroma" and CHROMA_AVAILABLE:
            ids = [f"doc_{i}_{len(self.texts) + i}" for i in range(len(texts))]
            
            # Clean metadata for ChromaDB - flatten nested dictionaries
            cleaned_metadata = []
            for meta in metadata:
                clean_meta = {}
                for key, value in meta.items():
                    if isinstance(value, dict):
                        # Flatten nested dictionaries
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, (str, int, float, bool, type(None))):
                                clean_meta[f"{key}_{nested_key}"] = nested_value
                            else:
                                clean_meta[f"{key}_{nested_key}"] = str(nested_value)
                    elif isinstance(value, (str, int, float, bool, type(None))):
                        clean_meta[key] = value
                    else:
                        clean_meta[key] = str(value)
                cleaned_metadata.append(clean_meta)
            
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=cleaned_metadata,
                ids=ids
            )
            
        elif self.store_type == "faiss" and FAISS_AVAILABLE:
            if self.faiss_index is None:
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)
            
            self.faiss_index.add(embeddings.astype('float32'))
            self.metadata.extend(metadata)
            self.texts.extend(texts)
    
    def save_local_store(self, filepath: str):
        """Save local vector store to file"""
        if self.store_type == "local":
            store_data = {
                'vectors': np.array(self.vectors),
                'metadata': self.metadata,
                'texts': self.texts,
                'created_at': datetime.now().isoformat(),
                'collection_name': self.collection_name,
                'embedding_model': 'text-embedding-ada-002',
                'embedding_dimensions': 1536
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(store_data, f)
            
            print(f"Local vector store saved to: {filepath}")
            
        elif self.store_type == "faiss" and self.faiss_index is not None:
            # Save FAISS index and metadata separately
            faiss.write_index(self.faiss_index, f"{filepath}.faiss")
            
            metadata_path = f"{filepath}.metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'texts': self.texts,
                    'created_at': datetime.now().isoformat(),
                    'embedding_model': 'text-embedding-ada-002',
                    'embedding_dimensions': 1536
                }, f)
            
            print(f"FAISS index saved to: {filepath}.faiss")
            print(f"Metadata saved to: {metadata_path}")

def main():
    """Main function to process JSON and create embeddings"""
    
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "test_extracted.json")
    output_dir = os.path.join(script_dir, "vector_store")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("RAG Embeddings Creator - OpenAI Ada-002")
    print("=" * 50)
    
    # Check if JSON file exists
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return
    
    # Load JSON data
    print(f"Loading JSON data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Initialize chunker
    chunk_size = int(os.getenv('MAX_CHUNK_SIZE', 512))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
    
    print(f"Initializing document chunker (size: {chunk_size}, overlap: {chunk_overlap})...")
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Process and chunk the data
    print("Processing and chunking document content...")
    chunks = chunker.process_json_data(json_data)
    print(f"Created {len(chunks)} chunks from the document")
    
    # Generate embeddings using OpenAI
    try:
        print("Generating embeddings using OpenAI text-embedding-ada-002...")
        embedding_generator = OpenAIEmbeddingGenerator()
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_generator.generate_embeddings(texts)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Store vectors
        print("Storing vectors...")
        
        # Local storage
        vector_store = VectorStore("local", "pdf_rag")
        vector_store.add_documents(chunks, embeddings)
        
        local_store_path = os.path.join(output_dir, "local_vector_store.pkl")
        vector_store.save_local_store(local_store_path)
        
        # FAISS storage (if available)
        if FAISS_AVAILABLE:
            print("Creating FAISS index...")
            faiss_store = VectorStore("faiss", "pdf_rag")
            faiss_store.add_documents(chunks, embeddings)
            faiss_path = os.path.join(output_dir, "faiss_vector_store")
            faiss_store.save_local_store(faiss_path)
        
        # ChromaDB storage (if available)
        if CHROMA_AVAILABLE:
            print("Creating ChromaDB collection...")
            chroma_store = VectorStore("chroma", "pdf_rag")
            chroma_store.add_documents(chunks, embeddings)
            print("ChromaDB collection created in ./chroma_db")
        
        # Save chunk analysis
        analysis_path = os.path.join(output_dir, "chunk_analysis.json")
        analysis = {
            'total_chunks': len(chunks),
            'total_characters': sum(len(chunk['text']) for chunk in chunks),
            'total_words': sum(len(chunk['text'].split()) for chunk in chunks),
            'chunk_types': {},
            'pages_processed': len(json_data.get('pages', [])),
            'embedding_dimensions': embeddings.shape[1] if embeddings is not None else 0,
            'embedding_model': 'text-embedding-ada-002',
            'created_at': datetime.now().isoformat()
        }
        
        # Analyze chunk types
        for chunk in chunks:
            chunk_type = chunk['metadata'].get('type', 'unknown')
            analysis['chunk_types'][chunk_type] = analysis['chunk_types'].get(chunk_type, 0) + 1
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Chunk analysis saved to: {analysis_path}")
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"  - Total chunks created: {len(chunks)}")
        print(f"  - Embedding model: text-embedding-ada-002")
        print(f"  - Embedding dimensions: {embeddings.shape[1]}")
        print(f"  - Total characters: {analysis['total_characters']:,}")
        print(f"  - Total words: {analysis['total_words']:,}")
        print(f"  - Pages processed: {analysis['pages_processed']}")
        
        print("\nChunk distribution by type:")
        for chunk_type, count in analysis['chunk_types'].items():
            print(f"  - {chunk_type}: {count}")
        
        print(f"\nVector stores created in: {output_dir}")
        print("RAG preparation completed successfully!")
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        print("Please check your Azure OpenAI configuration and network connectivity")

if __name__ == "__main__":
    main()