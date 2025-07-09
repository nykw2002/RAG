#!/usr/bin/env python3
"""
RAG Embeddings Creator
Processes the extracted PDF JSON and creates vector embeddings for RAG system
"""

import json
import os
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

# Vector storage options
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

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

class EmbeddingGenerator:
    """Generate embeddings using various models"""
    
    def __init__(self, model_type: str = "sentence-transformers", model_name: str = None):
        self.model_type = model_type
        self.model = None
        
        if model_type == "sentence-transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
            model_name = model_name or "all-MiniLM-L6-v2"
            print(f"Loading sentence transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            
        elif model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not available. Install with: pip install openai")
            self.model_name = model_name or "text-embedding-ada-002"
            # OpenAI client will be initialized when needed
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if self.model_type == "sentence-transformers":
            return self.model.encode(texts, convert_to_numpy=True)
            
        elif self.model_type == "openai":
            # This would require OpenAI API key
            raise NotImplementedError("OpenAI embeddings require API key configuration")
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

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
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadata,
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
                'collection_name': self.collection_name
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
                    'created_at': datetime.now().isoformat()
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
    
    print("RAG Embeddings Creator")
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
    print("Initializing document chunker...")
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    
    # Process and chunk the data
    print("Processing and chunking document content...")
    chunks = chunker.process_json_data(json_data)
    print(f"Created {len(chunks)} chunks from the document")
    
    # Generate embeddings
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Generating embeddings using sentence-transformers...")
        embedding_generator = EmbeddingGenerator("sentence-transformers", "all-MiniLM-L6-v2")
        
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
        print(f"  - Embedding dimensions: {embeddings.shape[1]}")
        print(f"  - Total characters: {analysis['total_characters']:,}")
        print(f"  - Total words: {analysis['total_words']:,}")
        print(f"  - Pages processed: {analysis['pages_processed']}")
        
        print("\nChunk distribution by type:")
        for chunk_type, count in analysis['chunk_types'].items():
            print(f"  - {chunk_type}: {count}")
        
        print(f"\nVector stores created in: {output_dir}")
        print("RAG preparation completed successfully!")
        
    else:
        print("Error: sentence-transformers not available.")
        print("Install it with: pip install sentence-transformers")

if __name__ == "__main__":
    main()
