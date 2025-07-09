#!/usr/bin/env python3
"""
Batched RAG System for Large Documents
Retrieves top 50 chunks and sends them in 5 separate batches to Claude
"""

import pickle
import numpy as np
import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

class BatchedRAG:
    """Batched RAG system for comprehensive document analysis"""
    
    def __init__(self, vector_store_path: str, api_key: str = None):
        self.vectors = None
        self.metadata = None
        self.texts = None
        self.embedding_model = None
        self.anthropic_client = None
        
        # Load embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("📥 Loading embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            raise ImportError("sentence-transformers required for querying")
        
        # Initialize Claude API client
        self._setup_claude_api(api_key)
        
        # Load vector store
        self.load_vector_store(vector_store_path)
    
    def _setup_claude_api(self, api_key: str = None):
        """Setup Claude API client"""
        if not ANTHROPIC_AVAILABLE:
            print("⚠️  anthropic package not available. Install with: pip install anthropic")
            return
        
        if api_key:
            claude_api_key = api_key
        else:
            claude_api_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('CLAUDE_API_KEY')
        
        if claude_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=claude_api_key)
                print("✅ Claude API client initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Claude API: {e}")
                self.anthropic_client = None
        else:
            print("⚠️  No Claude API key found. Set ANTHROPIC_API_KEY environment variable.")
    
    def load_vector_store(self, vector_store_path: str):
        """Load the vector store from file"""
        print(f"📂 Loading vector store from: {vector_store_path}")
        with open(vector_store_path, 'rb') as f:
            store_data = pickle.load(f)
        
        self.vectors = np.array(store_data['vectors'])
        self.metadata = store_data['metadata']
        self.texts = store_data['texts']
        
        print(f"✅ Loaded {len(self.texts)} chunks with {self.vectors.shape[1]} dimensions")
    
    def retrieve_top_chunks(self, question: str, top_k: int = 50, min_similarity: float = 0.2) -> List[Dict[str, Any]]:
        """Retrieve top K relevant chunks using vector similarity"""
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded")
        
        print(f"🔍 Searching for top {top_k} chunks for: '{question}'")
        
        # Generate embedding for the question
        question_embedding = self.embedding_model.encode([question])
        
        # Calculate cosine similarity
        similarities = np.dot(self.vectors, question_embedding.T).flatten()
        similarities = similarities / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(question_embedding))
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= min_similarity:
                results.append({
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarity),
                    'chunk_index': int(idx),
                    'rank': len(results) + 1
                })
        
        return results
    
    def create_batches(self, chunks: List[Dict[str, Any]], batch_size: int = 10) -> List[List[Dict[str, Any]]]:
        """Split chunks into batches"""
        batches = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def format_batch_context(self, batch: List[Dict[str, Any]], batch_num: int, total_batches: int) -> str:
        """Format a batch of chunks for Claude"""
        context = f"DOCUMENT CONTEXT - BATCH {batch_num}/{total_batches}\n"
        context += "="*60 + "\n\n"
        
        for i, chunk in enumerate(batch, 1):
            metadata = chunk['metadata']
            similarity = chunk['similarity']
            
            context += f"[CHUNK {i}/{len(batch)}] (Similarity: {similarity:.3f})\n"
            context += f"Source: Page {metadata.get('page_number', 'N/A')}"
            
            if metadata.get('type') == 'table':
                context += f" | Table {metadata.get('table_index', 'N/A')}"
                table_info = metadata.get('table_info', {})
                if table_info:
                    context += f" | {table_info.get('row_count', 0)} rows × {table_info.get('column_count', 0)} cols"
            
            context += f" | Type: {metadata.get('type', 'text')}\n"
            context += "-" * 50 + "\n"
            context += chunk['text'] + "\n\n"
        
        return context
    
    def query_claude_batch(self, question: str, batch_context: str, batch_num: int, 
                          total_batches: int, max_tokens: int = 1000) -> str:
        """Query Claude with a single batch of context"""
        if not self.anthropic_client:
            return f"Claude API not available for batch {batch_num}."
        
        prompt = f"""You are analyzing a large pharmaceutical document. I will provide you with a question and a batch of relevant context chunks (batch {batch_num} of {total_batches}).

QUESTION: {question}

{batch_context}

INSTRUCTIONS:
1. Answer the question based ONLY on the context provided in this batch
2. Be specific and cite page numbers, table numbers, or other source references
3. If this batch doesn't contain sufficient information to fully answer the question, indicate what information is available and what might be missing
4. Focus on concrete data, numbers, and facts from the provided context
5. If you find specific data points (like complaint numbers, batch counts, etc.), highlight them clearly
6. Mention which batch this is ({batch_num}/{total_batches}) in your response

BATCH {batch_num} ANALYSIS:"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error in batch {batch_num}: {str(e)}"
    
    def comprehensive_query(self, question: str, top_k: int = 50, batch_size: int = 10, 
                           min_similarity: float = 0.2, delay_between_requests: float = 1.0) -> Dict[str, Any]:
        """Perform comprehensive query with batched Claude requests"""
        print(f"🚀 Starting comprehensive analysis for: '{question}'")
        print(f"⚙️  Settings: top_k={top_k}, batch_size={batch_size}, min_similarity={min_similarity}")
        
        # Step 1: Retrieve top chunks
        top_chunks = self.retrieve_top_chunks(question, top_k, min_similarity)
        
        if not top_chunks:
            return {
                'question': question,
                'batch_responses': [],
                'total_chunks': 0,
                'batches_processed': 0,
                'error': 'No relevant chunks found'
            }
        
        print(f"📊 Found {len(top_chunks)} relevant chunks")
        
        # Step 2: Create batches
        batches = self.create_batches(top_chunks, batch_size)
        print(f"📦 Created {len(batches)} batches of up to {batch_size} chunks each")
        
        # Step 3: Process each batch with Claude
        batch_responses = []
        
        for i, batch in enumerate(batches, 1):
            print(f"\n🤖 Processing batch {i}/{len(batches)} ({len(batch)} chunks)...")
            
            # Format context for this batch
            batch_context = self.format_batch_context(batch, i, len(batches))
            
            # Query Claude
            if self.anthropic_client:
                claude_response = self.query_claude_batch(question, batch_context, i, len(batches))
            else:
                claude_response = f"Claude API not available. Batch {i} contains {len(batch)} relevant chunks."
            
            batch_responses.append({
                'batch_number': i,
                'chunks_in_batch': len(batch),
                'claude_response': claude_response,
                'chunk_details': [
                    {
                        'rank': chunk['rank'],
                        'similarity': chunk['similarity'],
                        'page': chunk['metadata'].get('page_number'),
                        'type': chunk['metadata'].get('type'),
                        'preview': chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                    }
                    for chunk in batch
                ]
            })
            
            # Delay between requests to be respectful to the API
            if i < len(batches) and delay_between_requests > 0:
                print(f"⏱️  Waiting {delay_between_requests}s before next request...")
                time.sleep(delay_between_requests)
        
        # Step 4: Analyze coverage
        pages_covered = set()
        chunk_types = {}
        
        for chunk in top_chunks:
            page = chunk['metadata'].get('page_number')
            if page:
                pages_covered.add(page)
            
            chunk_type = chunk['metadata'].get('type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            'question': question,
            'batch_responses': batch_responses,
            'total_chunks': len(top_chunks),
            'batches_processed': len(batches),
            'pages_covered': sorted(list(pages_covered)),
            'chunk_types': chunk_types,
            'similarity_range': {
                'min': min([c['similarity'] for c in top_chunks]),
                'max': max([c['similarity'] for c in top_chunks]),
                'avg': np.mean([c['similarity'] for c in top_chunks])
            },
            'processing_time': datetime.now().isoformat()
        }
    
    def print_comprehensive_results(self, results: Dict[str, Any]):
        """Print the comprehensive query results"""
        print("\n" + "="*100)
        print(f"📋 COMPREHENSIVE ANALYSIS RESULTS")
        print("="*100)
        print(f"❓ QUESTION: {results['question']}")
        print(f"📊 COVERAGE: {results['total_chunks']} chunks from {len(results.get('pages_covered', []))} pages")
        print(f"📦 BATCHES: {results['batches_processed']} separate Claude requests")
        
        if results.get('similarity_range'):
            sim = results['similarity_range']
            print(f"🎯 SIMILARITY: {sim['min']:.3f} - {sim['max']:.3f} (avg: {sim['avg']:.3f})")
        
        if results.get('pages_covered'):
            print(f"📄 PAGES: {results['pages_covered']}")
        
        if results.get('chunk_types'):
            print(f"📋 CONTENT TYPES: {results['chunk_types']}")
        
        print("\n" + "="*100)
        print("🤖 CLAUDE AI RESPONSES BY BATCH")
        print("="*100)
        
        for batch_response in results.get('batch_responses', []):
            batch_num = batch_response['batch_number']
            chunks_count = batch_response['chunks_in_batch']
            
            print(f"\n📦 BATCH {batch_num}/{results['batches_processed']} ({chunks_count} chunks)")
            print("-" * 80)
            
            # Show chunk details for this batch
            print("📋 Chunks in this batch:")
            for detail in batch_response['chunk_details']:
                print(f"   • Rank {detail['rank']} | Page {detail['page']} | Sim: {detail['similarity']:.3f} | {detail['type']}")
                print(f"     Preview: {detail['preview']}")
            
            print(f"\n🤖 CLAUDE RESPONSE FOR BATCH {batch_num}:")
            print("-" * 60)
            print(batch_response['claude_response'])
            print("\n" + "="*100)

def main():
    """Main function for batched RAG system"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vector_store_path = os.path.join(script_dir, "vector_store", "local_vector_store.pkl")
    
    print("🚀 BATCHED RAG SYSTEM FOR LARGE DOCUMENTS")
    print("="*80)
    print("📖 Optimized for 500+ page documents")
    print("🎯 Retrieves top 50 chunks, sends 5 batches of 10 chunks each")
    print("="*80)
    
    if not os.path.exists(vector_store_path):
        print("❌ Vector store not found. Please run create_embeddings.py first.")
        return
    
    # Initialize the batched RAG system
    try:
        rag = BatchedRAG(vector_store_path)
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return
    
    # Interactive query loop
    while True:
        try:
            print(f"\n" + "="*80)
            question = input("🤔 Your question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Optional: Customize settings
            print(f"\n⚙️  Current settings: top_k=50, batch_size=10")
            customize = input("🔧 Customize settings? (y/n, default n): ").strip().lower()
            
            top_k = 50
            batch_size = 10
            min_similarity = 0.2
            
            if customize == 'y':
                try:
                    top_k = int(input(f"📊 Top chunks to retrieve (default 50): ") or "50")
                    batch_size = int(input(f"📦 Chunks per batch (default 10): ") or "10")
                    min_similarity = float(input(f"🎯 Min similarity (default 0.2): ") or "0.2")
                except ValueError:
                    print("⚠️  Using default settings")
                    top_k, batch_size, min_similarity = 50, 10, 0.2
            
            # Perform comprehensive query
            print(f"\n🚀 Starting comprehensive analysis...")
            results = rag.comprehensive_query(
                question=question,
                top_k=top_k,
                batch_size=batch_size,
                min_similarity=min_similarity,
                delay_between_requests=1.0
            )
            
            # Display results
            rag.print_comprehensive_results(results)
            
            # Option to save results
            save_option = input(f"\n💾 Save results to file? (y/n): ").strip().lower()
            if save_option == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comprehensive_analysis_{timestamp}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Results saved to: {filename}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
