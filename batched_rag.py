#!/usr/bin/env python3
"""
Enhanced Batched RAG System for Large Documents - OpenAI Ada-002 + GPT-4o
Advanced pharmaceutical document analysis with multi-pass search and enhanced AI instructions
"""

import pickle
import numpy as np
import os
import json
import time
import requests
import re
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

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

class EnhancedBatchedRAG:
    """Enhanced batched RAG system for comprehensive document analysis"""
    
    def __init__(self, vector_store_path: str):
        self.vectors = None
        self.metadata = None
        self.texts = None
        self.auth = AzureOpenAIAuth()
        
        # Azure OpenAI configuration
        self.endpoint = os.getenv('KGW_ENDPOINT')
        self.api_version = os.getenv('AOAI_API_VERSION')
        self.embedding_deployment = os.getenv('EMBEDDING_MODEL_DEPLOYMENT_NAME')
        self.chat_deployment = os.getenv('CHAT_MODEL_DEPLOYMENT_NAME')
        
        if not all([self.endpoint, self.api_version, self.embedding_deployment, self.chat_deployment]):
            raise ValueError("Missing required Azure OpenAI configuration")
        
        print("✅ Enhanced Azure OpenAI clients initialized successfully")
        print(f"   - Endpoint: {self.endpoint}")
        print(f"   - Embedding Model: {self.embedding_deployment}")
        print(f"   - Chat Model: {self.chat_deployment}")
        
        # Load vector store
        self.load_vector_store(vector_store_path)
    
    def load_vector_store(self, vector_store_path: str):
        """Load the vector store from file"""
        print(f"📂 Loading vector store from: {vector_store_path}")
        with open(vector_store_path, 'rb') as f:
            store_data = pickle.load(f)
        
        self.vectors = np.array(store_data['vectors'])
        self.metadata = store_data['metadata']
        self.texts = store_data['texts']
        
        embedding_model = store_data.get('embedding_model', 'unknown')
        embedding_dims = store_data.get('embedding_dimensions', self.vectors.shape[1])
        
        print(f"✅ Loaded {len(self.texts)} chunks")
        print(f"   - Embedding model: {embedding_model}")
        print(f"   - Dimensions: {embedding_dims}")
    
    def generate_query_embedding(self, question: str) -> np.ndarray:
        """Generate embedding for the query using OpenAI ada-002"""
        access_token = self.auth.get_access_token()
        
        url = f"{self.endpoint}/openai/deployments/{self.embedding_deployment}/embeddings?api-version={self.api_version}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        payload = {
            'input': [question],
            'model': self.embedding_deployment
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return np.array(result['data'][0]['embedding'])
        else:
            raise Exception(f"Embedding request failed: {response.status_code} - {response.text}")
    
    def extract_keywords(self, question: str) -> List[str]:
        """Extract key terms from the question for pharmaceutical documents"""
        pharma_keywords = {
            'complaints': ['complaint', 'adverse event', 'AE', 'side effect', 'reaction', 'report'],
            'countries': ['Israel', 'USA', 'US', 'UK', 'Germany', 'France', 'Canada', 'Japan', 'Australia'],
            'quantities': ['number', 'count', 'total', 'sum', 'amount', 'how many', 'quantity'],
            'status': ['unsubstantiated', 'substantiated', 'pending', 'resolved', 'confirmed', 'unconfirmed'],
            'severity': ['serious', 'severe', 'mild', 'moderate', 'fatal', 'life-threatening'],
            'regulatory': ['FDA', 'EMA', 'PMDA', 'Health Canada', 'TGA', 'regulatory', 'compliance']
        }
        
        keywords = []
        question_lower = question.lower()
        
        for category, terms in pharma_keywords.items():
            for term in terms:
                if term.lower() in question_lower:
                    keywords.append(term)
        
        # Extract numbers and specific terms
        numbers = re.findall(r'\d+', question)
        keywords.extend(numbers)
        
        return list(set(keywords))
    
    def calculate_enhanced_similarity(self, question: str, chunk: Dict[str, Any]) -> float:
        """Calculate enhanced similarity score with multiple factors"""
        base_similarity = chunk['similarity']
        
        # Content type bonus
        content_bonus = 0
        if chunk['metadata'].get('type') == 'table':
            content_bonus += 0.1  # Tables often contain quantitative data
        
        # Keyword matching bonus
        keyword_bonus = self.calculate_keyword_bonus(question, chunk['text'])
        
        # Combine scores
        enhanced_similarity = base_similarity + content_bonus + keyword_bonus
        
        return min(enhanced_similarity, 1.0)  # Cap at 1.0
    
    def calculate_keyword_bonus(self, question: str, text: str) -> float:
        """Calculate bonus for keyword matches"""
        question_words = set(question.lower().split())
        text_words = set(text.lower().split())
        
        # Find exact matches
        exact_matches = question_words.intersection(text_words)
        
        # Find partial matches
        partial_matches = 0
        for q_word in question_words:
            for t_word in text_words:
                if len(q_word) > 3 and (q_word in t_word or t_word in q_word):
                    partial_matches += 1
                    break
        
        return (len(exact_matches) * 0.05) + (partial_matches * 0.02)
    
    def keyword_search(self, keywords: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """Perform keyword-based search"""
        keyword_scores = []
        
        for i, text in enumerate(self.texts):
            score = 0
            text_lower = text.lower()
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Boost score for exact matches
                    score += text_lower.count(keyword.lower()) * 0.1
            
            if score > 0:
                keyword_scores.append({
                    'text': text,
                    'metadata': self.metadata[i],
                    'similarity': score,
                    'chunk_index': i,
                    'rank': 0  # Will be set later
                })
        
        # Sort by keyword score
        keyword_scores.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Set ranks
        for i, result in enumerate(keyword_scores[:top_k]):
            result['rank'] = i + 1
        
        return keyword_scores[:top_k]
    
    def multi_pass_search(self, question: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Perform multiple search passes with different strategies"""
        all_results = []
        seen_indices = set()
        
        # Pass 1: Direct semantic search
        print("   🔍 Pass 1: Semantic similarity search...")
        semantic_results = self.retrieve_top_chunks(question, top_k//2, min_similarity=0.25)
        for result in semantic_results:
            if result['chunk_index'] not in seen_indices:
                all_results.append(result)
                seen_indices.add(result['chunk_index'])
        
        # Pass 2: Keyword-based search
        print("   🔍 Pass 2: Keyword-based search...")
        keywords = self.extract_keywords(question)
        if keywords:
            keyword_results = self.keyword_search(keywords, top_k//3)
            for result in keyword_results:
                if result['chunk_index'] not in seen_indices:
                    all_results.append(result)
                    seen_indices.add(result['chunk_index'])
        
        # Pass 3: Related concept search
        print("   🔍 Pass 3: Related concept search...")
        related_queries = self.generate_related_queries(question)
        for related_query in related_queries:
            related_results = self.retrieve_top_chunks(related_query, top_k//6, min_similarity=0.2)
            for result in related_results:
                if result['chunk_index'] not in seen_indices:
                    all_results.append(result)
                    seen_indices.add(result['chunk_index'])
        
        # Re-rank all results
        for result in all_results:
            result['enhanced_similarity'] = self.calculate_enhanced_similarity(question, result)
        
        # Sort by enhanced similarity
        all_results.sort(key=lambda x: x['enhanced_similarity'], reverse=True)
        
        return all_results[:top_k]
    
    def generate_related_queries(self, question: str) -> List[str]:
        """Generate related queries for broader search"""
        related_queries = []
        
        # Extract key concepts
        if 'complaint' in question.lower():
            related_queries.extend([
                'adverse event report',
                'side effect documentation',
                'safety report'
            ])
        
        if 'israel' in question.lower():
            related_queries.extend([
                'country specific data',
                'geographic distribution',
                'regional reports'
            ])
        
        if 'unsubstantiated' in question.lower():
            related_queries.extend([
                'investigation status',
                'verification process',
                'evidence assessment'
            ])
        
        return related_queries
    
    def retrieve_top_chunks(self, question: str, top_k: int = 50, min_similarity: float = 0.2) -> List[Dict[str, Any]]:
        """Retrieve top K relevant chunks using vector similarity"""
        # Generate embedding for the question
        question_embedding = self.generate_query_embedding(question)
        
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
    
    def group_chunks_by_relevance(self, chunks: List[Dict[str, Any]], question: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by relevance and content type"""
        grouped = {
            'high_relevance': [],
            'quantitative': [],
            'contextual': [],
            'regulatory': []
        }
        
        for chunk in chunks:
            similarity = chunk.get('enhanced_similarity', chunk['similarity'])
            text = chunk['text'].lower()
            
            # High relevance (direct answers)
            if similarity > 0.7:
                grouped['high_relevance'].append(chunk)
            
            # Quantitative data
            elif (chunk['metadata'].get('type') == 'table' or 
                  re.search(r'\d+', chunk['text']) or
                  any(word in text for word in ['count', 'number', 'total', 'percentage', '%'])):
                grouped['quantitative'].append(chunk)
            
            # Regulatory content
            elif any(word in text for word in ['regulatory', 'compliance', 'fda', 'ema', 'approval']):
                grouped['regulatory'].append(chunk)
            
            # Contextual information
            else:
                grouped['contextual'].append(chunk)
        
        return grouped
    
    def highlight_key_information(self, text: str, question: str) -> str:
        """Highlight key information in text"""
        # Extract key terms from question
        keywords = self.extract_keywords(question)
        
        highlighted_text = text
        for keyword in keywords:
            if keyword.lower() in text.lower():
                # Simple highlighting with uppercase
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                highlighted_text = pattern.sub(f"**{keyword.upper()}**", highlighted_text)
        
        # Highlight numbers
        highlighted_text = re.sub(r'\b(\d+)\b', r'**\1**', highlighted_text)
        
        return highlighted_text
    
    def format_chunk_with_analysis(self, chunk: Dict[str, Any], index: int, category: str) -> str:
        """Format individual chunk with analytical markers"""
        metadata = chunk['metadata']
        text = chunk['text']
        similarity = chunk.get('enhanced_similarity', chunk['similarity'])
        
        # Add analytical markers
        markers = {
            'HIGH_RELEVANCE': '🎯',
            'QUANTITATIVE': '📈',
            'CONTEXTUAL': '📝',
            'REGULATORY': '⚖️'
        }
        
        formatted = f"{markers.get(category, '•')} [{category} {index}] (Similarity: {similarity:.3f})\n"
        formatted += f"📍 Source: Page {metadata.get('page_number', 'N/A')}"
        
        if metadata.get('type') == 'table':
            formatted += f" | 📊 Table {metadata.get('table_index', 'N/A')}"
            table_info = metadata.get('table_info', {})
            if table_info:
                formatted += f" | {table_info.get('row_count', 0)}×{table_info.get('column_count', 0)}"
        
        formatted += f" | Type: {metadata.get('type', 'text')}\n"
        
        # Highlight key information
        highlighted_text = self.highlight_key_information(text, chunk.get('question', ''))
        formatted += f"{'-'*60}\n{highlighted_text}\n\n"
        
        return formatted
    
    def format_enhanced_batch_context(self, batch: List[Dict[str, Any]], batch_num: int, 
                                     total_batches: int, question: str) -> str:
        """Format batch context with enhanced analytical structure"""
        
        # Group chunks by type and relevance
        grouped_chunks = self.group_chunks_by_relevance(batch, question)
        
        context = f"""PHARMACEUTICAL DOCUMENT ANALYSIS - BATCH {batch_num}/{total_batches}
{'='*80}

QUERY: {question}

ANALYSIS PRIORITY:
🎯 HIGH RELEVANCE (Direct answers and highly relevant content)
📈 QUANTITATIVE DATA (Numbers, statistics, measurements)  
📝 CONTEXTUAL INFORMATION (Background and supporting details)
⚖️  REGULATORY CONTENT (Compliance and regulatory information)

"""
        
        # Format high-relevance chunks first
        if grouped_chunks['high_relevance']:
            context += "🎯 HIGH RELEVANCE CONTENT:\n" + "="*60 + "\n"
            for i, chunk in enumerate(grouped_chunks['high_relevance'], 1):
                chunk['question'] = question  # Pass question for highlighting
                context += self.format_chunk_with_analysis(chunk, i, "HIGH_RELEVANCE")
        
        # Format quantitative data
        if grouped_chunks['quantitative']:
            context += "\n📈 QUANTITATIVE DATA:\n" + "="*60 + "\n"
            for i, chunk in enumerate(grouped_chunks['quantitative'], 1):
                chunk['question'] = question
                context += self.format_chunk_with_analysis(chunk, i, "QUANTITATIVE")
        
        # Format regulatory content
        if grouped_chunks['regulatory']:
            context += "\n⚖️ REGULATORY CONTENT:\n" + "="*60 + "\n"
            for i, chunk in enumerate(grouped_chunks['regulatory'], 1):
                chunk['question'] = question
                context += self.format_chunk_with_analysis(chunk, i, "REGULATORY")
        
        # Format contextual information
        if grouped_chunks['contextual']:
            context += "\n📝 CONTEXTUAL INFORMATION:\n" + "="*60 + "\n"
            for i, chunk in enumerate(grouped_chunks['contextual'], 1):
                chunk['question'] = question
                context += self.format_chunk_with_analysis(chunk, i, "CONTEXTUAL")
        
        return context
    
    def adaptive_batch_sizing(self, chunks: List[Dict[str, Any]], question: str) -> List[List[Dict[str, Any]]]:
        """Create adaptive batches based on content complexity"""
        
        # Analyze content complexity
        high_relevance = [c for c in chunks if c.get('enhanced_similarity', c['similarity']) > 0.8]
        medium_relevance = [c for c in chunks if 0.5 < c.get('enhanced_similarity', c['similarity']) <= 0.8]
        low_relevance = [c for c in chunks if c.get('enhanced_similarity', c['similarity']) <= 0.5]
        
        batches = []
        
        # First batch: High-relevance content (smaller batch for focused analysis)
        if high_relevance:
            batches.append(high_relevance[:6])
        
        # Subsequent batches: Mix of medium and low relevance
        remaining = high_relevance[6:] + medium_relevance + low_relevance
        
        current_batch = []
        for chunk in remaining:
            current_batch.append(chunk)
            
            # Dynamic batch size based on content complexity
            batch_size = 8 if chunk['metadata'].get('type') == 'table' else 12
            
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def generate_comprehensive_analysis_prompt(self, question: str, batch_context: str, 
                                             batch_num: int, total_batches: int) -> str:
        """Generate a comprehensive analysis prompt"""
        
        return f"""You are a pharmaceutical regulatory expert with deep expertise in adverse event reporting, drug safety, and compliance analysis. You are analyzing batch {batch_num} of {total_batches} from a comprehensive regulatory document.

PRIMARY QUESTION: {question}

EXPERT ANALYSIS FRAMEWORK:

1. **DIRECT ANSWER EXTRACTION**
   - Search for explicit numerical data that directly answers the question
   - Identify specific counts, percentages, or statistical measures
   - Note exact page references, table numbers, and section locations
   - Verify data consistency across multiple sources

2. **REGULATORY CONTEXT ANALYSIS**
   - Consider regulatory requirements and reporting standards
   - Evaluate compliance implications of the findings
   - Identify any safety signals or patterns
   - Assess data completeness and quality

3. **QUANTITATIVE DATA VALIDATION**
   - Cross-reference numbers across different document sections
   - Identify trends, patterns, or anomalies in the data
   - Note any discrepancies or inconsistencies
   - Evaluate statistical significance where applicable

4. **COMPREHENSIVE INSIGHTS**
   - Provide context for the numerical findings
   - Explain regulatory implications
   - Identify potential safety concerns
   - Suggest areas requiring further investigation

5. **EVIDENCE-BASED CONCLUSIONS**
   - Summarize key findings with supporting evidence
   - Note confidence levels and data limitations
   - Provide actionable recommendations
   - Identify gaps in available information

{batch_context}

RESPONSE REQUIREMENTS:
✅ Start with a clear, direct answer if available
✅ Provide specific numbers with exact page/table citations
✅ Explain regulatory context and implications
✅ Highlight any safety concerns or compliance issues
✅ Note data quality, completeness, and limitations
✅ Suggest follow-up questions or investigations
✅ Maintain scientific objectivity and precision

BATCH {batch_num}/{total_batches} EXPERT ANALYSIS:"""
    
    def query_enhanced_gpt(self, question: str, batch_context: str, batch_num: int, 
                          total_batches: int, max_tokens: int = 1200) -> str:
        """Query GPT with enhanced pharmaceutical analysis prompt"""
        access_token = self.auth.get_access_token()
        
        url = f"{self.endpoint}/openai/deployments/{self.chat_deployment}/chat/completions?api-version={self.api_version}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        prompt = self.generate_comprehensive_analysis_prompt(question, batch_context, batch_num, total_batches)
        
        payload = {
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a pharmaceutical regulatory expert specializing in adverse event analysis, drug safety, and compliance reporting. Provide precise, evidence-based analysis with specific citations.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': max_tokens,
            'temperature': 0.1,
            'top_p': 0.95,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) + 3
                    print(f"   Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Error in batch {batch_num}: {response.status_code} - {response.text}"
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error in batch {batch_num}: {str(e)}"
                wait_time = (2 ** attempt) + 2
                print(f"   Request error, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return f"Failed to process batch {batch_num} after {max_retries} attempts"
    
    def comprehensive_query(self, question: str, top_k: int = 75, min_similarity: float = 0.2, 
                           delay_between_requests: float = 2.0) -> Dict[str, Any]:
        """Perform comprehensive query with enhanced multi-pass search"""
        print(f"🚀 Starting enhanced comprehensive analysis for: '{question}'")
        print(f"⚙️  Enhanced Settings: top_k={top_k}, min_similarity={min_similarity}")
        
        # Step 1: Multi-pass search for relevant chunks
        print("🔍 Performing multi-pass search...")
        top_chunks = self.multi_pass_search(question, top_k)
        
        if not top_chunks:
            return {
                'question': question,
                'batch_responses': [],
                'total_chunks': 0,
                'batches_processed': 0,
                'error': 'No relevant chunks found'
            }
        
        print(f"📊 Found {len(top_chunks)} relevant chunks from multi-pass search")
        
        # Step 2: Adaptive batch creation
        print("📦 Creating adaptive batches...")
        batches = self.adaptive_batch_sizing(top_chunks, question)
        print(f"📦 Created {len(batches)} adaptive batches")
        
        # Step 3: Process each batch with enhanced GPT analysis
        batch_responses = []
        
        for i, batch in enumerate(batches, 1):
            print(f"\n🧠 Processing batch {i}/{len(batches)} ({len(batch)} chunks) with enhanced analysis...")
            
            # Format enhanced context for this batch
            batch_context = self.format_enhanced_batch_context(batch, i, len(batches), question)
            
            # Query GPT with enhanced prompt
            gpt_response = self.query_enhanced_gpt(question, batch_context, i, len(batches))
            
            batch_responses.append({
                'batch_number': i,
                'chunks_in_batch': len(batch),
                'gpt_response': gpt_response,
                'chunk_details': [
                    {
                        'rank': chunk['rank'],
                        'similarity': chunk.get('enhanced_similarity', chunk['similarity']),
                        'base_similarity': chunk['similarity'],
                        'page': chunk['metadata'].get('page_number'),
                        'type': chunk['metadata'].get('type'),
                        'preview': chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
                    }
                    for chunk in batch
                ]
            })
            
            # Delay between requests
            if i < len(batches) and delay_between_requests > 0:
                print(f"⏱️  Waiting {delay_between_requests}s before next request...")
                time.sleep(delay_between_requests)
        
        # Step 4: Analyze coverage and generate insights
        pages_covered = set()
        chunk_types = defaultdict(int)
        similarity_scores = []
        
        for chunk in top_chunks:
            page = chunk['metadata'].get('page_number')
            if page:
                pages_covered.add(page)
            
            chunk_type = chunk['metadata'].get('type', 'unknown')
            chunk_types[chunk_type] += 1
            
            similarity_scores.append(chunk.get('enhanced_similarity', chunk['similarity']))
        
        return {
            'question': question,
            'batch_responses': batch_responses,
            'total_chunks': len(top_chunks),
            'batches_processed': len(batches),
            'pages_covered': sorted(list(pages_covered)),
            'chunk_types': dict(chunk_types),
            'similarity_range': {
                'min': min(similarity_scores),
                'max': max(similarity_scores),
                'avg': np.mean(similarity_scores)
            },
            'search_strategy': 'multi-pass-enhanced',
            'processing_time': datetime.now().isoformat()
        }
    
    def print_enhanced_results(self, results: Dict[str, Any]):
        """Print the enhanced comprehensive query results"""
        print("\n" + "="*120)
        print(f"📋 ENHANCED PHARMACEUTICAL DOCUMENT ANALYSIS")
        print("="*120)
        print(f"❓ QUESTION: {results['question']}")
        print(f"📊 COVERAGE: {results['total_chunks']} chunks from {len(results.get('pages_covered', []))} pages")
        print(f"📦 BATCHES: {results['batches_processed']} adaptive batches processed")
        print(f"🔍 STRATEGY: {results.get('search_strategy', 'standard')}")
        
        if results.get('similarity_range'):
            sim = results['similarity_range']
            print(f"🎯 SIMILARITY RANGE: {sim['min']:.3f} - {sim['max']:.3f} (avg: {sim['avg']:.3f})")
        
        if results.get('pages_covered'):
            pages = results['pages_covered']
            print(f"📄 PAGES ANALYZED: {pages[:10]}{'...' if len(pages) > 10 else ''} ({len(pages)} total)")
        
        if results.get('chunk_types'):
            print(f"📋 CONTENT TYPES: {results['chunk_types']}")
        
        print("\n" + "="*120)
        print("🧠 ENHANCED AI ANALYSIS BY BATCH")
        print("="*120)
        
        for batch_response in results.get('batch_responses', []):
            batch_num = batch_response['batch_number']
            chunks_count = batch_response['chunks_in_batch']
            
            print(f"\n📦 BATCH {batch_num}/{results['batches_processed']} - {chunks_count} chunks")
            print("-" * 100)
            
            # Show top chunk details for this batch
            print("📋 Top chunks in this batch:")
            for detail in batch_response['chunk_details'][:3]:  # Show top 3
                enhanced_sim = detail.get('similarity', detail.get('base_similarity', 0))
                print(f"   🎯 Rank {detail['rank']} | Page {detail['page']} | Enhanced Sim: {enhanced_sim:.3f} | {detail['type']}")
                print(f"      Preview: {detail['preview']}")
            
            if len(batch_response['chunk_details']) > 3:
                print(f"   ... and {len(batch_response['chunk_details']) - 3} more chunks")
            
            print(f"\n🧠 ENHANCED AI ANALYSIS FOR BATCH {batch_num}:")
            print("-" * 80)
            print(batch_response['gpt_response'])
            print("\n" + "="*120)

def main():
    """Main function for enhanced batched RAG system"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vector_store_path = os.path.join(script_dir, "vector_store", "local_vector_store.pkl")
    
    print("🚀 ENHANCED BATCHED RAG SYSTEM")
    print("="*100)
    print("🧠 Advanced Pharmaceutical Document Analysis")
    print("🔍 Multi-pass search with enhanced AI instructions")
    print("📊 Adaptive batching and comprehensive analysis")
    print("🎯 Optimized for regulatory compliance and safety reporting")
    print("🤖 Powered by OpenAI text-embedding-ada-002 + GPT-4o")
    print("="*100)
    
    if not os.path.exists(vector_store_path):
        print("❌ Vector store not found. Please run create_embeddings.py first.")
        return
    
    # Initialize the enhanced batched RAG system
    try:
        rag = EnhancedBatchedRAG(vector_store_path)
    except Exception as e:
        print(f"❌ Error initializing enhanced RAG system: {e}")
        return
    
    # Interactive query loop
    while True:
        try:
            print(f"\n" + "="*100)
            question = input("🤔 Your pharmaceutical document question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the Enhanced RAG System!")
                break
            
            if not question:
                continue
            
            # Optional: Customize settings
            print(f"\n⚙️  Enhanced Settings: top_k=75, adaptive_batching=enabled, multi_pass_search=enabled")
            customize = input("🔧 Customize analysis settings? (y/n, default n): ").strip().lower()
            
            top_k = 75
            min_similarity = 0.2
            delay_between_requests = 2.0
            
            if customize == 'y':
                try:
                    top_k = int(input(f"📊 Total chunks to analyze (default 75): ") or "75")
                    min_similarity = float(input(f"🎯 Minimum similarity threshold (default 0.2): ") or "0.2")
                    delay_between_requests = float(input(f"⏱️  Delay between API requests in seconds (default 2.0): ") or "2.0")
                except ValueError:
                    print("⚠️  Invalid input, using enhanced default settings")
                    top_k, min_similarity, delay_between_requests = 75, 0.2, 2.0
            
            # Show analysis preview
            print(f"\n🔍 ANALYSIS PREVIEW:")
            print(f"   • Multi-pass search strategy enabled")
            print(f"   • Analyzing top {top_k} most relevant chunks")
            print(f"   • Minimum similarity threshold: {min_similarity}")
            print(f"   • Adaptive batch sizing based on content complexity")
            print(f"   • Enhanced pharmaceutical domain expertise")
            print(f"   • Comprehensive regulatory analysis framework")
            
            # Perform enhanced comprehensive query
            print(f"\n🚀 Starting enhanced analysis...")
            start_time = time.time()
            
            results = rag.comprehensive_query(
                question=question,
                top_k=top_k,
                min_similarity=min_similarity,
                delay_between_requests=delay_between_requests
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Display enhanced results
            results['total_processing_time'] = processing_time
            rag.print_enhanced_results(results)
            
            # Show processing summary
            print(f"\n📈 PROCESSING SUMMARY:")
            print(f"   • Total processing time: {processing_time:.1f} seconds")
            print(f"   • Chunks analyzed: {results['total_chunks']}")
            print(f"   • Pages covered: {len(results.get('pages_covered', []))}")
            print(f"   • Batches processed: {results['batches_processed']}")
            print(f"   • Search strategy: {results.get('search_strategy', 'standard')}")
            
            # Option to save detailed results
            save_option = input(f"\n💾 Save detailed analysis results to file? (y/n): ").strip().lower()
            if save_option == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"enhanced_pharma_analysis_{timestamp}.json"
                
                # Add metadata to results
                results['analysis_metadata'] = {
                    'system_version': 'Enhanced Batched RAG v2.0',
                    'processing_time_seconds': processing_time,
                    'timestamp': timestamp,
                    'settings': {
                        'top_k': top_k,
                        'min_similarity': min_similarity,
                        'delay_between_requests': delay_between_requests
                    }
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Enhanced analysis results saved to: {filename}")
            
            # Option for follow-up analysis
            followup = input(f"\n🔍 Generate follow-up questions based on this analysis? (y/n): ").strip().lower()
            if followup == 'y':
                followup_questions = generate_followup_questions(results)
                print(f"\n💡 SUGGESTED FOLLOW-UP QUESTIONS:")
                for i, q in enumerate(followup_questions, 1):
                    print(f"   {i}. {q}")
            
        except KeyboardInterrupt:
            print("\n👋 Analysis interrupted. Thank you for using the Enhanced RAG System!")
            break
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            print("Please check your configuration and try again.")

def generate_followup_questions(results: Dict[str, Any]) -> List[str]:
    """Generate intelligent follow-up questions based on analysis results"""
    followup_questions = []
    
    # Extract key concepts from the original question
    original_question = results.get('question', '').lower()
    
    # Generate contextual follow-up questions
    if 'complaint' in original_question:
        followup_questions.extend([
            "What is the trend of complaints over time?",
            "Which are the most common types of complaints?",
            "What is the resolution rate for these complaints?",
            "Are there any seasonal patterns in complaint reports?"
        ])
    
    if 'israel' in original_question or 'country' in original_question:
        followup_questions.extend([
            "How does this compare to other countries in the region?",
            "What is the per-capita rate for this country?",
            "Are there any country-specific regulatory requirements?",
            "What is the reporting timeline for this geographic region?"
        ])
    
    if 'unsubstantiated' in original_question:
        followup_questions.extend([
            "What are the main reasons for lack of substantiation?",
            "How long does the investigation process typically take?",
            "What percentage of unsubstantiated cases become substantiated later?",
            "Are there any patterns in unsubstantiated reports?"
        ])
    
    # Add general pharmaceutical analysis questions
    followup_questions.extend([
        "What are the key regulatory compliance implications?",
        "Are there any safety signals that require immediate attention?",
        "What is the overall risk assessment for this product?",
        "How does this data compare to industry benchmarks?"
    ])
    
    # Return top 6 most relevant questions
    return followup_questions[:6]

if __name__ == "__main__":
    main()