#!/usr/bin/env python3
"""
AI-Powered Document Search Script - Enhanced with Multi-Query Support
Using Azure OpenAI o3-mini model
"""

import os
import json
import re
import time
import requests
from typing import List, Dict, Any
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.kgw_client_id,
            'client_secret': self.kgw_client_secret
        }
        
        response = requests.post(self.ping_fed_url, headers=headers, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.token_expires_at = time.time() + token_data.get('expires_in', 3600) - 300
            return self.access_token
        else:
            raise Exception(f"Failed to get access token: {response.status_code} - {response.text}")

class EnhancedAIDocumentSearcher:
    def __init__(self):
        """Initialize the Enhanced AI Document Searcher with Azure OpenAI"""
        self.auth = AzureOpenAIAuth()
        self.endpoint = os.getenv('KGW_ENDPOINT')
        self.api_version = os.getenv('AOAI_API_VERSION')
        self.chat_deployment = 'o3-mini'  # Using o3-mini as requested
        
        if not all([self.endpoint, self.api_version]):
            raise ValueError(
                "❌ Missing required Azure OpenAI configuration!\n"
                "   Please set KGW_ENDPOINT and AOAI_API_VERSION environment variables."
            )
        
        print("✅ Enhanced AI Document Searcher initialized")
        print(f"   - Endpoint: {self.endpoint}")
        print(f"   - Model: {self.chat_deployment}")
    
    def query_ai(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
        """Generic AI query function for Azure OpenAI o3-mini"""
        access_token = self.auth.get_access_token()
        
        url = f"{self.endpoint}/openai/deployments/{self.chat_deployment}/chat/completions?api-version={self.api_version}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        payload = {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'max_completion_tokens': max_tokens
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
                    return f"Error: {response.status_code} - {response.text}"
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error: {str(e)}"
                wait_time = (2 ** attempt) + 2
                print(f"   Request error, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return f"Failed to get response after {max_retries} attempts"
    
    def analyze_input(self, user_input: str) -> Dict[str, Any]:
        """AI determines if input contains multiple distinct questions"""
        system_prompt = """You are an expert at analyzing user queries to determine if they contain multiple distinct questions.

IMPORTANT: Only classify as multiple queries if there are clearly SEPARATE, DISTINCT questions separated by punctuation or conjunctions.

Single query examples:
- "Find all 14 complaints for Israel"
- "Can you find all complaints for Israel?"
- "How many complaints are for Israel and what are their issues?" (single complex query)
- "Show me rejected batches and their reasons" (single complex query)

Multiple query examples:
- "How many complaints for Israel? What's the CAPA status?" (two separate questions)
- "Find Israel complaints. Also show CAPA items." (two separate requests)

Return JSON:
{
    "query_type": "single|multiple_independent",
    "questions": ["question1", "question2", ...],
    "processing_strategy": "single|sequential",
    "reasoning": "brief explanation"
}

Default to "single" unless there are clearly separate questions.
Return ONLY the JSON object, no other text."""
        
        user_prompt = f"""Analyze this user input: "{user_input}"

Determine if this is a single query or multiple independent queries."""
        
        try:
            response = self.query_ai(system_prompt, user_prompt, max_tokens=500)
            result = json.loads(response.strip())
            
            # Validate the result
            if "query_type" not in result:
                raise ValueError("Invalid response format")
                
            return result
            
        except Exception as e:
            print(f"⚠️ Query analysis failed: {e}")
            # Fallback: treat as single query
            return {
                "query_type": "single",
                "questions": [user_input],
                "processing_strategy": "single",
                "reasoning": "Fallback due to analysis error"
            }
    
    def fix_line_breaks(self, text: str) -> str:
        """Fix line breaks in PDF text that may be missing proper formatting"""
        patterns_for_new_lines = [
            r'(\d{12})',  # 12-digit numbers (like complaint IDs)
            r'(000\d{9})',  # GSK complaint patterns
            r'(QE-\d{6})',  # QE complaint patterns
            r'(DEV-\d{6})',  # Deviation patterns
            r'(Table \d+)',  # Table references
            r'(Section \d+)',  # Section references
            r'(\d+\.\d+)',  # Numbered sections like 5.1, 5.2
            r'(Page \d+ of \d+)',  # Page references
        ]
        
        for pattern in patterns_for_new_lines:
            text = re.sub(pattern, r'\n\1', text)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s+', '\n', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text
        
    def read_document(self, file_path: str) -> str:
        """Read document content from file - handles both text and PDF files"""
        try:
            if file_path.lower().endswith('.pdf'):
                try:
                    import pdfplumber
                    text = ""
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    
                    text = self.fix_line_breaks(text)
                    return text
                    
                except ImportError:
                    print("📦 pdfplumber not installed. Installing now...")
                    os.system("pip install pdfplumber")
                    import pdfplumber
                    text = ""
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    
                    text = self.fix_line_breaks(text)
                    return text
            else:
                encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            return file.read()
                    except UnicodeDecodeError:
                        continue
                        
                with open(file_path, 'rb') as file:
                    content = file.read()
                    return content.decode('utf-8', errors='ignore')
                    
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return ""

    def simple_search(self, document_text: str, query: str) -> List[Dict[str, Any]]:
        """AI-powered fallback search that understands the query semantically"""
        system_prompt = """You are an expert at analyzing search queries to extract the most relevant search terms.

For Israel queries, focus ONLY on finding lines containing "Israel".
For other queries, extract relevant terms.

Return a JSON object with:
1. "search_terms": list of key terms/concepts to search for
2. "search_strategy": brief description of search approach
3. "case_sensitive": boolean if search should be case sensitive
4. "primary_term": the most important term to search for

Examples:
- "How many complaints for Israel?" → {"search_terms": ["israel"], "primary_term": "israel", "search_strategy": "find lines with Israel", "case_sensitive": false}

Return ONLY the JSON object, no other text."""
        
        user_prompt = f"""Analyze this search query and extract what the user is looking for: "{query}"

Extract the most relevant search terms for this query."""
        
        try:
            response = self.query_ai(system_prompt, user_prompt, max_tokens=500)
            search_config = json.loads(response.strip())
            search_terms = search_config.get("search_terms", [])
            primary_term = search_config.get("primary_term", search_terms[0] if search_terms else "")
            case_sensitive = search_config.get("case_sensitive", False)
            
        except Exception as e:
            print(f"⚠️ AI analysis failed, using basic text extraction: {e}")
            words = re.findall(r'\b\w+\b', query.lower())
            search_terms = [w for w in words if len(w) > 2 and w not in 
                          ['what', 'how', 'many', 'show', 'find', 'the', 'and', 'for', 'with', 'are', 'is', 'of', 'to', 'from']]
            primary_term = search_terms[0] if search_terms else ""
            case_sensitive = False
        
        results = []
        lines = document_text.split('\n')
        
        # For Israel queries, be very specific
        if 'israel' in query.lower():
            for i, line in enumerate(lines, 1):
                if 'israel' in line.lower():
                    results.append({
                        'line_number': i,
                        'content': line.strip(),
                        'details': {
                            'matched_terms': ['israel'],
                            'search_method': 'simple_israel_specific'
                        }
                    })
        else:
            # General search for other queries
            for i, line in enumerate(lines, 1):
                line_to_search = line if case_sensitive else line.lower()
                terms_to_find = search_terms if case_sensitive else [term.lower() for term in search_terms]
                
                if any(term in line_to_search for term in terms_to_find):
                    matched_terms = [term for term in terms_to_find if term in line_to_search]
                    results.append({
                        'line_number': i,
                        'content': line.strip(),
                        'details': {
                            'matched_terms': matched_terms,
                            'search_method': 'ai_semantic_search'
                        }
                    })
        
        return results
    
    def generate_search_function(self, query: str) -> str:
        """Ask AI to generate a Python search function that casts a broad net"""
        print(f"🤖 Analyzing query: '{query}'")
        
        system_prompt = """You are an expert Python programmer. Create a search function that casts a BROAD net to capture comprehensive information.

STRATEGY: Cast a BROAD net to capture comprehensive information, then rely on AI analysis to interpret results intelligently.

INSTRUCTIONS:
1. Extract key terms from the query
2. Search for lines containing ANY of those terms (OR logic)
3. Include related terms and variations
4. Be comprehensive rather than restrictive
5. Use case-insensitive matching

EXAMPLES:
- "How many complaints for Israel?" → Search for "israel" OR "complaint" 
- "CAPA status" → Search for "capa" OR "status"
- "rejected batches" → Search for "reject" OR "rejected" OR "batch"

This broad approach ensures we don't miss relevant context or summary information.

Write ONLY the Python function code. No explanations or markdown formatting."""
        
        user_prompt = f"""Create a Python function to search for: "{query}"

The function should cast a broad net and return ALL potentially relevant results.

Format:
def search_function(document_text: str) -> List[Dict[str, Any]]:
    results = []
    lines = document_text.split('\\n')
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Your search logic here
        if condition:
            results.append({{
                'line_number': i + 1,
                'content': line.strip(),
                'details': {{'search_method': 'ai_comprehensive'}}
            }})
    
    return results"""
        
        try:
            response = self.query_ai(system_prompt, user_prompt, max_tokens=1500)
            
            # Clean up the response to extract just the function code
            code = response.strip()
            
            # Remove any markdown formatting
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            
            print(f"✅ Generated comprehensive search function")
            return code.strip()
            
        except Exception as e:
            print(f"❌ Error generating search function: {e}")
            return ""
    
    def execute_search_function(self, function_code: str, document_text: str) -> List[Dict[str, Any]]:
        """Safely execute the generated search function with debugging"""
        print("\n" + "="*60)
        print("🔧 AI-GENERATED SEARCH FUNCTION:")
        print("="*60)
        print(function_code)
        print("="*60)
        
        try:
            exec_globals = {
                'List': List,
                'Dict': Dict,
                'Any': Any,
                're': re,
                'json': json
            }
            
            exec(function_code, exec_globals)
            search_function = exec_globals['search_function']
            results = search_function(document_text)
            
            print(f"✅ Function executed successfully, found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ Error executing AI search function: {e}")
            print(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            return []
    
    def debug_israel_search(self, document_text: str) -> None:
        """Debug method to show how many Israel mentions exist"""
        lines = document_text.split('\n')
        israel_lines = []
        
        for i, line in enumerate(lines, 1):
            if 'israel' in line.lower():
                israel_lines.append((i, line.strip()))
        
        print(f"\n🧪 DEBUG: Found {len(israel_lines)} lines containing 'Israel':")
        for i, (line_num, content) in enumerate(israel_lines, 1):
            print(f"  {i}. Line {line_num}: {content[:100]}...")
            if i >= 15:  # Show first 15
                print(f"  ... and {len(israel_lines) - 15} more")
                break
    
    def search_single_query(self, document_text: str, query: str) -> Dict[str, Any]:
        """Two-stage search: Stage 1 = broad collection, Stage 2 = intelligent analysis"""
        print(f"🔍 STAGE 1: Broad data collection for: '{query}'")
        
        # Debug for Israel queries to verify data exists
        if 'israel' in query.lower():
            self.debug_israel_search(document_text)
        
        # Stage 1: Use broad AI-generated search function
        print("🤖 Stage 1: Generating comprehensive search function...")
        function_code = self.generate_search_function(query)
        
        results = []
        search_method = ""
        
        if function_code:
            print("⚡ Stage 1: Executing comprehensive search...")
            results = self.execute_search_function(function_code, document_text)
            if results:
                search_method = "Two-stage (broad collection)"
        
        # Fallback to simple search if AI generation fails
        if not results:
            print("🔄 Stage 1 failed, using simple search fallback...")
            results = self.simple_search(document_text, query)
            search_method = "Two-stage (simple fallback)"
        
        print(f"✅ Stage 1 complete: Collected {len(results)} results")
        
        # Stage 2: Intelligent analysis (this will filter and analyze)
        analysis = self.analyze_results(query, results)
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results,
            "analysis": analysis,
            "search_method": search_method
        }
    
    def analyze_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Two-stage analysis: Send ALL results + query for intelligent filtering"""
        if not results:
            return f"No results found for query: '{query}'"
        
        # Stage 2: Send EVERYTHING to AI for intelligent filtering
        print(f"🧠 Stage 2: Sending {len(results)} results to o3-mini for intelligent analysis...")
        
        # For Israel queries, pre-filter to only Israel results to save tokens
        if 'israel' in query.lower():
            israel_results = [r for r in results if 'israel' in r['content'].lower()]
            if israel_results:
                print(f"🇮🇱 Pre-filtering: Found {len(israel_results)} Israel-specific results out of {len(results)} total")
                results_for_analysis = israel_results
            else:
                results_for_analysis = results[:50]  # Fallback limit
        else:
            results_for_analysis = results[:50]  # Limit to prevent token overflow
        
        # Prepare results for analysis (limit to avoid token limits)
        max_results_for_analysis = 50  # Limit to prevent token overflow
        results_for_analysis = results[:max_results_for_analysis]
        
        results_text = f"STAGE 2 ANALYSIS\n"
        results_text += f"Original User Query: '{query}'\n"
        results_text += f"Search Results Found: {len(results)} total results\n"
        results_text += f"Analyzing first {len(results_for_analysis)} results:\n\n"
        
        for i, result in enumerate(results_for_analysis, 1):
            results_text += f"{i}. Line {result['line_number']}: {result['content']}\n"
        
        if len(results) > max_results_for_analysis:
            results_text += f"\n... and {len(results) - max_results_for_analysis} more results not shown due to token limits"
        
        system_prompt = """You are in STAGE 2 of a two-stage AI search process.

STAGE 1 cast a broad net and found multiple results that might be relevant.
STAGE 2 (your job): Intelligently filter through ALL the results to answer the user's specific question.

INSTRUCTIONS:
1. Read through ALL results carefully
2. Identify which results actually answer the user's question
3. Filter out irrelevant noise (headers, general mentions, etc.)
4. Provide a precise answer based on the relevant results only

For example, if the user asked "How many complaints for Israel?":
- Look for actual individual complaint entries mentioning Israel
- Ignore general references to "complaints" that don't mention Israel
- Count the specific complaint instances
- Provide the exact number and details

Be thorough but focused. Use ALL the data, but answer the specific question asked."""
        
        user_prompt = f"""USER'S ORIGINAL QUESTION: "{query}"

YOUR TASK:
Read through ALL {len(results)} results carefully and answer the user's specific question.

{results_text}"""
        
        try:
            response = self.query_ai(system_prompt, user_prompt, max_tokens=2000)
            return response
            
        except Exception as e:
            return f"Stage 2 analysis failed: {e}. Found {len(results)} results but couldn't analyze them."
    
    def process_multiple_independent(self, document_text: str, questions: List[str]) -> Dict[str, Any]:
        """Process multiple independent questions separately"""
        print(f"📋 Processing {len(questions)} independent queries...")
        
        all_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n--- Query {i}/{len(questions)} ---")
            result = self.search_single_query(document_text, question)
            all_results.append(result)
        
        # Generate combined summary
        summary = self.generate_multi_query_summary(all_results)
        
        return {
            "query_type": "multiple_independent",
            "total_queries": len(questions),
            "individual_results": all_results,
            "combined_summary": summary
        }
    
    def generate_multi_query_summary(self, all_results: List[Dict[str, Any]]) -> str:
        """Generate a combined summary for multiple independent queries"""
        summary_text = "MULTI-QUERY SEARCH RESULTS:\n\n"
        
        for i, result in enumerate(all_results, 1):
            summary_text += f"Query {i}: {result['query']}\n"
            summary_text += f"Results: {result['total_results']} found\n"
            analysis_preview = result['analysis'][:200] + "..." if len(result['analysis']) > 200 else result['analysis']
            summary_text += f"Analysis: {analysis_preview}\n\n"
        
        system_prompt = """You are analyzing multiple search query results to create a comprehensive summary.

Provide:
1. **Executive Summary** - key findings across all queries
2. **Cross-Query Insights** - patterns or connections between results
3. **Actionable Takeaways** - what this means overall

Be concise but comprehensive."""
        
        user_prompt = f"""Create a comprehensive summary of these multiple search queries:

{summary_text}"""
        
        try:
            response = self.query_ai(system_prompt, user_prompt, max_tokens=800)
            return response
            
        except Exception as e:
            return "Multiple queries processed successfully. See individual results for details."
    
    def search_document(self, file_path: str, user_input: str) -> Dict[str, Any]:
        """Main method with multi-query support"""
        print(f"\n🔍 Processing input: '{user_input}'")
        print("📖 Reading document...")
        
        document_text = self.read_document(file_path)
        if not document_text:
            return {"error": "Could not read document"}
        
        print(f"✅ Document loaded ({len(document_text)} characters)")
        
        # Analyze the input to determine query structure
        print("🤖 Analyzing query structure...")
        query_analysis = self.analyze_input(user_input)
        
        print(f"📊 Query type: {query_analysis['query_type']}")
        print(f"💭 Reasoning: {query_analysis['reasoning']}")
        
        # Route to appropriate processor
        if query_analysis["query_type"] == "single":
            return self.search_single_query(document_text, user_input)
        elif query_analysis["query_type"] == "multiple_independent":
            return self.process_multiple_independent(document_text, query_analysis["questions"])
        else:
            # Fallback to single query processing
            return self.search_single_query(document_text, user_input)

def interactive_mode():
    """Enhanced interactive mode with multi-query support"""
    print("🚀 Enhanced AI Document Searcher - Multi-Query Support (Azure OpenAI o3-mini)")
    print("=" * 80)
    
    try:
        searcher = EnhancedAIDocumentSearcher()
    except ValueError as e:
        print(e)
        return
    
    while True:
        file_path = input("\n📁 Enter document path (or 'quit' to exit): ").strip()
        if file_path.lower() == 'quit':
            return
        
        if os.path.exists(file_path):
            break
        else:
            print(f"❌ File not found: {file_path}")
    
    print(f"✅ Using document: {file_path}")
    print("\n💡 Examples of multi-query inputs:")
    print("   • 'How many complaints for Israel? What's the CAPA status?'")
    print("   • 'Find rejected batches and their failure reasons'")
    print("   • 'Show me temperature deviations and their impact'")
    
    while True:
        print("\n" + "="*80)
        user_input = input("🔍 Enter your query/queries (or 'quit' to exit): ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if not user_input:
            print("❌ Please enter a search query")
            continue
        
        result = searcher.search_document(file_path, user_input)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        # Display results based on query type
        if result.get("query_type") == "multiple_independent":
            print(f"\n✅ Processed {result['total_queries']} independent queries")
            print(f"\n📋 COMBINED SUMMARY:")
            print(result['combined_summary'])
            
            show_individual = input(f"\n❓ Show individual query results? (y/n): ").strip().lower()
            if show_individual == 'y':
                for i, res in enumerate(result['individual_results'], 1):
                    print(f"\n--- Query {i}: {res['query']} ---")
                    print(f"Results: {res['total_results']}")
                    print(f"Analysis: {res['analysis']}")
                    
        else:
            # Single query result
            print(f"\n✅ Found {result['total_results']} results using {result.get('search_method', 'Unknown')} search")
            print(f"\n📋 Analysis:")
            print(result['analysis'])
        
        # Option to show raw results
        if result.get('results'):
            show_raw = input(f"\n❓ Show raw results? (y/n): ").strip().lower()
            if show_raw == 'y':
                results_to_show = result.get('results', [])
                if result.get("query_type") == "multiple_independent":
                    print("📝 Raw results from all queries:")
                    for query_result in result['individual_results']:
                        print(f"\n--- {query_result['query']} ---")
                        for i, res in enumerate(query_result['results'][:5], 1):
                            print(f"  {i}. Line {res['line_number']}: {res['content'][:100]}...")
                else:
                    print("📝 Raw results:")
                    for i, res in enumerate(results_to_show[:10], 1):
                        print(f"  {i}. Line {res['line_number']}: {res['content'][:100]}...")

def main():
    """Enhanced main function"""
    print("🚀 Enhanced AI Document Searcher with Multi-Query Support")
    print("🤖 Powered by Azure OpenAI o3-mini")
    print("=" * 80)
    print("New Features:")
    print("✨ Intelligent query detection")
    print("✨ Multiple independent questions processing")
    print("✨ Two-stage search: broad collection + intelligent filtering")
    print("✨ Cross-query insights and summaries")
    print("\nChoose your mode:")
    print("1. 💬 Interactive Mode (Multi-query support)")
    print("2. ❓ Help")
    
    while True:
        choice = input("\nSelect mode (1 or 2): ").strip()
        
        if choice == "1":
            interactive_mode()
            break
        elif choice == "2":
            print_help()
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")

def print_help():
    """Enhanced help information"""
    print("\n📚 Enhanced AI Document Searcher Help")
    print("=" * 80)
    print("🎯 Purpose:")
    print("   Advanced AI-powered document search with multi-query support using Azure OpenAI o3-mini.")
    print("   Intelligently handles single questions, multiple independent questions.")
    
    print("\n✨ Multi-Query Examples:")
    print("   🔸 Independent Questions:")
    print("     'How many complaints for Israel? What's the CAPA status?'")
    print("     'Show rejected batches. Find temperature deviations.'")
    
    print("\n   🔸 Single Questions:")
    print("     'Find all complaints from Israel'")
    print("     'Show me ongoing CAPA items'")
    
    print("\n🧠 Two-Stage Intelligence:")
    print("   • Stage 1: Broad data collection with comprehensive search")
    print("   • Stage 2: Intelligent filtering and analysis by o3-mini")
    print("   • Automatic query type detection")
    print("   • Cross-query pattern analysis")
    
    print("\n🔧 Setup:")
    print("   1. Set up Azure OpenAI environment variables:")
    print("      - PING_FED_URL")
    print("      - KGW_CLIENT_ID") 
    print("      - KGW_CLIENT_SECRET")
    print("      - KGW_ENDPOINT")
    print("      - AOAI_API_VERSION")
    print("   2. Install: pip install requests python-dotenv pdfplumber")
    print("   3. Run script and test with multi-query inputs")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()