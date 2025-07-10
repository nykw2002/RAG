#!/usr/bin/env python3
"""
Memory-Based Direct Processing - Enhanced with GPT-o3-mini for superior reasoning
Optimized for comprehensive pharmaceutical document analysis
"""

import json
import os
import time
import requests
from datetime import datetime
from typing import List, Dict, Any

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

class EnhancedMemoryProcessor:
    """Enhanced processor using GPT-o3-mini for superior reasoning and systematic analysis"""
    
    def __init__(self):
        self.auth = AzureOpenAIAuth()
        self.endpoint = os.getenv('KGW_ENDPOINT')
        self.api_version = os.getenv('AOAI_API_VERSION')
        
        # Check if o3-mini should be used
        self.use_o3_mini = os.getenv('USE_O3_MINI', 'false').lower() == 'true'
        
        if self.use_o3_mini:
            self.chat_deployment = os.getenv('GPT_O3_MINI_DEPLOYMENT_NAME')
            self.model_name = "o3-mini"
        else:
            self.chat_deployment = os.getenv('CHAT_MODEL_DEPLOYMENT_NAME')
            self.model_name = "gpt-4o"
        
        if not all([self.endpoint, self.api_version, self.chat_deployment]):
            raise ValueError("Missing required Azure OpenAI configuration")
        
        print("✅ Enhanced Memory Processor initialized")
        print(f"   - Endpoint: {self.endpoint}")
        print(f"   - Model: {self.model_name} ({self.chat_deployment})")
        print(f"   - Enhanced reasoning mode: {'ON' if self.use_o3_mini else 'OFF'}")
        print(f"   - Systematic analysis optimized for completeness")
    
    def load_document(self, json_path: str) -> Dict[str, Any]:
        """Load JSON document"""
        print(f"📂 Loading document from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def split_document(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into optimized parts for o3-mini processing"""
        if 'pages' not in data:
            return [data]
        
        pages = data['pages']
        total_pages = len(pages)
        
        # Optimize part size for o3-mini's reasoning capabilities
        if self.use_o3_mini:
            # o3-mini can handle larger contexts with better reasoning
            part_size = total_pages // 4  # 4 parts for more granular analysis
            num_parts = 4
        else:
            part_size = total_pages // 3  # Original 3 parts
            num_parts = 3
        
        parts = []
        for i in range(num_parts):
            start_idx = i * part_size
            if i == num_parts - 1:  # Last part takes remainder
                end_idx = total_pages
            else:
                end_idx = (i + 1) * part_size
            
            part = {
                'metadata': data.get('metadata', {}),
                'pages': pages[start_idx:end_idx],
                'part_info': f"Pages {start_idx + 1}-{end_idx} (Part {i+1} of {num_parts})",
                'part_number': i + 1,
                'total_parts': num_parts
            }
            parts.append(part)
        
        print(f"📄 Split document into {num_parts} parts for {self.model_name} processing:")
        for i, part in enumerate(parts, 1):
            start_page = part['pages'][0].get('page_number', 'Unknown') if part['pages'] else 'Empty'
            end_page = part['pages'][-1].get('page_number', 'Unknown') if part['pages'] else 'Empty'
            page_count = len(part['pages'])
            print(f"   Part {i}: Pages {start_page}-{end_page} ({page_count} pages)")
        
        return parts
    
    def format_part_to_text(self, part: Dict[str, Any]) -> str:
        """Convert document part to text with enhanced structure for o3-mini"""
        text_parts = []
        
        # Add structured header for o3-mini
        text_parts.append(f"=== DOCUMENT ANALYSIS: {part['part_info']} ===")
        text_parts.append(f"Part {part['part_number']} of {part['total_parts']}")
        text_parts.append("")
        
        # Add pages with enhanced structure
        for page in part['pages']:
            page_num = page.get('page_number', 'Unknown')
            text_parts.append(f"\n--- PAGE {page_num} START ---")
            
            # Add page text with markers
            if page.get('text'):
                text_parts.append("[TEXT CONTENT]")
                text_parts.append(page['text'])
                text_parts.append("[END TEXT CONTENT]")
            
            # Add tables with enhanced structure
            if page.get('tables'):
                for i, table in enumerate(page['tables']):
                    text_parts.append(f"\n[TABLE {i+1} ON PAGE {page_num} START]")
                    if table.get('rows'):
                        for row_idx, row in enumerate(table['rows']):
                            if row and any(cell for cell in row if cell):
                                clean_row = [str(cell).strip() if cell else "" for cell in row]
                                text_parts.append(f"Row {row_idx + 1}: {' | '.join(clean_row)}")
                    text_parts.append(f"[TABLE {i+1} END]")
            
            text_parts.append(f"--- PAGE {page_num} END ---")
        
        return "\n".join(text_parts)
    
    def query_ai(self, messages: List[Dict[str, str]], reasoning_effort: str = "medium") -> str:
        """Query AI with o3-mini optimized parameters"""
        access_token = self.auth.get_access_token()
        
        url = f"{self.endpoint}/openai/deployments/{self.chat_deployment}/chat/completions?api-version={self.api_version}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        # Optimize parameters based on model
        if self.use_o3_mini:
            payload = {
                'messages': messages,
                'max_completion_tokens': 4000,  # o3-mini uses max_completion_tokens instead of max_tokens
                'reasoning_effort': reasoning_effort  # o3-mini specific parameter
            }
        else:
            payload = {
                'messages': messages,
                'max_tokens': 3000,
                'temperature': 0.0,
                'top_p': 0.95
            }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=240)  # Longer timeout for o3-mini
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) + 5
                    print(f"   Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return f"Error: {response.status_code} - {response.text}"
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error querying AI: {str(e)}"
                wait_time = (2 ** attempt) + 3
                print(f"   Request error, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return f"Failed to get response after {max_retries} attempts"
    
    def analyze_document_structure(self, question: str) -> str:
        """Enhanced structure analysis using o3-mini's reasoning capabilities"""
        print("🧠 Step 1: Enhanced document structure analysis...")
        
        system_prompt = '''You are an expert pharmaceutical regulatory analyst with deep knowledge of drug safety documentation, adverse event reporting, and regulatory compliance. You have extensive experience analyzing Periodic Product Reviews (PPRs), safety reports, and regulatory submission documents.

Use systematic reasoning to create a comprehensive search strategy.'''
        
        user_prompt = f'''DOCUMENT CONTEXT:
- Type: Pharmaceutical regulatory document (123 pages)
- Content: Periodic Product Review (PPR) for Fluticasone Furoate Nasal Spray
- Scope: Review period May 2023 to April 2024
- Source: GSK regulatory submission

ANALYSIS QUESTION: "{question}"

TASK: Create a systematic, comprehensive search strategy using step-by-step reasoning.

REASONING FRAMEWORK:
1. QUESTION ANALYSIS
   - What specific information is being requested?
   - What type of data would answer this question?
   - Where in a PPR document would this information typically be located?

2. DOCUMENT STRUCTURE MAPPING
   - Identify key sections that would contain relevant information
   - Consider appendices, tables, and supplementary data sections
   - Think about regulatory reporting requirements and standard formats

3. SEARCH STRATEGY DEVELOPMENT
   - Define specific keywords and phrases to look for
   - Identify table structures and data patterns
   - Create systematic approach to ensure no instances are missed
   - Consider variations in terminology and reporting formats

4. QUALITY ASSURANCE APPROACH
   - How to verify completeness of findings
   - Cross-referencing methods
   - Validation checks to prevent missing data

Provide a detailed, systematic strategy that ensures comprehensive coverage.'''
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        reasoning_effort = "high" if self.use_o3_mini else "medium"
        return self.query_ai(messages, reasoning_effort)
    
    def process_part(self, part_text: str, question: str, part_num: int, total_parts: int, 
                    strategy: str, previous_findings: List[str]) -> str:
        """Enhanced part processing with o3-mini's systematic reasoning"""
        print(f"🧠 Step {part_num + 1}: Enhanced processing of Part {part_num + 1}/{total_parts}...")
        
        # Create condensed summary of previous findings for context
        if previous_findings:
            findings_summary = f"PREVIOUS FINDINGS SUMMARY: {len(previous_findings)} instances found in previous parts. Continue systematic search in this part."
        else:
            findings_summary = "FIRST PART: Beginning systematic search for all instances."
        
        system_prompt = '''You are a meticulous pharmaceutical regulatory analyst. Your task is to systematically identify and extract ALL instances of the requested information from this document section.

Use methodical reasoning to ensure complete coverage - missing even one instance could have regulatory implications.'''
        
        user_prompt = f'''ANALYSIS QUESTION: {question}

SEARCH STRATEGY TO FOLLOW:
{strategy}

CONTEXT: {findings_summary}

DOCUMENT SECTION TO ANALYZE:
{part_text}

SYSTEMATIC ANALYSIS REQUIREMENTS:

1. METHODICAL SCANNING
   - Read every line systematically
   - Check all tables row by row
   - Examine headers, footnotes, and annotations
   - Look for both explicit and implicit references

2. EXTRACTION PROTOCOL
   - For each finding, provide: ID number, exact page number, specific description
   - Quote relevant text portions
   - Note context and surrounding information
   - Identify data source (table, text, appendix, etc.)

3. VERIFICATION CHECKS
   - Double-check numerical values
   - Verify page references
   - Ensure no duplicates within this section
   - Cross-reference with strategy requirements

4. PROGRESS TRACKING
   - Count findings in this part
   - Maintain running total from all parts processed so far
   - Note any patterns or trends observed

RESPONSE FORMAT:
PART {part_num + 1} ANALYSIS:
- Findings in this part: [exact number]
- Detailed list:
  [ID] Page [number]: [detailed description with context]
  [ID] Page [number]: [detailed description with context]
  ...
- Running total: [total from all parts so far]
- Analysis notes: [any patterns, observations, or quality notes]

Execute systematic analysis now:'''
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        reasoning_effort = "high" if self.use_o3_mini else "medium"
        return self.query_ai(messages, reasoning_effort)
    
    def process_query(self, json_path: str, question: str) -> Dict[str, Any]:
        """Enhanced main processing function"""
        start_time = time.time()
        
        # Load and split document
        data = self.load_document(json_path)
        parts = self.split_document(data)
        
        # Step 1: Enhanced document structure analysis
        strategy = self.analyze_document_structure(question)
        
        # Step 2: Process each part with enhanced systematic analysis
        all_responses = []
        findings = []
        
        for i, part in enumerate(parts):
            part_text = self.format_part_to_text(part)
            part_size = len(part_text)
            estimated_tokens = part_size // 4
            print(f"      📊 Part {i+1} size: {part_size:,} chars (~{estimated_tokens:,} tokens)")
            
            response = self.process_part(part_text, question, i, len(parts), strategy, findings)
            all_responses.append(response)
            
            # Extract findings for context in next iteration
            if "Findings in this part:" in response:
                findings.append(response)
            
            # Adaptive delay based on model (o3-mini may need more time)
            if i < len(parts) - 1:
                delay = 4 if self.use_o3_mini else 2
                print(f"   ⏱️  Waiting {delay}s before next part...")
                time.sleep(delay)
        
        # Step 3: Enhanced final synthesis
        print("🔗 Creating enhanced final synthesis...")
        final_response = self.synthesize_final_answer(question, all_responses)
        
        end_time = time.time()
        
        return {
            'question': question,
            'model_used': self.model_name,
            'reasoning_enhanced': self.use_o3_mini,
            'strategy': strategy,
            'part_responses': all_responses,
            'final_answer': final_response,
            'parts_processed': len(parts),
            'processing_time': end_time - start_time,
            'timestamp': datetime.now().isoformat(),
            'method': 'enhanced_memory_processing_with_reasoning'
        }
    
    def synthesize_final_answer(self, question: str, part_responses: List[str]) -> str:
        """Enhanced final synthesis using o3-mini's reasoning capabilities"""
        
        # Create structured summary of all findings
        findings_summary = []
        for i, response in enumerate(part_responses, 1):
            # Extract key findings from each part
            if "Findings in this part:" in response:
                findings_summary.append(f"PART {i} FINDINGS:\n{response}\n")
            else:
                findings_summary.append(f"PART {i} RESPONSE:\n{response[:800]}...\n")
        
        system_prompt = '''You are a senior pharmaceutical regulatory expert conducting final analysis. Synthesize all findings with meticulous attention to completeness and accuracy.

Ensure no instances are missed and provide comprehensive regulatory context.'''
        
        user_prompt = f'''ANALYSIS QUESTION: {question}

ALL PART RESPONSES:
{chr(10).join(findings_summary)}

FINAL SYNTHESIS REQUIREMENTS:

1. COMPREHENSIVE CONSOLIDATION
   - Combine ALL findings from ALL parts
   - Eliminate any duplicates
   - Provide final accurate total count
   - List every instance with specific page references

2. VERIFICATION AND VALIDATION
   - Cross-check for consistency across parts
   - Verify all page numbers and descriptions
   - Ensure no gaps in coverage
   - Note any discrepancies or uncertainties

3. REGULATORY CONTEXT
   - Explain significance of findings
   - Consider compliance implications
   - Identify any patterns or trends
   - Note data quality and completeness

4. ACTIONABLE SUMMARY
   - Provide clear, definitive answer to the question
   - Include confidence level in completeness
   - Suggest any follow-up analysis needed
   - Highlight key insights for decision-making

FINAL COMPREHENSIVE ANSWER:'''
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        reasoning_effort = "high" if self.use_o3_mini else "medium"
        return self.query_ai(messages, reasoning_effort)

def main():
    """Enhanced main function"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "test_extracted.json")
    
    print("🚀 ENHANCED MEMORY-BASED PROCESSING")
    print("=" * 70)
    print("🧠 Powered by GPT-o3-mini advanced reasoning")
    print("📄 Systematic pharmaceutical document analysis")
    print("🔍 Optimized for comprehensive coverage")
    print("⚡ Enhanced accuracy and completeness")
    print("=" * 70)
    
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return
    
    try:
        processor = EnhancedMemoryProcessor()
    except Exception as e:
        print(f"❌ Error initializing enhanced processor: {e}")
        return
    
    while True:
        try:
            print(f"\n" + "=" * 70)
            question = input("🤔 Your pharmaceutical document question (or 'quit' to exit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using Enhanced Memory Processing!")
                break
            
            if not question:
                continue
            
            print(f"\n🔍 Processing with {processor.model_name} enhanced reasoning...")
            result = processor.process_query(json_path, question)
            
            print(f"\n" + "=" * 70)
            print(f"📋 ENHANCED PROCESSING RESULT")
            print("=" * 70)
            print(f"❓ QUESTION: {result['question']}")
            print(f"🤖 MODEL: {result['model_used']} (Reasoning: {'Enhanced' if result['reasoning_enhanced'] else 'Standard'})")
            print(f"📄 PARTS PROCESSED: {result['parts_processed']}")
            print(f"⏱️  PROCESSING TIME: {result['processing_time']:.1f} seconds")
            print(f"🔬 METHOD: {result['method']}")
            print(f"\n🎯 COMPREHENSIVE FINAL ANSWER:")
            print("-" * 70)
            print(result['final_answer'])
            print("=" * 70)
            
            # Enhanced options
            show_strategy = input(f"\n🔍 Show detailed analysis strategy? (y/n): ").strip().lower()
            if show_strategy == 'y':
                print(f"\n📋 SYSTEMATIC ANALYSIS STRATEGY:")
                print("-" * 50)
                print(result['strategy'])
            
            show_parts = input(f"\n🔍 Show individual part analyses? (y/n): ").strip().lower()
            if show_parts == 'y':
                for i, response in enumerate(result['part_responses'], 1):
                    print(f"\n📄 PART {i} DETAILED ANALYSIS:")
                    print("-" * 50)
                    print(response)
            
            # Save with enhanced metadata
            save_option = input(f"\n💾 Save detailed results? (y/n): ").strip().lower()
            if save_option == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"enhanced_reasoning_analysis_{timestamp}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Enhanced analysis saved to: {filename}")
            
        except KeyboardInterrupt:
            print("\n👋 Analysis interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()