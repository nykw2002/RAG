#!/usr/bin/env python3
"""
Simplified Deep Research Document Analysis - summary2.py
Two-phase approach: AI-driven section selection + Deep research investigation
"""

import os
import time
import requests
import json
import re
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

class SimplifiedDeepResearchAnalyzer:
    """Simplified deep research document analyzer with AI-driven section selection"""
    
    def __init__(self):
        self.auth = AzureOpenAIAuth()
        self.endpoint = os.getenv('KGW_ENDPOINT')
        self.api_version = os.getenv('AOAI_API_VERSION')
        self.chat_deployment = os.getenv('CHAT_MODEL_DEPLOYMENT_NAME')
        
        if not all([self.endpoint, self.api_version, self.chat_deployment]):
            raise ValueError("Missing required Azure OpenAI configuration")
        
        print("✅ Simplified Deep Research Analyzer initialized")
        print(f"   - Endpoint: {self.endpoint}")
        print(f"   - Model: {self.chat_deployment}")
        print(f"   - Method: AI-driven section selection + Deep research")
    
    def load_document_summary(self, summary_file: str = "doc_summary.json") -> Dict[str, Any]:
        """Load document summary from JSON file"""
        print(f"📂 Loading document summary: {summary_file}")
        
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            sections_count = len(summary_data.get('section_summaries', []))
            print(f"✅ Loaded summary with {sections_count} sections")
            return summary_data
        except Exception as e:
            raise Exception(f"Error loading summary: {e}")
    
    def load_original_document(self, file_path: str) -> str:
        """Load original document for deep analysis"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error loading original document: {e}")
    
    def query_ai(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
        """Generic AI query function"""
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
            'max_tokens': max_tokens,
            'temperature': 0.0
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
    
    def ai_select_sections(self, question: str, summary_data: Dict[str, Any]) -> List[int]:
        """Phase 1: AI analyzes JSON summary and selects top 3 sections"""
        print("🤖 Phase 1: AI analyzing summary to select relevant sections...")
        
        system_prompt = """You are an expert document analyst. Your task is to analyze a document summary and identify the TOP 3 sections most relevant to answering a specific question.

INSTRUCTIONS:
1. Carefully read the provided JSON document summary
2. Analyze each section summary for relevance to the research question
3. Select the TOP 3 sections that are most likely to contain the answer
4. Respond with ONLY the section numbers in this exact format: [1, 2, 3]
5. Do not include any other text, explanations, or formatting

Your response must be a simple list of 3 numbers representing section numbers."""
        
        # Convert summary to a clean format for AI
        summary_text = "DOCUMENT SUMMARY:\n\n"
        for section in summary_data.get('section_summaries', []):
            summary_text += f"Section {section['section_number']}:\n{section['summary']}\n\n"
        
        user_prompt = f"RESEARCH QUESTION: {question}\n\n{summary_text}\n\nSelect the TOP 3 sections most relevant to answering this question. Respond with only the section numbers in format: [1, 2, 3]"
        
        ai_response = self.query_ai(system_prompt, user_prompt, max_tokens=50)
        
        print(f"🤖 AI Response: {ai_response}")
        
        # Extract section numbers from AI response
        selected_sections = self.extract_section_numbers(ai_response)
        
        if not selected_sections:
            print("⚠️  Could not extract section numbers from AI response, using fallback...")
            # Fallback to first 3 sections
            available_sections = [s['section_number'] for s in summary_data.get('section_summaries', [])]
            selected_sections = available_sections[:3]
        
        print(f"✅ AI selected sections: {selected_sections}")
        return selected_sections
    
    def extract_section_numbers(self, ai_response: str) -> List[int]:
        """Extract section numbers from AI response"""
        try:
            # Look for patterns like [1, 2, 3] or 1, 2, 3
            import re
            
            # Try to find numbers in brackets
            bracket_match = re.search(r'\[([0-9, ]+)\]', ai_response)
            if bracket_match:
                numbers_str = bracket_match.group(1)
                numbers = [int(x.strip()) for x in numbers_str.split(',')]
                return numbers[:3]  # Ensure only 3 numbers
            
            # Try to find any sequence of numbers
            numbers = re.findall(r'\b\d+\b', ai_response)
            if numbers:
                return [int(x) for x in numbers[:3]]  # Take first 3 numbers
            
            return []
        except Exception as e:
            print(f"Error extracting section numbers: {e}")
            return []
    
    def extract_section_from_document(self, document_content: str, section_number: int, total_sections: int = 5) -> str:
        """Extract specific section from original document"""
        total_length = len(document_content)
        section_size = total_length // total_sections
        
        start_pos = (section_number - 1) * section_size
        if section_number == total_sections:
            end_pos = total_length
        else:
            end_pos = section_number * section_size
        
        return document_content[start_pos:end_pos]
    
    def ai_deep_research(self, question: str, section_content: str, section_number: int) -> str:
        """Phase 2: AI conducts deep research on selected section"""
        system_prompt = """You are an expert researcher conducting deep analysis on a document section. Your task is to thoroughly examine the provided content and extract all relevant information that answers the research question.

INSTRUCTIONS:
1. Analyze the section content carefully and systematically
2. Look for specific data, numbers, patterns, and relationships
3. Pay special attention to tabular data and structured information
4. Extract direct quotes when relevant
5. Identify any entities, IDs, or specific references
6. Provide detailed evidence-based findings
7. Note any patterns or anomalies you discover

Be thorough and precise in your analysis."""
        
        user_prompt = f"""RESEARCH QUESTION: {question}

SECTION {section_number} CONTENT:
{section_content}

Conduct a thorough analysis of this section to answer the research question. Provide detailed findings with specific evidence."""
        
        return self.query_ai(system_prompt, user_prompt, max_tokens=1500)
    
    def conduct_simplified_research(self, question: str, summary_file: str = "doc_summary.json", 
                                  original_file: str = "test.txt", delay: float = 2.0) -> Dict[str, Any]:
        """Main simplified research workflow"""
        start_time = time.time()
        
        # Load documents
        summary_data = self.load_document_summary(summary_file)
        original_content = self.load_original_document(original_file)
        
        # Phase 1: AI selects sections based on summary
        selected_sections = self.ai_select_sections(question, summary_data)
        
        if not selected_sections:
            raise Exception("Failed to select sections for analysis")
        
        print(f"\n🔍 Phase 2: Deep research on selected sections {selected_sections}...")
        
        # Phase 2: Deep research on selected sections
        research_findings = []
        
        for i, section_num in enumerate(selected_sections):
            print(f"   📄 Analyzing Section {section_num}... ({i+1}/{len(selected_sections)})")
            
            # Extract section content
            section_content = self.extract_section_from_document(original_content, section_num)
            
            # Conduct deep research
            analysis_result = self.ai_deep_research(question, section_content, section_num)
            
            finding = {
                'section_number': section_num,
                'analysis': analysis_result,
                'selection_rank': i + 1
            }
            
            research_findings.append(finding)
            
            # Delay between sections
            if i < len(selected_sections) - 1:
                print(f"⏱️  Waiting {delay}s before next section...")
                time.sleep(delay)
        
        end_time = time.time()
        
        return {
            'question': question,
            'summary_file': summary_file,
            'original_file': original_file,
            'selected_sections': selected_sections,
            'research_findings': research_findings,
            'total_sections_analyzed': len(selected_sections),
            'processing_time': end_time - start_time,
            'timestamp': datetime.now().isoformat(),
            'method': 'ai_driven_section_selection'
        }
    
    def print_research_results(self, results: Dict[str, Any]):
        """Print research results in a clean format"""
        print("\n" + "="*80)
        print("🔬 AI-DRIVEN DEEP RESEARCH RESULTS")
        print("="*80)
        print(f"❓ RESEARCH QUESTION: {results['question']}")
        print(f"📄 SUMMARY FILE: {results['summary_file']}")
        print(f"📁 ORIGINAL FILE: {results['original_file']}")
        print(f"⏱️  PROCESSING TIME: {results['processing_time']:.1f} seconds")
        print(f"🤖 METHOD: {results['method']}")
        
        print(f"\n🎯 AI SECTION SELECTION:")
        print(f"   Selected sections: {results['selected_sections']}")
        print(f"   Total sections analyzed: {results['total_sections_analyzed']}")
        
        print(f"\n🔍 DEEP RESEARCH FINDINGS:")
        print("="*80)
        
        for finding in results['research_findings']:
            section_num = finding['section_number']
            rank = finding['selection_rank']
            analysis = finding['analysis']
            
            print(f"\n📋 SECTION {section_num} (AI Priority Rank: {rank})")
            print("-" * 60)
            print(analysis)
            print("-" * 60)

def main():
    """Main function for simplified deep research system"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🚀 SIMPLIFIED AI-DRIVEN DEEP RESEARCH ANALYZER")
    print("="*80)
    print("🤖 Phase 1: AI analyzes JSON summary and selects top 3 sections")
    print("🔍 Phase 2: Deep research on AI-selected sections")
    print("⚡ Streamlined workflow with AI-driven intelligence")
    print("="*80)
    
    # Initialize analyzer
    try:
        analyzer = SimplifiedDeepResearchAnalyzer()
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return
    
    while True:
        try:
            print(f"\n" + "="*80)
            
            # Get question
            question = input("🤔 Your research question (or 'quit'): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using AI-Driven Deep Research!")
                break
            
            if not question:
                print("❌ Please enter a research question")
                continue
            
            # File settings
            summary_file = input("📄 Summary file (default 'doc_summary.json'): ").strip()
            if not summary_file:
                summary_file = "doc_summary.json"
            
            original_file = input("📁 Original document (default 'test.txt'): ").strip()
            if not original_file:
                original_file = "test.txt"
            
            # Make paths absolute
            if not os.path.isabs(summary_file):
                summary_file = os.path.join(script_dir, summary_file)
            if not os.path.isabs(original_file):
                original_file = os.path.join(script_dir, original_file)
            
            # Check files exist
            if not os.path.exists(summary_file):
                print(f"❌ Summary file not found: {summary_file}")
                print("💡 Run summary1.py first to create document summary")
                continue
            
            if not os.path.exists(original_file):
                print(f"❌ Original file not found: {original_file}")
                continue
            
            # Optional settings
            delay_input = input("⏱️  Delay between sections (default 2.0s): ").strip()
            try:
                delay = float(delay_input) if delay_input else 2.0
            except ValueError:
                delay = 2.0
            
            print(f"\n🔬 RESEARCH PREVIEW:")
            print(f"   • Question: {question}")
            print(f"   • Summary file: {summary_file}")
            print(f"   • Original document: {original_file}")
            print(f"   • Method: AI-driven section selection + Deep research")
            
            # Conduct research
            print(f"\n🚀 Starting AI-driven research...")
            results = analyzer.conduct_simplified_research(question, summary_file, original_file, delay)
            
            # Display results
            analyzer.print_research_results(results)
            
            # Auto-save to deep_research.json
            default_filename = "deep_research.json"
            
            # Ask for custom filename or use default
            save_filename = input(f"\n💾 Save as (default '{default_filename}'): ").strip()
            if not save_filename:
                save_filename = default_filename
            
            # Make path absolute if needed
            if not os.path.isabs(save_filename):
                save_filename = os.path.join(script_dir, save_filename)
            
            try:
                with open(save_filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Research results saved to: {save_filename}")
            except Exception as e:
                print(f"❌ Error saving results: {e}")
                # Fallback save with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fallback_filename = f"deep_research_backup_{timestamp}.json"
                try:
                    with open(fallback_filename, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"💾 Saved to backup file: {fallback_filename}")
                except Exception as e2:
                    print(f"❌ Failed to save backup: {e2}")
            
        except KeyboardInterrupt:
            print("\n👋 Research interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during research: {e}")

if __name__ == "__main__":
    main()