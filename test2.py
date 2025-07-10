#!/usr/bin/env python3
"""
Simple File Split Analysis - test2.py
Basic approach: Split file into 5 sections, make individual AI calls with simple instructions
"""

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

class SimpleFileSplitter:
    """Simple file splitter and AI processor"""
    
    def __init__(self):
        self.auth = AzureOpenAIAuth()
        self.endpoint = os.getenv('KGW_ENDPOINT')
        self.api_version = os.getenv('AOAI_API_VERSION')
        self.chat_deployment = os.getenv('CHAT_MODEL_DEPLOYMENT_NAME')
        
        if not all([self.endpoint, self.api_version, self.chat_deployment]):
            raise ValueError("Missing required Azure OpenAI configuration")
        
        print("✅ Simple File Splitter initialized")
        print(f"   - Endpoint: {self.endpoint}")
        print(f"   - Model: {self.chat_deployment}")
        print(f"   - Method: Simple 5-way split with basic instructions")
    
    def load_text_file(self, file_path: str) -> str:
        """Load text file content"""
        print(f"📂 Loading file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"✅ Loaded file: {len(content):,} characters")
            return content
        except Exception as e:
            raise Exception(f"Error loading file: {e}")
    
    def split_into_5_sections(self, content: str) -> List[Dict[str, Any]]:
        """Split content into exactly 5 equal sections"""
        total_length = len(content)
        section_size = total_length // 5
        
        sections = []
        
        for i in range(5):
            start_pos = i * section_size
            
            if i == 4:  # Last section gets remainder
                end_pos = total_length
            else:
                end_pos = (i + 1) * section_size
            
            section_content = content[start_pos:end_pos]
            
            sections.append({
                'section_number': i + 1,
                'start_position': start_pos,
                'end_position': end_pos,
                'content': section_content,
                'character_count': len(section_content)
            })
        
        print(f"📄 Split file into 5 sections:")
        for section in sections:
            print(f"   Section {section['section_number']}: {section['character_count']:,} characters "
                  f"(pos {section['start_position']:,}-{section['end_position']:,})")
        
        return sections
    
    def query_ai_simple(self, user_question: str, section_content: str, section_number: int) -> str:
        """Simple AI query with basic instructions"""
        access_token = self.auth.get_access_token()
        
        url = f"{self.endpoint}/openai/deployments/{self.chat_deployment}/chat/completions?api-version={self.api_version}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        # Enhanced simple instructions focusing on tabular data
        system_prompt = "When answering the user question look in the given data very very closely. Pay special attention to any tabular data, tables, rows, columns, and structured data formats as they often contain the specific numbers and counts being requested."
        
        user_prompt = f"""User Question: {user_question}

Given Data (Section {section_number} of 5):
{section_content}"""
        
        payload = {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'max_tokens': 1000,
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
                    return f"Error querying AI: {str(e)}"
                wait_time = (2 ** attempt) + 2
                print(f"   Request error, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return f"Failed to get response after {max_retries} attempts"
    
    def process_question(self, file_path: str, user_question: str, delay_between_calls: float = 2.0) -> Dict[str, Any]:
        """Process user question against all 5 sections"""
        start_time = time.time()
        
        # Load file
        content = self.load_text_file(file_path)
        
        # Split into 5 sections
        sections = self.split_into_5_sections(content)
        
        # Process each section
        section_responses = []
        
        for section in sections:
            section_num = section['section_number']
            print(f"\n🧠 Processing Section {section_num}/5 ({section['character_count']:,} chars)...")
            
            # Make AI call for this section
            response = self.query_ai_simple(
                user_question, 
                section['content'], 
                section_num
            )
            
            section_responses.append({
                'section_number': section_num,
                'character_count': section['character_count'],
                'start_position': section['start_position'],
                'end_position': section['end_position'],
                'ai_response': response
            })
            
            # Delay between calls
            if section_num < 5:
                print(f"⏱️  Waiting {delay_between_calls}s before next section...")
                time.sleep(delay_between_calls)
        
        end_time = time.time()
        
        return {
            'question': user_question,
            'file_path': file_path,
            'total_characters': len(content),
            'sections': section_responses,
            'processing_time': end_time - start_time,
            'timestamp': datetime.now().isoformat(),
            'method': 'simple_5_section_split'
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print results in a clear format"""
        print("\n" + "="*80)
        print("📋 SIMPLE 5-SECTION ANALYSIS RESULTS")
        print("="*80)
        print(f"❓ QUESTION: {results['question']}")
        print(f"📁 FILE: {results['file_path']}")
        print(f"📄 TOTAL SIZE: {results['total_characters']:,} characters")
        print(f"⏱️  PROCESSING TIME: {results['processing_time']:.1f} seconds")
        print(f"🔬 METHOD: {results['method']}")
        
        print(f"\n🧠 AI RESPONSES BY SECTION:")
        print("="*80)
        
        for section in results['sections']:
            section_num = section['section_number']
            char_count = section['character_count']
            start_pos = section['start_position']
            end_pos = section['end_position']
            
            print(f"\n📄 SECTION {section_num}/5")
            print(f"📊 Size: {char_count:,} chars (positions {start_pos:,}-{end_pos:,})")
            print("-" * 60)
            print(section['ai_response'])
            print("-" * 60)

def main():
    """Main function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🚀 SIMPLE FILE SPLIT ANALYSIS")
    print("="*60)
    print("📄 Split file into 5 equal sections")
    print("🧠 Simple AI instructions: 'look in the data very very closely'")
    print("🔍 Individual AI calls per section")
    print("⚡ No RAG, no embeddings, no complex processing")
    print("="*60)
    
    # Initialize processor
    try:
        processor = SimpleFileSplitter()
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
        return
    
    while True:
        try:
            print(f"\n" + "="*60)
            
            # Get file path
            file_input = input("📁 Enter text file path (or 'quit'): ").strip()
            
            if file_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Default to test.txt if no path given
            if not file_input:
                file_input = "test.txt"
            
            # Make path absolute if not already
            if not os.path.isabs(file_input):
                file_path = os.path.join(script_dir, file_input)
            else:
                file_path = file_input
            
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
            
            # Get question
            question = input("🤔 Your question about the file: ").strip()
            
            if not question:
                print("❌ Please enter a question")
                continue
            
            # Optional delay setting
            delay_input = input("⏱️  Delay between AI calls in seconds (default 2.0): ").strip()
            try:
                delay = float(delay_input) if delay_input else 2.0
            except ValueError:
                delay = 2.0
            
            print(f"\n🔍 ANALYSIS PREVIEW:")
            print(f"   • File: {file_path}")
            print(f"   • Question: {question}")
            print(f"   • Method: Split into 5 equal sections")
            print(f"   • AI Instructions: Very simple - 'look closely'")
            print(f"   • Delay between calls: {delay}s")
            
            # Process the question
            print(f"\n🚀 Starting simple analysis...")
            results = processor.process_question(file_path, question, delay)
            
            # Display results
            processor.print_results(results)
            
            # Option to save results
            save_option = input(f"\n💾 Save results to JSON? (y/n): ").strip().lower()
            if save_option == 'y':
                import json
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"simple_analysis_{timestamp}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Results saved to: {filename}")
            
            # Show section summaries
            summaries = input(f"\n📊 Show section content summaries? (y/n): ").strip().lower()
            if summaries == 'y':
                print(f"\n📄 SECTION CONTENT PREVIEWS:")
                print("="*60)
                
                # Re-load the file to show previews
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                section_size = len(content) // 5
                for i in range(5):
                    start_pos = i * section_size
                    end_pos = (i + 1) * section_size if i < 4 else len(content)
                    section_content = content[start_pos:end_pos]
                    
                    print(f"\nSection {i+1} preview (first 200 chars):")
                    print(f"'{section_content[:200]}...'")
            
        except KeyboardInterrupt:
            print("\n👋 Analysis interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()