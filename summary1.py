#!/usr/bin/env python3
"""
Document Summarization Analysis - test_summarize.py
Basic approach: Split file into 5 sections, create meaningful summaries, save to JSON
"""

import os
import time
import requests
import json
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

class DocumentSummarizer:
    """Document summarizer with 5-section approach"""
    
    def __init__(self):
        self.auth = AzureOpenAIAuth()
        self.endpoint = os.getenv('KGW_ENDPOINT')
        self.api_version = os.getenv('AOAI_API_VERSION')
        self.chat_deployment = os.getenv('CHAT_MODEL_DEPLOYMENT_NAME')
        
        if not all([self.endpoint, self.api_version, self.chat_deployment]):
            raise ValueError("Missing required Azure OpenAI configuration")
        
        print("✅ Document Summarizer initialized")
        print(f"   - Endpoint: {self.endpoint}")
        print(f"   - Model: {self.chat_deployment}")
        print(f"   - Method: 5-section summarization with JSON output")
    
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
    
    def create_section_summary(self, section_content: str, section_number: int, total_sections: int) -> str:
        """Create a meaningful summary for a document section"""
        access_token = self.auth.get_access_token()
        
        url = f"{self.endpoint}/openai/deployments/{self.chat_deployment}/chat/completions?api-version={self.api_version}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        # Comprehensive summarization instructions
        system_prompt = """You are an expert document analyst. Create a comprehensive, meaningful summary of the given document section. 

Your summary should:
- Capture all key information, facts, and data points
- Identify important tables, lists, and structured data
- Note significant numbers, dates, names, and identifiers
- Highlight main themes and topics covered
- Preserve critical details that someone would need to know
- Be well-organized and easy to understand
- Include specific examples and concrete details when relevant

Focus on creating a summary that would allow someone to understand the essential content without reading the full section."""

        user_prompt = f"""Please create a comprehensive summary of this document section.

This is Section {section_number} of {total_sections} from the document.

Document Section Content:
{section_content}

Provide a detailed summary that captures all the important information from this section."""
        
        payload = {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'max_tokens': 1500,
            'temperature': 0.1
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
                    return f"Error creating summary: {str(e)}"
                wait_time = (2 ** attempt) + 2
                print(f"   Request error, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return f"Failed to create summary after {max_retries} attempts"
    
    def summarize_document(self, file_path: str, delay_between_calls: float = 2.0) -> Dict[str, Any]:
        """Create summaries for all 5 sections and compile results"""
        start_time = time.time()
        
        # Load file
        content = self.load_text_file(file_path)
        
        # Split into 5 sections
        sections = self.split_into_5_sections(content)
        
        # Create summaries for each section
        section_summaries = []
        
        print(f"\n🔍 Creating comprehensive summaries...")
        
        for section in sections:
            section_num = section['section_number']
            print(f"\n📝 Summarizing Section {section_num}/5 ({section['character_count']:,} chars)...")
            
            # Create summary for this section
            summary = self.create_section_summary(
                section['content'], 
                section_num, 
                len(sections)
            )
            
            section_summary = {
                'section_number': section_num,
                'start_position': section['start_position'],
                'end_position': section['end_position'],
                'character_count': section['character_count'],
                'content_preview': section['content'][:200] + "..." if len(section['content']) > 200 else section['content'],
                'summary': summary,
                'summary_created': datetime.now().isoformat()
            }
            
            section_summaries.append(section_summary)
            
            # Show progress
            summary_length = len(summary)
            print(f"   ✅ Summary created: {summary_length:,} characters")
            
            # Delay between calls
            if section_num < 5:
                print(f"⏱️  Waiting {delay_between_calls}s before next section...")
                time.sleep(delay_between_calls)
        
        end_time = time.time()
        
        # Compile final result
        result = {
            'document_info': {
                'file_path': file_path,
                'total_characters': len(content),
                'sections_count': len(sections),
                'processing_time': end_time - start_time,
                'created_timestamp': datetime.now().isoformat(),
                'method': 'five_section_summarization'
            },
            'section_summaries': section_summaries,
            'summary_stats': {
                'total_sections': len(section_summaries),
                'total_summary_length': sum(len(s['summary']) for s in section_summaries),
                'average_summary_length': sum(len(s['summary']) for s in section_summaries) // len(section_summaries),
                'compression_ratio': round(len(content) / sum(len(s['summary']) for s in section_summaries), 2)
            }
        }
        
        return result
    
    def save_summary_to_json(self, summary_data: Dict[str, Any], output_filename: str = "doc_summary.json"):
        """Save the summary data to JSON file"""
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Summary saved to: {output_filename}")
            
            # Show file info
            file_size = os.path.getsize(output_filename)
            print(f"   📁 File size: {file_size:,} bytes")
            
            return True
        except Exception as e:
            print(f"❌ Error saving summary: {e}")
            return False
    
    def print_summary_overview(self, summary_data: Dict[str, Any]):
        """Print an overview of the created summaries"""
        doc_info = summary_data['document_info']
        stats = summary_data['summary_stats']
        
        print("\n" + "="*80)
        print("📋 DOCUMENT SUMMARIZATION RESULTS")
        print("="*80)
        print(f"📁 FILE: {doc_info['file_path']}")
        print(f"📄 ORIGINAL SIZE: {doc_info['total_characters']:,} characters")
        print(f"⏱️  PROCESSING TIME: {doc_info['processing_time']:.1f} seconds")
        print(f"🔬 METHOD: {doc_info['method']}")
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   • Sections processed: {stats['total_sections']}")
        print(f"   • Total summary length: {stats['total_summary_length']:,} characters")
        print(f"   • Average summary length: {stats['average_summary_length']:,} characters")
        print(f"   • Compression ratio: {stats['compression_ratio']:1}:1")
        
        print(f"\n📝 SECTION SUMMARIES OVERVIEW:")
        print("="*80)
        
        for i, section in enumerate(summary_data['section_summaries'], 1):
            summary_length = len(section['summary'])
            compression = round(section['character_count'] / summary_length, 1)
            
            print(f"\n📄 SECTION {i}/5")
            print(f"📊 Original: {section['character_count']:,} chars → Summary: {summary_length:,} chars ({compression}:1)")
            print(f"📋 Preview: {section['content_preview']}")
            print("-" * 60)
            # Show first 200 characters of summary
            summary_preview = section['summary'][:200] + "..." if len(section['summary']) > 200 else section['summary']
            print(f"📝 Summary: {summary_preview}")
            print("-" * 60)

def main():
    """Main function for document summarization"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🚀 DOCUMENT SUMMARIZATION SYSTEM")
    print("="*70)
    print("📄 Split document into 5 sections")
    print("📝 Create comprehensive summaries for each section")
    print("💾 Save all summaries to doc_summary.json")
    print("⚡ Fast and efficient document comprehension")
    print("="*70)
    
    # Initialize summarizer
    try:
        summarizer = DocumentSummarizer()
    except Exception as e:
        print(f"❌ Error initializing summarizer: {e}")
        return
    
    while True:
        try:
            print(f"\n" + "="*70)
            
            # Get file path
            file_input = input("📁 Enter text file path (or 'quit'): ").strip()
            
            if file_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using Document Summarization!")
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
            
            # Optional settings
            delay_input = input("⏱️  Delay between AI calls in seconds (default 2.0): ").strip()
            try:
                delay = float(delay_input) if delay_input else 2.0
            except ValueError:
                delay = 2.0
            
            # Custom output filename option
            output_name = input("💾 Output filename (default 'doc_summary.json'): ").strip()
            if not output_name:
                output_name = "doc_summary.json"
            elif not output_name.endswith('.json'):
                output_name += '.json'
            
            print(f"\n🔍 SUMMARIZATION PREVIEW:")
            print(f"   • File: {file_path}")
            print(f"   • Method: 5-section comprehensive summarization")
            print(f"   • Output: {output_name}")
            print(f"   • Delay between calls: {delay}s")
            
            # Create summaries
            print(f"\n🚀 Starting document summarization...")
            summary_data = summarizer.summarize_document(file_path, delay)
            
            # Display overview
            summarizer.print_summary_overview(summary_data)
            
            # Save to JSON
            print(f"\n💾 Saving summary to {output_name}...")
            success = summarizer.save_summary_to_json(summary_data, output_name)
            
            if success:
                print(f"\n✅ Document summarization completed successfully!")
                print(f"📄 Summary saved to: {output_name}")
                
                # Option to view specific section
                view_section = input(f"\n👀 View detailed summary for specific section (1-5, or 'n'): ").strip()
                if view_section.isdigit() and 1 <= int(view_section) <= 5:
                    section_idx = int(view_section) - 1
                    section = summary_data['section_summaries'][section_idx]
                    
                    print(f"\n📄 DETAILED SECTION {view_section} SUMMARY:")
                    print("="*60)
                    print(section['summary'])
                    print("="*60)
                
                # Option to load and query existing summary
                query_option = input(f"\n🔍 Load existing summary for analysis? (y/n): ").strip().lower()
                if query_option == 'y':
                    try:
                        with open(output_name, 'r', encoding='utf-8') as f:
                            loaded_data = json.load(f)
                        print(f"✅ Loaded summary from {output_name}")
                        print(f"📊 Contains {len(loaded_data['section_summaries'])} section summaries")
                    except Exception as e:
                        print(f"❌ Error loading summary: {e}")
            
        except KeyboardInterrupt:
            print("\n👋 Summarization interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during summarization: {e}")

if __name__ == "__main__":
    main()