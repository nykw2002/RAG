#!/usr/bin/env python3
"""
Research Synthesis Engine - summary3.py
Takes deep_research.json and synthesizes findings into comprehensive final answers
Combines all section analyses into coherent, well-structured responses
"""

import os
import time
import requests
import json
from datetime import datetime
from typing import Dict, Any, List

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

class ResearchSynthesizer:
    """Synthesizes deep research findings into comprehensive final answers"""
    
    def __init__(self):
        self.auth = AzureOpenAIAuth()
        self.endpoint = os.getenv('KGW_ENDPOINT')
        self.api_version = os.getenv('AOAI_API_VERSION')
        self.chat_deployment = os.getenv('CHAT_MODEL_DEPLOYMENT_NAME')
        
        if not all([self.endpoint, self.api_version, self.chat_deployment]):
            raise ValueError("Missing required Azure OpenAI configuration")
        
        print("✅ Research Synthesizer initialized")
        print(f"   - Endpoint: {self.endpoint}")
        print(f"   - Model: {self.chat_deployment}")
        print(f"   - Purpose: Synthesize deep research findings into final answers")
    
    def load_deep_research(self, research_file: str = "deep_research.json") -> Dict[str, Any]:
        """Load deep research results from JSON file"""
        print(f"📂 Loading deep research results: {research_file}")
        
        try:
            with open(research_file, 'r', encoding='utf-8') as f:
                research_data = json.load(f)
            
            findings_count = len(research_data.get('research_findings', []))
            question = research_data.get('question', 'Unknown')
            
            print(f"✅ Loaded research data:")
            print(f"   - Question: {question}")
            print(f"   - Findings from {findings_count} sections")
            print(f"   - Selected sections: {research_data.get('selected_sections', [])}")
            
            return research_data
        except Exception as e:
            raise Exception(f"Error loading deep research: {e}")
    
    def query_ai_synthesizer(self, question: str, all_findings: List[Dict], research_metadata: Dict) -> str:
        """AI synthesis of all research findings"""
        access_token = self.auth.get_access_token()
        
        url = f"{self.endpoint}/openai/deployments/{self.chat_deployment}/chat/completions?api-version={self.api_version}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        system_prompt = """You are an expert research synthesizer with strong analytical and mathematical skills. Your task is to analyze multiple research findings from different document sections and create a comprehensive, well-structured final answer.

CRITICAL SYNTHESIS RULES:
1. MATHEMATICAL ACCURACY: When counting items across sections, ADD ALL NUMBERS together. Do not pick one section as "most reliable" unless there are clear contradictions.
2. COMPREHENSIVE AGGREGATION: Combine ALL data from ALL sections - each section may contain different items that need to be totaled.
3. CAREFUL COUNTING: If Section A has 2 items, Section B has 7 items, and Section C has 4 items, the total is 2+7+4=13, NOT 7.
4. EVIDENCE-BASED: Include ALL specific data, numbers, IDs, and references from ALL findings.
5. NO SELECTIVE REPORTING: Do not ignore sections or choose one as "primary" without explicit justification.
6. DUPLICATE DETECTION: Only exclude items if they are clearly the same (same ID appearing in multiple sections).
7. TRANSPARENT CALCULATIONS: Show your math when totaling numbers across sections.

SYNTHESIS GUIDELINES:
1. COMPREHENSIVE: Combine all relevant information from all sections
2. COHERENT: Create a logical flow that tells the complete story
3. STRUCTURED: Organize information clearly with headings and bullet points where appropriate
4. CROSS-REFERENCED: Identify patterns, consistencies, or contradictions across sections
5. ACTIONABLE: Provide clear, definitive answers when possible

FORMAT YOUR RESPONSE AS:
## Summary
[Brief executive summary with CORRECT totals answering the core question]

## Section-by-Section Analysis
[Break down findings from each section clearly]

## Total Calculations
[Show the math: Section X: # items + Section Y: # items = Total]

## Complete Data Points
[ALL specific numbers, dates, IDs from ALL sections]

## Conclusions
[Final conclusions based on complete data]

REMEMBER: Your job is to COMBINE and ADD UP information from multiple sections, not to pick favorites!"""
        
        # Prepare findings text
        findings_text = "RESEARCH FINDINGS FROM MULTIPLE SECTIONS:\n\n"
        
        for i, finding in enumerate(all_findings, 1):
            section_num = finding.get('section_number', 'Unknown')
            rank = finding.get('selection_rank', i)
            analysis = finding.get('analysis', 'No analysis available')
            
            findings_text += f"FINDING {i} (Section {section_num}, AI Priority Rank {rank}):\n"
            findings_text += f"{analysis}\n"
            findings_text += f"{'-' * 80}\n\n"
        
        # Add metadata context
        metadata_text = f"""
RESEARCH CONTEXT:
- Original Question: {question}
- Sections Analyzed: {research_metadata.get('selected_sections', [])}
- Analysis Method: {research_metadata.get('method', 'Unknown')}
- Processing Time: {research_metadata.get('processing_time', 0):.1f} seconds
- Timestamp: {research_metadata.get('timestamp', 'Unknown')}
"""
        
        user_prompt = f"""{metadata_text}

{findings_text}

SYNTHESIS TASK:
Synthesize all the above research findings to provide a comprehensive, well-structured answer to the original research question: "{question}"

Create a complete response that combines insights from all sections while maintaining clarity and logical organization."""
        
        payload = {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'max_tokens': 2000,
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
                    return f"Error in synthesis: {str(e)}"
                wait_time = (2 ** attempt) + 2
                print(f"   Request error, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return f"Failed to get synthesis after {max_retries} attempts"
    
    def create_synthesis_report(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive synthesis report"""
        start_time = time.time()
        
        # Extract key information
        question = research_data.get('question', 'Unknown question')
        findings = research_data.get('research_findings', [])
        metadata = {
            'selected_sections': research_data.get('selected_sections', []),
            'method': research_data.get('method', 'Unknown'),
            'processing_time': research_data.get('processing_time', 0),
            'timestamp': research_data.get('timestamp', 'Unknown'),
            'original_summary_file': research_data.get('summary_file', 'Unknown'),
            'original_document_file': research_data.get('original_file', 'Unknown')
        }
        
        if not findings:
            raise Exception("No research findings found in the data")
        
        print(f"🧠 Synthesizing {len(findings)} research findings...")
        
        # AI synthesis
        synthesis_result = self.query_ai_synthesizer(question, findings, metadata)
        
        # Create comprehensive report
        synthesis_time = time.time() - start_time
        
        report = {
            'original_question': question,
            'synthesis_result': synthesis_result,
            'source_research_data': {
                'total_findings': len(findings),
                'sections_analyzed': metadata['selected_sections'],
                'original_processing_time': metadata['processing_time'],
                'original_timestamp': metadata['timestamp'],
                'original_method': metadata['method'],
                'source_files': {
                    'summary_file': metadata['original_summary_file'],
                    'document_file': metadata['original_document_file']
                }
            },
            'synthesis_metadata': {
                'synthesis_time': synthesis_time,
                'synthesis_timestamp': datetime.now().isoformat(),
                'synthesizer_version': 'summary3.py v1.0',
                'ai_model': self.chat_deployment
            }
        }
        
        return report
    
    def print_synthesis_results(self, report: Dict[str, Any]):
        """Print synthesis results in a clean, readable format"""
        print("\n" + "="*100)
        print("🧠 RESEARCH SYNTHESIS REPORT")
        print("="*100)
        print(f"❓ ORIGINAL QUESTION: {report['original_question']}")
        
        # Source data info
        source_data = report['source_research_data']
        print(f"\n📊 SOURCE RESEARCH DATA:")
        print(f"   • Sections analyzed: {source_data['sections_analyzed']}")
        print(f"   • Total findings: {source_data['total_findings']}")
        print(f"   • Original processing time: {source_data['original_processing_time']:.1f}s")
        print(f"   • Research method: {source_data['original_method']}")
        
        # Synthesis metadata
        synthesis_meta = report['synthesis_metadata']
        print(f"\n🔬 SYNTHESIS METADATA:")
        print(f"   • Synthesis time: {synthesis_meta['synthesis_time']:.1f}s")
        print(f"   • AI model: {synthesis_meta['ai_model']}")
        print(f"   • Synthesizer: {synthesis_meta['synthesizer_version']}")
        
        # Main synthesis result
        print(f"\n🎯 COMPREHENSIVE SYNTHESIZED ANSWER:")
        print("="*100)
        print(report['synthesis_result'])
        print("="*100)
    
    def save_synthesis_report(self, report: Dict[str, Any], output_file: str = "final_answer.json") -> str:
        """Save synthesis report to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Synthesis report saved to: {output_file}")
            return output_file
        except Exception as e:
            # Fallback save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_file = f"final_answer_backup_{timestamp}.json"
            
            try:
                with open(fallback_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                print(f"⚠️  Could not save to {output_file}: {e}")
                print(f"💾 Saved to backup file: {fallback_file}")
                return fallback_file
            except Exception as e2:
                print(f"❌ Failed to save synthesis report: {e2}")
                return None

def main():
    """Main function for research synthesis"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🧠 RESEARCH SYNTHESIS ENGINE")
    print("="*80)
    print("📥 Input: deep_research.json (from summary2.py)")
    print("🔄 Process: AI synthesis of all research findings")
    print("📤 Output: final_answer.json (comprehensive final answer)")
    print("🎯 Purpose: Transform research findings into coherent final answers")
    print("="*80)
    
    # Initialize synthesizer
    try:
        synthesizer = ResearchSynthesizer()
    except Exception as e:
        print(f"❌ Error initializing synthesizer: {e}")
        return
    
    while True:
        try:
            print(f"\n" + "="*80)
            
            # Get input file
            research_file = input("📂 Deep research file (default 'deep_research.json'): ").strip()
            if not research_file:
                research_file = "deep_research.json"
            
            # Make path absolute if needed
            if not os.path.isabs(research_file):
                research_file = os.path.join(script_dir, research_file)
            
            # Check file exists
            if not os.path.exists(research_file):
                print(f"❌ Research file not found: {research_file}")
                print("💡 Run summary2.py first to create deep research results")
                continue
            
            # Get output file
            output_file = input("📤 Output file (default 'final_answer.json'): ").strip()
            if not output_file:
                output_file = "final_answer.json"
            
            # Make path absolute if needed
            if not os.path.isabs(output_file):
                output_file = os.path.join(script_dir, output_file)
            
            print(f"\n🧠 SYNTHESIS PREVIEW:")
            print(f"   • Input: {research_file}")
            print(f"   • Output: {output_file}")
            print(f"   • Process: AI synthesis of research findings")
            
            # Load and synthesize
            print(f"\n🚀 Starting synthesis process...")
            
            research_data = synthesizer.load_deep_research(research_file)
            report = synthesizer.create_synthesis_report(research_data)
            
            # Display results
            synthesizer.print_synthesis_results(report)
            
            # Save results
            print(f"\n💾 Saving synthesis report...")
            saved_file = synthesizer.save_synthesis_report(report, output_file)
            
            if saved_file:
                print(f"\n🎉 SUCCESS! Final synthesized answer available in: {saved_file}")
            
            # Continue option
            continue_option = input(f"\n🔄 Synthesize another research file? (y/n): ").strip().lower()
            if continue_option != 'y':
                break
            
        except KeyboardInterrupt:
            print("\n👋 Synthesis interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error during synthesis: {e}")

if __name__ == "__main__":
    main()