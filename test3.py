#!/usr/bin/env python3
"""
Advanced Memory-Enhanced File Split Analysis - test2.py
Combines: Progressive Questioning + Document Memory + Iterative Deep-Dive Analysis
"""

import os
import time
import requests
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Set
from collections import defaultdict

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

class DocumentMemorySystem:
    """Option 2: Document Memory with Cross-References"""
    
    def __init__(self):
        self.findings_database = {}
        self.cross_references = defaultdict(list)
        self.entity_tracker = defaultdict(set)
        self.section_findings = defaultdict(list)
        self.analysis_patterns = []
        self.session_insights = []
    
    def store_finding(self, question: str, section_num: int, response: str, entities: Set[str]):
        """Store analysis findings with metadata"""
        finding_id = f"q{len(self.findings_database)}_{section_num}"
        
        finding = {
            'id': finding_id,
            'question': question,
            'section': section_num,
            'response': response,
            'entities': list(entities),
            'timestamp': datetime.now().isoformat(),
            'related_findings': []
        }
        
        self.findings_database[finding_id] = finding
        self.section_findings[section_num].append(finding_id)
        
        # Update entity tracker
        for entity in entities:
            self.entity_tracker[entity].add(section_num)
    
    def find_related_findings(self, question: str, entities: Set[str]) -> List[Dict]:
        """Find related findings based on entities and keywords"""
        related = []
        question_words = set(question.lower().split())
        
        for finding_id, finding in self.findings_database.items():
            # Check entity overlap
            finding_entities = set(finding['entities'])
            entity_overlap = entities.intersection(finding_entities)
            
            # Check keyword overlap
            finding_words = set(finding['question'].lower().split())
            keyword_overlap = question_words.intersection(finding_words)
            
            if entity_overlap or len(keyword_overlap) >= 2:
                related.append({
                    'finding': finding,
                    'relevance_score': len(entity_overlap) + len(keyword_overlap),
                    'entity_overlap': list(entity_overlap),
                    'keyword_overlap': list(keyword_overlap)
                })
        
        return sorted(related, key=lambda x: x['relevance_score'], reverse=True)[:3]
    
    def get_cross_section_insights(self) -> str:
        """Generate insights across sections"""
        insights = []
        
        # Find entities mentioned in multiple sections
        multi_section_entities = {entity: sections for entity, sections in self.entity_tracker.items() 
                                if len(sections) > 1}
        
        if multi_section_entities:
            insights.append("CROSS-SECTION ENTITIES:")
            for entity, sections in multi_section_entities.items():
                insights.append(f"  • {entity}: appears in sections {sorted(sections)}")
        
        # Recent patterns
        if self.analysis_patterns:
            insights.append("\nEMERGING PATTERNS:")
            for pattern in self.analysis_patterns[-3:]:
                insights.append(f"  • {pattern}")
        
        return "\n".join(insights) if insights else "No cross-section patterns identified yet."

class ProgressiveQuestioningSystem:
    """Option 1: Progressive Questioning System"""
    
    def __init__(self):
        self.conversation_history = []
        self.question_context = {}
        self.follow_up_suggestions = []
        self.analysis_depth = 0
    
    def is_follow_up_question(self, question: str) -> bool:
        """Determine if this is a follow-up to previous questions"""
        if not self.conversation_history:
            return False
        
        # Check for follow-up indicators
        follow_up_terms = [
            "also", "additionally", "furthermore", "now", "next", "then",
            "what about", "how about", "can you", "show me", "tell me more",
            "elaborate", "expand", "detail", "explain", "clarify"
        ]
        
        question_lower = question.lower()
        return any(term in question_lower for term in follow_up_terms)
    
    def build_progressive_context(self, current_question: str) -> str:
        """Build context from conversation history"""
        if not self.conversation_history:
            return ""
        
        context_parts = ["CONVERSATION CONTEXT:"]
        
        # Add recent conversation history
        for i, (prev_q, summary) in enumerate(self.conversation_history[-3:], 1):
            context_parts.append(f"Previous Q{i}: {prev_q}")
            context_parts.append(f"Key Findings: {summary}")
        
        # Add follow-up guidance if applicable
        if self.is_follow_up_question(current_question):
            context_parts.append(f"\nCURRENT QUESTION builds on previous analysis.")
            context_parts.append("Focus on: connecting findings, expanding details, or exploring new angles.")
        
        return "\n".join(context_parts)
    
    def suggest_follow_ups(self, question: str, findings: List[str]) -> List[str]:
        """Generate intelligent follow-up questions"""
        suggestions = []
        
        # Extract key entities from findings
        entities = set()
        numbers = []
        
        for finding in findings:
            # Extract numbers
            numbers.extend(re.findall(r'\d+', finding))
            # Extract potential entities (capitalized words)
            entities.update(re.findall(r'\b[A-Z][a-z]+\b', finding))
        
        # Generate contextual follow-ups
        if numbers:
            suggestions.append(f"What patterns can you find in these numbers: {', '.join(set(numbers))}?")
        
        if entities:
            suggestions.append(f"How do these entities relate to each other: {', '.join(list(entities)[:3])}?")
        
        # Question-specific follow-ups
        if "complaint" in question.lower():
            suggestions.extend([
                "What are the most common types of complaints?",
                "Are there any patterns in complaint timing or geography?",
                "Which complaints might be related to the same underlying issue?"
            ])
        
        if "count" in question.lower() or "how many" in question.lower():
            suggestions.extend([
                "What trends can you identify in these counts?",
                "Are there any anomalies or outliers in the data?",
                "How do these numbers compare to expected ranges?"
            ])
        
        return suggestions[:4]  # Return top 4 suggestions

class IterativeDeepDiveAnalyzer:
    """Option 3: Iterative Deep-Dive Analysis"""
    
    def __init__(self):
        self.analysis_iterations = []
        self.refinement_cycles = 0
        self.emerging_themes = []
        self.validation_checks = []
    
    def plan_analysis_iterations(self, question: str, previous_findings: List[str]) -> List[Dict]:
        """Plan multiple analysis iterations with increasing depth"""
        iterations = []
        
        # Iteration 1: Broad discovery
        iterations.append({
            'iteration': 1,
            'focus': 'Discovery',
            'instruction': 'Identify all relevant information and potential answers',
            'depth': 'broad'
        })
        
        # Iteration 2: Pattern identification
        if previous_findings:
            iterations.append({
                'iteration': 2,
                'focus': 'Pattern Analysis',
                'instruction': 'Look for patterns, connections, and relationships in the findings',
                'depth': 'analytical'
            })
        
        # Iteration 3: Validation and refinement
        if len(previous_findings) > 1:
            iterations.append({
                'iteration': 3,
                'focus': 'Validation',
                'instruction': 'Cross-check findings, identify inconsistencies, and refine conclusions',
                'depth': 'deep'
            })
        
        return iterations
    
    def generate_iterative_prompt(self, iteration: Dict, question: str, context: str) -> str:
        """Generate prompts for iterative analysis"""
        base_instruction = """When answering the user question look in the given data very very closely. 
Pay special attention to any tabular data, tables, rows, columns, and structured data formats."""
        
        iteration_guidance = {
            'Discovery': "Focus on finding ALL relevant information. Don't worry about analysis yet - just extract everything related to the question.",
            'Pattern Analysis': "Now analyze the patterns. Look for relationships, trends, commonalities, and anything that connects the findings.",
            'Validation': "Verify the accuracy and consistency of findings. Check for contradictions, validate numbers, and ensure completeness."
        }
        
        return f"""{base_instruction}

ANALYSIS ITERATION {iteration['iteration']}: {iteration['focus']}
{iteration_guidance[iteration['focus']]}

{context}

Question: {question}
"""

class MemoryEnhancedFileSplitter:
    """Advanced File Splitter with all three memory/analysis options"""
    
    def __init__(self):
        self.auth = AzureOpenAIAuth()
        self.endpoint = os.getenv('KGW_ENDPOINT')
        self.api_version = os.getenv('AOAI_API_VERSION')
        self.chat_deployment = os.getenv('CHAT_MODEL_DEPLOYMENT_NAME')
        
        # Initialize all three systems
        self.memory_system = DocumentMemorySystem()
        self.questioning_system = ProgressiveQuestioningSystem()
        self.deep_dive_analyzer = IterativeDeepDiveAnalyzer()
        
        if not all([self.endpoint, self.api_version, self.chat_deployment]):
            raise ValueError("Missing required Azure OpenAI configuration")
        
        print("✅ Memory-Enhanced File Splitter initialized")
        print(f"   - Endpoint: {self.endpoint}")
        print(f"   - Model: {self.chat_deployment}")
        print(f"   - Features: Progressive Questioning + Document Memory + Iterative Deep-Dive")
        print(f"   - Method: Enhanced 5-way split with advanced memory systems")
    
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
    
    def extract_entities(self, text: str) -> Set[str]:
        """Extract key entities from text for memory system"""
        entities = set()
        
        # Extract numbers (potential IDs, counts, etc.)
        numbers = re.findall(r'\b\d+\b', text)
        entities.update(numbers)
        
        # Extract capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.update(proper_nouns)
        
        # Extract specific patterns (IDs, codes, etc.)
        ids = re.findall(r'\b[A-Z]{2,}-\d+\b|\b\d{10,}\b', text)
        entities.update(ids)
        
        return entities
    
    def query_ai_with_memory(self, user_question: str, section_content: str, section_number: int, 
                           iteration: int = 1) -> str:
        """Enhanced AI query with all memory systems"""
        access_token = self.auth.get_access_token()
        
        url = f"{self.endpoint}/openai/deployments/{self.chat_deployment}/chat/completions?api-version={self.api_version}"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        
        # Build comprehensive context using all systems
        context_parts = []
        
        # Option 1: Progressive questioning context
        progressive_context = self.questioning_system.build_progressive_context(user_question)
        if progressive_context:
            context_parts.append(progressive_context)
        
        # Option 2: Document memory context
        entities = self.extract_entities(section_content)
        related_findings = self.memory_system.find_related_findings(user_question, entities)
        if related_findings:
            context_parts.append("RELATED FINDINGS FROM MEMORY:")
            for finding in related_findings:
                context_parts.append(f"  • {finding['finding']['question']}: {finding['finding']['response'][:100]}...")
        
        cross_section_insights = self.memory_system.get_cross_section_insights()
        if cross_section_insights:
            context_parts.append(cross_section_insights)
        
        # Option 3: Iterative analysis context
        iterations = self.deep_dive_analyzer.plan_analysis_iterations(user_question, 
                                                                    [f['response'] for f in self.memory_system.findings_database.values()])
        
        current_iteration = iterations[min(iteration - 1, len(iterations) - 1)] if iterations else {
            'iteration': 1, 'focus': 'Discovery', 'instruction': 'Comprehensive analysis'
        }
        
        # Combine all context
        full_context = "\n\n".join(context_parts) if context_parts else ""
        
        # Generate the enhanced prompt
        system_prompt = self.deep_dive_analyzer.generate_iterative_prompt(
            current_iteration, user_question, full_context
        )
        
        user_prompt = f"""Section {section_number} of 5 - Iteration {iteration}:
{section_content}"""
        
        payload = {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'max_tokens': 1200,
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
    
    def process_question_with_memory(self, file_path: str, user_question: str, 
                                   delay_between_calls: float = 2.0) -> Dict[str, Any]:
        """Process user question with all memory and analysis systems"""
        start_time = time.time()
        
        # Load file
        content = self.load_text_file(file_path)
        
        # Split into 5 sections
        sections = self.split_into_5_sections(content)
        
        # Determine analysis approach
        is_follow_up = self.questioning_system.is_follow_up_question(user_question)
        analysis_type = "follow_up" if is_follow_up else "initial"
        
        print(f"\n🔍 Analysis Type: {analysis_type.upper()}")
        if is_follow_up:
            print("🔗 Building on previous conversation context")
        
        # Process each section with enhanced memory
        section_responses = []
        all_findings = []
        
        for section in sections:
            section_num = section['section_number']
            print(f"\n🧠 Processing Section {section_num}/5 ({section['character_count']:,} chars) with memory...")
            
            # Make enhanced AI call
            response = self.query_ai_with_memory(
                user_question, 
                section['content'], 
                section_num
            )
            
            # Extract entities and store in memory
            entities = self.extract_entities(response)
            self.memory_system.store_finding(user_question, section_num, response, entities)
            
            section_responses.append({
                'section_number': section_num,
                'character_count': section['character_count'],
                'start_position': section['start_position'],
                'end_position': section['end_position'],
                'ai_response': response,
                'entities_found': list(entities),
                'analysis_type': analysis_type
            })
            
            all_findings.append(response)
            
            # Delay between calls
            if section_num < 5:
                print(f"⏱️  Waiting {delay_between_calls}s before next section...")
                time.sleep(delay_between_calls)
        
        # Generate follow-up suggestions
        follow_up_suggestions = self.questioning_system.suggest_follow_ups(user_question, all_findings)
        
        # Update conversation history
        findings_summary = f"Found {len([f for f in all_findings if f and 'Error' not in f])} responses across sections"
        self.questioning_system.conversation_history.append((user_question, findings_summary))
        
        end_time = time.time()
        
        return {
            'question': user_question,
            'file_path': file_path,
            'analysis_type': analysis_type,
            'total_characters': len(content),
            'sections': section_responses,
            'follow_up_suggestions': follow_up_suggestions,
            'cross_section_insights': self.memory_system.get_cross_section_insights(),
            'memory_stats': {
                'total_findings': len(self.memory_system.findings_database),
                'entities_tracked': len(self.memory_system.entity_tracker),
                'conversation_length': len(self.questioning_system.conversation_history)
            },
            'processing_time': end_time - start_time,
            'timestamp': datetime.now().isoformat(),
            'method': 'memory_enhanced_5_section_split'
        }
    
    def print_enhanced_results(self, results: Dict[str, Any]):
        """Print results with memory insights"""
        print("\n" + "="*90)
        print("📋 MEMORY-ENHANCED ANALYSIS RESULTS")
        print("="*90)
        print(f"❓ QUESTION: {results['question']}")
        print(f"📁 FILE: {results['file_path']}")
        print(f"🧩 ANALYSIS TYPE: {results['analysis_type'].upper()}")
        print(f"📄 TOTAL SIZE: {results['total_characters']:,} characters")
        print(f"⏱️  PROCESSING TIME: {results['processing_time']:.1f} seconds")
        print(f"🔬 METHOD: {results['method']}")
        
        # Memory statistics
        stats = results['memory_stats']
        print(f"\n🧠 MEMORY SYSTEM STATUS:")
        print(f"   • Total findings stored: {stats['total_findings']}")
        print(f"   • Entities being tracked: {stats['entities_tracked']}")
        print(f"   • Conversation history: {stats['conversation_length']} exchanges")
        
        print(f"\n🧠 AI RESPONSES BY SECTION:")
        print("="*90)
        
        for section in results['sections']:
            section_num = section['section_number']
            char_count = section['character_count']
            start_pos = section['start_position']
            end_pos = section['end_position']
            entities = section.get('entities_found', [])
            
            print(f"\n📄 SECTION {section_num}/5")
            print(f"📊 Size: {char_count:,} chars (positions {start_pos:,}-{end_pos:,})")
            if entities:
                print(f"🏷️  Entities: {', '.join(entities[:10])}{'...' if len(entities) > 10 else ''}")
            print("-" * 70)
            print(section['ai_response'])
            print("-" * 70)
        
        # Cross-section insights
        if results.get('cross_section_insights'):
            print(f"\n🔗 CROSS-SECTION INSIGHTS:")
            print("="*70)
            print(results['cross_section_insights'])
        
        # Follow-up suggestions
        if results.get('follow_up_suggestions'):
            print(f"\n💡 SUGGESTED FOLLOW-UP QUESTIONS:")
            print("="*70)
            for i, suggestion in enumerate(results['follow_up_suggestions'], 1):
                print(f"{i}. {suggestion}")

def main():
    """Main function with memory-enhanced features"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🚀 MEMORY-ENHANCED FILE SPLIT ANALYSIS")
    print("="*80)
    print("📄 Split file into 5 equal sections")
    print("🧠 Advanced AI with Progressive Questioning + Document Memory + Iterative Analysis")
    print("🔍 Individual AI calls per section with conversation memory")
    print("💡 Intelligent follow-up suggestions and cross-section insights")
    print("⚡ Enhanced context awareness and pattern recognition")
    print("="*80)
    
    # Initialize enhanced processor
    try:
        processor = MemoryEnhancedFileSplitter()
    except Exception as e:
        print(f"❌ Error initializing enhanced processor: {e}")
        return
    
    while True:
        try:
            print(f"\n" + "="*80)
            
            # Get file path
            file_input = input("📁 Enter text file path (or 'quit'): ").strip()
            
            if file_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using Memory-Enhanced Analysis!")
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
            
            # Show memory status
            memory_stats = processor.memory_system
            if memory_stats.findings_database:
                print(f"\n🧠 Memory Status: {len(memory_stats.findings_database)} findings, "
                      f"{len(memory_stats.entity_tracker)} entities tracked")
            
            # Optional delay setting
            delay_input = input("⏱️  Delay between AI calls in seconds (default 2.0): ").strip()
            try:
                delay = float(delay_input) if delay_input else 2.0
            except ValueError:
                delay = 2.0
            
            print(f"\n🔍 ENHANCED ANALYSIS PREVIEW:")
            print(f"   • File: {file_path}")
            print(f"   • Question: {question}")
            print(f"   • Method: Memory-enhanced 5-section analysis")
            print(f"   • Features: Progressive + Memory + Iterative")
            print(f"   • Delay between calls: {delay}s")
            
            # Process the question with all enhancements
            print(f"\n🚀 Starting memory-enhanced analysis...")
            results = processor.process_question_with_memory(file_path, question, delay)
            
            # Display enhanced results
            processor.print_enhanced_results(results)
            
            # Memory management options
            print(f"\n🧠 MEMORY MANAGEMENT:")
            memory_action = input("View memory (v), Clear memory (c), or Continue (enter): ").strip().lower()
            
            if memory_action == 'v':
                print(f"\n📚 STORED FINDINGS:")
                for finding_id, finding in processor.memory_system.findings_database.items():
                    print(f"  {finding_id}: {finding['question'][:50]}...")
            
            elif memory_action == 'c':
                processor.memory_system = DocumentMemorySystem()
                processor.questioning_system = ProgressiveQuestioningSystem()
                print("✅ Memory cleared!")
            
            # Option to save results
            save_option = input(f"\n💾 Save enhanced results to JSON? (y/n): ").strip().lower()
            if save_option == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"memory_enhanced_analysis_{timestamp}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Enhanced results saved to: {filename}")
            
            # Quick follow-up option
            if results.get('follow_up_suggestions'):
                follow_up = input(f"\n🔄 Ask a suggested follow-up question? (1-{len(results['follow_up_suggestions'])}, or 'n'): ").strip()
                if follow_up.isdigit() and 1 <= int(follow_up) <= len(results['follow_up_suggestions']):
                    suggested_q = results['follow_up_suggestions'][int(follow_up) - 1]
                    print(f"\n🔄 Following up with: {suggested_q}")
                    # Continue loop with suggested question
                    continue
            
        except KeyboardInterrupt:
            print("\n👋 Analysis interrupted. Memory preserved for next session!")
            break
        except Exception as e:
            print(f"❌ Error during enhanced analysis: {e}")

if __name__ == "__main__":
    main()