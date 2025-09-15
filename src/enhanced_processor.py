"""
Enhanced Document Processor with Clause Extraction and Q&A
Integrates clause extraction and smart search capabilities
"""

import logging
import os
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from .text_extractor import DocumentAIExtractor, GeminiTextExtractor

from .summarizer import GeminiSummarizer
from .entity_extractor import GeminiEntityExtractor
from .clause_extractor import GeminiClauseExtractor
from .qa_system import GeminiQASystem
from .config import Config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedLegalDocumentProcessor:
    """Enhanced processor with clause extraction and Q&A capabilities"""
    
    def __init__(self):
        self.config = Config()
        self.config.validate_config()
        
        # Initialize components
        self.text_extractor = DocumentAIExtractor()
        self.gemini_extractor = GeminiTextExtractor()
        self.summarizer = GeminiSummarizer()
        self.entity_extractor = GeminiEntityExtractor()
        self.clause_extractor = GeminiClauseExtractor()  # New component
        self.qa_system = GeminiQASystem()  # New component
        
        logger.info("Enhanced Legal Document Processor initialized with clause extraction and Q&A")
    
    def process_document(self, file_path: str, analysis_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a legal document with enhanced capabilities
        
        Args:
            file_path: Path to the document file
            analysis_options: Dictionary of analysis options
            
        Returns:
            Dictionary containing all analysis results including clauses and Q&A
        """
        if analysis_options is None:
            analysis_options = {
                'extract_text': True,
                'generate_summary': True,
                'extract_entities': True,
                'extract_clauses': True,  # New option
                'generate_qa_suggestions': True,  # New option
                'summary_type': 'comprehensive',
                'entity_extraction_type': 'comprehensive',
                'analyze_risks': True,
                'generate_bullet_points': True
            }
        
        try:
            # Validate file
            if not self._validate_file(file_path):
                raise ValueError(f"Invalid file: {file_path}")
            
            results = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'analysis_options': analysis_options,
                'status': 'processing'
            }
            
            # Step 1: Extract text from document
            if analysis_options.get('extract_text', True):
                logger.info("Extracting text from document...")
                text_data = self._extract_document_text(file_path)
                results['text_extraction'] = text_data
                
                if not text_data.get('text'):
                    results['status'] = 'failed'
                    results['error'] = 'No text could be extracted from document'
                    return results
            
            document_text = text_data['text']
            
            # Step 2: Generate summary
            if analysis_options.get('generate_summary', True):
                logger.info("Generating document summary...")
                summary_type = analysis_options.get('summary_type', 'comprehensive')
                summary_data = self.summarizer.summarize_document(document_text, summary_type)
                results['summary'] = summary_data
            
            # Step 3: Extract entities
            if analysis_options.get('extract_entities', True):
                logger.info("Extracting named entities...")
                extraction_type = analysis_options.get('entity_extraction_type', 'comprehensive')
                entity_data = self.entity_extractor.extract_entities(document_text, extraction_type)
                results['entities'] = entity_data
            
            # Step 4: Extract clauses (NEW)
            if analysis_options.get('extract_clauses', True):
                logger.info("Extracting legal clauses...")
                clause_types = analysis_options.get('clause_types', None)
                clause_data = self.clause_extractor.extract_clauses(document_text, clause_types)
                results['clauses'] = clause_data
                
                # Generate highlighted text
                if clause_data.get('clauses'):
                    highlighted_text = self.clause_extractor.highlight_clauses_in_text(
                        document_text, clause_data['clauses']
                    )
                    results['highlighted_text'] = highlighted_text
                    
                    # Generate clause summary
                    clause_summary = self.clause_extractor.generate_clause_summary(clause_data['clauses'])
                    results['clause_summary'] = clause_summary
            
            # Step 5: Generate Q&A suggestions (NEW)
            if analysis_options.get('generate_qa_suggestions', True):
                logger.info("Generating Q&A suggestions...")
                suggested_questions = self.qa_system.get_suggested_questions(document_text)
                results['suggested_questions'] = suggested_questions
                
                # Store document text for Q&A (truncated for performance)
                results['qa_ready_text'] = document_text[:10000]  # First 10k characters
            
            # Step 6: Generate bullet points
            if analysis_options.get('generate_bullet_points', True):
                logger.info("Generating bullet points...")
                bullet_points = self.summarizer.generate_bullet_points(document_text)
                results['bullet_points'] = bullet_points
            
            # Step 7: Analyze risks
            if analysis_options.get('analyze_risks', True):
                logger.info("Analyzing legal risks...")
                risk_analysis = self.summarizer.analyze_legal_risks(document_text)
                results['risk_analysis'] = risk_analysis
            
            results['status'] = 'completed'
            logger.info(f"Successfully processed document: {file_path}")
            
            return results
        
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                'file_path': file_path,
                'status': 'failed',
                'error': str(e),
                'analysis_options': analysis_options
            }
    
    def answer_question(self, document_text: str, question: str, context_clauses: List[Dict] = None) -> Dict[str, Any]:
        """
        Answer a question about the processed document
        
        Args:
            document_text: Document text
            question: User's question
            context_clauses: Optional relevant clauses for context
            
        Returns:
            Answer data dictionary
        """
        try:
            return self.qa_system.answer_question(document_text, question, context_clauses)
        except Exception as e:
            logger.error(f"Q&A failed: {e}")
            return {
                'question': question,
                'answer': f"Error processing question: {str(e)}",
                'confidence': 0.0,
                'relevant_sections': [],
                'context_clauses_used': 0
            }
    
    def search_document(self, document_text: str, search_query: str) -> Dict[str, Any]:
        """
        Perform smart search within the document
        
        Args:
            document_text: Document text
            search_query: Search query
            
        Returns:
            Search results dictionary
        """
        try:
            return self.qa_system.search_document(document_text, search_query)
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return {
                'query': search_query,
                'results': [],
                'total_results': 0,
                'error': str(e)
            }
    
    def get_clause_context_for_question(self, clauses: List[Dict], question: str) -> List[Dict]:
        """
        Find relevant clauses that might help answer a question
        
        Args:
            clauses: List of extracted clauses
            question: User's question
            
        Returns:
            List of relevant clauses
        """
        try:
            relevant_clauses = []
            question_lower = question.lower()
            
            # Simple keyword matching for clause relevance
            keyword_mapping = {
                'payment': ['PAYMENT'],
                'terminate': ['TERMINATION'],
                'end': ['TERMINATION'],
                'liability': ['LIMITATION_LIABILITY', 'INDEMNIFICATION'],
                'confidential': ['CONFIDENTIALITY'],
                'intellectual property': ['INTELLECTUAL_PROPERTY'],
                'ip': ['INTELLECTUAL_PROPERTY'],
                'law': ['GOVERNING_LAW'],
                'jurisdiction': ['GOVERNING_LAW'],
                'force majeure': ['FORCE_MAJEURE'],
                'assignment': ['ASSIGNMENT'],
                'warranty': ['WARRANTIES'],
                'deliver': ['DELIVERY']
            }
            
            # Find clauses based on keywords in question
            for keyword, clause_types in keyword_mapping.items():
                if keyword in question_lower:
                    for clause in clauses:
                        if clause.get('clause_type') in clause_types:
                            relevant_clauses.append(clause)
            
            # If no keyword matches, return high importance clauses
            if not relevant_clauses:
                relevant_clauses = [c for c in clauses if c.get('importance') == 'HIGH'][:3]
            
            return relevant_clauses[:3]  # Limit to 3 most relevant
            
        except Exception as e:
            logger.error(f"Failed to find relevant clauses: {e}")
            return []
    
    def _extract_document_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text from document using appropriate method"""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return self.text_extractor.extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                return self.text_extractor.extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return {
                    'text': text,
                    'pages': 1,
                    'entities': [],
                    'tables': [],
                    'confidence': 1.0
                }
            else:
                # Try Gemini extractor as fallback
                return self.gemini_extractor.extract_text_from_file(file_path)
                
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            # Try Gemini extractor as final fallback
            try:
                return self.gemini_extractor.extract_text_from_file(file_path)
            except Exception as fallback_error:
                logger.error(f"Fallback extraction also failed: {fallback_error}")
                return {
                    'text': '',
                    'pages': 0,
                    'entities': [],
                    'tables': [],
                    'confidence': 0.0
                }
    
    def _validate_file(self, file_path: str) -> bool:
        """Validate if file exists and is supported"""
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False
        
        file_size = os.path.getsize(file_path)
        max_size = self.config.MAX_FILE_SIZE_MB * 1024 * 1024
        
        if file_size > max_size:
            logger.error(f"File too large: {file_size} bytes (max: {max_size} bytes)")
            return False
        
        file_extension = Path(file_path).suffix.lower().lstrip('.')
        if file_extension not in self.config.SUPPORTED_FILE_TYPES:
            logger.warning(f"File type may not be supported: {file_extension}")
        
        return True
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive analysis report
        
        Args:
            results: Processing results dictionary
            
        Returns:
            Formatted report string
        """
        try:
            report_lines = []
            report_lines.append("# LEGAL DOCUMENT ANALYSIS REPORT")
            report_lines.append("="*50)
            report_lines.append("")
            
            # Document info
            report_lines.append(f"**Document:** {results.get('file_name', 'Unknown')}")
            report_lines.append(f"**Status:** {results.get('status', 'Unknown').upper()}")
            report_lines.append("")
            
            # Summary
            if 'summary' in results:
                report_lines.append("## EXECUTIVE SUMMARY")
                report_lines.append("-"*30)
                report_lines.append(results['summary'].get('summary', 'No summary available'))
                report_lines.append("")
            
            # Key clauses
            if 'clauses' in results and results['clauses'].get('clauses'):
                report_lines.append("## KEY LEGAL CLAUSES")
                report_lines.append("-"*30)
                
                for clause in results['clauses']['clauses']:
                    clause_type = clause.get('clause_type', 'Unknown')
                    importance = clause.get('importance', 'MEDIUM')
                    context = clause.get('context', 'No context available')
                    
                    report_lines.append(f"### {clause_type} [{importance}]")
                    report_lines.append(f"**Context:** {context}")
                    report_lines.append(f"**Text:** {clause.get('clause_text', '')[:200]}...")
                    report_lines.append("")
            
            # Entities
            if 'entities' in results and results['entities'].get('entities'):
                report_lines.append("## EXTRACTED ENTITIES")
                report_lines.append("-"*30)
                
                for category, entity_list in results['entities']['entities'].items():
                    if entity_list:
                        report_lines.append(f"**{category.replace('_', ' ').title()}:** {', '.join(entity_list[:5])}")
                report_lines.append("")
            
            # Risk analysis
            if 'risk_analysis' in results:
                risk_data = results['risk_analysis']
                report_lines.append("## RISK ANALYSIS")
                report_lines.append("-"*30)
                
                if risk_data.get('high_risks'):
                    report_lines.append("**HIGH RISKS:**")
                    for risk in risk_data['high_risks']:
                        report_lines.append(f"• {risk}")
                    report_lines.append("")
                
                if risk_data.get('recommendations'):
                    report_lines.append("**RECOMMENDATIONS:**")
                    for rec in risk_data['recommendations']:
                        report_lines.append(f"• {rec}")
                    report_lines.append("")
            
            # Suggested questions
            if 'suggested_questions' in results:
                report_lines.append("## SUGGESTED QUESTIONS")
                report_lines.append("-"*30)
                for i, question in enumerate(results['suggested_questions'][:5], 1):
                    report_lines.append(f"{i}. {question}")
                report_lines.append("")
            
            report_lines.append("="*50)
            report_lines.append("Report generated by Legal Document Analysis System")
            
            return "\n".join(report_lines)
        
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Error generating report: {str(e)}"
