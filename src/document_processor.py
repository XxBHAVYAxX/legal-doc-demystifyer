"""
Main Document Processor Module
Coordinates text extraction, summarization, and entity extraction
"""

import logging
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path

from text_extractor import DocumentAIExtractor, GeminiTextExtractor
from summarizer import GeminiSummarizer  
from entity_extractor import GeminiEntityExtractor
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDocumentProcessor:
    """Main processor for legal document analysis using Google GenAI tools"""
    
    def __init__(self):
        self.config = Config()
        self.config.validate_config()
        
        # Initialize components
        self.text_extractor = DocumentAIExtractor()
        self.gemini_extractor = GeminiTextExtractor()  # Fallback extractor
        self.summarizer = GeminiSummarizer()
        self.entity_extractor = GeminiEntityExtractor()
        
        logger.info("Legal Document Processor initialized with Google GenAI tools")
    
    def process_document(self, file_path: str, analysis_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a legal document through the complete pipeline
        
        Args:
            file_path: Path to the document file
            analysis_options: Dictionary of analysis options
            
        Returns:
            Dictionary containing all analysis results
        """
        if analysis_options is None:
            analysis_options = {
                'extract_text': True,
                'generate_summary': True,
                'extract_entities': True,
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
            
            # Step 4: Generate bullet points
            if analysis_options.get('generate_bullet_points', True):
                logger.info("Generating bullet points...")
                bullet_points = self.summarizer.generate_bullet_points(document_text)
                results['bullet_points'] = bullet_points
            
            # Step 5: Analyze risks
            if analysis_options.get('analyze_risks', True):
                logger.info("Analyzing legal risks...")
                risk_analysis = self.summarizer.analyze_legal_risks(document_text)
                results['risk_analysis'] = risk_analysis
            
            # Step 6: Extract legal relationships
            if analysis_options.get('extract_relationships', False):
                logger.info("Extracting legal relationships...")
                relationships = self.entity_extractor.extract_legal_relationships(document_text)
                results['relationships'] = relationships
            
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
    
    def process_multiple_documents(self, file_paths: list, analysis_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of file paths
            analysis_options: Dictionary of analysis options
            
        Returns:
            Dictionary containing results for all documents
        """
        results = {
            'total_documents': len(file_paths),
            'processed_successfully': 0,
            'failed': 0,
            'individual_results': {},
            'summary_comparison': None
        }
        
        individual_results = []
        
        for file_path in file_paths:
            logger.info(f"Processing document {len(individual_results) + 1}/{len(file_paths)}: {file_path}")
            
            result = self.process_document(file_path, analysis_options)
            results['individual_results'][file_path] = result
            
            if result['status'] == 'completed':
                results['processed_successfully'] += 1
                individual_results.append(result)
            else:
                results['failed'] += 1
        
        # Generate comparison if multiple documents were processed successfully
        if len(individual_results) >= 2 and analysis_options and analysis_options.get('compare_documents', False):
            logger.info("Generating document comparison...")
            results['summary_comparison'] = self._compare_multiple_documents(individual_results)
        
        return results
    
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
    
    def _compare_multiple_documents(self, results: list) -> Dict[str, Any]:
        """Generate comparison analysis for multiple documents"""
        try:
            if len(results) < 2:
                return {'error': 'Need at least 2 documents for comparison'}
            
            # Compare first two documents as example
            doc1_result = results[0]
            doc2_result = results[1]
            
            doc1_text = doc1_result['text_extraction']['text']
            doc2_text = doc2_result['text_extraction']['text']
            
            comparison = self.summarizer.compare_documents(
                doc1_text, doc2_text,
                doc1_result['file_name'], doc2_result['file_name']
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Document comparison failed: {e}")
            return {'error': f'Comparison failed: {str(e)}'}
    
    def get_processing_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing statistics from results"""
        stats = {
            'document_info': {},
            'text_stats': {},
            'summary_stats': {},
            'entity_stats': {},
            'processing_time': 0
        }
        
        try:
            if 'text_extraction' in results:
                text_data = results['text_extraction']
                stats['text_stats'] = {
                    'character_count': len(text_data.get('text', '')),
                    'word_count': len(text_data.get('text', '').split()),
                    'page_count': text_data.get('pages', 0),
                    'confidence': text_data.get('confidence', 0.0)
                }
            
            if 'summary' in results:
                summary_data = results['summary']
                stats['summary_stats'] = {
                    'summary_length': summary_data.get('summary_length', 0),
                    'compression_ratio': summary_data.get('compression_ratio', 0.0),
                    'key_points_count': len(summary_data.get('key_points', []))
                }
            
            if 'entities' in results:
                entity_data = results['entities']
                stats['entity_stats'] = {
                    'total_entities': entity_data.get('total_entities', 0),
                    'categories_found': len(entity_data.get('categories_found', [])),
                    'entities_by_category': {
                        category: len(entities) 
                        for category, entities in entity_data.get('entities', {}).items()
                    }
                }
            
            stats['document_info'] = {
                'file_name': results.get('file_name', ''),
                'status': results.get('status', ''),
                'analysis_options': results.get('analysis_options', {})
            }
            
        except Exception as e:
            logger.error(f"Statistics generation failed: {e}")
            stats['error'] = str(e)
        
        return stats