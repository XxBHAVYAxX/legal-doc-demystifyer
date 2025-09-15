"""
Utility functions for the Legal Document Analysis application
"""

import os
import logging
import hashlib
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path exists and is accessible
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False

def get_file_hash(file_path: str) -> Optional[str]:
    """
    Generate MD5 hash of a file for caching purposes
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash string or None if error
    """
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error generating file hash: {e}")
        return None

def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s\.,;:!?\-\'\"()[\]{}]', ' ', text)
    
    # Normalize quotes
    text = re.sub(r'[""''`]', '"', text)
    
    return text

def chunk_text(text: str, max_length: int = 8000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks for processing large documents
    
    Args:
        text: Text to split
        max_length: Maximum length per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending within the last 200 characters
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + max_length - 200:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = max(end - overlap, start + 1)  # Prevent infinite loop
    
    return chunks

def format_currency(amount_str: str) -> Optional[str]:
    """
    Format currency amounts consistently
    
    Args:
        amount_str: String containing currency amount
        
    Returns:
        Formatted currency string or None
    """
    try:
        # Extract numeric value
        numeric = re.findall(r'[\d,]+\.?\d*', amount_str)
        if not numeric:
            return None
        
        # Get currency symbol
        currency_symbols = ['$', '€', '£', '¥', '₹']
        currency = None
        for symbol in currency_symbols:
            if symbol in amount_str:
                currency = symbol
                break
        
        if not currency:
            currency = '$'  # Default to dollar
        
        # Format the number
        number = numeric[0].replace(',', '')
        if '.' in number:
            formatted = f"{currency}{float(number):,.2f}"
        else:
            formatted = f"{currency}{int(number):,}"
        
        return formatted
        
    except Exception:
        return amount_str  # Return original if formatting fails

def extract_dates(text: str) -> List[str]:
    """
    Extract date patterns from text
    
    Args:
        text: Text to search for dates
        
    Returns:
        List of found dates
    """
    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
        r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # YYYY-MM-DD
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    
    return list(set(dates))  # Remove duplicates

def save_results_to_json(results: Dict[str, Any], file_path: str) -> bool:
    """
    Save analysis results to JSON file
    
    Args:
        results: Analysis results dictionary
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Add metadata
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'tool': 'legal_document_analysis_google_genai'
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

def load_results_from_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load analysis results from JSON file
    
    Args:
        file_path: Input file path
        
    Returns:
        Results dictionary or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Results loaded from {file_path}")
        return results
        
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return None

def generate_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics from analysis results
    
    Args:
        results: Analysis results
        
    Returns:
        Dictionary of summary statistics
    """
    stats = {
        'processing_info': {},
        'content_stats': {},
        'entity_stats': {},
        'summary_stats': {}
    }
    
    try:
        # Processing info
        stats['processing_info'] = {
            'status': results.get('status', 'unknown'),
            'file_name': results.get('file_name', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Content statistics
        if 'text_extraction' in results:
            text_data = results['text_extraction']
            text_content = text_data.get('text', '')
            
            stats['content_stats'] = {
                'character_count': len(text_content),
                'word_count': len(text_content.split()),
                'sentence_count': len(re.findall(r'[.!?]+', text_content)),
                'paragraph_count': len([p for p in text_content.split('\n\n') if p.strip()]),
                'page_count': text_data.get('pages', 0),
                'confidence': text_data.get('confidence', 0.0)
            }
        
        # Entity statistics
        if 'entities' in results:
            entity_data = results['entities']
            entities = entity_data.get('entities', {})
            
            stats['entity_stats'] = {
                'total_entities': sum(len(entity_list) for entity_list in entities.values()),
                'categories_found': len(entities),
                'entities_by_category': {
                    category: len(entity_list) 
                    for category, entity_list in entities.items()
                }
            }
        
        # Summary statistics
        if 'summary' in results:
            summary_data = results['summary']
            
            stats['summary_stats'] = {
                'summary_type': summary_data.get('summary_type', 'unknown'),
                'original_length': summary_data.get('original_length', 0),
                'summary_length': summary_data.get('summary_length', 0),
                'compression_ratio': summary_data.get('compression_ratio', 0.0),
                'key_points_count': len(summary_data.get('key_points', []))
            }
        
    except Exception as e:
        logger.error(f"Error generating statistics: {e}")
        stats['error'] = str(e)
    
    return stats

def create_directory_structure(base_path: str) -> bool:
    """
    Create necessary directory structure for the project
    
    Args:
        base_path: Base directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        directories = [
            'src',
            'notebooks',
            'sample_pdfs',
            'output',
            'logs'
        ]
        
        base_path = Path(base_path)
        
        for directory in directories:
            dir_path = base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py in src directory
        init_file = base_path / 'src' / '__init__.py'
        init_file.touch(exist_ok=True)
        
        logger.info(f"Directory structure created at {base_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating directory structure: {e}")
        return False

def format_entity_report(entities: Dict[str, List[str]]) -> str:
    """
    Format entity extraction results as a readable report
    
    Args:
        entities: Dictionary of extracted entities
        
    Returns:
        Formatted string report
    """
    report_lines = []
    report_lines.append("LEGAL DOCUMENT ENTITY EXTRACTION REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    total_entities = sum(len(entity_list) for entity_list in entities.values())
    report_lines.append(f"Total entities found: {total_entities}")
    report_lines.append(f"Categories identified: {len(entities)}")
    report_lines.append("")
    
    for category, entity_list in entities.items():
        if entity_list:
            report_lines.append(f"{category.replace('_', ' ').upper()} ({len(entity_list)})")
            report_lines.append("-" * 30)
            
            for i, entity in enumerate(entity_list, 1):
                report_lines.append(f"  {i:2d}. {entity}")
            
            report_lines.append("")
    
    report_lines.append("=" * 50)
    report_lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(report_lines)

def validate_environment() -> Dict[str, bool]:
    """
    Validate the environment setup and dependencies
    
    Returns:
        Dictionary of validation results
    """
    validation = {
        'gemini_api_key': False,
        'google_cloud_available': False,
        'required_packages': False
    }
    
    # Check Gemini API key
    if os.getenv('GEMINI_API_KEY'):
        validation['gemini_api_key'] = True
    
    # Check Google Cloud availability
    try:
        from google.cloud import documentai_v1
        validation['google_cloud_available'] = True
    except ImportError:
        pass
    
    # Check required packages
    try:
        import google.generativeai
        import streamlit
        import pandas
        validation['required_packages'] = True
    except ImportError:
        pass
    
    return validation