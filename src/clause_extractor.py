"""
Enhanced Clause Extraction Module using Gemini API
Identifies and categorizes specific legal clauses in documents
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import google.generativeai as genai
from .config import Config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClauseExtractor:
    """Legal clause extraction and highlighting using Google Gemini API"""
    
    def __init__(self):
        self.config = Config()
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
        self.flash_model = genai.GenerativeModel(self.config.GEMINI_FLASH_MODEL)
        
        # Define common legal clause types
        self.clause_types = {
            'TERMINATION': 'Clauses related to contract termination, expiry, or cancellation',
            'PAYMENT': 'Payment terms, fees, invoicing, and financial obligations',
            'INDEMNIFICATION': 'Indemnity, liability, and hold harmless provisions',
            'CONFIDENTIALITY': 'Non-disclosure and confidentiality requirements',
            'INTELLECTUAL_PROPERTY': 'IP rights, ownership, and licensing terms',
            'FORCE_MAJEURE': 'Force majeure and unforeseeable circumstances',
            'GOVERNING_LAW': 'Jurisdiction, governing law, and dispute resolution',
            'WARRANTIES': 'Warranties, representations, and guarantees',
            'LIMITATION_LIABILITY': 'Liability limitations and damage caps',
            'ASSIGNMENT': 'Assignment and transfer of rights provisions',
            'AMENDMENT': 'Contract modification and amendment procedures',
            'DELIVERY': 'Delivery terms, timelines, and performance obligations'
        }
    
    def extract_clauses(self, text: str, clause_types: List[str] = None) -> Dict[str, Any]:
        """
        Extract and categorize legal clauses from document text
        
        Args:
            text: The document text to analyze
            clause_types: Specific clause types to look for (default: all)
            
        Returns:
            Dictionary containing extracted clauses with positions
        """
        try:
            if clause_types is None:
                clause_types = list(self.clause_types.keys())
            
            prompt = self._build_clause_extraction_prompt(text, clause_types)
            response = self.model.generate_content(prompt)
            
            # Parse the response
            extracted_clauses = self._parse_clause_response(response.text)
            
            # Add metadata
            clause_data = {
                'clauses': extracted_clauses,
                'total_clauses_found': len(extracted_clauses),
                'clause_types_searched': clause_types,
                'document_length': len(text)
            }
            
            logger.info(f"Successfully extracted {clause_data['total_clauses_found']} clauses")
            return clause_data
            
        except Exception as e:
            logger.error(f"Clause extraction failed: {e}")
            return {
                'clauses': [],
                'total_clauses_found': 0,
                'clause_types_searched': clause_types or [],
                'document_length': len(text)
            }
    
    def _build_clause_extraction_prompt(self, text: str, clause_types: List[str]) -> str:
        """Build the prompt for clause extraction"""
        
        clause_descriptions = []
        for clause_type in clause_types:
            description = self.clause_types.get(clause_type, f"Clauses related to {clause_type}")
            clause_descriptions.append(f"- {clause_type}: {description}")
        
        prompt = f"""
        You are a legal expert analyzing a contract. Extract specific legal clauses from this document and provide their exact text with context.

        Document text:
        {text}

        Extract the following types of clauses:
        {chr(10).join(clause_descriptions)}

        For each clause found, return a JSON object with this structure:
        {{
            "clause_type": "TYPE_NAME",
            "clause_text": "The exact text of the clause",
            "context": "Brief explanation of what this clause means",
            "importance": "HIGH/MEDIUM/LOW",
            "section": "Section or paragraph where found (if identifiable)"
        }}

        Return all found clauses as a JSON array. If no clauses of a specific type are found, omit that type from the results.

        Example format:
        [
            {{
                "clause_type": "PAYMENT",
                "clause_text": "Payment shall be due within 30 days of invoice date...",
                "context": "Establishes 30-day payment terms",
                "importance": "HIGH",
                "section": "Section 3.1"
            }},
            {{
                "clause_type": "TERMINATION",
                "clause_text": "Either party may terminate this agreement with 60 days written notice...",
                "context": "Allows termination with 60-day notice",
                "importance": "HIGH",
                "section": "Section 8.2"
            }}
        ]
        """
        
        return prompt
    
    def _parse_clause_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse JSON response from Gemini and clean up clauses"""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                clauses = json.loads(json_str)
            else:
                # Try parsing entire response as JSON
                clauses = json.loads(response_text)
            
            # Validate and clean clauses
            cleaned_clauses = []
            for clause in clauses:
                if isinstance(clause, dict) and 'clause_type' in clause and 'clause_text' in clause:
                    # Ensure all required fields exist
                    cleaned_clause = {
                        'clause_type': clause.get('clause_type', ''),
                        'clause_text': clause.get('clause_text', '').strip(),
                        'context': clause.get('context', ''),
                        'importance': clause.get('importance', 'MEDIUM'),
                        'section': clause.get('section', 'Unknown')
                    }
                    
                    # Skip if clause text is too short or empty
                    if len(cleaned_clause['clause_text']) > 20:
                        cleaned_clauses.append(cleaned_clause)
            
            return cleaned_clauses
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, attempting fallback parsing")
            return self._fallback_clause_parsing(response_text)
        except Exception as e:
            logger.error(f"Clause parsing failed: {e}")
            return []
    
    def _fallback_clause_parsing(self, response_text: str) -> List[Dict[str, Any]]:
        """Fallback method to extract clauses when JSON parsing fails"""
        clauses = []
        
        # Look for patterns that might indicate clauses
        lines = response_text.split('\n')
        current_clause = {}
        
        for line in lines:
            line = line.strip()
            
            # Look for clause type indicators
            for clause_type in self.clause_types.keys():
                if clause_type.lower() in line.lower() and ':' in line:
                    if current_clause:
                        clauses.append(current_clause)
                    current_clause = {
                        'clause_type': clause_type,
                        'clause_text': '',
                        'context': '',
                        'importance': 'MEDIUM',
                        'section': 'Unknown'
                    }
                    break
            
            # Add to current clause text
            if current_clause and line and not any(keyword in line.lower() for keyword in ['type:', 'context:', 'importance:']):
                if len(line) > 20:  # Likely clause text
                    current_clause['clause_text'] = line
        
        if current_clause:
            clauses.append(current_clause)
        
        return clauses
    
    def highlight_clauses_in_text(self, text: str, clauses: List[Dict[str, Any]]) -> str:
        """
        Create highlighted version of text with clause markers
        
        Args:
            text: Original document text
            clauses: List of extracted clauses
            
        Returns:
            Text with HTML highlighting markers
        """
        try:
            highlighted_text = text
            
            # Color coding for different clause types
            color_map = {
                'TERMINATION': '#ffcccc',      # Light red
                'PAYMENT': '#ccffcc',          # Light green
                'INDEMNIFICATION': '#ffcc99',  # Light orange
                'CONFIDENTIALITY': '#ccccff',  # Light blue
                'INTELLECTUAL_PROPERTY': '#ffccff', # Light purple
                'FORCE_MAJEURE': '#ffffcc',    # Light yellow
                'GOVERNING_LAW': '#ccffff',    # Light cyan
                'WARRANTIES': '#f0ccff',       # Light magenta
                'LIMITATION_LIABILITY': '#ffccdd', # Light pink
                'ASSIGNMENT': '#ccffdd',       # Light mint
                'AMENDMENT': '#ddccff',        # Light lavender
                'DELIVERY': '#ddffcc'          # Light lime
            }
            
            # Sort clauses by text length (longest first) to avoid nested highlighting issues
            sorted_clauses = sorted(clauses, key=lambda x: len(x.get('clause_text', '')), reverse=True)
            
            for i, clause in enumerate(sorted_clauses):
                clause_text = clause.get('clause_text', '').strip()
                clause_type = clause.get('clause_type', '')
                
                if clause_text and clause_text in highlighted_text:
                    color = color_map.get(clause_type, '#f0f0f0')
                    
                    # Create highlight span
                    highlight_span = f'<span style="background-color: {color}; padding: 2px; border-left: 3px solid #333; margin: 2px;" title="{clause_type}: {clause.get("context", "")}">{clause_text}</span>'
                    
                    # Replace first occurrence
                    highlighted_text = highlighted_text.replace(clause_text, highlight_span, 1)
            
            return highlighted_text
            
        except Exception as e:
            logger.error(f"Text highlighting failed: {e}")
            return text  # Return original text if highlighting fails
    
    def generate_clause_summary(self, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary report of extracted clauses
        
        Args:
            clauses: List of extracted clauses
            
        Returns:
            Dictionary containing clause analysis summary
        """
        try:
            if not clauses:
                return {
                    'total_clauses': 0,
                    'high_importance': 0,
                    'clause_distribution': {},
                    'key_findings': [],
                    'recommendations': []
                }
            
            # Analyze clause distribution
            clause_distribution = {}
            importance_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            
            for clause in clauses:
                clause_type = clause.get('clause_type', 'UNKNOWN')
                importance = clause.get('importance', 'MEDIUM')
                
                clause_distribution[clause_type] = clause_distribution.get(clause_type, 0) + 1
                importance_counts[importance] = importance_counts.get(importance, 0) + 1
            
            # Generate key findings
            key_findings = []
            high_importance_clauses = [c for c in clauses if c.get('importance') == 'HIGH']
            
            if high_importance_clauses:
                key_findings.append(f"Found {len(high_importance_clauses)} high-importance clauses requiring attention")
            
            if 'TERMINATION' in clause_distribution:
                key_findings.append("Document contains termination provisions")
            
            if 'PAYMENT' in clause_distribution:
                key_findings.append("Payment terms are specified in the document")
            
            # Generate basic recommendations
            recommendations = []
            if importance_counts['HIGH'] > 0:
                recommendations.append("Review all high-importance clauses carefully")
            
            if 'LIMITATION_LIABILITY' not in clause_distribution:
                recommendations.append("Consider adding liability limitation clauses")
            
            if 'GOVERNING_LAW' not in clause_distribution:
                recommendations.append("Ensure governing law and jurisdiction are specified")
            
            return {
                'total_clauses': len(clauses),
                'high_importance': importance_counts['HIGH'],
                'clause_distribution': clause_distribution,
                'importance_distribution': importance_counts,
                'key_findings': key_findings,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Clause summary generation failed: {e}")
            return {
                'total_clauses': len(clauses) if clauses else 0,
                'error': str(e)
            }