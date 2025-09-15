"""
Gemini API-based Summarization Module
Replaces FlanT5 model with Google Gemini for legal document summarization
"""

import logging
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from .config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiSummarizer:
    """Legal document summarization using Google Gemini API"""
    
    def __init__(self):
        self.config = Config()
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
        self.flash_model = genai.GenerativeModel(self.config.GEMINI_FLASH_MODEL)
    
    def summarize_document(self, text: str, summary_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Generate a summary of the legal document
        
        Args:
            text: The document text to summarize
            summary_type: Type of summary ('comprehensive', 'brief', 'executive')
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # Choose appropriate prompt based on summary type
            prompt = self._get_summary_prompt(text, summary_type)
            
            # Use flash model for brief summaries, regular model for comprehensive
            model = self.flash_model if summary_type == "brief" else self.model
            
            response = model.generate_content(prompt)
            
            summary_data = {
                'summary': response.text,
                'summary_type': summary_type,
                'original_length': len(text),
                'summary_length': len(response.text),
                'compression_ratio': len(response.text) / len(text) if len(text) > 0 else 0,
                'key_points': self._extract_key_points(response.text)
            }
            
            logger.info(f"Successfully generated {summary_type} summary")
            return summary_data
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {
                'summary': "Error generating summary",
                'summary_type': summary_type,
                'original_length': len(text),
                'summary_length': 0,
                'compression_ratio': 0,
                'key_points': []
            }
    
    def _get_summary_prompt(self, text: str, summary_type: str) -> str:
        """Get appropriate prompt based on summary type"""
        
        base_prompt = """
        You are an expert legal analyst. Analyze the following legal document and provide a summary.
        
        Document text:
        {text}
        """
        
        if summary_type == "brief":
            specific_instruction = """
            Provide a brief summary (2-3 paragraphs) focusing on:
            1. Document type and purpose
            2. Main parties involved
            3. Key obligations and terms
            """
        
        elif summary_type == "executive":
            specific_instruction = """
            Provide an executive summary (1 paragraph) highlighting:
            1. Document essence in business terms
            2. Critical legal implications
            3. Key decision points
            """
        
        else:  # comprehensive
            specific_instruction = """
            Provide a comprehensive summary including:
            1. Document type, purpose, and context
            2. All parties involved and their roles
            3. Key provisions, clauses, and obligations
            4. Important dates, deadlines, and milestones
            5. Financial terms and considerations
            6. Legal implications and potential risks
            7. Compliance requirements
            8. Termination and dispute resolution clauses
            """
        
        return base_prompt.format(text=text) + "\n" + specific_instruction
    
    def generate_bullet_points(self, text: str) -> List[str]:
        """
        Generate bullet point summary of key legal provisions
        
        Args:
            text: The document text
            
        Returns:
            List of bullet points
        """
        try:
            prompt = f"""
            Extract the most important legal provisions from this document as concise bullet points.
            Focus on actionable items, obligations, rights, and critical terms.
            
            Document text: {text}
            
            Return as a numbered list of bullet points.
            """
            
            response = self.flash_model.generate_content(prompt)
            
            # Parse bullet points
            bullet_points = []
            for line in response.text.split('\n'):
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*') or any(line.startswith(f"{i}.") for i in range(1, 100))):
                    bullet_points.append(line)
            
            return bullet_points
            
        except Exception as e:
            logger.error(f"Bullet point generation failed: {e}")
            return []
    
    def analyze_legal_risks(self, text: str) -> Dict[str, Any]:
        """
        Analyze potential legal risks in the document
        
        Args:
            text: The document text
            
        Returns:
            Dictionary containing risk analysis
        """
        try:
            prompt = f"""
            As a legal expert, analyze this document for potential legal risks and concerns.
            
            Document text: {text}
            
            Provide analysis in the following format:
            
            HIGH RISK AREAS:
            - [List high-risk provisions or clauses]
            
            MEDIUM RISK AREAS:
            - [List medium-risk provisions]
            
            RECOMMENDATIONS:
            - [List recommended actions or considerations]
            
            COMPLIANCE NOTES:
            - [List any compliance requirements or regulatory considerations]
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse the response
            analysis = self._parse_risk_analysis(response.text)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {
                'high_risks': [],
                'medium_risks': [],
                'recommendations': [],
                'compliance_notes': []
            }
    
    def compare_documents(self, text1: str, text2: str, doc1_name: str = "Document 1", doc2_name: str = "Document 2") -> Dict[str, Any]:
        """
        Compare two legal documents and highlight differences
        
        Args:
            text1: First document text
            text2: Second document text
            doc1_name: Name of first document
            doc2_name: Name of second document
            
        Returns:
            Dictionary containing comparison analysis
        """
        try:
            prompt = f"""
            Compare these two legal documents and provide a detailed analysis of their differences:
            
            {doc1_name}:
            {text1}
            
            {doc2_name}:
            {text2}
            
            Provide analysis in the following format:
            
            KEY DIFFERENCES:
            - [List major differences between documents]
            
            SIMILAR PROVISIONS:
            - [List similar or identical provisions]
            
            UNIQUE TO {doc1_name.upper()}:
            - [List provisions only in document 1]
            
            UNIQUE TO {doc2_name.upper()}:
            - [List provisions only in document 2]
            
            RECOMMENDATIONS:
            - [Recommendations based on the comparison]
            """
            
            response = self.model.generate_content(prompt)
            
            return {
                'comparison_text': response.text,
                'doc1_name': doc1_name,
                'doc2_name': doc2_name,
                'analysis_date': self._get_current_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Document comparison failed: {e}")
            return {
                'comparison_text': "Error comparing documents",
                'doc1_name': doc1_name,
                'doc2_name': doc2_name,
                'analysis_date': self._get_current_timestamp()
            }
    
    def _extract_key_points(self, summary_text: str) -> List[str]:
        """Extract key points from summary text"""
        key_points = []
        lines = summary_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered points, bullet points, or lines that start with key indicators
            if (line.startswith(('1.', '2.', '3.', '4.', '5.', '•', '-', '*')) or 
                'key' in line.lower() or 'important' in line.lower() or 'critical' in line.lower()):
                if len(line) > 10:  # Avoid very short lines
                    key_points.append(line)
        
        return key_points[:10]  # Limit to top 10 key points
    
    def _parse_risk_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse risk analysis response into structured format"""
        analysis = {
            'high_risks': [],
            'medium_risks': [],
            'recommendations': [],
            'compliance_notes': []
        }
        
        current_section = None
        lines = analysis_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Identify sections
            if 'HIGH RISK' in line.upper():
                current_section = 'high_risks'
            elif 'MEDIUM RISK' in line.upper():
                current_section = 'medium_risks'
            elif 'RECOMMENDATIONS' in line.upper():
                current_section = 'recommendations'
            elif 'COMPLIANCE' in line.upper():
                current_section = 'compliance_notes'
            elif line.startswith('-') and current_section:
                analysis[current_section].append(line[1:].strip())
        
        return analysis
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")