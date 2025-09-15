"""
Gemini API-based Entity Extraction Module
Replaces BERT NER model with Google Gemini for legal named entity recognition
"""

import json
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from .config import Config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiEntityExtractor:
    """Legal named entity recognition using Google Gemini API"""
    
    def __init__(self):
        self.config = Config()
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
        self.flash_model = genai.GenerativeModel(self.config.GEMINI_FLASH_MODEL)
        
        # Define legal entity categories
        self.entity_categories = {
            'PERSONS': 'Names of individuals, lawyers, judges, witnesses',
            'ORGANIZATIONS': 'Company names, law firms, courts, government bodies',
            'LOCATIONS': 'Addresses, cities, states, countries, jurisdictions',
            'DATES': 'Important dates, deadlines, effective dates',
            'MONETARY_VALUES': 'Financial amounts, fees, damages, penalties',
            'LEGAL_REFERENCES': 'Statutes, regulations, case law, legal codes',
            'AGREEMENTS': 'Contract types, agreement names, legal instruments',
            'LEGAL_CONCEPTS': 'Legal terms, causes of action, legal principles'
        }
    
    def extract_entities(self, text: str, extraction_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Extract named entities from legal document text
        
        Args:
            text: The document text to analyze
            extraction_type: Type of extraction ('comprehensive', 'basic', 'specific')
            
        Returns:
            Dictionary containing extracted entities organized by category
        """
        try:
            prompt = self._get_entity_extraction_prompt(text, extraction_type)
            
            # Use flash model for basic extraction, regular model for comprehensive
            model = self.flash_model if extraction_type == "basic" else self.model
            
            response = model.generate_content(prompt)
            
            # Parse the JSON response
            entities = self._parse_entity_response(response.text)
            
            # Add metadata
            entity_data = {
                'entities': entities,
                'extraction_type': extraction_type,
                'total_entities': sum(len(category_entities) for category_entities in entities.values()),
                'categories_found': list(entities.keys()),
                'text_length': len(text)
            }
            
            logger.info(f"Successfully extracted {entity_data['total_entities']} entities")
            return entity_data
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {
                'entities': {},
                'extraction_type': extraction_type,
                'total_entities': 0,
                'categories_found': [],
                'text_length': len(text)
            }
    
    def _get_entity_extraction_prompt(self, text: str, extraction_type: str) -> str:
        """Generate appropriate prompt for entity extraction"""
        
        base_prompt = f"""
        Extract named entities from this legal document text. Return the results in valid JSON format.
        
        Document text:
        {text}
        
        """
        
        if extraction_type == "basic":
            categories = ['PERSONS', 'ORGANIZATIONS', 'DATES', 'LOCATIONS']
            instruction = f"""
            Extract entities in these categories: {', '.join(categories)}
            
            Return in this JSON format:
            {{
                "PERSONS": ["name1", "name2"],
                "ORGANIZATIONS": ["org1", "org2"],
                "DATES": ["date1", "date2"],
                "LOCATIONS": ["location1", "location2"]
            }}
            """
        
        elif extraction_type == "specific":
            instruction = """
            Focus on extracting:
            1. Contract parties and signatories
            2. Legal deadlines and effective dates
            3. Financial amounts and monetary terms
            4. Legal references and citations
            
            Return in JSON format with these categories:
            {
                "CONTRACT_PARTIES": ["party1", "party2"],
                "LEGAL_DATES": ["date1", "date2"],
                "FINANCIAL_TERMS": ["amount1", "amount2"],
                "LEGAL_CITATIONS": ["citation1", "citation2"]
            }
            """
        
        else:  # comprehensive
            instruction = f"""
            Extract entities in ALL these categories: {', '.join(self.entity_categories.keys())}
            
            Category definitions:
            {chr(10).join([f"- {cat}: {desc}" for cat, desc in self.entity_categories.items()])}
            
            Return in this JSON format:
            {{
                "PERSONS": ["name1", "name2"],
                "ORGANIZATIONS": ["org1", "org2"],
                "LOCATIONS": ["location1", "location2"],
                "DATES": ["date1", "date2"],
                "MONETARY_VALUES": ["amount1", "amount2"],
                "LEGAL_REFERENCES": ["ref1", "ref2"],
                "AGREEMENTS": ["agreement1", "agreement2"],
                "LEGAL_CONCEPTS": ["concept1", "concept2"]
            }}
            """
        
        return base_prompt + instruction
    
    def _parse_entity_response(self, response_text: str) -> Dict[str, List[str]]:
        """Parse JSON response from Gemini and clean up entities"""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                entities = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                entities = json.loads(response_text)
            
            # Clean and validate entities
            cleaned_entities = {}
            for category, entity_list in entities.items():
                if isinstance(entity_list, list):
                    # Remove duplicates and empty strings, limit length
                    clean_list = list(set([
                        entity.strip() 
                        for entity in entity_list 
                        if entity and len(entity.strip()) > 1
                    ]))[:20]  # Limit to 20 entities per category
                    
                    if clean_list:
                        cleaned_entities[category] = clean_list
            
            return cleaned_entities
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, attempting fallback parsing")
            return self._fallback_entity_parsing(response_text)
        except Exception as e:
            logger.error(f"Entity parsing failed: {e}")
            return {}
    
    def _fallback_entity_parsing(self, response_text: str) -> Dict[str, List[str]]:
        """Fallback method to extract entities when JSON parsing fails"""
        entities = {}
        current_category = None
        
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if this line defines a category
            for category in self.entity_categories.keys():
                if category in line.upper() and ':' in line:
                    current_category = category
                    entities[category] = []
                    break
            
            # Check if this line contains an entity (starts with -, *, or number)
            if current_category and (line.startswith('-') or line.startswith('*') or 
                                   any(line.startswith(f"{i}.") for i in range(1, 20))):
                entity = line.lstrip('-*0123456789. ').strip()
                if entity and len(entity) > 1:
                    entities[current_category].append(entity)
        
        return entities
    
    def extract_legal_relationships(self, text: str) -> Dict[str, Any]:
        """
        Extract relationships between legal entities
        
        Args:
            text: The document text to analyze
            
        Returns:
            Dictionary containing relationship analysis
        """
        try:
            prompt = f"""
            Analyze this legal document and identify key relationships between entities.
            
            Document text: {text}
            
            Identify:
            1. CONTRACTUAL_RELATIONSHIPS: Who has agreements with whom
            2. LEGAL_OBLIGATIONS: Who owes what to whom
            3. AUTHORITY_RELATIONSHIPS: Who has authority over whom
            4. FINANCIAL_RELATIONSHIPS: Who pays what to whom
            
            Return as JSON with relationship descriptions:
            {{
                "CONTRACTUAL_RELATIONSHIPS": ["Party A contracts with Party B for services"],
                "LEGAL_OBLIGATIONS": ["Company X must deliver Y by date Z"],
                "AUTHORITY_RELATIONSHIPS": ["Court has jurisdiction over the matter"],
                "FINANCIAL_RELATIONSHIPS": ["Buyer pays $X to Seller"]
            }}
            """
            
            response = self.model.generate_content(prompt)
            relationships = self._parse_entity_response(response.text)
            
            return {
                'relationships': relationships,
                'analysis_type': 'legal_relationships',
                'total_relationships': sum(len(rel_list) for rel_list in relationships.values())
            }
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return {
                'relationships': {},
                'analysis_type': 'legal_relationships',
                'total_relationships': 0
            }
    
    def validate_entity_consistency(self, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Validate consistency of extracted entities
        
        Args:
            entities: Dictionary of extracted entities
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'consistent_entities': [],
            'potential_duplicates': [],
            'formatting_issues': [],
            'completeness_score': 0.0
        }
        
        try:
            # Check for potential duplicates across categories
            all_entities = []
            for category, entity_list in entities.items():
                all_entities.extend(entity_list)
            
            # Find potential duplicates (similar strings)
            potential_dupes = []
            for i, entity1 in enumerate(all_entities):
                for j, entity2 in enumerate(all_entities[i+1:], i+1):
                    if self._similarity_score(entity1.lower(), entity2.lower()) > 0.8:
                        potential_dupes.append((entity1, entity2))
            
            validation_results['potential_duplicates'] = potential_dupes
            
            # Check formatting issues
            formatting_issues = []
            for category, entity_list in entities.items():
                for entity in entity_list:
                    if len(entity) < 2:
                        formatting_issues.append(f"Too short: {entity}")
                    elif len(entity) > 100:
                        formatting_issues.append(f"Too long: {entity[:50]}...")
                    elif entity.isupper() or entity.islower():
                        formatting_issues.append(f"Capitalization issue: {entity}")
            
            validation_results['formatting_issues'] = formatting_issues
            
            # Calculate completeness score
            expected_categories = len(self.entity_categories)
            found_categories = len([cat for cat in entities.keys() if entities[cat]])
            validation_results['completeness_score'] = found_categories / expected_categories
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Entity validation failed: {e}")
            return validation_results
    
    def _similarity_score(self, str1: str, str2: str) -> float:
        """Calculate similarity score between two strings"""
        # Simple Jaccard similarity
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def generate_entity_report(self, entities: Dict[str, List[str]]) -> str:
        """
        Generate a formatted report of extracted entities
        
        Args:
            entities: Dictionary of extracted entities
            
        Returns:
            Formatted string report
        """
        report_lines = ["# Legal Document Entity Analysis Report", ""]
        
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        report_lines.append(f"**Total Entities Found:** {total_entities}")
        report_lines.append("")
        
        for category, entity_list in entities.items():
            if entity_list:
                report_lines.append(f"## {category.replace('_', ' ').title()} ({len(entity_list)})")
                for i, entity in enumerate(entity_list, 1):
                    report_lines.append(f"{i}. {entity}")
                report_lines.append("")
        
        return "\n".join(report_lines)