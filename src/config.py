import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the Legal Document Analysis application"""
    
    # Google API Keys
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyAdmjoDR8vw9RZV_77JPlreXgOZRx5p5dc')
    GOOGLE_CLOUD_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT_ID', '')
    
    # Document AI Configuration
    DOCUMENT_AI_LOCATION = os.getenv('DOCUMENT_AI_LOCATION', 'us')  # 'us' or 'eu'
    DOCUMENT_AI_PROCESSOR_ID = os.getenv('DOCUMENT_AI_PROCESSOR_ID', '')
    
    # Gemini Model Configuration
    GEMINI_MODEL = 'gemini-1.5-pro'
    GEMINI_FLASH_MODEL = 'gemini-1.5-flash'
    
    # Application Settings
    MAX_FILE_SIZE_MB = 20
    SUPPORTED_FILE_TYPES = ['pdf', 'docx', 'txt']
    
    # Prompt Templates
    SUMMARIZATION_PROMPT = """
    Please provide a comprehensive summary of this legal document. 
    Focus on:
    1. Key legal provisions and clauses
    2. Important parties involved
    3. Main obligations and responsibilities
    4. Critical dates and deadlines
    5. Potential legal implications
    
    Document text: {text}
    """
    
    ENTITY_EXTRACTION_PROMPT = """
    Extract and categorize the following entities from this legal document:
    
    1. PERSONS: Names of individuals mentioned
    2. ORGANIZATIONS: Company names, institutions, government bodies
    3. LOCATIONS: Addresses, cities, states, countries
    4. DATES: Important dates and deadlines
    5. MONETARY_VALUES: Financial amounts, fees, penalties
    6. LEGAL_REFERENCES: Statutes, regulations, case law citations
    7. AGREEMENTS: Contract types, agreement names
    
    Return the results in JSON format with clear categorization.
    
    Document text: {text}
    """
    
    # Validation
    @classmethod
    def validate_config(cls):
        """Validate required configuration settings"""
        required_settings = [
            'GEMINI_API_KEY',
        ]
        
        missing = [setting for setting in required_settings if not getattr(cls, setting)]
        
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        
        return True
