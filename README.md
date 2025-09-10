# Legal Document Analysis - Google GenAI Edition

A comprehensive legal document analysis application built using Google's GenAI tools including Gemini API and Document AI, replacing the original BERT/FlanT5 implementation with modern Google cloud-based AI services.

## 🌟 Features

### Core Functionality
- **📄 Multi-format Document Support**: PDF, DOCX, and TXT files
- **🤖 AI-Powered Text Extraction**: Google Document AI for OCR and text extraction
- **📝 Intelligent Summarization**: Gemini API for comprehensive document summaries
- **🏷️ Named Entity Recognition**: Legal-specific entity extraction using Gemini
- **⚠️ Risk Analysis**: Automated legal risk assessment and recommendations
- **🔗 Relationship Mapping**: Extract legal relationships between entities
- **📊 Interactive Web Interface**: Modern Streamlit-based UI

### Analysis Types
- **Comprehensive Analysis**: Full document breakdown with all features
- **Brief Summaries**: Quick overviews for rapid document review
- **Executive Summaries**: Business-focused analysis for decision makers
- **Risk Assessment**: Identifies potential legal concerns and compliance issues
- **Entity Extraction**: People, organizations, dates, financial terms, and legal references

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))
- Optional: Google Cloud Project with Document AI enabled

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd legal_document_analysis_google_genai
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT_ID=your_project_id  # Optional
DOCUMENT_AI_LOCATION=us  # Optional: 'us' or 'eu'
DOCUMENT_AI_PROCESSOR_ID=your_processor_id  # Optional
```

4. **Run the application:**
```bash
streamlit run streamlit_app.py
```

5. **Open your browser to `http://localhost:8501`**

## 📁 Project Structure

```
legal_document_analysis_google_genai/
├── requirements.txt              # Python dependencies
├── config.py                    # Configuration settings
├── streamlit_app.py            # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── document_processor.py   # Main processing coordinator
│   ├── text_extractor.py      # Google Document AI integration
│   ├── summarizer.py          # Gemini-based summarization
│   ├── entity_extractor.py    # Gemini-based NER
│   └── utils.py               # Utility functions
├── notebooks/                  # Jupyter notebooks for development
├── sample_pdfs/               # Sample documents for testing
└── README.md                  # This file
```

## 🔧 Configuration Options

### Analysis Settings
- **Summary Type**: Choose between comprehensive, brief, or executive summaries
- **Entity Extraction**: Basic, comprehensive, or specific legal entity extraction
- **Risk Analysis**: Enable/disable automated legal risk assessment
- **Relationship Analysis**: Extract legal relationships between entities
- **Document Comparison**: Compare multiple documents side-by-side

### API Configuration
- **Gemini Model**: Default uses `gemini-1.5-pro` for comprehensive analysis
- **Flash Model**: Uses `gemini-1.5-flash` for faster, lighter operations
- **Document AI**: Optional integration for enhanced OCR capabilities

## 🎯 Usage Examples

### Command Line Usage (Python)
```python
from src.document_processor import LegalDocumentProcessor

# Initialize processor
processor = LegalDocumentProcessor()

# Process a single document
result = processor.process_document(
    file_path="sample_pdfs/contract.pdf",
    analysis_options={
        'summary_type': 'comprehensive',
        'entity_extraction_type': 'comprehensive',
        'analyze_risks': True,
        'generate_bullet_points': True
    }
)

# Print summary
print(result['summary']['summary'])

# Print extracted entities
for category, entities in result['entities']['entities'].items():
    print(f"{category}: {entities}")
```

### Web Interface Usage
1. Start the Streamlit app: `streamlit run streamlit_app.py`
2. Enter your Gemini API key in the sidebar
3. Upload legal documents (PDF, DOCX, or TXT)
4. Configure analysis options
5. Click "Analyze Documents"
6. Review results and download reports

## 📊 Output Formats

### Analysis Results Include:
- **Document Summary**: Comprehensive analysis of legal content
- **Key Points**: Bullet-point breakdown of important provisions
- **Named Entities**: Categorized extraction of people, organizations, dates, etc.
- **Risk Assessment**: High/medium risk areas with recommendations
- **Legal Relationships**: Contractual and legal connections between entities
- **Statistics**: Processing metrics and document insights

### Export Options:
- **JSON**: Complete analysis results in structured format
- **Text Report**: Human-readable summary document
- **Entity Report**: Detailed entity extraction results
- **CSV**: Tabular data for further analysis

## 🔄 Migration from Original Repository

This version replaces the original BERT/FlanT5 implementation with Google GenAI tools:

| Original Component | Google GenAI Replacement |
|-------------------|-------------------------|
| PyPDF2/pytesseract | Google Document AI OCR |
| FlanT5 Summarizer | Gemini API Summarization |
| BERT NER | Gemini API Entity Extraction |
| Custom Training | Pre-trained Google Models |

### Key Improvements:
- **No Model Training Required**: Uses pre-trained Google models
- **Better Accuracy**: State-of-the-art performance on legal documents
- **Scalable**: Cloud-based processing with enterprise-grade reliability
- **Multilingual**: Built-in support for multiple languages
- **Faster Setup**: No local model downloads or GPU requirements

## 💡 Advanced Features

### Document AI Integration
For enhanced OCR capabilities, optionally set up Google Document AI:

1. Enable Document AI API in Google Cloud Console
2. Create an OCR processor
3. Set environment variables for project ID and processor ID
4. The application will automatically use Document AI for better text extraction

### Batch Processing
Process multiple documents simultaneously:
```python
file_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
batch_results = processor.process_multiple_documents(
    file_paths, 
    analysis_options={'compare_documents': True}
)
```

### Custom Prompts
Modify analysis prompts in `config.py` to customize the analysis for specific legal domains or requirements.

## 📝 API Reference

### LegalDocumentProcessor
Main class for document processing.

**Methods:**
- `process_document(file_path, analysis_options)`: Process a single document
- `process_multiple_documents(file_paths, analysis_options)`: Process multiple documents
- `get_processing_statistics(results)`: Generate processing statistics

### GeminiSummarizer
Handles document summarization using Gemini API.

**Methods:**
- `summarize_document(text, summary_type)`: Generate document summary
- `generate_bullet_points(text)`: Extract key points
- `analyze_legal_risks(text)`: Assess legal risks
- `compare_documents(text1, text2)`: Compare two documents

### GeminiEntityExtractor
Manages named entity recognition using Gemini API.

**Methods:**
- `extract_entities(text, extraction_type)`: Extract named entities
- `extract_legal_relationships(text)`: Find legal relationships
- `validate_entity_consistency(entities)`: Validate extracted entities

## 🛠️ Development

### Setting Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/
```

### Adding Custom Features
1. Extend the relevant classes in `src/`
2. Update configuration in `config.py`
3. Modify the Streamlit interface as needed
4. Test with sample documents

## 📋 Troubleshooting

### Common Issues

**API Key Errors:**
- Ensure your Gemini API key is valid and has sufficient quota
- Check that the API key environment variable is set correctly

**Document Processing Errors:**
- Verify file format is supported (PDF, DOCX, TXT)
- Check file size is under 20MB limit
- Ensure document is not corrupted or password-protected

**Installation Issues:**
- Use Python 3.8 or higher
- Install in a virtual environment to avoid conflicts
- Check that all dependencies are installed correctly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google AI team for the powerful GenAI tools
- Original repository contributors for the foundation
- Streamlit team for the excellent web framework
- Legal tech community for feedback and suggestions

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the Google GenAI documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Disclaimer**: This application is for educational and research purposes only. It is not intended to provide legal advice. Always consult with qualified legal professionals for legal matters.