"""
Enhanced Streamlit MVP App with Clause Extraction and Smart Q&A
Features clause highlighting and interactive document Q&A
"""

import streamlit as st
import pandas as pd
import json
import os
from io import BytesIO
from pathlib import Path
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import enhanced modules
from src.enhanced_processor import EnhancedLegalDocumentProcessor
from src.config import Config

# Configure Streamlit page
st.set_page_config(
    page_title="Legal Document Analysis MVP - Enhanced",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.clause-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3498db;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.clause-high {
    border-left-color: #e74c3c;
    background-color: #fdf2f2;
}
.clause-medium {
    border-left-color: #f39c12;
    background-color: #fef9f3;
}
.clause-low {
    border-left-color: #27ae60;
    background-color: #f1f8f4;
}
.qa-question {
    background-color: #e8f4fd;
    padding: 0.8rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    cursor: pointer;
    border: 1px solid #bee5eb;
}
.qa-question:hover {
    background-color: #d4edda;
    border-color: #c3e6cb;
}
.qa-answer {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
}
.search-result {
    background-color: #fff3cd;
    padding: 0.8rem;
    border-radius: 0.5rem;
    border-left: 3px solid #ffc107;
    margin: 0.5rem 0;
}
.highlighted-text {
    max-height: 400px;
    overflow-y: auto;
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""

def setup_api_keys():
    """Setup and validate API keys"""
    st.sidebar.header("🔑 API Configuration")
    
    gemini_api_key = os.getenv('GEMINI_API_KEY', '')
    
    if not gemini_api_key:
        gemini_api_key = st.sidebar.text_input(
            "Enter Gemini API Key:",
            type="password",
            help="Get your API key from Google AI Studio"
        )
    
    if gemini_api_key:
        os.environ['AIzaSyAdmjoDR8vw9RZV_77JPlreXgOZRx5p5dc'] = gemini_api_key
        
        try:
            if st.session_state.processor is None:
                st.session_state.processor = EnhancedLegalDocumentProcessor()
            st.sidebar.success("✅ API Key configured successfully!")
            return True
        except Exception as e:
            st.sidebar.error(f"❌ API Key validation failed: {str(e)}")
            return False
    else:
        st.sidebar.warning("⚠️ Please enter your Gemini API Key to continue")
        return False

def analysis_options_section():
    """Configure analysis options"""
    st.sidebar.header("⚙️ Analysis Options")
    
    # Core options
    summary_type = st.sidebar.selectbox(
        "Summary Type:",
        ["comprehensive", "brief", "executive"],
        help="Choose the type of summary to generate"
    )
    
    # Enhanced options
    extract_clauses = st.sidebar.checkbox("Extract Legal Clauses", value=True)
    generate_qa = st.sidebar.checkbox("Generate Q&A Suggestions", value=True)
    
    # Clause type selection
    clause_types = []
    if extract_clauses:
        with st.sidebar.expander("Select Clause Types"):
            clause_options = {
                'PAYMENT': st.checkbox("Payment Terms", value=True),
                'TERMINATION': st.checkbox("Termination Clauses", value=True),
                'CONFIDENTIALITY': st.checkbox("Confidentiality", value=True),
                'INDEMNIFICATION': st.checkbox("Indemnification", value=True),
                'INTELLECTUAL_PROPERTY': st.checkbox("IP Rights", value=False),
                'GOVERNING_LAW': st.checkbox("Governing Law", value=True)
            }
            clause_types = [k for k, v in clause_options.items() if v]
    
    return {
        'summary_type': summary_type,
        'extract_clauses': extract_clauses,
        'generate_qa_suggestions': generate_qa,
        'clause_types': clause_types if clause_types else None,
        'extract_entities': True,
        'analyze_risks': True
    }

def file_upload_section():
    """Handle file upload section"""
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a legal document to analyze",
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, DOCX, TXT. Max file size: 20MB"
    )
    
    return uploaded_file

def process_document(uploaded_file, analysis_options):
    """Process uploaded document"""
    if not uploaded_file:
        return None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        status_text.text("Processing document...")
        progress_bar.progress(0.2)
        
        # Process the document
        result = st.session_state.processor.process_document(tmp_file_path, analysis_options)
        progress_bar.progress(1.0)
        
        # Store document text for Q&A
        if 'qa_ready_text' in result:
            st.session_state.document_text = result['qa_ready_text']
        elif 'text_extraction' in result:
            st.session_state.document_text = result['text_extraction'].get('text', '')[:10000]
        
        result['original_filename'] = uploaded_file.name
        
        progress_bar.empty()
        status_text.empty()
        return result
        
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)

def display_clause_analysis(result):
    """Display clause extraction results with highlighting"""
    if 'clauses' not in result or not result['clauses'].get('clauses'):
        return
    
    st.markdown('<div class="section-header">📋 Legal Clause Analysis</div>', unsafe_allow_html=True)
    
    clauses = result['clauses']['clauses']
    clause_summary = result.get('clause_summary', {})
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Clauses", len(clauses))
    with col2:
        st.metric("High Importance", clause_summary.get('high_importance', 0))
    with col3:
        clause_types_found = len(set(c.get('clause_type') for c in clauses))
        st.metric("Clause Types", clause_types_found)
    
    # Clause distribution chart
    if clause_summary.get('clause_distribution'):
        fig = px.bar(
            x=list(clause_summary['clause_distribution'].keys()),
            y=list(clause_summary['clause_distribution'].values()),
            title="Clause Distribution",
            labels={'x': 'Clause Type', 'y': 'Count'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual clause cards
    st.subheader("Extracted Clauses")
    
    for i, clause in enumerate(clauses):
        importance = clause.get('importance', 'MEDIUM').upper()
        clause_type = clause.get('clause_type', 'Unknown')
        clause_text = clause.get('clause_text', '')
        context = clause.get('context', '')
        section = clause.get('section', 'Unknown')
        
        # Style based on importance
        card_class = f"clause-card clause-{importance.lower()}"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4>{clause_type.replace('_', ' ').title()} [{importance}]</h4>
            <p><strong>Section:</strong> {section}</p>
            <p><strong>Context:</strong> {context}</p>
            <p><strong>Text:</strong> {clause_text[:300]}{'...' if len(clause_text) > 300 else ''}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show highlighted text if available
    if 'highlighted_text' in result:
        with st.expander("View Highlighted Document"):
            st.markdown(
                f'<div class="highlighted-text">{result["highlighted_text"]}</div>',
                unsafe_allow_html=True
            )

def display_qa_section(result):
    """Display Q&A section with suggested questions and search"""
    st.markdown('<div class="section-header">🤔 Smart Q&A and Search</div>', unsafe_allow_html=True)
    
    # Create two columns for Q&A and search
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Ask Questions")
        
        # Custom question input
        user_question = st.text_input("Ask a question about this document:", key="user_question")
        
        if st.button("Get Answer", key="ask_button") and user_question and st.session_state.document_text:
            with st.spinner("Finding answer..."):
                # Get relevant clauses for context
                context_clauses = []
                if 'clauses' in result and result['clauses'].get('clauses'):
                    context_clauses = st.session_state.processor.get_clause_context_for_question(
                        result['clauses']['clauses'], user_question
                    )
                
                # Get answer
                answer_data = st.session_state.processor.answer_question(
                    st.session_state.document_text, user_question, context_clauses
                )
                
                # Store in history
                st.session_state.qa_history.append(answer_data)
        
        # Suggested questions
        if 'suggested_questions' in result:
            st.subheader("Suggested Questions")
            for i, question in enumerate(result['suggested_questions'][:6]):
                if st.button(question, key=f"suggested_{i}"):
                    with st.spinner("Finding answer..."):
                        # Get relevant clauses
                        context_clauses = []
                        if 'clauses' in result and result['clauses'].get('clauses'):
                            context_clauses = st.session_state.processor.get_clause_context_for_question(
                                result['clauses']['clauses'], question
                            )
                        
                        answer_data = st.session_state.processor.answer_question(
                            st.session_state.document_text, question, context_clauses
                        )
                        st.session_state.qa_history.append(answer_data)
                        st.experimental_rerun()
    
    with col2:
        st.subheader("Smart Search")
        
        # Search input
        search_query = st.text_input("Search the document:", key="search_query")
        
        if st.button("Search", key="search_button") and search_query and st.session_state.document_text:
            with st.spinner("Searching..."):
                search_results = st.session_state.processor.search_document(
                    st.session_state.document_text, search_query
                )
                
                if search_results['total_results'] > 0:
                    st.write(f"Found {search_results['total_results']} results:")
                    
                    for i, search_result in enumerate(search_results['results']):
                        st.markdown(f"""
                        <div class="search-result">
                            <h5>Result {i+1}</h5>
                            <p><strong>Text:</strong> {search_result.get('text', '')}</p>
                            <p><strong>Relevance:</strong> {search_result.get('relevance', '')}</p>
                            <p><strong>Context:</strong> {search_result.get('context', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No relevant results found.")
    
    # Display Q&A history
    if st.session_state.qa_history:
        st.subheader("Q&A History")
        
        for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5
            with st.expander(f"Q: {qa['question'][:60]}..."):
                st.markdown(f"""
                <div class="qa-answer">
                    <p><strong>Question:</strong> {qa['question']}</p>
                    <p><strong>Answer:</strong> {qa['answer']}</p>
                    <p><strong>Confidence:</strong> {qa.get('confidence', 0.0):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if qa.get('relevant_sections'):
                    st.write("**Supporting sections:**")
                    for section in qa['relevant_sections']:
                        st.write(f"• {section}")

def display_enhanced_summary(result):
    """Display enhanced summary with key insights"""
    if 'summary' not in result:
        return
    
    st.markdown('<div class="section-header">📝 Document Summary</div>', unsafe_allow_html=True)
    
    summary_data = result['summary']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'text_extraction' in result:
            word_count = len(result['text_extraction'].get('text', '').split())
            st.metric("Word Count", f"{word_count:,}")
    
    with col2:
        st.metric("Summary Type", summary_data.get('summary_type', 'Unknown').title())
    
    with col3:
        compression = summary_data.get('compression_ratio', 0)
        st.metric("Compression", f"{compression:.1%}")
    
    with col4:
        if 'clauses' in result:
            clause_count = len(result['clauses'].get('clauses', []))
            st.metric("Clauses Found", clause_count)
    
    # Summary text
    st.write(summary_data.get('summary', 'No summary available'))
    
    # Key points if available
    if 'bullet_points' in result and result['bullet_points']:
        with st.expander("Key Points"):
            for point in result['bullet_points'][:8]:
                st.write(f"• {point}")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">⚖️ Legal Document Analysis MVP</div>', unsafe_allow_html=True)
    st.markdown("### Enhanced with Clause Extraction & Smart Q&A")
    
    # Setup API keys
    api_configured = setup_api_keys()
    
    if not api_configured:
        st.warning("Please configure your Gemini API key in the sidebar to continue.")
        st.info("""
        **MVP Features:**
        - 📋 **Legal Clause Extraction**: Automatically identifies and categorizes key legal clauses
        - 🤔 **Smart Q&A**: Ask questions about your document and get AI-powered answers
        - 🔍 **Intelligent Search**: Search through document content with context-aware results
        - 🎯 **Clause Highlighting**: Visual highlighting of important contract provisions
        - 📊 **Enhanced Analytics**: Comprehensive analysis with interactive visualizations
        
        Get started by entering your Gemini API key in the sidebar!
        """)
        return
    
    # Analysis options
    analysis_options = analysis_options_section()
    
    # File upload
    uploaded_file = file_upload_section()
    
    # Process document
    if uploaded_file and st.button("🚀 Analyze Document", type="primary"):
        with st.spinner("Processing document with enhanced AI analysis..."):
            result = process_document(uploaded_file, analysis_options)
            st.session_state.analysis_results = result
    
    # Display results
    if st.session_state.analysis_results:
        result = st.session_state.analysis_results
        
        if result.get('status') == 'completed':
            st.success("✅ Document analysis completed!")
            
            # Display enhanced summary
            display_enhanced_summary(result)
            
            # Display clause analysis
            display_clause_analysis(result)
            
            # Display Q&A section
            display_qa_section(result)
            
            # Download section
            st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON download
                json_data = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📄 Download Complete Analysis (JSON)",
                    data=json_data,
                    file_name=f"legal_analysis_{uploaded_file.name}.json",
                    mime="application/json"
                )
            
            with col2:
                # Generate comprehensive report
                if st.button("📋 Generate Comprehensive Report"):
                    with st.spinner("Generating report..."):
                        report = st.session_state.processor.generate_comprehensive_report(result)
                        st.download_button(
                            label="📋 Download Report",
                            data=report,
                            file_name=f"legal_report_{uploaded_file.name}.txt",
                            mime="text/plain"
                        )
        
        else:
            st.error(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")
    
    # Clear history button
    if st.sidebar.button("🗑️ Clear Q&A History"):
        st.session_state.qa_history = []
        st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <strong>MVP Features:</strong> Clause Extraction • Smart Q&A • Document Search<br>
        Built with ❤️ using Google GenAI Tools • Streamlit • Python<br>
        <small>Perfect for hackathons and legal document analysis</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()