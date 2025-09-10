"""
Streamlit Web Application for Legal Document Analysis
Uses Google GenAI tools for document processing
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

# Import our custom modules
from src.document_processor import LegalDocumentProcessor
from src.config import Config

# Configure Streamlit page
st.set_page_config(
    page_title="Legal Document Analysis - Google GenAI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
.metric-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3498db;
    margin: 1rem 0;
}
.entity-category {
    background-color: #e8f4fd;
    padding: 0.5rem;
    margin: 0.5rem 0;
    border-radius: 0.3rem;
    border-left: 3px solid #3498db;
}
.risk-high {
    background-color: #ffe6e6;
    border-left: 3px solid #e74c3c;
    padding: 0.5rem;
    margin: 0.5rem 0;
    border-radius: 0.3rem;
}
.risk-medium {
    background-color: #fff3cd;
    border-left: 3px solid #f39c12;
    padding: 0.5rem;
    margin: 0.5rem 0;
    border-radius: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

def setup_api_keys():
    """Setup and validate API keys"""
    st.sidebar.header("🔑 API Configuration")
    
    # Get API key from environment or user input
    gemini_api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyAdmjoDR8vw9RZV_77JPlreXgOZRx5p5dc')
    
    if not gemini_api_key:
        gemini_api_key = st.sidebar.text_input(
            "Enter Gemini API Key:",
            type="password",
            help="Get your API key from Google AI Studio"
        )
    
    if gemini_api_key:
        os.environ['GEMINI_API_KEY'] = gemini_api_key
        
        try:
            if st.session_state.processor is None:
                st.session_state.processor = LegalDocumentProcessor()
            st.sidebar.success("✅ API Key configured successfully!")
            return True
        except Exception as e:
            st.sidebar.error(f"❌ API Key validation failed: {str(e)}")
            return False
    else:
        st.sidebar.warning("⚠️ Please enter your Gemini API Key to continue")
        return False

def file_upload_section():
    """Handle file upload section"""
    st.header("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose legal documents to analyze",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT. Max file size: 20MB"
    )
    
    return uploaded_files

def analysis_options_section():
    """Configure analysis options"""
    st.sidebar.header("⚙️ Analysis Options")
    
    # Summary options
    summary_type = st.sidebar.selectbox(
        "Summary Type:",
        ["comprehensive", "brief", "executive"],
        help="Choose the type of summary to generate"
    )
    
    # Entity extraction options  
    entity_extraction_type = st.sidebar.selectbox(
        "Entity Extraction:",
        ["comprehensive", "basic", "specific"],
        help="Choose the depth of entity extraction"
    )
    
    # Additional analysis options
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        analyze_risks = st.checkbox("Risk Analysis", value=True)
        generate_bullet_points = st.checkbox("Bullet Points", value=True)
    
    with col2:
        extract_relationships = st.checkbox("Legal Relationships", value=False)
        compare_documents = st.checkbox("Compare Docs", value=False)
    
    return {
        'summary_type': summary_type,
        'entity_extraction_type': entity_extraction_type,
        'analyze_risks': analyze_risks,
        'generate_bullet_points': generate_bullet_points,
        'extract_relationships': extract_relationships,
        'compare_documents': compare_documents
    }

def process_documents(uploaded_files, analysis_options):
    """Process uploaded documents"""
    if not uploaded_files:
        return None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Process the document
            result = st.session_state.processor.process_document(tmp_file_path, analysis_options)
            result['original_filename'] = uploaded_file.name
            results.append(result)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def display_document_summary(result):
    """Display document analysis summary"""
    st.markdown('<div class="section-header">📋 Document Summary</div>', unsafe_allow_html=True)
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", result.get('status', 'Unknown').title())
    
    if 'text_extraction' in result:
        text_data = result['text_extraction']
        
        with col2:
            st.metric("Pages", text_data.get('pages', 0))
        
        with col3:
            word_count = len(text_data.get('text', '').split())
            st.metric("Word Count", f"{word_count:,}")
        
        with col4:
            confidence = text_data.get('confidence', 0.0)
            st.metric("Extraction Confidence", f"{confidence:.1%}")

def display_summary_analysis(result):
    """Display summary analysis"""
    if 'summary' not in result:
        return
    
    summary_data = result['summary']
    
    st.markdown('<div class="section-header">📝 Document Analysis</div>', unsafe_allow_html=True)
    
    # Summary text
    st.subheader("Summary")
    st.write(summary_data.get('summary', 'No summary available'))
    
    # Key points
    if 'bullet_points' in result and result['bullet_points']:
        st.subheader("Key Points")
        for point in result['bullet_points'][:10]:  # Limit to 10 points
            st.write(f"• {point}")
    
    # Summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Original Length", f"{summary_data.get('original_length', 0):,} chars")
    
    with col2:
        compression_ratio = summary_data.get('compression_ratio', 0)
        st.metric("Compression Ratio", f"{compression_ratio:.1%}")

def display_entity_analysis(result):
    """Display entity extraction results"""
    if 'entities' not in result:
        return
    
    entity_data = result['entities']
    entities = entity_data.get('entities', {})
    
    if not entities:
        return
    
    st.markdown('<div class="section-header">🏷️ Named Entity Recognition</div>', unsafe_allow_html=True)
    
    # Entity statistics
    total_entities = entity_data.get('total_entities', 0)
    categories_found = len(entities)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Entities", total_entities)
    with col2:
        st.metric("Categories Found", categories_found)
    
    # Entity visualization
    if entities:
        entity_counts = {category: len(entity_list) for category, entity_list in entities.items()}
        
        fig = px.bar(
            x=list(entity_counts.keys()),
            y=list(entity_counts.values()),
            title="Entities by Category",
            labels={'x': 'Entity Category', 'y': 'Count'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed entity display
    st.subheader("Extracted Entities")
    
    # Create tabs for each entity category
    if entities:
        tabs = st.tabs(list(entities.keys()))
        
        for tab, (category, entity_list) in zip(tabs, entities.items()):
            with tab:
                if entity_list:
                    for i, entity in enumerate(entity_list, 1):
                        st.write(f"{i}. {entity}")
                else:
                    st.write("No entities found in this category")

def display_risk_analysis(result):
    """Display risk analysis results"""
    if 'risk_analysis' not in result:
        return
    
    risk_data = result['risk_analysis']
    
    st.markdown('<div class="section-header">⚠️ Risk Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("High Risk Areas")
        high_risks = risk_data.get('high_risks', [])
        if high_risks:
            for risk in high_risks:
                st.markdown(f'<div class="risk-high">🔴 {risk}</div>', unsafe_allow_html=True)
        else:
            st.write("No high-risk areas identified")
    
    with col2:
        st.subheader("Medium Risk Areas")
        medium_risks = risk_data.get('medium_risks', [])
        if medium_risks:
            for risk in medium_risks:
                st.markdown(f'<div class="risk-medium">🟡 {risk}</div>', unsafe_allow_html=True)
        else:
            st.write("No medium-risk areas identified")
    
    # Recommendations
    st.subheader("Recommendations")
    recommendations = risk_data.get('recommendations', [])
    if recommendations:
        for rec in recommendations:
            st.write(f"• {rec}")
    
    # Compliance notes
    compliance_notes = risk_data.get('compliance_notes', [])
    if compliance_notes:
        st.subheader("Compliance Notes")
        for note in compliance_notes:
            st.write(f"• {note}")

def display_relationships(result):
    """Display legal relationships"""
    if 'relationships' not in result:
        return
    
    relationship_data = result['relationships']
    relationships = relationship_data.get('relationships', {})
    
    if not relationships:
        return
    
    st.markdown('<div class="section-header">🔗 Legal Relationships</div>', unsafe_allow_html=True)
    
    for category, relationship_list in relationships.items():
        if relationship_list:
            st.subheader(category.replace('_', ' ').title())
            for relationship in relationship_list:
                st.write(f"• {relationship}")

def download_results(results):
    """Provide download options for results"""
    st.markdown('<div class="section-header">💾 Download Results</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON download
        json_data = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name="legal_analysis_results.json",
            mime="application/json"
        )
    
    with col2:
        # Summary report download
        if results and len(results) > 0:
            summary_text = ""
            for result in results:
                summary_text += f"Document: {result.get('original_filename', 'Unknown')}\n"
                summary_text += "="*50 + "\n"
                if 'summary' in result:
                    summary_text += result['summary'].get('summary', '') + "\n\n"
                if 'bullet_points' in result:
                    summary_text += "Key Points:\n"
                    for point in result['bullet_points']:
                        summary_text += f"• {point}\n"
                summary_text += "\n" + "="*50 + "\n\n"
            
            st.download_button(
                label="📋 Download Summary",
                data=summary_text,
                file_name="legal_analysis_summary.txt",
                mime="text/plain"
            )
    
    with col3:
        # Entity report download
        if results and len(results) > 0:
            entity_report = ""
            for result in results:
                if 'entities' in result:
                    entity_report += f"Document: {result.get('original_filename', 'Unknown')}\n"
                    entity_report += "="*50 + "\n"
                    entities = result['entities'].get('entities', {})
                    for category, entity_list in entities.items():
                        if entity_list:
                            entity_report += f"\n{category}:\n"
                            for entity in entity_list:
                                entity_report += f"• {entity}\n"
                    entity_report += "\n" + "="*50 + "\n\n"
            
            st.download_button(
                label="🏷️ Download Entities",
                data=entity_report,
                file_name="legal_analysis_entities.txt",
                mime="text/plain"
            )

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">⚖️ Legal Document Analysis</div>', unsafe_allow_html=True)
    st.markdown("### Powered by Google GenAI Tools")
    
    # Setup API keys
    api_configured = setup_api_keys()
    
    if not api_configured:
        st.warning("Please configure your Gemini API key in the sidebar to continue.")
        st.info("""
        To get started:
        1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enter the API key in the sidebar
        3. Upload your legal documents
        4. Configure analysis options
        5. Start analyzing!
        """)
        return
    
    # Analysis options
    analysis_options = analysis_options_section()
    
    # File upload
    uploaded_files = file_upload_section()
    
    # Process documents button
    if uploaded_files and st.button("🚀 Analyze Documents", type="primary"):
        with st.spinner("Processing documents..."):
            results = process_documents(uploaded_files, analysis_options)
            st.session_state.analysis_results = results
            st.session_state.processed_files = uploaded_files
    
    # Display results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.success(f"✅ Successfully processed {len(results)} document(s)!")
        
        # Create tabs for each document
        if len(results) == 1:
            # Single document - no tabs needed
            result = results[0]
            display_document_summary(result)
            display_summary_analysis(result)
            display_entity_analysis(result)
            display_risk_analysis(result)
            display_relationships(result)
        else:
            # Multiple documents - use tabs
            tab_names = [f"📄 {result.get('original_filename', f'Doc {i+1}')}" for i, result in enumerate(results)]
            tabs = st.tabs(tab_names)
            
            for tab, result in zip(tabs, results):
                with tab:
                    display_document_summary(result)
                    display_summary_analysis(result)
                    display_entity_analysis(result)
                    display_risk_analysis(result)
                    display_relationships(result)
        
        # Download section
        download_results(results)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Google GenAI Tools • Streamlit • Python<br>
        <small>For educational and research purposes only. Not intended as legal advice.</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()