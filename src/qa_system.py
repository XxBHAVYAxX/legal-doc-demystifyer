"""
Smart Search and Q&A Module using Gemini API
Provides document-specific question answering and search capabilities
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import google.generativeai as genai
from .config import Config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiQASystem:
    """Document Q&A and smart search using Google Gemini API"""
    
    def __init__(self):
        self.config = Config()
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.config.GEMINI_MODEL)
        self.flash_model = genai.GenerativeModel(self.config.GEMINI_FLASH_MODEL)
        
        # Predefined question categories for legal documents
        self.question_categories = {
            'parties': [
                "Who are the parties to this agreement?",
                "What are the roles of each party?",
                "Who is the primary contractor/client?"
            ],
            'terms': [
                "What is the duration of this agreement?",
                "When does this contract expire?",
                "What are the key dates and deadlines?"
            ],
            'payment': [
                "What are the payment terms?",
                "How much will be paid and when?",
                "Are there any penalty fees mentioned?"
            ],
            'termination': [
                "How can this agreement be terminated?",
                "What notice period is required for termination?",
                "What happens upon termination?"
            ],
            'obligations': [
                "What are the main obligations of each party?",
                "What deliverables are expected?",
                "What are the performance requirements?"
            ],
            'legal': [
                "What law governs this agreement?",
                "How are disputes resolved?",
                "Are there any liability limitations?"
            ]
        }
    
    def answer_question(self, document_text: str, question: str, context_clauses: List[Dict] = None) -> Dict[str, Any]:
        """
        Answer a specific question about the document
        
        Args:
            document_text: Full document text
            question: User's question
            context_clauses: Optional list of relevant clauses for context
            
        Returns:
            Dictionary containing answer and supporting information
        """
        try:
            # Build context from clauses if provided
            context_info = ""
            if context_clauses:
                context_info = "\n\nRelevant clauses for additional context:\n"
                for clause in context_clauses[:3]:  # Limit to 3 most relevant
                    context_info += f"- {clause.get('clause_type', '')}: {clause.get('clause_text', '')}\n"
            
            prompt = self._build_qa_prompt(document_text, question, context_info)
            response = self.model.generate_content(prompt)
            
            answer_data = {
                'question': question,
                'answer': response.text.strip(),
                'confidence': self._estimate_confidence(response.text),
                'relevant_sections': self._extract_relevant_sections(document_text, question, response.text),
                'context_clauses_used': len(context_clauses) if context_clauses else 0
            }
            
            logger.info(f"Successfully answered question: {question[:50]}...")
            return answer_data
            
        except Exception as e:
            logger.error(f"Q&A failed for question '{question}': {e}")
            return {
                'question': question,
                'answer': f"Sorry, I couldn't answer this question. Error: {str(e)}",
                'confidence': 0.0,
                'relevant_sections': [],
                'context_clauses_used': 0
            }
    
    def _build_qa_prompt(self, document_text: str, question: str, context_info: str = "") -> str:
        """Build the prompt for question answering"""
        
        prompt = f"""
        You are a legal document analyst. Answer the following question about this legal document accurately and concisely.

        Document text:
        {document_text}
        {context_info}

        Question: {question}

        Instructions:
        1. Provide a clear, direct answer based on the document content
        2. Quote specific text from the document when possible
        3. If the information is not in the document, clearly state "This information is not specified in the document"
        4. Focus on factual information from the document, not legal advice
        5. If multiple interpretations are possible, mention the key alternatives

        Answer:
        """
        
        return prompt
    
    def search_document(self, document_text: str, search_query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform intelligent search within the document
        
        Args:
            document_text: Full document text
            search_query: Search query or keywords
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results
        """
        try:
            prompt = f"""
            Search through this legal document for information related to: "{search_query}"

            Document text:
            {document_text}

            Find and extract up to {max_results} relevant sections or paragraphs that relate to the search query.
            For each result, provide:
            1. The relevant text excerpt (keep it concise but meaningful)
            2. A brief explanation of why this section is relevant
            3. The context or section where it appears (if identifiable)

            Format as:
            **Result 1:**
            Text: [relevant excerpt]
            Relevance: [explanation]
            Context: [section/paragraph info]

            **Result 2:**
            [continue format...]

            If no relevant information is found, state "No relevant information found for this search query."
            """
            
            response = self.flash_model.generate_content(prompt)  # Use flash for faster search
            
            # Parse search results
            search_results = self._parse_search_results(response.text, search_query)
            
            search_data = {
                'query': search_query,
                'results': search_results,
                'total_results': len(search_results),
                'document_length': len(document_text)
            }
            
            logger.info(f"Search completed for '{search_query}': found {len(search_results)} results")
            return search_data
            
        except Exception as e:
            logger.error(f"Document search failed for '{search_query}': {e}")
            return {
                'query': search_query,
                'results': [],
                'total_results': 0,
                'error': str(e),
                'document_length': len(document_text)
            }
    
    def _parse_search_results(self, response_text: str, query: str) -> List[Dict[str, Any]]:
        """Parse search results from Gemini response"""
        results = []
        
        try:
            # Split by result markers
            sections = response_text.split('**Result')
            
            for section in sections[1:]:  # Skip first empty section
                result_data = {'text': '', 'relevance': '', 'context': '', 'query': query}
                
                lines = section.strip().split('\n')
                current_field = None
                
                for line in lines:
                    line = line.strip()
                    
                    if line.startswith('Text:'):
                        current_field = 'text'
                        result_data['text'] = line.replace('Text:', '').strip()
                    elif line.startswith('Relevance:'):
                        current_field = 'relevance'
                        result_data['relevance'] = line.replace('Relevance:', '').strip()
                    elif line.startswith('Context:'):
                        current_field = 'context'
                        result_data['context'] = line.replace('Context:', '').strip()
                    elif current_field and line:
                        # Continue previous field
                        result_data[current_field] += ' ' + line
                
                # Only add if we have meaningful text
                if result_data['text'] and len(result_data['text']) > 10:
                    results.append(result_data)
        
        except Exception as e:
            logger.warning(f"Failed to parse search results: {e}")
        
        return results[:5]  # Limit to 5 results
    
    def get_suggested_questions(self, document_text: str, document_type: str = "contract") -> List[str]:
        """
        Generate suggested questions based on document content
        
        Args:
            document_text: Document text to analyze
            document_type: Type of document (contract, agreement, etc.)
            
        Returns:
            List of suggested questions
        """
        try:
            prompt = f"""
            Analyze this legal document and suggest 8-10 important questions that someone might want to ask about it.

            Document text:
            {document_text[:2000]}...  # First 2000 characters for analysis

            Based on the content, suggest practical questions that would help someone understand:
            - Key terms and conditions
            - Important dates and deadlines  
            - Responsibilities and obligations
            - Financial terms
            - Legal implications

            Return as a simple numbered list of questions.
            """
            
            response = self.flash_model.generate_content(prompt)
            
            # Extract questions from response
            suggested_questions = []
            lines = response.text.split('\n')
            
            for line in lines:
                line = line.strip()
                # Look for numbered questions
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Clean up the question
                    question = line.split('.', 1)[-1].strip() if '.' in line else line.strip()
                    question = question.lstrip('- •').strip()
                    
                    if question and len(question) > 10 and question.endswith('?'):
                        suggested_questions.append(question)
            
            # Add category-based questions if we don't have enough
            if len(suggested_questions) < 6:
                category_questions = []
                for category, questions in self.question_categories.items():
                    category_questions.extend(questions[:2])  # Take 2 from each category
                
                # Fill up to 8 total questions
                for q in category_questions:
                    if len(suggested_questions) < 8 and q not in suggested_questions:
                        suggested_questions.append(q)
            
            return suggested_questions[:10]  # Limit to 10 questions
            
        except Exception as e:
            logger.error(f"Failed to generate suggested questions: {e}")
            # Return default questions
            return [
                "Who are the parties to this agreement?",
                "What are the main terms and conditions?",
                "What are the payment obligations?",
                "How can this agreement be terminated?",
                "What are the key dates and deadlines?",
                "What law governs this agreement?"
            ]
    
    def _estimate_confidence(self, answer_text: str) -> float:
        """Estimate confidence in the answer based on response characteristics"""
        if not answer_text:
            return 0.0
        
        # Simple heuristic for confidence estimation
        confidence = 0.8  # Base confidence
        
        # Lower confidence for uncertain language
        uncertain_phrases = ['might', 'could', 'possibly', 'unclear', 'not specified', 'not mentioned']
        for phrase in uncertain_phrases:
            if phrase in answer_text.lower():
                confidence -= 0.2
        
        # Higher confidence for specific quotes and references
        if '"' in answer_text or 'section' in answer_text.lower():
            confidence += 0.1
        
        # Ensure confidence stays within bounds
        return max(0.1, min(1.0, confidence))
    
    def _extract_relevant_sections(self, document_text: str, question: str, answer: str) -> List[str]:
        """Extract relevant document sections that support the answer"""
        sections = []
        
        try:
            # Simple approach: find sentences in the document that appear in the answer
            doc_sentences = [s.strip() for s in document_text.split('.') if len(s.strip()) > 20]
            answer_words = set(answer.lower().split())
            
            for sentence in doc_sentences:
                sentence_words = set(sentence.lower().split())
                # If there's significant overlap between answer and document sentence
                overlap = len(answer_words.intersection(sentence_words))
                if overlap >= 3:  # At least 3 words in common
                    sections.append(sentence.strip()[:200] + '...' if len(sentence) > 200 else sentence.strip())
                
                if len(sections) >= 3:  # Limit to 3 sections
                    break
        
        except Exception as e:
            logger.warning(f"Failed to extract relevant sections: {e}")
        
        return sections
    
    def batch_answer_questions(self, document_text: str, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Answer multiple questions about the document
        
        Args:
            document_text: Document text
            questions: List of questions to answer
            
        Returns:
            List of answer dictionaries
        """
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"Answering question {i+1}/{len(questions)}")
            answer_data = self.answer_question(document_text, question)
            answers.append(answer_data)
        
        return answers