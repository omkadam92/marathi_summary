import re
import networkx as nx
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import tempfile
import os
from pathlib import Path
import pandas as pd

class MarathiSummarizer:
    def __init__(self):
        self.stop_words = [
            'आहे', 'आहेत', 'येथे', 'सर्व', 'परंतु', 'तर', 'तरी', 'की', 'म्हणून',
            'मग', 'या', 'व', 'ते', 'तो', 'ती', 'आणि', 'होते', 'करून', 'अथवा',
            'परत', 'एक', 'होता', 'होती', 'हे', 'पण', 'मी', 'तू', 'आम्ही', 'तुम्ही',
            'ह्या', 'जो', 'जी', 'जे', 'मला', 'तुला', 'त्याला', 'तिला', 'त्यांना'
        ]
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        
    def preprocess_text(self, text):
        """Clean and preprocess the text"""
        # Remove special characters except Devanagari
        text = re.sub(r'[^\u0900-\u097F\s।,.!?]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def split_sentences(self, text):
        """Split text into meaningful sentences"""
        # Split on sentence boundaries
        sentences = re.split('[।!?\n]+', text)
        
        final_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short segments
                continue
                
            # Split long sentences on conjunctions and punctuation
            if len(sentence.split()) > 20:
                conjunctions = 'आणि|परंतु|किंवा|म्हणून|तसेच|पण|अथवा'
                parts = re.split(f'(?<=[।,.])\s+(?={conjunctions})', sentence)
                final_sentences.extend([p.strip() for p in parts if len(p.strip()) > 10])
            else:
                final_sentences.append(sentence)
        
        return final_sentences

    def calculate_sentence_scores(self, sentences):
        """Calculate importance scores for sentences using TF-IDF and TextRank"""
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create graph and calculate TextRank scores
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        return scores

    def generate_summary(self, text):
        """Generate summary with 25-40% compression ratio"""
        # Split text into sentences
        sentences = self.split_sentences(text)
        
        # Calculate sentence scores
        sentence_scores = {}
        for sentence in sentences:
            words = sentence.split()
            # Score based on position and length
            position_score = 1.0 / (1 + sentences.index(sentence))
            length_score = min(1.0, len(words) / 20)  # Normalize length score
            sentence_scores[sentence] = position_score + length_score
        
        # Calculate target summary length (25-40% of original)
        target_length = int(len(text.split()) * 0.3)  # Aim for 30% compression
        
        # Select sentences for summary
        summary_sentences = []
        current_length = 0
        
        # Sort sentences by score
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Add sentences until we reach target length
        for sentence, _ in sorted_sentences:
            if current_length >= target_length:
                break
            summary_sentences.append(sentence)
            current_length += len(sentence.split())
        
        # Sort sentences by original position
        summary_sentences.sort(key=lambda x: sentences.index(x))
        
        return ' '.join(summary_sentences)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    with open(tmp_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    os.unlink(tmp_file_path)
    return text

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
        tmp_file.write(docx_file.read())
        tmp_file_path = tmp_file.name

    doc = docx.Document(tmp_file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    
    os.unlink(tmp_file_path)
    return text

def main():
    st.title("Marathi Text Summarization System")
    
    # Text input box for direct text input
    user_input_text = st.text_area("Or, enter text directly to summarize:", height=200)
    
    uploaded_files = st.file_uploader(
        "Upload Files (PDF/DOCX/TXT)", 
        type=['pdf', 'docx', 'txt'], 
        accept_multiple_files=True
    )

    if st.button("Generate Summaries"):
        summarizer = MarathiSummarizer()
        results = []
        
        # Process user input text if provided
        if user_input_text.strip():
            st.subheader("Summary for Direct Text Input")
            summary = summarizer.generate_summary(user_input_text)
            st.write("Original Text Length:", len(user_input_text.split()), "words")
            st.write("Summary:")
            st.write(summary)
            
            # Calculate and display statistics
            summary_length = len(summary.split())
            compression_ratio = (summary_length / len(user_input_text.split())) * 100
            st.write("Summary Statistics:")
            st.write(f"- Summary length: {summary_length} words")
            st.write(f"- Compression ratio: {compression_ratio:.1f}%")
            
            # Store results for visualization
            results.append({
                'Document': 'User Input',
                'Original Length': len(user_input_text.split()),
                'Generated Summary Length': summary_length,
                'Compression Ratio': compression_ratio,
                'Similarity Score': None,  # No reference summary for direct input
                'Generated Summary': summary
            })
            st.markdown("---")
        
        # Process uploaded files
        if uploaded_files:
            st.header("Document Summaries")
            for file in uploaded_files:
                st.subheader(f"Summary for {file.name}")
                
                # Extract text based on file type
                if file.name.endswith('.pdf'):
                    text = extract_text_from_pdf(file)
                elif file.name.endswith('.docx'):
                    text = extract_text_from_docx(file)
                elif file.name.endswith('.txt'):
                    text = file.getvalue().decode('utf-8')
                
                if text.strip():
                    # Generate and display summary
                    summary = summarizer.generate_summary(text)
                    
                    st.write("Original Text Length:", len(text.split()), "words")
                    st.write("Summary:")
                    st.write(summary)
                    
                    # Calculate similarity score
                    vectorizer = TfidfVectorizer(stop_words=summarizer.stop_words)
                    vectors = vectorizer.fit_transform([summary, text])
                    similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
                    
                    # Display statistics
                    summary_length = len(summary.split())
                    compression_ratio = (summary_length / len(text.split())) * 100
                    st.write("Summary Statistics:")
                    st.write(f"- Summary length: {summary_length} words")
                    st.write(f"- Compression ratio: {compression_ratio:.1f}%")
                    st.write(f"- Similarity score: {similarity_score:.1f}%")
                    
                    # Store results for visualization
                    results.append({
                        'Document': file.name,
                        'Original Length': len(text.split()),
                        'Generated Summary Length': summary_length,
                        'Compression Ratio': compression_ratio,
                        'Similarity Score': similarity_score,
                        'Generated Summary': summary
                    })
                else:
                    st.write(f"No text could be extracted from {file.name}")
                
                st.markdown("---")
        else:
            st.write("Please upload documents to summarize or enter text above!")

if __name__ == "__main__":
    main()