import os
from pathlib import Path
from marathi_summarizer import MarathiSummarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_document(file_path):
    """Load text from document"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def evaluate_summarizer():
    # Initialize summarizer
    summarizer = MarathiSummarizer()
    
    # Get paths
    docs_path = Path('documents')
    summaries_path = Path('summaries')
    
    # Store results
    results = []
    
    # Get all document files
    doc_files = list(docs_path.glob('*.txt'))
    
    print("Starting evaluation of Marathi Summarizer...")
    print("-" * 50)
    
    for doc_file in doc_files:
        print(f"\nProcessing: {doc_file.name}")
        
        # Get corresponding summary file
        summary_file = summaries_path / doc_file.name
        
        if not summary_file.exists():
            print(f"No reference summary found for {doc_file.name}")
            continue
            
        # Load documents
        original_text = load_document(doc_file)
        reference_summary = load_document(summary_file)
        
        # Generate summary
        generated_summary = summarizer.generate_summary(original_text)
        
        # Calculate similarity score
        vectorizer = TfidfVectorizer(stop_words=summarizer.stop_words)
        vectors = vectorizer.fit_transform([generated_summary, reference_summary])
        similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        # Calculate metrics
        original_length = len(original_text.split())
        generated_length = len(generated_summary.split())
        reference_length = len(reference_summary.split())
        compression_ratio = (generated_length / original_length) * 100
        
        # Store results
        results.append({
            'Document': doc_file.name,
            'Original Length': original_length,
            'Generated Summary Length': generated_length,
            'Reference Summary Length': reference_length,
            'Compression Ratio': compression_ratio,
            'Similarity Score': similarity_score * 100,
            'Generated Summary': generated_summary,
            'Reference Summary': reference_summary
        })
        
        # Print detailed results for each document
        print(f"Original Length: {original_length} words")
        print(f"Generated Summary Length: {generated_length} words")
        print(f"Reference Summary Length: {reference_length} words")
        print(f"Compression Ratio: {compression_ratio:.1f}%")
        print(f"Similarity Score: {similarity_score*100:.1f}%")
    
    return results

def generate_report(results):
    """Generate and save detailed report"""
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate average metrics
    avg_metrics = {
        'Average Original Length': df['Original Length'].mean(),
        'Average Generated Length': df['Generated Summary Length'].mean(),
        'Average Reference Length': df['Reference Summary Length'].mean(),
        'Average Compression Ratio': df['Compression Ratio'].mean(),
        'Average Similarity Score': df['Similarity Score'].mean()
    }
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Compression Ratio vs Similarity Score
    plt.subplot(2, 2, 1)
    plt.scatter(df['Compression Ratio'], df['Similarity Score'])
    plt.xlabel('Compression Ratio (%)')
    plt.ylabel('Similarity Score (%)')
    plt.title('Compression vs Similarity')
    
    # Length Comparisons
    plt.subplot(2, 2, 2)
    df[['Generated Summary Length', 'Reference Summary Length']].plot(kind='bar')
    plt.title('Summary Lengths Comparison')
    plt.xticks(rotation=45)
    
    # Save plots
    plt.tight_layout()
    plt.savefig('summarization_analysis.png')
    
    # Generate detailed report
    with open('summarization_report.txt', 'w', encoding='utf-8') as f:
        f.write("Marathi Summarizer Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        
        # Write average metrics
        f.write("Average Metrics:\n")
        f.write("-" * 20 + "\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.2f}\n")
        
        # Write detailed results for each document
        f.write("\nDetailed Results:\n")
        f.write("=" * 40 + "\n\n")
        
        for result in results:
            f.write(f"Document: {result['Document']}\n")
            f.write(f"Original Length: {result['Original Length']} words\n")
            f.write(f"Generated Summary Length: {result['Generated Summary Length']} words\n")
            f.write(f"Reference Summary Length: {result['Reference Summary Length']} words\n")
            f.write(f"Compression Ratio: {result['Compression Ratio']:.1f}%\n")
            f.write(f"Similarity Score: {result['Similarity Score']:.1f}%\n")
            f.write("\nGenerated Summary:\n")
            f.write(result['Generated Summary'] + "\n")
            f.write("\nReference Summary:\n")
            f.write(result['Reference Summary'] + "\n")
            f.write("\n" + "=" * 40 + "\n\n")

def generate_visualizations(results):
    """Generate visualizations for summarization results."""
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(15, 10))
    
    # Compression Ratio vs Similarity Score
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='Compression Ratio', y='Similarity Score', data=df)
    plt.title('Compression Ratio vs Similarity Score')
    plt.xlabel('Compression Ratio (%)')
    plt.ylabel('Similarity Score (%)')
    
    # Distribution of Compression Ratios
    plt.subplot(2, 2, 2)
    sns.histplot(df['Compression Ratio'], bins=20, kde=True)
    plt.title('Distribution of Compression Ratios')
    plt.xlabel('Compression Ratio (%)')
    
    # Distribution of Similarity Scores
    plt.subplot(2, 2, 3)
    sns.histplot(df['Similarity Score'], bins=20, kde=True)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score (%)')
    
    # Summary Length Comparison
    plt.subplot(2, 2, 4)
    df[['Generated Summary Length', 'Reference Summary Length']].plot(kind='bar', ax=plt.gca())
    plt.title('Summary Lengths Comparison')
    plt.xlabel('Document Index')
    plt.ylabel('Length (words)')
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('summarization_visualizations.png')
    plt.show()

def main():
    # Run evaluation
    results = evaluate_summarizer()
    
    # Generate report
    generate_report(results)
    
    # Generate visualizations
    generate_visualizations(results)
    
    print("\nEvaluation complete!")
    print("Check 'summarization_report.txt' for detailed results")
    print("Check 'summarization_analysis.png' for visualizations")

if __name__ == "__main__":
    main() 