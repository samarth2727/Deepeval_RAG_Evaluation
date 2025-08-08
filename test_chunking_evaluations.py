#!/usr/bin/env python3
"""
Test script to demonstrate chunking evaluations during document processing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.enhanced_document_processor import EnhancedDocumentProcessor
from evaluation.chunking_evaluator import ChunkingEvaluator

def create_test_documents():
    """Create test documents for evaluation"""
    test_docs = [
        """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and make decisions from data without being explicitly programmed for every task. It involves the development of algorithms and statistical models that can identify patterns in data and make predictions or decisions based on that data.
        
        There are three main types of machine learning:
        1. Supervised Learning: Uses labeled training data to learn a function that maps inputs to outputs
        2. Unsupervised Learning: Finds hidden patterns in data without labeled examples
        3. Reinforcement Learning: Learns through interaction with an environment using rewards and penalties
        
        Common applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles.
        """,
        """
        Deep Learning and Neural Networks
        
        Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence "deep") to progressively extract higher-level features from raw input. These networks are inspired by the structure and function of the human brain.
        
        Key components of neural networks:
        - Neurons (nodes): Basic processing units
        - Layers: Input, hidden, and output layers
        - Weights and Biases: Parameters that determine the network's behavior
        - Activation Functions: Determine the output of neurons
        
        Deep learning has revolutionized fields like computer vision, natural language processing, and speech recognition. Popular frameworks include TensorFlow, PyTorch, and Keras.
        """
    ]
    
    # Save test documents
    for i, doc in enumerate(test_docs, 1):
        with open(f"test_doc_{i}.txt", "w", encoding="utf-8") as f:
            f.write(doc)
    
    return [f"test_doc_{i}.txt" for i in range(1, len(test_docs) + 1)]

def test_chunking_evaluations():
    """Test chunking evaluations"""
    print("🔍 Testing Chunking Evaluations")
    print("=" * 50)
    
    # Create test documents
    doc_paths = create_test_documents()
    print(f"Created {len(doc_paths)} test documents")
    
    # Initialize enhanced processor
    processor = EnhancedDocumentProcessor(
        chunk_size=300,
        chunk_overlap=50,
        enable_evaluations=True,
        quality_threshold=0.7
    )
    
    print("\n📄 Processing documents with evaluations...")
    
    # Process documents with evaluations
    result = processor.process_documents_with_evaluation(
        doc_paths, optimize_chunks=True
    )
    
    # Display results
    print(f"\n✅ Processing completed!")
    print(f"  Total chunks: {result.total_chunks}")
    print(f"  Processing time: {result.processing_time:.2f}s")
    print(f"  Evaluation time: {result.evaluation_time:.2f}s")
    
    if result.chunking_metrics:
        print(f"\n📊 Chunking Quality Metrics:")
        print(f"  Overall Quality: {result.chunking_metrics.overall_chunk_quality:.3f}")
        print(f"  Coherence: {result.chunking_metrics.chunk_coherence:.3f}")
        print(f"  Completeness: {result.chunking_metrics.chunk_completeness:.3f}")
        print(f"  Overlap Quality: {result.chunking_metrics.chunk_overlap_quality:.3f}")
        print(f"  Size Consistency: {result.chunking_metrics.chunk_size_consistency:.3f}")
        print(f"  Semantic Boundaries: {result.chunking_metrics.semantic_boundary_respect:.3f}")
    
    if result.processing_metrics:
        print(f"\n🔧 Processing Quality Metrics:")
        print(f"  Overall Quality: {result.processing_metrics.overall_processing_quality:.3f}")
        print(f"  Text Cleanliness: {result.processing_metrics.text_cleanliness:.3f}")
        print(f"  Content Preservation: {result.processing_metrics.content_preservation:.3f}")
        print(f"  Format Handling: {result.processing_metrics.format_handling:.3f}")
        print(f"  Metadata Quality: {result.processing_metrics.metadata_quality:.3f}")
        print(f"  Processing Efficiency: {result.processing_metrics.processing_efficiency:.3f}")
    
    # Show some sample chunks
    print(f"\n📝 Sample Chunks:")
    for i, chunk in enumerate(result.chunks[:3], 1):
        print(f"  Chunk {i}:")
        print(f"    Content: {chunk['content'][:100]}...")
        print(f"    Meta: {chunk['meta']}")
        print()
    
    # Get evaluation summary
    summary = processor.get_evaluation_summary()
    print(f"\n📈 Evaluation Summary:")
    print(f"  Total evaluations: {summary.get('total_evaluations', 0)}")
    print(f"  Average chunking quality: {summary.get('average_chunking_quality', 0):.3f}")
    print(f"  Average processing quality: {summary.get('average_processing_quality', 0):.3f}")
    
    # Cleanup
    for doc_path in doc_paths:
        Path(doc_path).unlink(missing_ok=True)
    
    print("\n🎉 Test completed successfully!")

def test_evaluator_directly():
    """Test the chunking evaluator directly"""
    print("\n🔍 Testing Chunking Evaluator Directly")
    print("=" * 50)
    
    evaluator = ChunkingEvaluator()
    
    # Test chunking evaluation
    test_chunks = [
        "This is a complete sentence about machine learning.",
        "Another sentence that follows naturally from the previous one.",
        "This chunk contains multiple sentences. It maintains coherence.",
        "Incomplete sentence without proper ending"
    ]
    
    original_text = " ".join(test_chunks)
    
    print("Evaluating chunking quality...")
    chunking_metrics = evaluator.evaluate_chunking_quality(
        test_chunks, original_text, chunk_size=100, chunk_overlap=20
    )
    
    print(f"Chunking Quality Results:")
    print(f"  Overall Quality: {chunking_metrics.overall_chunk_quality:.3f}")
    print(f"  Coherence: {chunking_metrics.chunk_coherence:.3f}")
    print(f"  Completeness: {chunking_metrics.chunk_completeness:.3f}")
    print(f"  Overlap Quality: {chunking_metrics.chunk_overlap_quality:.3f}")
    print(f"  Size Consistency: {chunking_metrics.chunk_size_consistency:.3f}")
    print(f"  Semantic Boundaries: {chunking_metrics.semantic_boundary_respect:.3f}")
    
    # Test processing evaluation
    test_processed_chunks = [
        {
            "content": "This is a well-processed chunk with clean text.",
            "meta": {"source": "test.txt", "chunk_id": 0, "total_chunks": 4}
        },
        {
            "content": "Another processed chunk with proper metadata.",
            "meta": {"source": "test.txt", "chunk_id": 1, "total_chunks": 4}
        }
    ]
    
    print("\nEvaluating processing quality...")
    processing_metrics = evaluator.evaluate_processing_quality(
        test_processed_chunks, ["test.txt"]
    )
    
    print(f"Processing Quality Results:")
    print(f"  Overall Quality: {processing_metrics.overall_processing_quality:.3f}")
    print(f"  Text Cleanliness: {processing_metrics.text_cleanliness:.3f}")
    print(f"  Content Preservation: {processing_metrics.content_preservation:.3f}")
    print(f"  Format Handling: {processing_metrics.format_handling:.3f}")
    print(f"  Metadata Quality: {processing_metrics.metadata_quality:.3f}")
    print(f"  Processing Efficiency: {processing_metrics.processing_efficiency:.3f}")

if __name__ == "__main__":
    print("🚀 Starting Chunking Evaluation Tests")
    print("=" * 50)
    
    try:
        test_chunking_evaluations()
        test_evaluator_directly()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 