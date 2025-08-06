"""
Document Processing Module
Handles various document formats for RAG system
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Document processor for various file formats
    
    Features:
    - Text file processing
    - PDF document processing
    - Word document processing
    - Text chunking and preprocessing
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
            separators: List of text separators for chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]
        
        logger.info("Document processor initialized")
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single file
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of processed document chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and process accordingly
        if file_path.suffix.lower() == '.txt':
            return self._process_text_file(file_path)
        elif file_path.suffix.lower() == '.pdf':
            return self._process_pdf_file(file_path)
        elif file_path.suffix.lower() in ['.doc', '.docx']:
            return self._process_word_file(file_path)
        else:
            # Try to process as text file
            logger.warning(f"Unknown file type {file_path.suffix}, treating as text")
            return self._process_text_file(file_path)
    
    def process_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple files
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of all processed document chunks
        """
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
                logger.info(f"Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        return all_chunks
    
    def _process_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self._chunk_text(content, str(file_path))
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    return self._chunk_text(content, str(file_path))
                except UnicodeDecodeError:
                    continue
            
            raise UnicodeDecodeError(f"Could not decode file: {file_path}")
    
    def _process_pdf_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process PDF file"""
        try:
            import PyPDF2
            
            content = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
            
            return self._chunk_text(content, str(file_path))
            
        except ImportError:
            logger.error("PyPDF2 not installed. Cannot process PDF files.")
            return []
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []
    
    def _process_word_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process Word document"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            content = ""
            
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            
            return self._chunk_text(content, str(file_path))
            
        except ImportError:
            logger.error("python-docx not installed. Cannot process Word documents.")
            return []
        except Exception as e:
            logger.error(f"Error processing Word document {file_path}: {e}")
            return []
    
    def _chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text content to chunk
            source: Source file path
            
        Returns:
            List of text chunks with metadata
        """
        # Clean the text
        text = self._clean_text(text)
        
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If we're not at the end, try to break at a separator
            if end < len(text):
                best_break = end
                
                for separator in self.separators:
                    # Look for separator near the end position
                    sep_pos = text.rfind(separator, start, end)
                    if sep_pos > start:
                        best_break = sep_pos + len(separator)
                        break
                
                end = best_break
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'content': chunk_text,
                    'metadata': {
                        'source': source,
                        'chunk_index': len(chunks),
                        'start_pos': start,
                        'end_pos': end,
                        'chunk_size': len(chunk_text)
                    }
                })
            
            # Move to next position with overlap
            if end >= len(text):
                break
            
            start = max(end - self.chunk_overlap, start + 1)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text content
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove special characters that might cause issues
        text = text.replace('\x00', '')
        text = text.replace('\ufffd', '')
        
        return text.strip()
    
    def get_processing_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about processed chunks
        
        Args:
            chunks: List of processed chunks
            
        Returns:
            Processing statistics
        """
        if not chunks:
            return {'total_chunks': 0}
        
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        sources = set(chunk['metadata']['source'] for chunk in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_sources': len(sources),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'sources': list(sources)
        } 