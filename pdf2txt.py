import pdfplumber
import re
import unicodedata
from typing import Optional, Dict, List
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFTextExtractor:
    def __init__(self, keep_page_breaks: bool = True, min_line_length: int = 1):
        """
        Initialize the PDF text extractor with configuration options.
        
        Args:
            keep_page_breaks: Whether to keep page break markers in the output
            min_line_length: Minimum number of characters for a line to be kept
        """
        self.keep_page_breaks = keep_page_breaks
        self.min_line_length = min_line_length
        
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving meaningful characters.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Cleaned text with preserved special characters
        """
        if not text:
            return ""
            
        # Replace various types of whitespace with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters while preserving newlines
        text = ''.join(char if unicodedata.category(char)[0] != 'C' 
                      or char in '\n\t' else ' ' 
                      for char in text)
        
        # Remove zero-width spaces and other invisible characters
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        
        # Normalize unicode characters (e.g., convert different forms of quotes)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove repeated punctuation
        text = re.sub(r'([!?,.:;])\1+', r'\1', text)
        
        return text.strip()
    
    def is_meaningful_line(self, line: str) -> bool:
        """
        Check if a line contains meaningful content.
        
        Args:
            line: Text line to check
            
        Returns:
            Boolean indicating if line should be kept
        """
        # Remove whitespace
        line = line.strip()
        
        # Check minimum length
        if len(line) < self.min_line_length:
            return False
            
        # Check if line contains only punctuation or special characters
        text_chars = sum(1 for c in line if c.isalnum() or unicodedata.category(c).startswith('L'))
        return text_chars > 0
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """
        Extract and clean text from each page of a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        page_texts = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        # Extract text from page
                        raw_text = page.extract_text() or ""
                        
                        # Clean the extracted text
                        cleaned_text = self.clean_text(raw_text)
                        
                        # Split into lines and filter meaningful content
                        lines = cleaned_text.split('\n')
                        meaningful_lines = [line for line in lines 
                                         if self.is_meaningful_line(line)]
                        
                        # Join filtered lines
                        processed_text = '\n'.join(meaningful_lines)
                        
                        if processed_text:
                            if self.keep_page_breaks:
                                processed_text = f"\n\n--- Page {page_num} ---\n\n{processed_text}"
                            page_texts[page_num] = processed_text
                            
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_path}: {str(e)}")
            raise
            
        return page_texts
    
    def save_extracted_text(self, 
                          page_texts: Dict[int, str], 
                          output_path: str,
                          split_pages: bool = False) -> None:
        """
        Save extracted text to file(s).
        
        Args:
            page_texts: Dictionary of page numbers and their text
            output_path: Path to save the output
            split_pages: Whether to save each page as a separate file
        """
        output_path = Path(output_path)
        
        try:
            if split_pages:
                # Save each page to a separate file
                output_dir = output_path.parent / output_path.stem
                output_dir.mkdir(exist_ok=True)
                
                for page_num, text in page_texts.items():
                    page_path = output_dir / f"page_{page_num}.txt"
                    with open(page_path, "w", encoding="utf-8") as f:
                        f.write(text)
            else:
                # Combine all pages and save to a single file
                combined_text = "\n\n".join(page_texts.values())
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(combined_text)
                    
        except Exception as e:
            logger.error(f"Error saving output: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    pdf_path = "ml.pdf"
    output_path = "extracted_text.txt"
    
    # Initialize extractor
    extractor = PDFTextExtractor(
        keep_page_breaks=True,
        min_line_length=1
    )
    
    # Extract text
    try:
        page_texts = extractor.extract_text_from_pdf(pdf_path)
        
        # Save extracted text
        extractor.save_extracted_text(
            page_texts=page_texts,
            output_path=output_path,
            split_pages=False
        )
        
        logger.info(f"Text extraction complete. Saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
