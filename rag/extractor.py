from pypdf import PdfReader
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pdf_text_extractor(filepath: str) -> None:
    """
    Extracts text content from a PDF file and saves it as a .txt file.

    Args:
        filepath (str): The path to the PDF file.

    Returns:
        None
    """
    try:
        # Initialize the PdfReader and content accumulator
        content = ""
        pdf_reader = PdfReader(filepath, strict=True)

        # Extract text from each page
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                content += f"{page_text}\n\n"

        # Save the extracted text to a .txt file
        txt_filepath = filepath.replace(".pdf", ".txt")
        with open(txt_filepath, "w", encoding="utf-8") as file:
            file.write(content)

        logger.info(f"Text successfully extracted and saved to {txt_filepath}")

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise RuntimeError(f"File not found: {filepath}")

    except Exception as e:
        logger.error(f"An error occurred while extracting text: {e}")
        raise RuntimeError(f"Failed to extract text from {filepath}: {e}")
