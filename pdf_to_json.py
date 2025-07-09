#!/usr/bin/env python3
"""
PDF to JSON Converter using pdfplumber
Extracts text, tables, and metadata from PDF and saves as JSON
"""

import pdfplumber
import json
import os
from datetime import datetime

def extract_pdf_content(pdf_path):
    """
    Extract comprehensive content from PDF using pdfplumber
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict: Extracted content including text, tables, and metadata
    """
    extracted_data = {
        "metadata": {},
        "pages": [],
        "total_pages": 0,
        "extraction_timestamp": datetime.now().isoformat(),
        "source_file": os.path.basename(pdf_path)
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extract metadata
            if pdf.metadata:
                extracted_data["metadata"] = pdf.metadata
            
            extracted_data["total_pages"] = len(pdf.pages)
            
            # Process each page
            for page_num, page in enumerate(pdf.pages, 1):
                page_data = {
                    "page_number": page_num,
                    "text": "",
                    "tables": [],
                    "page_info": {
                        "width": page.width,
                        "height": page.height,
                        "rotation": getattr(page, 'rotation', 0)
                    }
                }
                
                # Extract text
                text = page.extract_text()
                if text:
                    page_data["text"] = text.strip()
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        table_data = {
                            "table_index": table_idx,
                            "rows": table,
                            "row_count": len(table),
                            "column_count": len(table[0]) if table else 0
                        }
                        page_data["tables"].append(table_data)
                
                extracted_data["pages"].append(page_data)
                
        return extracted_data
        
    except Exception as e:
        return {
            "error": f"Failed to process PDF: {str(e)}",
            "extraction_timestamp": datetime.now().isoformat(),
            "source_file": os.path.basename(pdf_path)
        }

def save_to_json(data, output_path):
    """
    Save extracted data to JSON file
    
    Args:
        data (dict): Data to save
        output_path (str): Path for output JSON file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved extracted data to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")
        return False

def main():
    """Main function to process PDF and save as JSON"""
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "test.pdf")
    json_path = os.path.join(script_dir, "test_extracted.json")
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    print(f"Processing PDF: {pdf_path}")
    print("Extracting content using pdfplumber...")
    
    # Extract content
    extracted_data = extract_pdf_content(pdf_path)
    
    # Check for errors
    if "error" in extracted_data:
        print(f"Error: {extracted_data['error']}")
        return
    
    # Print summary
    print(f"Extraction Summary:")
    print(f"   - Total pages: {extracted_data['total_pages']}")
    print(f"   - Total characters: {sum(len(page['text']) for page in extracted_data['pages'])}")
    print(f"   - Tables found: {sum(len(page['tables']) for page in extracted_data['pages'])}")
    
    # Save to JSON
    print(f"Saving to JSON: {json_path}")
    if save_to_json(extracted_data, json_path):
        print("Process completed successfully!")
    else:
        print("Failed to save JSON file")

if __name__ == "__main__":
    main()
