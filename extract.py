#!/usr/bin/env python3
"""
PDF to Text and XML Extractor using pdfplumber
Converts test.pdf to both test.txt and test.xml for comprehensive analysis
"""

import pdfplumber
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

def extract_pdf_to_text(pdf_path: str, output_path: str):
    """
    Extract all text from PDF and save to text file
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Path for output text file
    """
    print(f"📄 Extracting text from: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            
            print(f"📊 Processing {len(pdf.pages)} pages for text extraction...")
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"   Processing page {page_num}/{len(pdf.pages)}", end='\r')
                
                # Add page header
                all_text.append(f"\n--- PAGE {page_num} ---\n")
                
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)
                
                # Extract tables as text
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        all_text.append(f"\n[TABLE {table_idx + 1} ON PAGE {page_num}]\n")
                        if table:
                            for row in table:
                                if row and any(cell for cell in row if cell):
                                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                                    all_text.append(" | ".join(clean_row))
                        all_text.append("")  # Empty line after table
            
            print()  # New line after progress indicator
            
        # Write to text file
        print(f"💾 Saving text to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
        
        # Print statistics
        total_text = '\n'.join(all_text)
        char_count = len(total_text)
        word_count = len(total_text.split())
        line_count = len(all_text)
        
        print(f"✅ Text extraction completed!")
        print(f"   📊 Text Statistics:")
        print(f"      - Characters: {char_count:,}")
        print(f"      - Words: {word_count:,}")
        print(f"      - Lines: {line_count:,}")
        print(f"      - Pages: {len(pdf.pages)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting PDF to text: {str(e)}")
        return False

def extract_pdf_to_xml(pdf_path: str, output_path: str):
    """
    Extract PDF content and save as structured XML
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Path for output XML file
    """
    print(f"📄 Extracting PDF to XML: {pdf_path}")
    
    try:
        # Create root XML element
        root = ET.Element("document")
        root.set("source", os.path.basename(pdf_path))
        
        with pdfplumber.open(pdf_path) as pdf:
            # Add metadata
            metadata = ET.SubElement(root, "metadata")
            if pdf.metadata:
                for key, value in pdf.metadata.items():
                    meta_elem = ET.SubElement(metadata, "field")
                    meta_elem.set("name", str(key))
                    meta_elem.text = str(value) if value else ""
            
            # Add page count
            doc_info = ET.SubElement(root, "document_info")
            ET.SubElement(doc_info, "total_pages").text = str(len(pdf.pages))
            
            print(f"📊 Processing {len(pdf.pages)} pages for XML extraction...")
            
            # Process each page
            pages_elem = ET.SubElement(root, "pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"   Processing page {page_num}/{len(pdf.pages)} for XML", end='\r')
                
                # Create page element
                page_elem = ET.SubElement(pages_elem, "page")
                page_elem.set("number", str(page_num))
                
                # Add page dimensions
                page_info = ET.SubElement(page_elem, "page_info")
                ET.SubElement(page_info, "width").text = str(page.width)
                ET.SubElement(page_info, "height").text = str(page.height)
                
                # Extract and add text
                page_text = page.extract_text()
                if page_text:
                    text_elem = ET.SubElement(page_elem, "text")
                    text_elem.text = page_text
                
                # Extract and add tables
                tables = page.extract_tables()
                if tables:
                    tables_elem = ET.SubElement(page_elem, "tables")
                    for table_idx, table in enumerate(tables):
                        table_elem = ET.SubElement(tables_elem, "table")
                        table_elem.set("index", str(table_idx))
                        
                        if table:
                            for row_idx, row in enumerate(table):
                                if row and any(cell for cell in row if cell):
                                    row_elem = ET.SubElement(table_elem, "row")
                                    row_elem.set("index", str(row_idx))
                                    
                                    for cell_idx, cell in enumerate(row):
                                        cell_elem = ET.SubElement(row_elem, "cell")
                                        cell_elem.set("index", str(cell_idx))
                                        cell_elem.text = str(cell).strip() if cell else ""
            
            print()  # New line after progress indicator
        
        # Pretty print and save XML
        print(f"💾 Saving XML to: {output_path}")
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        # Calculate XML statistics
        xml_size = len(pretty_xml)
        element_count = len(root.findall(".//*"))
        page_count = len(root.findall(".//page"))
        table_count = len(root.findall(".//table"))
        
        print(f"✅ XML extraction completed!")
        print(f"   📊 XML Statistics:")
        print(f"      - File size: {xml_size:,} characters")
        print(f"      - Elements: {element_count:,}")
        print(f"      - Pages: {page_count}")
        print(f"      - Tables: {table_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting PDF to XML: {str(e)}")
        return False

def main():
    """Main function"""
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "test.pdf")
    txt_path = os.path.join(script_dir, "test.txt")
    xml_path = os.path.join(script_dir, "test.xml")
    
    print("🚀 PDF TO TEXT AND XML EXTRACTOR")
    print("=" * 60)
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    # Extract PDF to text
    print("\n📝 EXTRACTING TO TEXT FORMAT")
    print("-" * 40)
    text_success = extract_pdf_to_text(pdf_path, txt_path)
    
    # Extract PDF to XML
    print("\n🏗️  EXTRACTING TO XML FORMAT")
    print("-" * 40)
    xml_success = extract_pdf_to_xml(pdf_path, xml_path)
    
    # Summary
    print(f"\n📋 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"✅ Text extraction: {'SUCCESS' if text_success else 'FAILED'}")
    print(f"✅ XML extraction: {'SUCCESS' if xml_success else 'FAILED'}")
    
    if text_success or xml_success:
        print(f"\n🎯 ANALYSIS OPTIONS:")
        if text_success:
            print(f"   📄 Text file: {txt_path}")
            print(f"      - Use Ctrl+F to search for 'Israel', 'unsubstantiated', 'Appendix 3'")
        if xml_success:
            print(f"   🏗️  XML file: {xml_path}")
            print(f"      - Structured format for programmatic analysis")
            print(f"      - Can use XPath queries to find specific elements")
        
        print(f"\n💡 TROUBLESHOOTING TIPS:")
        print(f"   1. Search the text file for patterns like 'IL-GSK' or 'Israel'")
        print(f"   2. Look for 'Appendix 3' section with complaint data")
        print(f"   3. Check if table data is properly extracted")
        print(f"   4. Compare JSON vs TXT vs XML to see formatting differences")
    else:
        print(f"\n❌ Both extractions failed. Check the error messages above.")

if __name__ == "__main__":
    main()