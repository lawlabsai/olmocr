# This script prepares Library of congress transcriptions for use with olmOCR training
# Ex. Find proper transcription datasets here: https://www.loc.gov/search/?q=transcription+dataset&st=list&c=150
# Now, download the archives, extract them, and point this script to a list of all the CSVs
# This script will go through each CSV file, convert each page to PDF format, clean up the transcription using a grounded prompt in chatgpt-4o
# and then output data in olmocr-format, where you have a .md file and a .pdf file named with the ItemID in a folder structure for 
# each initial CSV
# We use https://pypi.org/project/img2pdf/ to convert the images to PDFs losslessly.

import os
import csv
import argparse
import requests
import img2pdf
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import time
import hashlib


def fix_image_url(url: str) -> str:
    """Fix image URL to use full resolution instead of percentage-based sizing."""
    import re
    # Replace any pct:XX pattern with just "full"
    pattern = r'full/pct:\d+/0/default\.jpg'
    if re.search(pattern, url):
        return re.sub(pattern, 'full/full/0/default.jpg', url)
    return url


def download_image(url: str, output_path: Path, max_retries: int = 3) -> bool:
    """Download image from URL with exponential backoff retry logic."""
    # Fix URL if needed
    url = fix_image_url(url)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                # Exponential backoff: 2^attempt seconds (2, 4, 8, ...)
                wait_time = 2 ** (attempt + 1)
                time.sleep(wait_time)
    return False


def convert_image_to_pdf(image_path: Path, pdf_path: Path) -> bool:
    """Convert image to PDF using img2pdf."""
    try:
        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert(str(image_path)))
        return True
    except Exception as e:
        print(f"Failed to convert {image_path} to PDF: {e}")
        return False


def create_markdown_file(transcription: str, md_path: Path) -> None:
    """Create markdown file with transcription."""
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(transcription)


def get_safe_filename(item_id: str) -> str:
    """Create safe filename from item ID."""
    # Replace problematic characters
    safe_name = item_id.replace('/', '_').replace('\\', '_').replace(':', '_')
    # If the name is too long, hash it
    if len(safe_name) > 200:
        hash_suffix = hashlib.md5(safe_name.encode()).hexdigest()[:8]
        safe_name = safe_name[:150] + '_' + hash_suffix
    return safe_name


def process_csv_file(csv_path: Path, output_dir: Path, skip_cleanup: bool = True) -> None:
    """Process a single CSV file containing LOC transcription data."""
    csv_name = csv_path.stem
    dataset_output_dir = output_dir / csv_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {csv_path.name}")
    
    # Read CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Process each row
    processed = 0
    skipped = 0
    
    for row in tqdm(rows, desc=f"Processing {csv_name}"):
        # Check required fields
        if not all(key in row for key in ['ItemId', 'DownloadUrl', 'Transcription']):
            print(f"Skipping row - missing required fields")
            skipped += 1
            continue
        
        item_id = row['ItemId']
        download_url = row['DownloadUrl']
        transcription = row['Transcription']
        
        if not item_id or not download_url or not transcription:
            skipped += 1
            continue
        
        # Create safe filename
        safe_filename = get_safe_filename(item_id)
        
        # Define output paths
        pdf_path = dataset_output_dir / f"{safe_filename}.pdf"
        md_path = dataset_output_dir / f"{safe_filename}.md"
        
        # Skip if already processed
        if pdf_path.exists() and md_path.exists():
            processed += 1
            continue
        
        # Create temp directory for downloads
        temp_dir = dataset_output_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        # Download image
        image_path = temp_dir / f"{safe_filename}.jpg"
        if download_image(download_url, image_path):
            # Convert to PDF
            if convert_image_to_pdf(image_path, pdf_path):
                # Clean up transcription if needed (skipping for now)
                if skip_cleanup:
                    cleaned_transcription = transcription
                else:
                    # TODO: Add transcription cleanup using GPT-4o
                    cleaned_transcription = transcription
                
                # Create markdown file
                create_markdown_file(cleaned_transcription, md_path)
                processed += 1
                
                # Clean up temp image
                image_path.unlink(missing_ok=True)
            else:
                skipped += 1
        else:
            skipped += 1
    
    # Clean up temp directory
    temp_dir = dataset_output_dir / 'temp'
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except:
            pass
    
    print(f"Completed {csv_name}: {processed} processed, {skipped} skipped")


def main():
    parser = argparse.ArgumentParser(description='Prepare LOC transcriptions for olmOCR training')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing CSV files from LOC transcription datasets')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for processed files')
    parser.add_argument('--skip-cleanup', action='store_true', default=True,
                        help='Skip transcription cleanup with GPT-4o (default: True)')
    parser.add_argument('--csv-pattern', type=str, default='*.csv',
                        help='Pattern to match CSV files (default: *.csv)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = sorted(input_dir.glob(args.csv_pattern))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir} matching pattern {args.csv_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in csv_files:
        process_csv_file(csv_file, output_dir, args.skip_cleanup)
    
    print(f"\nAll processing complete. Output saved to {output_dir}")


if __name__ == '__main__':
    main()
