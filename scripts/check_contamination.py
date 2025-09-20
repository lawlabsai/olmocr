#!/usr/bin/env python3
# Input arguments:
# path to olmocr-bench/bench_data directory
# Path to metadata jsonl file
# Path to sqlite db
# Steps:
# Find all jsonl files in bench_data directory, read all "url" fields and make a set
# In metadata jsonl file, read all lines, get source_url field
# Do mapping between source_url and real_url by
# first turning ex. s3://ai2-s2-pdfs/b2d8/3a50695174f1de4973248fcf03c681ba1218.pdf into b2d83a50695174f1de4973248fcf03c681ba1218
# Then, in sqlite db with schema below, look up the real uri
# CREATE TABLE pdf_mapping (
#                 pdf_hash TEXT PRIMARY KEY,
#                 uri TEXT
#             );
# Report if any of the final uri's match with original set
#
# Also support things if the source_url is in the following format, starting with ./
# ex ./synth_tables/56441bdefb2397d956da725903948e0893c9_pg1.pdf, then get the 56441bdefb2397d956da725903948e0893c9
# Then, using the schema below in the same db, look up the full hash first some this given hash, then get the full uri to continue the lookup
# CREATE TABLE substr_to_full_hash (
#     pdf_hash TEXT PRIMARY KEY,  -- this will be the shortened hash
#     full_hash TEXT              -- this is the original hash
# );

import json
import sqlite3
import argparse
from pathlib import Path
import re


def get_bench_urls(bench_data_dir):
    """Read all JSONL files in bench_data directory and extract URLs."""
    bench_urls = set()
    bench_data_path = Path(bench_data_dir)
    
    for jsonl_file in bench_data_path.rglob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'url' in data:
                        bench_urls.add(data['url'])
                except json.JSONDecodeError:
                    continue
    
    return bench_urls


def s3_url_to_hash(s3_url):
    """Convert S3 URL to hash format.
    e.g., s3://ai2-s2-pdfs/b2d8/3a50695174f1de4973248fcf03c681ba1218.pdf -> b2d83a50695174f1de4973248fcf03c681ba1218
    """
    match = re.search(r's3://[^/]+/([^/]+)/([^.]+)', s3_url)
    if match:
        prefix = match.group(1)
        hash_part = match.group(2)
        return prefix + hash_part
    return None


def local_path_to_short_hash(local_path):
    """Extract short hash from local path format.
    e.g., ./synth_tables/56441bdefb2397d956da725903948e0893c9_pg1.pdf -> 56441bdefb2397d956da725903948e0893c9
    """
    match = re.search(r'([a-f0-9]+)(?:_pg\d+)?\.pdf', local_path)
    if match:
        return match.group(1)
    return None


def check_contamination(bench_data_dir, metadata_jsonl_path, sqlite_db_path):
    """Main function to check for contamination between bench data and training data."""
    print(f"Checking contamination...")
    print(f"Bench data directory: {bench_data_dir}")
    print(f"Metadata JSONL: {metadata_jsonl_path}")
    print(f"SQLite database: {sqlite_db_path}\n")
    
    # Step 1: Get all URLs from bench data
    print("Step 1: Reading URLs from bench data...")
    bench_urls = get_bench_urls(bench_data_dir)
    print(f"Found {len(bench_urls)} unique URLs in bench data\n")
    
    # Step 2: Read metadata JSONL and process source URLs
    print("Step 2: Processing metadata JSONL...")
    source_urls = []
    with open(metadata_jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'source_url' in data:
                    source_urls.append(data['source_url'])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num}")
    
    print(f"Found {len(source_urls)} source URLs in metadata\n")
    
    # Step 3: Map URLs to hashes and query database
    print("Step 3: Mapping URLs and querying database...")
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()
    
    real_urls = set()
    unmapped_count = 0
    s3_count = 0
    local_count = 0
    empty_result_count = 0
    
    for source_url in source_urls:
        pdf_hash = None
        
        # Handle S3 URLs
        if source_url.startswith("s3://"):
            s3_count += 1
            pdf_hash = s3_url_to_hash(source_url)
        
        # Handle local paths starting with ./
        elif source_url.startswith("./"):
            local_count += 1
            short_hash = local_path_to_short_hash(source_url)
            if short_hash:
                # First lookup: get full hash from short hash
                cursor.execute("SELECT full_hash FROM substr_to_full_hash WHERE pdf_hash = ?", (short_hash,))
                result = cursor.fetchone()
                if result:
                    pdf_hash = result[0]
        
        # If we have a hash, look up the real URI
        if pdf_hash:
            cursor.execute("SELECT uri FROM pdf_mapping WHERE pdf_hash = ?", (pdf_hash,))
            result = cursor.fetchone()
            if result:
                # Check if the looked up URL is empty/blank
                if result[0] == "" or result[0] is None:
                    empty_result_count += 1
                else:
                    real_urls.add(result[0])
        else:
            unmapped_count += 1
    
    conn.close()
    
    print(list(real_urls)[:5])

    print(f"Successfully mapped {len(real_urls)} URLs from database")
    print(f"  - S3 URLs processed: {s3_count}")
    print(f"  - Local paths processed: {local_count}")
    print(f"  - Empty/blank URLs from database: {empty_result_count}")
    if unmapped_count > 0:
        print(f"Warning: {unmapped_count} URLs could not be mapped\n")
    
    # Step 4: Check for contamination
    print("Step 4: Checking for contamination...")
    contaminated_urls = bench_urls.intersection(real_urls)
    
    if contaminated_urls:
        print(f"\n⚠️  CONTAMINATION DETECTED! Found {len(contaminated_urls)} matching URLs:")
        for url in sorted(contaminated_urls)[:10]:  # Show first 10
            print(f"  - {url}")
        if len(contaminated_urls) > 10:
            print(f"  ... and {len(contaminated_urls) - 10} more")
    else:
        print("\n✅ No contamination detected. Bench URLs and training URLs are disjoint.")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Bench URLs: {len(bench_urls)}")
    print(f"  Training URLs (mapped): {len(real_urls)}")
    print(f"  Contaminated URLs: {len(contaminated_urls)}")
    if bench_urls:
        contamination_rate = (len(contaminated_urls) / len(bench_urls)) * 100
        print(f"  Contamination rate: {contamination_rate:.2f}%")
    
    return len(contaminated_urls)


def main():
    parser = argparse.ArgumentParser(
        description="Check for contamination between benchmark data and training data"
    )
    parser.add_argument(
        "bench_data_dir",
        help="Path to olmocr-bench/bench_data directory"
    )
    parser.add_argument(
        "metadata_jsonl",
        help="Path to metadata JSONL file"
    )
    parser.add_argument(
        "sqlite_db",
        help="Path to SQLite database with pdf_mapping table"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.bench_data_dir).is_dir():
        print(f"Error: {args.bench_data_dir} is not a directory")
        return 1
    
    if not Path(args.metadata_jsonl).is_file():
        print(f"Error: {args.metadata_jsonl} is not a file")
        return 1
    
    if not Path(args.sqlite_db).is_file():
        print(f"Error: {args.sqlite_db} is not a file")
        return 1
    
    # Run contamination check
    contaminated_count = check_contamination(
        args.bench_data_dir,
        args.metadata_jsonl,
        args.sqlite_db
    )
    
    # Return non-zero exit code if contamination found
    return 1 if contaminated_count > 0 else 0


if __name__ == "__main__":
    exit(main())