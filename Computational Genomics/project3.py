import pysam
import pandas as pd
from collections import defaultdict
import multiprocessing
import os

# Function to read SAM/BAM files
def read_sam_bam(file_path):
    samfile = pysam.AlignmentFile(file_path, "rb")
    return samfile

# Function to detect structural variants
def detect_structural_variants(samfile):
    variants = []
    read_pairs = defaultdict(list)

    for read in samfile.fetch():
        if not read.is_unmapped and read.is_paired:
            read_pairs[read.query_name].append(read)

    for read_name, reads in read_pairs.items():
        if len(reads) == 2:
            read1, read2 = reads
            if read1.reference_name == read2.reference_name:
                pos1, pos2 = read1.reference_start, read2.reference_start
                start, end = sorted([pos1, pos2])
                length = abs(end - start)

                # Deletion
                if length > 1000 and read1.is_reverse != read2.is_reverse:
                    variants.append({
                        'CHROM': read1.reference_name,
                        'POS': start,
                        'ID': '.',
                        'REF': 'N',
                        'ALT': f'<DEL:{length}>',
                        'QUAL': '.',
                        'FILTER': 'PASS',
                        'INFO': f'SVTYPE=DEL;END={end};SVLEN=-{length}'
                    })
                # Inversion
                elif read1.is_reverse == read2.is_reverse:
                    variants.append({
                        'CHROM': read1.reference_name,
                        'POS': start,
                        'ID': '.',
                        'REF': 'N',
                        'ALT': f'<INV:{length}>',
                        'QUAL': '.',
                        'FILTER': 'PASS',
                        'INFO': f'SVTYPE=INV;END={end};SVLEN={length}'
                    })
                # Duplication
                elif read1.is_reverse != read2.is_reverse and length < 1000:
                    variants.append({
                        'CHROM': read1.reference_name,
                        'POS': start,
                        'ID': '.',
                        'REF': 'N',
                        'ALT': f'<DUP:{length}>',
                        'QUAL': '.',
                        'FILTER': 'PASS',
                        'INFO': f'SVTYPE=DUP;END={end};SVLEN={length}'
                    })
            else:
                # Translocation
                variants.append({
                    'CHROM': read1.reference_name,
                    'POS': read1.reference_start,
                    'ID': '.',
                    'REF': 'N',
                    'ALT': f'<TRA>',
                    'QUAL': '.',
                    'FILTER': 'PASS',
                    'INFO': f'SVTYPE=TRA;CHR2={read2.reference_name};POS2={read2.reference_start}'
                })
    return variants

# Function to write VCF file
def write_vcf(variants, output_file):
    with open(output_file, 'w') as vcf:
        vcf.write("##fileformat=VCFv4.2\n")
        vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for variant in variants:
            vcf.write(
                f"{variant['CHROM']}\t{variant['POS']}\t{variant['ID']}\t{variant['REF']}\t{variant['ALT']}\t{variant['QUAL']}\t{variant['FILTER']}\t{variant['INFO']}\n")

# Function to determine chunk boundaries
def determine_chunk_boundaries(samfile, chunk_size):
    boundaries = []
    current_pos = 0
    for read in samfile.fetch():
        if read.reference_start > current_pos + chunk_size:
            boundaries.append(read.reference_start)
            current_pos = read.reference_start
    return boundaries

# Function to split the BAM file into chunks
def split_bam_file(input_file, chunk_size):
    samfile = pysam.AlignmentFile(input_file, "rb")
    boundaries = determine_chunk_boundaries(samfile, chunk_size)
    chunks = []
    for i, boundary in enumerate(boundaries):
        chunk_file = f"{input_file}_chunk_{i}.bam"
        chunks.append(chunk_file)
        with pysam.AlignmentFile(chunk_file, "wb", header=samfile.header) as chunk:
            for read in samfile.fetch(region=f"{samfile.references[0]}:{boundary}-{boundary + chunk_size}"):
                chunk.write(read)
        # Index the chunk
        pysam.index(chunk_file)
    return chunks

# Parallel processing function
def process_file_chunk(chunk, output_queue):
    samfile = pysam.AlignmentFile(chunk, "rb")
    variants = detect_structural_variants(samfile)
    output_queue.put(variants)

# Main function
def main():
    input_file = "bam files/SRR_final_sorted.bam"
    output_file = "output.vcf"
    chunk_size = 1000000  # Example chunk size, adjust based on your data and resources

    # Check if the BAM file is indexed
    if not os.path.exists(input_file + ".bai"):
        print(f"Index file for {input_file} not found. Creating index...")
        pysam.index(input_file)

    # Split input file into chunks
    chunks = split_bam_file(input_file, chunk_size)

    # Create a multiprocessing queue
    output_queue = multiprocessing.Queue()

    # Create and start multiprocessing processes
    processes = []
    for chunk in chunks:
        p = multiprocessing.Process(target=process_file_chunk, args=(chunk, output_queue))
        processes.append(p)
        p.start()

    # Collect results from all processes
    all_variants = []
    for p in processes:
        p.join()
        while not output_queue.empty():
            all_variants.extend(output_queue.get())

    # Write all variants to the VCF file
    write_vcf(all_variants, output_file)

if __name__ == "__main__":
    main()
