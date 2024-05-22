import pysam
import vcf


def detect_deletions(bam_file, min_insert_size):
    bamfile = pysam.AlignmentFile(bam_file, "rb")
    deletions = []

    for read in bamfile.fetch():
        if not read.is_proper_pair:
            continue

        insert_size = abs(read.template_length)
        if insert_size > min_insert_size:
            deletions.append((read.reference_name, read.reference_start, read.reference_start + insert_size))

    bamfile.close()
    return deletions


def write_vcf(deletions, output_file):
    # Create a VCF header
    vcf_header = vcf.parser._HeaderParser().parse("""
    ##fileformat=VCFv4.2
    ##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
    ##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the structural variant">
    ##contig=<ID=1,length=249250621>
    ##contig=<ID=2,length=243199373>
    ##contig=<ID=3,length=198022430>
    ##contig=<ID=4,length=191154276>
    ##contig=<ID=5,length=180915260>
    ##contig=<ID=6,length=171115067>
    ##contig=<ID=7,length=159138663>
    ##contig=<ID=8,length=146364022>
    ##contig=<ID=9,length=141213431>
    ##contig=<ID=10,length=135534747>
    ##contig=<ID=11,length=135006516>
    ##contig=<ID=12,length=133851895>
    ##contig=<ID=13,length=115169878>
    ##contig=<ID=14,length=107349540>
    ##contig=<ID=15,length=102531392>
    ##contig=<ID=16,length=90354753>
    ##contig=<ID=17,length=81195210>
    ##contig=<ID=18,length=78077248>
    ##contig=<ID=19,length=59128983>
    ##contig=<ID=20,length=63025520>
    ##contig=<ID=21,length=48129895>
    ##contig=<ID=22,length=51304566>
    ##contig=<ID=X,length=155270560>
    ##contig=<ID=Y,length=59373566>
    """.split('\n'))

    # Open the output VCF file
    vcf_writer = vcf.Writer(open(output_file, 'w'), vcf_header)

    for deletion in deletions:
        rec = vcf.model._Record(
            CHROM=deletion[0],
            POS=deletion[1],
            ID='.',
            REF='N',
            ALT=[vcf.model._SV('DEL')],
            QUAL='.',
            FILTER='PASS',
            INFO={'SVTYPE': 'DEL', 'END': deletion[2]}
        )
        vcf_writer.write_record(rec)

    vcf_writer.close()


if __name__ == "__main__":
    bam_file = "your_alignment_file.bam"  # Path to your BAM file
    min_insert_size = 1000  # Minimum insert size to consider a deletion
    output_vcf_file = "output.vcf"  # Path to the output VCF file

    # Detect deletions
    deletions = detect_deletions(bam_file, min_insert_size)

    # Write deletions to VCF
    write_vcf(deletions, output_vcf_file)

    print(f"Detected {len(deletions)} deletions and saved to {output_vcf_file}")
