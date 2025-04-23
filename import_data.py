#%%

from Bio import SeqIO, SeqRecord, Seq
import glob
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence





#%%


# 1. Find all FASTA files in your directory
fasta_paths = glob.glob(
    "/Users/ronakagarwal/Library/CloudStorage/OneDrive-Personal/"
    "Desktop/S4 washu/Foundations of biomedical computing/"
    "Final project/Data/*.fa"
)

all_records = []
for fp in fasta_paths:
    # Read and filter in one go, storing into a list
    allowed = {"A", "C", "G", "T"}
    filtered = [
        r
        for r in SeqIO.parse(fp, "fasta")
        if 500 <= len(r.seq) <= 750
           and len(r.seq) % 3 == 0
           and set(str(r.seq).upper()) <= allowed
    ]
    all_records.extend(filtered)

print(f"Total CDS loaded: {len(all_records)}")

# Inspect the first record
first = all_records[0]
print(first.id, len(first.seq), "bp")
print(first.seq[:30], "…", first.seq[-30:])

#%% Converting from strings to amino acids

codon2aa = {
    'TTT':'F', 'TTC':'F', 'TTA':'L', 'TTG':'L',
    'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L',
    'ATT':'I', 'ATC':'I', 'ATA':'I', 'ATG':'M',
    'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V',
    'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S',
    'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P',
    'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T',
    'GCT':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
    'TAT':'Y', 'TAC':'Y', 'TAA':'*', 'TAG':'*',
    'CAT':'H', 'CAC':'H', 'CAA':'Q', 'CAG':'Q',
    'AAT':'N', 'AAC':'N', 'AAA':'K', 'AAG':'K',
    'GAT':'D', 'GAC':'D', 'GAA':'E', 'GAG':'E',
    'TGT':'C', 'TGC':'C', 'TGA':'*', 'TGG':'W',
    'CGT':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R',
    'AGT':'S', 'AGC':'S', 'AGA':'R', 'AGG':'R',
    'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G'
}

# 2. Build amino acid → index mapping
aa_list = sorted(set(codon2aa.values()))
aa2idx = {aa: idx for idx, aa in enumerate(aa_list)}

# 3. Build codon → index mapping (collapsing synonyms to same amino acid index)
codon2idx = {codon: aa2idx[aa] for codon, aa in codon2aa.items()}

# 4. Build reverse maps for reference
idx2aa     = {idx: aa for aa, idx in aa2idx.items()}
idx2codons = {}
for codon, idx in codon2idx.items():
    idx2codons.setdefault(idx, []).append(codon)

# 5. Conversion function

def seq_to_codon_indices(seq: str) -> list[int]:
    """
    Split the DNA sequence into codons and map each to its amino-acid index.
    - Requires len(seq) % 3 == 0.
    - Raises ValueError on unknown or ambiguous codons.
    """
    s = seq.upper().replace("\n","").replace(" ","")
    if len(s) % 3 != 0:
        raise ValueError("Sequence length is not a multiple of 3")
    indices = []
    for i in range(0, len(s), 3):
        codon = s[i:i+3]
        if codon not in codon2idx:
            raise ValueError(f"Unknown or ambiguous codon: '{codon}'")
        indices.append(codon2idx[codon])
    return indices

# === Example Usage ===
# example_seq = "ATGAAATGCTAGTTT"
# # Convert to indices
# codon_ids = seq_to_codon_indices(example_seq)
# print("Codon IDs:", codon_ids)
# # Map back to amino-acid symbols
# print("AAs:", [idx2aa[i] for i in codon_ids])

# Reference table (index → amino acid, codons)
print("\nIndex mapping:")
for idx in sorted(idx2aa):
    aa = idx2aa[idx]
    print(f" {idx:2d}: '{aa}'", "→", idx2codons[idx])


#%% Create list of amino acids
exons = []
print(len(all_records))
for rec in all_records:
    sequence = str(rec.seq)
    amino_acids = seq_to_codon_indices(sequence)
    if amino_acids[0] == 11 and amino_acids[-1] == 0:
        exons.append(amino_acids)
print(len(exons))

#%% Padding

# 1) Turning each list into a 1D LongTensor
tensor_seqs = [torch.tensor(seq, dtype=torch.long) for seq in exons]
# 2) Choosing a padding index that is **not** used by the real codons
pad_idx = vocab_size  # e.g. 21 if your real indices run 0–20
# 3) Pad them into a single (batch_size × max_len) tensor
#      shorter sequences get filled with pad_idx
padded = pad_sequence(tensor_seqs,
                      batch_first=True,
                      padding_value=pad_idx)
vocab_size += 1
sequence_len = len(padded[0])

#%% Graph
# import matplotlib.pyplot as plt
#
# # 1. Extract lengths (in bp)
# lengths = [len(gene) for gene in exons]
#
# # 2. Histogram of lengths
# plt.figure()
# plt.hist(lengths, bins=250)           # you can adjust bin count
# plt.xlabel("Sequence length (codons)")
# plt.ylabel("Count")
# plt.title("Distribution of Sequence Lengths")
# plt.show()


#%%






