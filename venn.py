import sys
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3, venn3_circles
from venny4py.venny4py import *

def load_mirna_list(filename):
    """Load miRNA names from a text file (one per line)."""
    with open(filename, "r") as f:
        return set(line.strip() for line in f if line.strip())

if len(sys.argv) < 3:
    print("Usage: python venn_diagram.py <file1.txt> <file2.txt> [<file3.txt> ... <file4.txt>]")
    sys.exit(1)

print("FIles:", sys.argv[1:])
# Load miRNA lists from files
mirna_lists = [load_mirna_list(filename) for filename in sys.argv[1:]]

# Create Venn diagram
plt.figure(figsize=(8, 8))

if len(mirna_lists) == 2:
    venn2(mirna_lists, (sys.argv[1], sys.argv[2]))
elif len(mirna_lists) == 3:
    sets = {
        sys.argv[1].removesuffix('.txt'): mirna_lists[0],
        sys.argv[2].removesuffix('.txt'): mirna_lists[1],
        "CFS": mirna_lists[2]
    }
    venny4py(sets=sets)
elif len(mirna_lists) == 4:
    sets = {
        sys.argv[1].removesuffix('.txt'): mirna_lists[0],
        sys.argv[2].removesuffix('.txt'): mirna_lists[1],
        sys.argv[3].removesuffix('.txt'): mirna_lists[2],
        sys.argv[4].removesuffix('.txt'): mirna_lists[3]
    }
    venny4py(sets=sets)
else:
    print("Currently, only 2, 3, or 4 files are supported for Venn diagrams.")
    sys.exit(1)

# Show plot
#plt.title("Venn Diagram of miRNA Lists")
plt.savefig("venn_diagram.png", dpi=600, bbox_inches='tight')
