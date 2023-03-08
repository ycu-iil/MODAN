from itertools import chain

# Amino acid name
L_AA_dict = {
    'A': 'Alanine',
    'C': 'Cysteine',
    'D': 'Aspartic acid',
    'E': 'Glutamic acid',
    'F': 'Phenylalanine',
    'G': 'Glycine',
    'H': 'Histidine',
    'I': 'Isoleucine',
    'K': 'Lysine',
    'L': 'Leucine',
    'M': 'Methionine',
    'N': 'Asparagine',
    'P': 'Proline',
    'Q': 'Glutamine',
    'R': 'Arginine',
    'S': 'Serine',
    'T': 'Threonine',
    'V': 'Valine',
    'W': 'Tryptophane',
    'Y': 'Tyrosine',
    'O': 'Orthinine',
    'X0': 'L-homoserine-(O-allyl)',
    'J': 'Dab',
    'X2': 'Sarcosine',
    'B': 'Ac5c',
    'U': 'Aib',
    'Z': 'Ac6c',
    'S5': '(S)-2-(4-pentenyl)Alanine',
    'R8': '(R)-2-(7-pentenyl)Alanine'
}
D_AA_dict = {
    'a': 'D-Alanine',
    'c': 'D-Cysteine',
    'd': 'D-Aspartic acid',
    'e': 'D-Glutamic acid',
    'f': 'D-Phenylalanine',
    'g': 'D-Glycine',
    'h': 'D-Histidine',
    'i': 'D-Isoleucine',
    'k': 'D-Lysine',
    'l': 'D-Leucine',
    'm': 'D-Methionine',
    'n': 'D-Asparagine',
    'p': 'D-Proline',
    'q': 'D-Glutamine',
    'r': 'D-Arginine',
    's': 'D-Serine',
    't': 'D-Threonine',
    'v': 'D-Valine',
    'w': 'D-Tryptophane',
    'y': 'D-Tyrosine',
    'o': 'D-Orthinine',
    'x0': 'D-L-homoserine-(O-allyl)',
    'x1': 'D-Dab',
    'x2': 'D-Sarcosine',
    'b': 'D-Ac5c',
    'u': 'D-Aib',
    'z': 'D-Ac6c',
    's5': 'D-(S)-2-(4-pentenyl)Alanine',
    'r8': 'D-(R)-2-(7-pentenyl)Alanine',
}

#Side-chain stapling
CL_dict = {
    '=': 'Link'
}
AA_dict = dict(
    chain.from_iterable(d.items() for d in (
        L_AA_dict,
        D_AA_dict,
        CL_dict))
)

#SMILES representation of side chain of L-amino acids
AA_joint = {
    'A': '[1*]C',
    'C': '[1*]CS',
    'D': 'OC(=O)C[1*]',
    'E': 'OC(=O)CC[1*]',
    'F': '[1*]CC1=CC=CC=C1',
    'G': '[1*][H]',
    'H': '[1*]CC1=CN=CN1',
    'I': 'CC[C@H](C)[1*]',
    'K': 'NCCCC[1*]',
    'L': 'CC(C)C[1*]',
    'M': 'CSCC[1*]',
    'N': 'NC(=O)C[1*]',
    'P': 'Proline',
    'Q': 'NC(=O)CC[1*]',
    'R': 'NC(N)=NCCC[1*]',
    'S': 'OC[1*]',
    'T': 'C[C@@H](O)[1*]',
    'V': 'CC(C)[1*]',
    'W': '[1*]CC1=CNC2=CC=CC=C12',
    'Y': 'OC1=CC=C(C[1*])C=C1',
    'O': 'NCCC[1*]',
    'X0': '[1*]CCOCC=C',
    'J': 'NCC[1*]',
    'X2': 'Sarcosine',
    'B': 'C1CC[1*]C1',
    'U': 'Aib',
    'Z': 'C1CC[1*]CC1',  
    'S5': 'C[1*]CCC\C=[300*]',
    'R8': 'C[1*]CCCCCCC=[300*]'
}