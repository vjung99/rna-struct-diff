import biotite.structure.io.pdb as bspdb
from biotite.structure import Atom, array
from dataset import INT_TO_BASE

BASE_ATOMS = {
    'A': ['P','OP1','OP2',"O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'",
          'N9','C8','N7','C5','C6','N6','N1','C2','N3','C4'],
    'C': ['P','OP1','OP2',"O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'",
          'N1','C2','O2','N3','C4','N4','C5','C6'],
    'G': ['P','OP1','OP2',"O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'",
          'N9','C8','N7','C5','C6','O6','N1','C2','N2','N3','C4'],
    'U': ['P','OP1','OP2',"O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'",
          'N1','C2','O2','N3','C4','O4','C5','C6'],
}

def save_pdb(path, seq, x_A, v_A):
    """seq: (L,) int tokens 0-3; x_A: (L,3) Ångströms; v_A: (L,72) Ångströms"""
    atoms = []
    for i, tok in enumerate(seq, start=1):
        base = INT_TO_BASE[int(tok)]                                # 'A'/'C'/'G'/'U'
        for slot, name in enumerate(BASE_ATOMS[base]):
            coord = x_A[i-1] + v_A[i-1, slot*3 : slot*3+3]
            atoms.append(Atom(
                coord=coord,
                atom_name=name,
                res_name=base,                                # 1-letter is standard for RNA
                res_id=i,
                chain_id='A',
                element=name[0],                              # P / O / C / N
            ))
    pdb_file = bspdb.PDBFile()
    pdb_file.set_structure(array(atoms))
    pdb_file.write(path)
