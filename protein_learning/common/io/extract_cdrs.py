from typing import List

from protein_learning.common.io.pdb_utils import extract_pdb_seq_from_pdb_file


def extract_cdr_posns(pdb_file: str, heavy_chain_ids: List[str], light_chain_ids: List[str]):
    """Extract 0-indexed positions of CDRs
    Params:
        pdb_file: Antibody pdb file (chothia format)
        heavy_chain_ids: List of chain ids for heavy chains in pdb
        light_chain_ids: List of chain ids for light chains

    Returns:
        tuple of dicts
        heavy_chain_cdrs : chain_id -> Tuple((cdr1_start,cdr_2_end),...,(cdr3_start,cdr3_end))
        light_chain_cdrs (analogous)

        The cdr start/end positions are INCLUSIVE, i.e. cdr1_end is part of cdr1.
        The cdr start/end positions are given w.r.t the 0-indexed residues in the respective chain.
        i.e. if chain residues are indexed as [1...L], and res_{cdr3_start},...,res_{cdr3_end}
        comprise cdr3.
    """
    seqs, residues, chains = extract_pdb_seq_from_pdb_file(pdb_file)
    seqs = {c.get_id(): seqs[i] for i, c in enumerate(chains)}
    heavy_cdrs, light_cdrs = {}, {}
    for chain in chains:
        chain_id = chain.get_id()
        if chain_id in heavy_chain_ids:
            heavy_cdrs[chain.get_id()] = detect_heavy_chain_cdrs(
                seqs[chain_id], [r for r in chain.get_residues()]
            )
        if chain_id in light_chain_ids:
            light_cdrs[chain_id] = detect_light_chain_cdrs(
                seqs[chain_id], [r for r in chain.get_residues()]
            )
    return heavy_cdrs, light_cdrs


def detect_heavy_chain_cdrs(seq, residues):
    start_posns, end_posns = [-1, -1, -1], [-1, -1, -1]
    # cdr_keys = set("25 26 32 33 35 50 52 52A 52a 53 56 58 95 96 101 102 104".split())
    for res_idx, res in enumerate(residues):
        res_label = str(res.get_id()[1])
        # if res_label not in cdr_keys:
        #    continue

        # CDR-H1 start
        if res_label == "25" or res_label == "26":
            # if seq[res_idx - 4] != "C":
            #    continue
            # if start_posns[0] > 0:
            #    continue
            start_posns[0] = res_idx

        # CDR-H1 end pos (alt. definitions)
        if res_label in ["32", "33", "35"]:
            if end_posns[0] > 0:
                continue
            # if seq[res_idx+1]!="W":
            #    continue
            end_posns[0] = res_idx

        # CDR H2 start
        if res_label in ["52", "52A", "52a"]:
            if start_posns[1] >= 0:
                continue
            start_posns[1] = res_idx

        # CDR H2 End
        if res_label in ['56']:
            # take the last residue with label 56
            end_posns[1] = res_idx

        # CDR H3 Start
        if res_label in ['92', '95', '96',"95A"]:
            if res_label == "96" and start_posns[2] < 0:
                start_posns[2] = res_idx - 1
            if seq[res_idx - 3] != "C":
                continue
            if start_posns[2] > 0:
                continue
            start_posns[2] = res_idx

        # CDR-H3 End
        if res_label in ["102", "104"]:
            # if seq[res_idx+1]!="W":
            #    continue
            if end_posns[2] > 0:
                continue
            end_posns[2] = res_idx

    cdrs = [(s, e) for s, e in zip(start_posns, end_posns)]
    # cdr_seqs = [seq[s:e + 1] if e > s > 0 else None for s, e in zip(start_posns, end_posns)]
    # return {"cdrs": cdrs, "seqs": cdr_seqs}
    return cdrs


def detect_light_chain_cdrs(seq, residues):
    start_posns, end_posns = [-1, -1, -1], [-1, -1, -1]
    cdr_keys = set("26 32 50 52 91 96".split())
    for res_idx, res in enumerate(residues):
        res_label = str(res.get_id()[1])
        if res_label not in cdr_keys:
            continue

        # CDR-L1 start
        if res_label == "26":
            if start_posns[0] > 0:
                continue
            start_posns[0] = res_idx

        # CDR-L1 end pos
        if res_label in ["32", ]:
            if end_posns[0] > 0:
                continue
            end_posns[0] = res_idx

        # CDR L2 start
        if res_label in ["50"]:
            if start_posns[1] >= 0:
                continue
            start_posns[1] = res_idx

        # CDR L2 End
        if res_label in ['52']:
            if len(res.get_id()) > 2:
                if res.get_id()[2] == "":
                    end_posns[1] = res_idx
            if end_posns[1] >= 0:
                continue
            # take the last residue with label 56
            end_posns[1] = res_idx

        # CDR L3 Start
        if res_label in ['91']:
            if start_posns[2] > 0:
                continue
            start_posns[2] = res_idx

        # CDR-L3 End
        if res_label in ["96"]:
            if end_posns[2] > 0:
                continue
            end_posns[2] = res_idx

    cdrs = [(s, e) for s, e in zip(start_posns, end_posns)]
    # cdr_seqs = [seq[s:e + 1] if e > s > 0 else None for s, e in zip(start_posns, end_posns)]
    # return {"cdrs": cdrs, "seqs": cdr_seqs}
    return cdrs
