"""Utility functions for working with protein sequences
"""
import os
from Bio import SeqIO # noqa
from typing import Union, List


def load_fasta_file(seq_file, returnStr=True) -> Union[str, List]:
    """Load a fasta file.

    :param seq_file: file to read (fasta) sequence from.
    :param returnStr: whether to return string representation (default) or list.
    :return: sequence as string or list.
    """
    if not os.path.isfile(seq_file) or not seq_file.endswith(".fasta"):
        raise Exception('ERROR: an invalid sequence file: ', seq_file)
    record = SeqIO.read(seq_file, "fasta")
    return str(record.seq) if returnStr else record.seq
