"""Complete reference alternatives and source-wide leakage dependencies."""

from collections import defaultdict
from hashlib import sha256


def fasta_records(path):
    """Yield original headers and unwrapped sequences, rejecting malformed data."""
    header, parts = None, []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:].strip(), []
            elif line.strip():
                if header is None:
                    raise ValueError("Sequence precedes FASTA header")
                parts.append(line.strip())
        if header is not None:
            yield header, "".join(parts)


def candidate_spans(references, peptides):
    """Enumerate every literal and I/L-equivalent span, including repeated spans."""
    queries = defaultdict(list)
    for peptide in peptides:
        sequence = peptide["peptide_sequence"]
        if not sequence or not sequence.isalpha() or sequence != sequence.upper():
            raise ValueError("Invalid source peptide sequence")
        queries[sequence.replace("L", "I")].append(peptide)
    width = min(5, min(map(len, queries), default=5))
    prefixes = defaultdict(list)
    for query in queries:
        prefixes[query[:width]].append(query)
    for reference in references:
        original = reference["reference_sequence"]
        normalized = original.replace("L", "I")
        for start in range(len(normalized) - width + 1):
            for query in prefixes.get(normalized[start : start + width], ()):
                if normalized.startswith(query, start):
                    for peptide in queries[query]:
                        yield (
                            peptide,
                            reference,
                            start + 1,
                            start + len(query),
                            (
                                "literal"
                                if original[start : start + len(query)]
                                == peptide["peptide_sequence"]
                                else "isoleucine_leucine_equivalent"
                            ),
                        )


class Dependencies:
    """All alternatives enter before eligibility; no preferred origin is chosen."""

    def __init__(self):
        self.parents = {}

    def find(self, node):
        self.parents.setdefault(node, node)
        while self.parents[node] != node:
            self.parents[node] = self.parents[self.parents[node]]
            node = self.parents[node]
        return node

    def link(self, peptide_id, parent):
        self.parents[self.find("peptide:" + peptide_id)] = self.find(parent)

    def groups(self, peptide_ids):
        for peptide_id in peptide_ids:
            self.find("peptide:" + peptide_id)
        members = defaultdict(list)
        for node in self.parents:
            members[self.find(node)].append(node)
        keys = {
            root: sha256("\n".join(sorted(nodes)).encode()).hexdigest()
            for root, nodes in members.items()
        }
        return {p: keys[self.find("peptide:" + p)] for p in peptide_ids}
