"""Execution-result comparison for Spider evaluation.

Matches the semantics of the Spider test-suite evaluation: two result tables
are equal if the predicted rows can be matched to the gold rows as unordered
multisets, allowing the columns to be permuted (SELECT a, b vs SELECT b, a is
not an error). Extra or missing columns are still wrong.
"""

from collections import Counter
from itertools import permutations

_MAX_PERM_COLS = 6  # brute-force column permutations only for narrow tables


def results_match(pred_rows: list[tuple], gold_rows: list[tuple]) -> bool:
    if len(pred_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return True

    ncols_gold = len(gold_rows[0])
    ncols_pred = len(pred_rows[0])
    if ncols_gold != ncols_pred:
        return False

    gold = Counter(tuple(str(v) for v in row) for row in gold_rows)
    pred = [tuple(str(v) for v in row) for row in pred_rows]

    if Counter(pred) == gold:
        return True
    if ncols_gold == 1 or ncols_gold > _MAX_PERM_COLS:
        return False

    # try every column reordering of the prediction
    for perm in permutations(range(ncols_gold)):
        if Counter(tuple(row[i] for i in perm) for row in pred) == gold:
            return True
    return False
