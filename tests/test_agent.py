from text2sql.agent import majority_pick, result_signature


def test_signature_is_order_insensitive():
    assert result_signature([(1, "a"), (2, "b")]) == result_signature([(2, "b"), (1, "a")])


def test_signature_distinguishes_different_results():
    assert result_signature([(1,)]) != result_signature([(2,)])


def test_majority_pick_prefers_agreeing_results():
    sig_a = result_signature([("USA", 13)])
    sig_b = result_signature([("Canada", 8)])
    candidates = [
        (sig_a, "sql1", ["c"], [("USA", 13)]),
        (sig_b, "sql2", ["c"], [("Canada", 8)]),
        (sig_a, "sql3", ["c"], [("USA", 13)]),
    ]
    _, sql, _, rows = majority_pick(candidates)
    assert sql == "sql1"  # first candidate of the winning group
    assert rows == [("USA", 13)]


def test_majority_pick_tie_takes_first():
    candidates = [
        (result_signature([(1,)]), "sql1", ["c"], [(1,)]),
        (result_signature([(2,)]), "sql2", ["c"], [(2,)]),
    ]
    assert majority_pick(candidates)[1] == "sql1"
