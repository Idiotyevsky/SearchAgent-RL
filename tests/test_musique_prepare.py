from argparse import Namespace

import pytest

from scripts.prepare_verl_musique import select_examples


def _args(hop_limits):
    return Namespace(
        hops=[2, 3, 4],
        hop_limits=hop_limits,
        per_hop_limit=None,
        start_index=0,
        limit=None,
    )


def test_selection_accepts_explicit_hop_quotas():
    examples = [
        Namespace(example_id=f"{hops}-{index}", hop_count=hops)
        for hops, count in ((2, 4), (3, 3), (4, 2))
        for index in range(count)
    ]

    selected = select_examples(examples, _args([3, 2, 1]))

    assert [example.hop_count for example in selected] == [2, 2, 2, 3, 3, 4]


def test_selection_requires_one_quota_per_hop():
    examples = [Namespace(example_id="2-0", hop_count=2)]

    with pytest.raises(ValueError, match="one quota for every requested hop"):
        select_examples(examples, _args([1, 1]))
