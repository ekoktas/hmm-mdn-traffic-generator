"""
Train/test split by flow, used by every model.

Flows, not packets, are split at random. A few flows can hold a large share of all packets,
so a purely random split can leave the training set with fewer packets than the test set.
The split is therefore redrawn until the training flows hold at least `min_keep_share` of the
packets. Draw 0 uses random_state=seed, i.e. it is the plain random split.
"""
import numpy as np
from sklearn.model_selection import train_test_split


def constrained_split(ids, sizes, test_size, seed, min_keep_share, max_draws=1000):
    """Split `ids` into a kept and a held-out part; return (keep, held, draw, keep_share).

    ids             flow ids to split
    sizes           pandas Series with the number of packets per flow, indexed by flow id
    test_size       share of the flows that goes to the held-out part (test or validation)
    seed            seed of draw 0; draw d uses seed + 1000 * d
    min_keep_share  minimum share of the packets of `ids` that the kept part must hold
    """
    ids = np.asarray(ids)
    total = float(sizes.loc[ids].sum())
    for draw in range(max_draws):
        keep, held = train_test_split(ids, test_size=test_size, random_state=seed + 1000 * draw)
        share = float(sizes.loc[keep].sum()) / total
        if share >= min_keep_share:
            return keep, held, draw, share
    raise RuntimeError(f"no split keeps >= {min_keep_share:.0%} of the packets in {max_draws} draws")
