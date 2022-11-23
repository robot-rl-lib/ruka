import numpy as np

from typing import List


def ascii_hists(
    values: List[List[float]],
    labels: List[str],
    bins: int = 10,
    bin_len: int = 25,
) -> str:
    """
    Returns ASCII stacked histograms representation.
    Args:
        values (List[List[float]]): list of lists of vamples
        labels (List[str]): histogram labels, for example ['*', '-']
        bins (int): bin count
        bin_len (int): bin length in chars

    Returns:
        str: ascii representation
    """
    minval = min([min(item_vals) for item_vals in values])
    maxval = max([max(item_vals) for item_vals in values])
    hists = [
        np.histogram(item_vals, bins=bins, range=(minval, maxval))
        for item_vals in values
    ]
    hist_vals = hists[0][1][:-1]
    maxcount = max([hs[0].max() for hs in hists])
    plot = ""
    for vali, val in enumerate(hist_vals):
        bin = ""
        counts = []
        for hi, hist in enumerate(hists):
            bin += labels[hi] * int(bin_len * hist[0][vali] / maxcount)
            counts.append(hist[0][vali])
        counts_str = ",".join([f"{cn:3d}" for cn in counts])
        plot += f"{val:.2f} : {counts_str} {bin}\n"
    return plot
