import numpy as np

REVERSE_GLOBAL_LABEL_DICT = {
    0: "NO_SP",
    1: "SP",
    2: "LIPO",
    3: "TAT",
    4: "TATLIPO",
    5: "PILIN",
}
SIGNALP_KINGDOM_DICT = {"EUKARYA": 0, "POSITIVE": 1, "NEGATIVE": 2, "ARCHAEA": 3}
REV_SIGNALP_KINGDOM_DICT = {v: k for k, v in SIGNALP_KINGDOM_DICT.items()}

def tagged_seq_to_cs_multiclass(tagged_seqs: np.ndarray, sp_tokens=[0, 4, 5]):
    """Convert a sequences of tokens to the index of the cleavage site.
    Inputs:
        tagged_seqs: (batch_size, seq_len) integer array of position-wise labels
        sp_tokens: label tokens that indicate a signal peptide
    Returns:
        cs_sites: (batch_size) integer array of last position that is a SP. NaN if no SP present in sequence.
    """

    def get_last_sp_idx(x: np.ndarray) -> int:
        """Func1d to get the last index that is tagged as SP. use with np.apply_along_axis. """
        sp_idx = np.where(np.isin(x, sp_tokens))[0]
        if len(sp_idx) > 0:
            max_idx = sp_idx.max() + 1
        else:
            max_idx = np.nan
        return max_idx

    cs_sites = np.apply_along_axis(get_last_sp_idx, 1, tagged_seqs)
    return cs_sites
