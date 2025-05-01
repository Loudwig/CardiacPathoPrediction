import numpy as np
from scipy.stats import entropy

def certainty_score(probs: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute a certainty score for each row of class-probabilities.
    """
    
    probs = np.asarray(probs)
    # 1) compute entropy H for each sample
    #    scipy.stats.entropy sums p * log(p), default base=e
    H = entropy(probs.T)  # .T so we entropy over rows
    
    if normalize:
        # maximum entropy is log(n_classes)
        H_max = np.log(probs.shape[1])
        # certainty = 1 − (H / H_max)
        cert = 1 - H / H_max
    else:
        # raw negative entropy (higher is more certain)
        cert = -H

    return cert

def margin_score(probs: np.ndarray) -> np.ndarray:
    
    # sort each row descending, then subtract second-best from best
    top2 = -np.sort(-probs, axis=1)[:, :2]
    return top2[:, 0] - top2[:, 1]