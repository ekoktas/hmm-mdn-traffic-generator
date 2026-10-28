"""
GMMHMM of hmmlearn with two guards in the M-step.

1. A state or mixture component that receives no posterior mass keeps its previous emission
   parameters. hmmlearn would compute 0/0 there; this happens because the MAP update of the
   transition matrix, max(N + Lambda - 1, 0), can set transition probabilities to exactly zero.
2. A variance that collapses to zero or becomes non-finite is reset to min_covar. hmmlearn
   applies min_covar only at initialization, not during EM.

When neither case occurs, the fit is identical to that of hmmlearn's GMMHMM.
"""
import numpy as np
from hmmlearn import hmm


class SafeGMMHMM(hmm.GMMHMM):
    def _do_mstep(self, stats):
        dead_state = stats["post_sum"] <= 0          # (K,)
        dead_comp = stats["post_mix_sum"] <= 0       # (K, n_mix)
        w, m, c = self.weights_.copy(), self.means_.copy(), self.covars_.copy()
        super()._do_mstep(stats)
        if dead_state.any():
            self.weights_[dead_state] = w[dead_state]
        if dead_comp.any():
            self.means_[dead_comp] = m[dead_comp]
            self.covars_[dead_comp] = c[dead_comp]
        bad = ~np.isfinite(self.covars_) | (self.covars_ <= 0)
        if bad.any():
            self.covars_[bad] = self.min_covar
