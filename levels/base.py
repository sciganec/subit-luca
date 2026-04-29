# levels/base.py
"""Abstract base class for evolutionary levels in SUBIT-LUCA."""

from abc import ABC, abstractmethod
from typing import List


class SubitLevel(ABC):
    """
    Abstract base class for an evolutionary level.
    
    Each level defines the set of SUBIT states (archetypes) that were
    accessible at that stage of evolution, from LUCA (level 0) to
    modern eukaryotes (level 5).
    """
    
    @abstractmethod
    def allowed_states(self) -> List[int]:
        """
        Return a list of state indices (0..63) that are possible at this level.
        
        The list must contain only integer indices representing valid
        6‑bit SUBIT states. The set of allowed states should be monotonic:
        each higher level includes all states of lower levels plus new ones.
        """
        pass
    
    def project(self, p64):
        """
        Project a probability distribution over all 64 states onto the
        allowed subspace of this level.
        
        Parameters
        ----------
        p64 : np.ndarray
            Array of shape (n_cells, 64) or (64,) containing probabilities.
        
        Returns
        -------
        np.ndarray
            Same shape as input but with forbidden states set to zero
            and renormalised so that each row sums to 1.
        """
        import numpy as np
        p = np.asarray(p64)
        allowed = set(self.allowed_states())
        # zero out forbidden states
        if p.ndim == 1:
            p_out = p.copy()
            for i in range(64):
                if i not in allowed:
                    p_out[i] = 0.0
            total = p_out.sum()
            if total > 0:
                p_out /= total
            else:
                # fallback: uniform over allowed states
                p_out[:] = 0.0
                for i in allowed:
                    p_out[i] = 1.0 / len(allowed)
            return p_out
        else:
            # 2D case (n_cells, 64)
            p_out = p.copy()
            # vectorised masking: column indices not in allowed set to zero
            mask = np.ones(64, dtype=bool)
            mask[list(allowed)] = False
            p_out[:, mask] = 0.0
            row_sums = p_out.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            p_out = p_out / row_sums
            return p_out