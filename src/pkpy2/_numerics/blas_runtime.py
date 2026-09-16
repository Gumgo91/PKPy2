"""One BLAS thread inside PKAgent's outer subject parallelism.

BLAS limits are process-wide. Overlapping/nested PKAgent pools share one lease
so closing one pool cannot restore internal threads while another is working.
"""
from threading import Lock
from threadpoolctl import threadpool_limits

_lock = Lock()
_users = 0
_limit = None


class BlasLease:
    def __init__(self):
        # Load both NumPy and SciPy BLAS before discovering libraries. Numba's
        # linear algebra also uses SciPy's BLAS bindings.
        import numpy
        import scipy.linalg
        global _users, _limit
        self.closed = True
        with _lock:
            if _users == 0:
                _limit = threadpool_limits(limits=1, user_api='blas')
            _users += 1
            self.closed = False

    def close(self):
        global _users, _limit
        with _lock:
            if self.closed:
                return
            self.closed = True
            _users -= 1
            if _users == 0:
                try:
                    _limit.restore_original_limits()
                finally:
                    _limit = None
