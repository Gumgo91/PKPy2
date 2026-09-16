"""Ordered, bounded subject work with a pool owned by one refinement call."""
from concurrent.futures import ThreadPoolExecutor
from .blas_runtime import BlasLease


def ordered_map(work, function, items):
    return list(map(function, items)) if work is None else work.map(function, items)


class SubjectWork:
    def __init__(self, workers, subjects):
        if type(workers) is not int or workers < 1:
            raise ValueError('positive integer importance workers required')
        self.workers = min(workers, subjects)
        self.lease = BlasLease() if self.workers > 1 else None
        try:
            self.pool = (ThreadPoolExecutor(max_workers=self.workers,
                         thread_name_prefix='pkagent-importance') if self.workers > 1 else None)
        except BaseException:
            if self.lease is not None: self.lease.close()
            raise

    def map(self, function, items):
        # Consume all futures before returning/raising: no worker may mutate an
        # evaluator after its caller starts recovery or another evaluation.
        if self.pool is None:
            return list(map(function, items))
        futures = [self.pool.submit(function, item) for item in items]
        results = []
        error = None
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error
        return results

    def close(self):
        try:
            if self.pool is not None:
                self.pool.shutdown(wait=True)
        finally:
            if self.lease is not None:
                self.lease.close()
