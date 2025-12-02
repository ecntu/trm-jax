import numpy as np
import jax
from collections import deque
from datasets import Dataset


class Loader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size,
        epochs=None,
        shuffle_seed=None,
        prefetch_size=2,
    ):
        if shuffle_seed is not None:
            dataset = dataset.shuffle(seed=shuffle_seed)
        dataset = dataset.with_format("numpy")

        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.prefetch_size = prefetch_size

    def iter_batch(self):
        N, bs = len(self.dataset), self.batch_size
        epochs = self.epochs
        while epochs is None or epochs > 0:
            for i in range(0, N, bs):
                j = min(i + bs, N)
                if (j - i) < bs:  # always drop incomplete batch
                    break
                batch = self.dataset[i:j]
                yield {
                    "inputs": np.asarray(batch["inputs"], dtype=np.int32),
                    "labels": np.asarray(batch["labels"], dtype=np.int32),
                }
            if epochs is not None:
                epochs -= 1

    def prefetch_to_device(self):  # TODO benchmark to see if necessary
        size = self.prefetch_size
        it = iter(self.iter_batch())
        put_in_device = jax.device_put

        if size <= 0:
            yield from self.iter_batch()
            return

        # prime
        q = deque(
            (put_in_device(batch) for _, batch in zip(range(size), it)), maxlen=size
        )

        for batch in it:
            if q:
                yield q.popleft()
            q.append(put_in_device(batch))

        while q:
            yield q.popleft()

    def __iter__(self):
        return self.prefetch_to_device()
