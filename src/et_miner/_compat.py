"""Internal compatibility utilities - avoids circular imports."""

# Soft tqdm import
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    class tqdm:
        """No-op fallback when tqdm not installed."""

        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

        def close(self):
            pass
