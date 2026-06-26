import os

def set_deterministic_env():
    # Force deterministic hashing and single-thread BLAS to avoid order-dependent nondeterminism
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
