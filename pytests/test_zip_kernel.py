import pytest
from refnd.kernels import KernelVariant, zip_kernel

DATA1 = ["AEP", "PEA", "ACDEFGHIKLM"]
DATA2 = ["PPP", "AEA", "ACDEFGHIKLM"]


def test_zip_kernel_thread_consistency():
    s1 = zip_kernel(KernelVariant.AlignmentGlobal, DATA1, DATA2, n_threads=1, progress=False)
    s2 = zip_kernel(KernelVariant.AlignmentGlobal, DATA1, DATA2, n_threads=2, progress=False)
    assert len(s1) == len(DATA1)
    assert s1 == s2


def test_zip_kernel_length_mismatch():
    with pytest.raises(ValueError):
        zip_kernel(KernelVariant.AlignmentGlobal, DATA1, DATA2[:2], progress=False)


if __name__ == "__main__":
    test_zip_kernel_thread_consistency()
    print("Thread consistency OK")
    test_zip_kernel_length_mismatch()
    print("Length mismatch OK")
