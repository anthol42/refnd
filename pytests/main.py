

def main():
    import refnd
    print(dir(py_proto))
    from py_proto import kernels
    print(dir(kernels))
    from refnd.kernels import protein
    print(dir(protein))
    from refnd.kernels.protein import sequence
    print(dir(sequence))
    from refnd.kernels.protein.sequence import LocalAligner, GlobalAligner
    print(LocalAligner(), GlobalAligner())


if __name__ == "__main__":
    main()
