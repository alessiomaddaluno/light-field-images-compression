import argparse
from enum import Enum
from supported_algos import Algorithm
from compressor import Compressor,Decompressor



def main(args):
    if args.mode == "compress":
        for algo_name in args.algos:
            algo = Algorithm[algo_name]
            print(f"Compressing using {algo.name}...")
            Compressor.compress(algo, args.dataset, f"output/")
    else:  # decompress
        print(f"Decompressing file {args.compressed_file}...")
        Decompressor.decompress(args.compressed_file, f"decompressed/")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compress and decompress Light Field Images using various codecs and algorithms.')

    parser.add_argument(
        "--mode",
        choices=["compress", "decompress"],
        default="compress",
        help="Mode: compress or decompress (default: compress)"
    )

    parser.add_argument(
        "--algos",
        nargs="+",
        choices=[algo.name for algo in Algorithm],
        help="Algorithms to use (required in compress mode)"
    )

    parser.add_argument(
        "--dataset",
        help="Path to the dataset (required in compress mode)"
    )

    parser.add_argument(
        "--compressed-file",
        help="Path to the compressed file (required in decompress mode)"
    )

    args = parser.parse_args()

    # Validazione
    if args.mode == "compress":
        if not args.algos:
            parser.error("--algos is required when --mode is 'compress'")
        if not args.dataset:
            parser.error("--dataset is required when --mode is 'compress'")

        print("Algorithms:", args.algos)
        print("Dataset:", args.dataset)
       

    else:  # decompress
        if not args.compressed_file:
            parser.error("--compressed-file is required when --mode is 'decompress'")

    main(args)
