import csv
from collections import defaultdict


def clean_edgelist_csv(raw_file, cleaned_file, drop_first=0, sep=None):
    """
    Cleans a raw edgelist file, converting node IDs (strings, gene symbols, etc.)
    to zero-based integers, and saves it in CSV format (two columns).
    """
    node_map = defaultdict(lambda: len(node_map))

    with (
        open(raw_file, "r", encoding="utf-8") as rf,
        open(cleaned_file, "w", encoding="utf-8", newline="") as cf,
    ):
        # Skip the first `drop_first` lines (e.g., header lines)
        for _ in range(drop_first):
            next(rf, None)

        writer = csv.writer(cf, delimiter=",")
        for line in rf:
            line = line.strip()
            if not line:
                continue  # skip blank lines
            parts = line.split(sep=sep)  # If sep=None, splits on any whitespace
            # Expect exactly two columns in each line
            node1_str, node2_str = parts[0], parts[1]

            # Use the dictionary to assign an integer ID for each node label
            node1_id = node_map[node1_str]
            node2_id = node_map[node2_str]

            # Write to CSV. For example: "0,1"
            writer.writerow([node1_id, node2_id])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean edge list and convert to CSV format."
    )
    parser.add_argument("--input", required=True, help="Path to the raw edge list.")
    parser.add_argument("--output", required=True, help="Path to the CSV file to save.")
    parser.add_argument(
        "--drop", type=int, default=0, help="Lines to skip from the top of input."
    )
    parser.add_argument(
        "--sep", default=None, help="Delimiter: e.g. ' ', '\\t' or None for whitespace."
    )
    args = parser.parse_args()

    clean_edgelist_csv(
        raw_file=args.input,
        cleaned_file=args.output,
        drop_first=args.drop,
        sep=args.sep,
    )
