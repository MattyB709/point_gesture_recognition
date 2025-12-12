#!/usr/bin/env python3
import os
import argparse

def average_8th_line(folder):
    
    total = 0.0
    count = 0

    # Loop over all files in the folder
    for name in os.listdir(folder):
        if not name.endswith(".txt"):
            continue

        path = os.path.join(folder, name)

        # Read lines and check if there is an 8th line
        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except OSError as e:
            print(f"Skipping {name}: could not open file ({e})")
            continue

        if len(lines) < 8:
            # No 8th line
            continue

        raw = lines[7].strip()
        if not raw:
            # Empty 8th line
            continue

        try:
            value = float(raw)
        except ValueError:
            print(f"Skipping {name}: 8th line is not a number -> {raw!r}")
            continue

        total += value
        count += 1

    if count == 0:
        print("No valid 8th-line values found.")
    else:
        avg = total / count
        print(f"Used {count} file(s).")
        print(f"Average of 8th-line values: {avg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Average the 8th line across all .txt files in a folder."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing .txt files"
    )
    args = parser.parse_args()
    average_8th_line(args.folder)
