"""Merge case index files into one de-duplicated list.

The three files produced by ``crawl_case_list.py`` overlap, because a case can
appear in more than one section. Cases are de-duplicated by URL.
"""

import argparse
import json

DEFAULT_INPUTS = ["recommended_cases.json", "selected_cases.json", "newest_cases.json"]
DEFAULT_OUTPUT = "merged_cases.json"


def merge_json_files(input_files, output_file):
    """Merge ``input_files`` into ``output_file`` and return the merged list."""

    unique = {}

    for path in input_files:
        with open(path, encoding="utf-8") as f:
            for case in json.load(f):
                # The URL is the stable identity of a case.
                unique[case["url"]] = case

    result = list(unique.values())

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"merged {len(result)} cases -> {output_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Merge case index files.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=DEFAULT_INPUTS,
        help="index files to merge, defaults to the three section files",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="merged output file")

    args = parser.parse_args()

    merge_json_files(args.inputs, args.output)


if __name__ == "__main__":
    main()
