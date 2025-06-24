from pathlib import Path
import argparse


def main():
    """
    Convert trigger codes to eyes open and close (4 and 6).
    """

    parser = argparse.ArgumentParser(description=("""
    Notes
    -----
    This script is mainly designed for old Regensburg data.
    """
    ),
    formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("fname", help="file path to the vhdr eeg recording.")
    args = parser.parse_args()
    fname = vars(args)["fname"]
    fname = Path(fname)

    #######
    with open(f"{fname.with_suffix('')}.vmrk", "r") as file:
        lines = file.readlines()

    new_lines = lines[:11]
    for line in lines:

        for tr_ec_id in ["111", "112", "113", "114", "115"]:
            if f"=Stimulus,S {tr_ec_id}" in line:
                new_line = line.replace(f"=Stimulus,S {tr_ec_id}", "=Stimulus,S 6")
                new_lines.append(new_line)

        for tr_eo_id in ["331", "332", "333", "334", "335"]:
            if f"=Stimulus,S {tr_eo_id}" in line:
                new_line = line.replace(f"=Stimulus,S {tr_eo_id}", "=Stimulus,S 4")
                new_lines.append(new_line)

    with open(f"{fname.with_suffix('')}.vmrk", "w") as file:
        for line in new_lines:
            file.write(line)
    
    print("trigger IDs changed to 4 (eo) and 6 (ec), now you are safe to move ...")


if __name__ == "__main__":
    main()