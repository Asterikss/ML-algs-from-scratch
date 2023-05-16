import logging
import pathlib


def ask_for_data_loc():
    while True:
        answer = input("For default data location type 1. Otherwise type 0: ")
        if answer == "1":
            # not including "/" does not work when
            # oppening a file with: with open...
            # Will look into it later
            return pathlib.Path("data/" + "iris_training.txt")
        elif answer == "0":
            while True:
                custom_path = pathlib.Path((input("Enter custom data location: ")))
                if custom_path.exists():
                    return custom_path
                print("Path not found")


def download_datasets(data_loc: pathlib.Path):
    return [[(3, 7), (1, 4)], [(2, 3), (4, 5)]]


def main():
    data_loc: pathlib.Path = ask_for_data_loc()
    dataset_examples: list[list[]] = download_datasets(data_loc)
    

if __name__ == "__main__":
    main()
