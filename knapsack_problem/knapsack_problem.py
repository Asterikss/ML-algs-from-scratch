import logging
from pathlib import Path
import random


def ask_for_data_loc():
    while True:
        answer = input("For default data location type 1. Otherwise type 0: ")
        if answer == "1":
            # not including "/" does not work when
            # oppening a file with .open()
            # Will look into it later
            return Path("data/" + "knapsacks.txt")
        elif answer == "0":
            while True:
                custom_path = Path((input("Enter custom data location: ")))
                if custom_path.exists():
                    return custom_path
                print("Path not found")


def download_datasets(data_loc: Path) -> list[list[tuple[int, int]]]: # pure
    datasets = []

    desired_lines = lambda l: l.find("{") != -1
    # checking the index of { twice. Maby could use a function that changes the state
    # outside of it, so it can "return" two values and still can be used inside filter()

    with data_loc.open(mode="r") as f:
        tmp_sizes = []
        tmp_values = []

        for i, line in enumerate(filter(desired_lines, f)):
            print(i)

            for val in line[line.find("{") + 1:line.find("}")].replace(",", " ").split():
                if i%2 != 0:
                    tmp_values.append(val)
                else:
                    tmp_sizes.append(val)
                    
            if i%2 != 0:
                datasets.append([(tmp_sizes[j], tmp_values[j]) for j in range(len(tmp_values))])
                tmp_values.clear()
                tmp_sizes.clear()
            
    # logging.debug(datasets)

    return datasets


def brute_force(dataset: list[tuple[int, int]]):
    ...


def init():
    # level = logging.INFO
    level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt) # filename = "log.log", filemode = "w"


def main():
    init()
    data_loc: Path = ask_for_data_loc()
    # v - [ [(3 - size, 7 - value), ..., (1, 4)], ..., [(2, 3), ..., (4, 5)] ]
    dataset_examples: list[list[tuple[int, int]]] = download_datasets(data_loc)
    dataset = random.choice(dataset_examples)
    print(dataset)
    brute_force(dataset)
    

if __name__ == "__main__":
    main()
