import logging
from pathlib import Path
import random
import itertools


class CapacityNotFound(Exception):
    pass


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


def download_datasets(data_loc: Path) -> tuple[list[list[tuple[int, int]]], int]: # pure
    datasets = []
    capacity = 0

    desired_lines = lambda l: l.find("{") != -1
    # checking the index of { twice. Maby could use a function that changes the state
    # outside of it, so it can "return" two values and still can be used inside filter()

    with data_loc.open(mode="r") as f:
        tmp_sizes = []
        tmp_values = []
    
        first_line = f.readline()
        cap_idx = first_line.index("capacity")
        first_line_capacity = first_line[cap_idx + 8 + 1:-1]

        if first_line_capacity.isdecimal():
            capacity = int(first_line_capacity)
            logging.debug(capacity)
        else:
            raise CapacityNotFound("Capacity could no be found. Be sure to use an int")


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

    return datasets, capacity


def brute_force(dataset: list[tuple[int, int]], capacity: int):
    all_combinations = [[True, False] for _ in range(len(dataset))]

    score = 0
    size = 0

    last_addition_size = 0
    last_addition_score = 0

    max_score = 0
    max_size = 0
    
    # This assumes that not all objects can fit in the knapsack
    for combination in itertools.product(*all_combinations):

        for i, switch in enumerate(combination):

            last_addition_score = switch * dataset[i][1]
            score += last_addition_score

            last_addition_size = switch * dataset[i][0]
            size += last_addition_size

            if size > capacity:
                max_score = score - last_addition_score
                max_size = size - last_addition_size
                
    
    return max_score, max_size


def init():
    # level = logging.INFO
    level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt) # filename = "log.log", filemode = "w"


def main():
    init()
    data_loc: Path = ask_for_data_loc()
    # v - [ [(3 - size, 7 - value), ..., (1, 4)], ...]
    dataset_examples, capacity = download_datasets(data_loc)
    dataset = random.choice(dataset_examples)
    print(dataset)

    max_score, max_size = brute_force(dataset, capacity)
    print(max_score, max_size)
    

if __name__ == "__main__":
    main()
