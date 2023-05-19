import logging
from pathlib import Path
import random
import itertools
import time


class CapacityNotFound(Exception):
    pass


def ask_for_data_loc():
    while True:
        answer = input("For default data location type 1. Otherwise type 0: ")
        if answer == "1":
            # not including "/" does not work when
            # oppening a file with .open()
            # Will look into it later
            return Path("data/" + "knapsacks2.txt")
        elif answer == "0":
            while True:
                custom_path = Path((input("Enter custom data location: ")))
                if custom_path.exists():
                    return custom_path
                print("Path not found")


def download_datasets(data_loc: Path) -> tuple[list[list[tuple[int, int]]], int]: # pure
    # v - [ [(3 - size, 7 - value), ...], ...]
    datasets = []
    capacity = 0

    desired_lines = lambda l: l.find("{") != -1
    # checking the index of { twice. Maby could use a function that changes the state
    # outside of it, so it can "return" "two values" and still could be used inside filter()

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
            logging.debug(i)

            for val in line[line.find("{") + 1:line.find("}")].replace(",", " ").split():
                if i%2 != 0:
                    tmp_values.append(int(val))
                else:
                    tmp_sizes.append(int(val))
                    
            if i%2 != 0:
                datasets.append([(tmp_sizes[j], tmp_values[j]) for j in range(len(tmp_values))])
                tmp_values.clear()
                tmp_sizes.clear()
            

    return datasets, capacity


def brute_force(dataset: list[tuple[int, int]], capacity: int) -> tuple[int, int, list[int], float]: # pure
    all_combinations = [[True, False] for _ in range(len(dataset))]
    # all_combinations = [[0, 1] for _ in range(len(dataset))]

    score = 0
    size = 0

    last_addition_size = 0
    last_addition_score = 0

    max_score = 0
    max_size = 0

    final_object_idxs = []
    
    # This assumes that not all objects can fit in the knapsack
    t0 = time.monotonic_ns()
    for combination in itertools.product(*all_combinations):
        object_idxs = []

        for i, switch in enumerate(combination):
            
            # print(switch)
            last_addition_score = switch * dataset[i][1]
            # print(last_addition_score)
            score += int(last_addition_score)
            # print(score)

            last_addition_size = switch * dataset[i][0]
            # print(last_addition_size)
            size += last_addition_size
            # print(size)

            if switch:
                object_idxs.append(i)

            if size > capacity:
                if (score - last_addition_score) > max_score :
                    max_score = score - last_addition_score
                    max_size = size - last_addition_size
                    object_idxs.pop()
                    final_object_idxs = object_idxs
                break
               

        score = 0
        size = 0
        

    t1 = time.monotonic_ns()
    seconds_needed = (t1-t0)/1000000000

    return max_score, max_size, final_object_idxs, seconds_needed


def print_results(dataset: list[tuple[int, int]], max_score: int, max_size: int,
                  final_object_idxs: list[int], capacity: int, time_sec_brute: float): 

    print("Dataset choosen:")
    print(dataset)
    print("Selected items:")
    for idx in final_object_idxs:
        print(f"idx - {idx} size - {dataset[idx][0]} value - {dataset[idx][1]}")

    print(f"Final score: {max_score}")
    print(f"Capacity used: {max_size}/{capacity}")
    print(f"Time taken: {time_sec_brute}")


def init():
    # level = logging.INFO
    level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt) # filename = "log.log", filemode = "w"


def main():
    init()
    data_loc: Path = ask_for_data_loc()
    dataset_examples, capacity = download_datasets(data_loc)

    dataset = random.choice(dataset_examples)

    max_score, max_size, final_object_idxs, time_sec_brute = brute_force(dataset, capacity)
    print_results(dataset, max_score, max_size, final_object_idxs, capacity, time_sec_brute)
    

if __name__ == "__main__":
    main()
