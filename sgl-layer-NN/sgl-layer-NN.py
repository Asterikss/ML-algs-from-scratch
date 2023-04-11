"Single layer neural network used for predicting the language of a piece of text"
from enum import Enum
from dataclasses import dataclass
import logging
import random
import os


@dataclass(frozen=True)
class DefaultVars:
    n_neurons = 4 # display it later
    # level = logging.INFO
    level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt) # filename = 'log_x.log', filemode = "w"


class Vars:
    data_loc = ""
    train_data = []


class TypeOfRead(Enum):
    TRAINING = 0
    PREDICTING = 1


class Layer:
    ...


class Neuron:
    ...


def ask_for_data_loc() -> str: # pure
    while True:
        answer = int(input("For default data location type 1. Otherwise type 0: "))
        if answer == 1:
            return "data/training"
        elif answer == 0:
            return str(input("Enter custom data location: "))
        else:
            print("Enter valid input")


def download_data_set(root_directory: str, read_type: TypeOfRead) -> list[list[int]]:
    logging.info("v")
    logging.info("downloading data set")
    collected_data = []
    # Iterate over all files and directories in the root directory recursively
    # for dirpath, dirnames, filenames in os.walk(root_directory):
    for dirpath, _, filenames in os.walk(root_directory):
        for fname in filenames:
            if fname.endswith(".txt"): # check later if this is needed
                dir_name = os.path.basename(dirpath)

                with open(os.path.join(dirpath, fname), 'r') as file:
                    data = file.read() # Maby there is smth more effc
                
                vec = [0 for _ in range(27)]
                for char in data:
                    # logging.StreamHandler.terminator = "  " #
                    # logging.debug(char) #
                    # logging.StreamHandler.terminator = "\n" #
                    in_ascii = ord(char.lower())
                    if 96 < in_ascii < 123:
                        vec[in_ascii - 97] += 1
    
                logging.debug(dir_name)
                if dir_name == "english":
                    vec[-1] = 0
                if dir_name == "polish":
                    vec[-1] = 1
                logging.debug(vec)
                collected_data.append(vec)
                break

    return collected_data


def train() -> Layer:
    Vars.train_data = download_data_set(Vars.data_loc, TypeOfRead.TRAINING)
    return Layer()


def main():
    Vars.data_loc = ask_for_data_loc()
    layer: Layer = train()


if __name__ == "__main__":
    main()
