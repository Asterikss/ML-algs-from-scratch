"Single layer neural network used for predicting the language of a piece of text"
from enum import Enum
from dataclasses import dataclass
import logging
import random

@dataclass(frozen=True)
class DefaultVars:
    n_neurons = 4
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
    missing_input = True
    data_loc = ""
    while missing_input:
        answer = int(input("For default data location type 1. Otherwise type 0: "))
        if answer == 1:
            data_loc = "data/iris_training.txt"
            missing_input = False
        elif answer == 0:
            data_loc = str(input("Enter custom data location: "))
            data_loc = data_loc
            missing_input = False
        else:
            print("Enter valid input")
    return data_loc


def download_data_set(data_loc: str, read_type: TypeOfRead):
    logging.info("v")
    logging.info("downloading data set")
    ...

def train() -> Layer:
    download_data_set(Vars.data_loc, TypeOfRead.TRAINING)
    return Layer()


def main():
    Vars.data_loc = ask_for_data_loc()
    layer: Layer = train()


if __name__ == "__main__":
    main()
