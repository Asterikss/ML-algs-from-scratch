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



def main():
    ...


if __name__ == "__main__":
    main()
