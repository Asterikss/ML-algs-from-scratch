"Single layer neural network used for predicting the language of a piece of text"
# from enum import Enum
from dataclasses import dataclass
import logging
import random
import os
import math


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


# class TypeOfRead(Enum):
#     TRAINING = 0
#     PREDICTING = 1


def sigmoid_func(x) -> float: # pure
    logging.debug(f"sig: {x/(1+ math.exp(-x))}")
    return x/(1+ math.exp(-x))


def dot_product(X: list, weights) -> float: # pure
    result = 0
    for x_n, w_n in zip(X, weights):
        result += x_n * w_n
    logging.StreamHandler.terminator = "  "
    logging.debug(round(result, 3))
    logging.StreamHandler.terminator = "\n"
    return round(result, 3) # maby without round


class Neuron:
    def __init__(self, n_inputs, activ_func=sigmoid_func, learning_rate=0.02) -> None:
        self.lr: float = learning_rate
        self.activation_func = activ_func
        self.weights = [round(random.uniform(-1, 1), 3) for _ in range(n_inputs)]
        self.bias = round(random.uniform(-1, 1), 3) # maby without round
        self.n_inputs = n_inputs


    def output(self, X) -> float:
        return self.activation_func(dot_product(X[:-1], self.weights) + self.bias)

    
    def get_n_of_inputs(self) -> int:
        return self.n_inputs


class Layer:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.neurons: list[Neuron] = [Neuron(n_inputs) for _ in range(n_neurons)]

    
    def output(self, X) -> list[float]:
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output(X)) 
        return outputs


def normalization(X) -> list[float]: # pure
    logging.debug(f"norm {X}")
    sq_sum = 0
    for i in range(len(X) - 1):
        sq_sum += math.pow(X[i], 2)
    # logging.debug(sq_sum)
    magnitude = math.sqrt(sq_sum)
    # logging.debug(magnitude)
    for i in range(len(X) - 1):
        X[i] /= magnitude
    logging.debug(f"norm {X}")
    return X


class NeuralNetwork():
    def __init__(self, n_inputs, layers: list[int]) -> None:
        self.layers = [Layer(n_inputs, layers[0])]
        self.layers += [Layer(layers[i], layers[i+1]) for i in range(len(layers) - 1)]
        self.n_inputs = n_inputs


    def feed_forward(self, X) -> list[float]:
        input = normalization(X)
        for layer in self.layers:
            input = layer.output(input)
        return input


    def train(self, train_data: list[list[int]]):
        output = []
        for X in train_data:
            output: list[float] = self.feed_forward(X)

        print(output)


    def show_arch(self):
        print(f"{self.n_inputs} -> ", end="")
        for layer in self.layers:
            print("| ", end="")
            for neuron in layer.neurons:
                print(f"N({neuron.get_n_of_inputs()})", end=" ")
        print("| ", end="")



def ask_for_data_loc() -> str: # pure
    while True:
        answer = int(input("For default data location type 1. Otherwise type 0: "))
        if answer == 1:
            return "data/training"
        elif answer == 0:
            return str(input("Enter custom data location: "))
        else:
            print("Enter valid input")


def download_data_set(root_directory: str) -> list[list[int]]: # pure
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


# def init_layer():
#     return Layer(26)


# def train(data_loc: str) -> NeuralNetwork:
#     train_data = download_data_set(data_loc)
#     neural_network: NeuralNetwork = 
#     return NeuralNetwork([3])


def main():
    data_loc = ask_for_data_loc()
    train_data = download_data_set(data_loc)
    neural_network: NeuralNetwork = NeuralNetwork(26, [3, 4, 2])
    neural_network.show_arch()
    # print("SDDDDDDDDDDDDDDDDDDDDDDD")
    neural_network.train(train_data)


if __name__ == "__main__":
    main()
