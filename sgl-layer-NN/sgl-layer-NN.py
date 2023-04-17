"Single layer neural network used for predicting the language of a piece of text"
from enum import Enum
from dataclasses import dataclass
import logging
import random
import os
import math


@dataclass(frozen=True)
class DefaultVars:
    langs = ["english", "polish"]
    level = logging.INFO
    # level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt) # filename = 'log_x.log', filemode = "w"


class Vars:
    data_loc = ""
    train_data = []


class ActivationType(Enum):
    SIGMOID = 0


def get_activation_and_derivative(activation_type: ActivationType): # pure
    def sigmoid_func(x) -> float: # pure
        # logging.StreamHandler.terminator = "  "
        # logging.debug(f"sig: {x}")
        # logging.StreamHandler.terminator = "\n"
        # logging.debug(f"sig: {1/(1+ math.exp(-x))}")
        return 1/(1+ math.exp(-x))

    def sigmoid_derivative(a): # pure
        return a * (1-a)

    if activation_type == ActivationType.SIGMOID:
        return sigmoid_func, sigmoid_derivative


def dot_product(X: list, weights) -> float: # pure
    result = 0
    for x_n, w_n in zip(X, weights):
        result += x_n * w_n
    # logging.StreamHandler.terminator = "  "
    # logging.debug(round(result, 3))
    # logging.StreamHandler.terminator = "\n"
    return result


class Neuron:
    def __init__(self, n_inputs, activ_type=ActivationType.SIGMOID) -> None:
        self.activation_func, self.activation_derivative = get_activation_and_derivative(activ_type)
        self.weights: list[float] = [round(random.uniform(-0.5, 0.5), 3) for _ in range(n_inputs)]
        self.bias: float = round(random.uniform(-0.5, 0.5), 3) # maby without round
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
    # logging.debug(f"norm {X}")
    sq_sum = 0
    for i in range(len(X) - 1):
        sq_sum += math.pow(X[i], 2)
    # logging.debug(sq_sum)
    magnitude = math.sqrt(sq_sum)
    # logging.debug(magnitude)
    for i in range(len(X) - 1):
        X[i] /= magnitude
    # logging.debug(f"norm {X}")
    return X


def expected_output(label: int, n_outputs: int) -> list[int]: # pure
    if label >= n_outputs:
        logging.warning("Number of outputs (number of neurons in the last layer) is too small to represent this label")
        logging.warning("Please update the neural network and repeat the process")
    return [1 if i == label else 0 for i in range(n_outputs)]


def calc_error(output: float, expected_output: int) -> float: # pure
    return math.pow(output - expected_output, 2)


# rename to calc_layer_error maby
def calc_full_error(output: list[float], expected_output: list[int]) -> float: # pure
    sum = 0
    for i in range(len(output)):
        sum += calc_error(output[i], expected_output[i])
    return sum


class NeuralNetwork():
    def __init__(self, n_inputs, layers: list[int]) -> None:
        self.layers = [Layer(n_inputs, layers[0])]
        self.layers += [Layer(layers[i], layers[i+1]) for i in range(len(layers) - 1)]
        self.n_inputs = n_inputs
        self.n_outputs = layers[-1]


    def feed_forward(self, X) -> list[float]:
        logging.debug("feed forward")
        input: list = normalization(X)
        for layer in self.layers:
            input = layer.output(input)
        return input


    def train(self, train_data: list[list[int]], learning_rate_w=3.0, learning_rate_b=0.2, error_gate=0.1, max_iterations=12):
        for j in range(max_iterations):
            logging.info(f"v {j+1} iteration")
            total_error = 0
            for X in train_data:
                # logging.debug(f"{i}")
                output: list[float] = self.feed_forward(X)
                expected_out: list[int] = expected_output(X[-1], self.n_outputs)
                full_error = calc_full_error(output, expected_out) 
                total_error += full_error
                logging.debug(f"output ---> {output}  expected output: {expected_out} full error: {full_error}")
                # print(f"output ---> {output}  expected output: {expected_out} full error: {full_error}")


                for layer in reversed(self.layers):
                    for i, neuron in enumerate(layer.neurons):
                        # error = calc_error(layer.neurons[i], expected_out[i]) 
                            # error = calc_error(output[i], expected_out[i]) 
                        # logging.debug(error)
        
                                # logging.debug(f"old b: {neuron.bias}")
                        # or -= ?
                        # neuron.bias += learning_rate * (expected_out[i] - output[i]) #costGradientB[neuron]

                        # (expected_out[i] - output[i]) could be swaped for actuall derivative. Then the sign is lost
                        neuron.bias += learning_rate_b * (expected_out[i] - output[i]) * neuron.activation_derivative(output[i])
                                # logging.debug(f"new b: {neuron.bias}")
                                # logging.debug(f"old w: {neuron.weights}")
                        for idx in range(len(neuron.weights)):
                            # neuron.weights[idx] += learning_rate * (expected_out[i] - output[i]) * neuron.derivative(output[i]) * X[idx]
                            neuron.weights[idx] += learning_rate_w * (expected_out[i] - output[i]) * neuron.activation_derivative(output[i]) * X[idx]
                                # logging.debug(f"new w: {neuron.weights}")


            logging.info(f"^ error for {j+1} iteration -> {total_error}")

            if total_error <= error_gate:
                logging.info("error threshold reached")
                break
 

    def custom_prediction(self, X: list):
        output = self.feed_forward(X)
        display_hr_output(output)


    def show_arch(self):
        print(f"{self.n_inputs} -> ", end="")
        for layer in self.layers:
            print("| ", end="")
            for neuron in layer.neurons:
                print(f"({neuron.get_n_of_inputs()})N", end=" ")
        print("|")


def ask_for_data_loc() -> str: # pure
    while True:
        answer = int(input("For default data location type 1. Otherwise type 0: "))
        if answer == 1:
            return "data/training"
        elif answer == 0:
            return str(input("Enter custom data location: "))
        else:
            print("Enter valid input")


def download_data_set(root_directory: str, langs=DefaultVars.langs) -> list[list[int]]: # pure
    logging.info("v - downloading data set")
    collected_data = []
    # Iterate over all files and directories in the root directory recursively
    for dirpath, _, filenames in os.walk(root_directory): # _ = dirnames
        for fname in filenames:
            if fname.endswith(".txt"):
                dir_name = os.path.basename(dirpath)

                with open(os.path.join(dirpath, fname), 'r') as file:
                    data = file.read() # Maby there is smth more effc
                
                vec = convert_txt_to_vector(data)
    
                logging.debug(dir_name)
                
                # If you remove a lang data set that is in the beggining or in the middle
                # of the lang list (DefaultVars.langs) and then you reduce the number
                # of neurons in the final layer this appproach will couse problems.
                # It is "handeled" in expected_output()
                for idx, lang in enumerate(langs):
                    if dir_name == lang:
                        vec.append(idx)
                
                logging.debug(vec)
                collected_data.append(vec)

    logging.info("^")
    return collected_data


def convert_txt_to_vector(txt: str) -> list[int]: # pure
    vec = [0 for _ in range(26)]
    for char in txt:
        # logging.StreamHandler.terminator = "  " #
        # logging.debug(char) #
        # logging.StreamHandler.terminator = "\n" #
        in_ascii = ord(char.lower())
        if 96 < in_ascii < 123:
            vec[in_ascii - 97] += 1
    return vec


def translate_output(vector: list) -> str: # pure
    langs = DefaultVars.langs
    idx = vector.index(max(vector))
    if idx < len(langs):
        return langs[idx]
    return "unknown"


def display_hr_output(output: list[float]):
    print(f"Prediction --> {translate_output(output)}")
    print("Confidence:")
    langs = DefaultVars.langs
    for i, out in enumerate(output):
        if i < len(langs):
            print(f"  {langs[i]} - {out}")
        else:
            print(f"  unknown - {out}")


def custom_prediction(NN: NeuralNetwork):
    while True:
        txt = str(input("Paste the text here: "))
        input_vec = convert_txt_to_vector(txt)
        NN.custom_prediction(input_vec)

        if int(input("q to quit. Otherwise hit enter") == "q"):
            break
    

def main():
    data_loc = ask_for_data_loc()
    train_data = download_data_set(data_loc)
    neural_network: NeuralNetwork = NeuralNetwork(26, [2])
    neural_network.show_arch()
    logging.debug(train_data)
    neural_network.train(train_data)


    logging.debug("##############################")
    for example in train_data:
        output: list[float] = neural_network.feed_forward(example)
        expected_out = expected_output(example[-1], 4)
        full_error = calc_full_error(output, expected_out) 
        logging.debug(f"output: {output}  -- {translate_output(output)}; expect -- {translate_output(expected_out)}; err - {full_error}")

    logging.debug("##############################")
    # custom_prediction(neural_network)
    

if __name__ == "__main__":
    main()
