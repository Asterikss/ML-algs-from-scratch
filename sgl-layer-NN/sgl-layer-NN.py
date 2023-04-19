"Single layer neural network used for predicting the language of a piece of text"
from enum import Enum
import logging
import random
import os
import math


class ActivationType(Enum):
    SIGMOID = 0


class NumberOfOutputsError(Exception):
    pass


class InadequateInputLen(Exception):
    pass


class TooManyLayers(Exception):
    pass


def get_activation_and_derivative(activation_type: ActivationType): # pure
    def sigmoid_func(x) -> float: # pure
        return 1/(1+ math.exp(-x))

    def sigmoid_derivative(a): # pure
        return a * (1-a)

    if activation_type == ActivationType.SIGMOID:
        return sigmoid_func, sigmoid_derivative


def dot_product(X: list, weights: list) -> float: # pure
    result = 0
    for x_n, w_n in zip(X, weights):
        result += x_n * w_n
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
    def __init__(self, n_inputs, n_neurons, activation_type=ActivationType.SIGMOID) -> None:
        self.neurons: list[Neuron] = [Neuron(n_inputs, activ_type=activation_type) for _ in range(n_neurons)]

    
    def output(self, X) -> list[float]:
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output(X)) 
        return outputs


def normalization(X) -> list[float]: # pure
    sq_sum = 0
    for i in range(len(X) - 1):
        sq_sum += math.pow(X[i], 2)

    magnitude = math.sqrt(sq_sum)
    for i in range(len(X) - 1):
        X[i] /= magnitude

    return X


def expected_output(label: int, n_outputs: int) -> list[int]: # pure?
    return [1 if i == label else 0 for i in range(n_outputs)]


def calc_error(output: float, expected_output: int) -> float:# pure
    return math.pow(output - expected_output, 2)


def calc_full_error(output: list[float], expected_output: list[int]) -> float: # pure
    sum = 0
    for i in range(len(output)):
        sum += calc_error(output[i], expected_output[i])
    return sum


class NeuralNetwork():
    def __init__(self, n_inputs, layers: list[int]) -> None:
        self.layers = [Layer(n_inputs, layers[0])]
        self.layers += [Layer(layers[i], layers[i+1]) for i in range(len(layers) - 1)]
        # Could use the softmax activation for the last layer
        self.n_inputs = n_inputs
        self.n_outputs = layers[-1]


    def feed_forward(self, X) -> list[float]:
        logging.debug("feed forward")
        input: list = normalization(X)
        for layer in self.layers:
            input = layer.output(input)
        return input


    # Currenlty supports training of only a single layer network
    def train(self, train_data: list[list[int]], lang_table: list[str], learning_rate_w=4.0, learning_rate_b=0.4, error_gate=0.5, max_iterations=32):
        self.lang_table = lang_table

        # Check for potential errors
        if len(self.layers) > 1:
            raise TooManyLayers("Currenlty train() supports only a sigle layer network")


        if len(train_data[0]) - 1 != self.n_inputs:
            raise InadequateInputLen("The number of inputs to the network does no match the length of the input vector " +
                                     f"length of the vector must be {self.n_inputs} plus one for the label " +
                                     "Aborting the training")


        if len(lang_table) != self.n_outputs:
            raise NumberOfOutputsError("Number of outputs (number of neurons in the last layer) " +
                                       "does not much the number of detected languages in the data set. " +
                                       "Please update the neural network accordingly and repeat the process")


        for j in range(max_iterations):
            logging.info(f"v {j+1} iteration")
            total_error = 0
            for X in train_data:
                output: list[float] = self.feed_forward(X)
                expected_out: list[int] = expected_output(X[-1], self.n_outputs)
                full_error = calc_full_error(output, expected_out) 
                total_error += full_error
                logging.debug(f"output ---> {output}  expected output: {expected_out} full error: {full_error}")

                for layer in reversed(self.layers):
                    for i, neuron in enumerate(layer.neurons):
                        # error = calc_error(output[i], expected_out[i]) 
                        # logging.debug(error)
        
                        # (expected_out[i] - output[i]) could be swaped for actuall derivative. Then the sign is lost
                        logging.debug(f"old b: {neuron.bias}")
                        neuron.bias += learning_rate_b * (expected_out[i] - output[i]) * neuron.activation_derivative(output[i])
                        logging.debug(f"new b: {neuron.bias}")
                        logging.debug(f"old w: {neuron.weights}")
                        for idx in range(len(neuron.weights)):
                            neuron.weights[idx] += learning_rate_w * (expected_out[i] - output[i]) * neuron.activation_derivative(output[i]) * X[idx]
                        logging.debug(f"new w: {neuron.weights}")


            logging.info(f"^ error for {j+1} iteration -> {total_error}")

            if total_error <= error_gate:
                logging.info("error threshold reached")
                break
 

    def custom_prediction(self, X: list):
        output = self.feed_forward(X)
        display_hr_output(output, self.lang_table)


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


def download_data_set(root_directory: str) -> tuple[list[list[int]], list[str]]: # pure
    logging.info("v - downloading data set")
    collected_data = []
    lang_table = []
    # Iterate over all files and directories in the root directory recursively
    for dirpath, _, filenames in os.walk(root_directory): # _ = dirnames
        for fname in filenames:
            if fname.endswith(".txt"):
                dir_name = os.path.basename(dirpath)

                logging.debug(os.path.join(dirpath, fname))
                with open(os.path.join(dirpath, fname), 'r', encoding="utf-8") as file:
                    data = file.read() # Maby there is smth more effc
                
                vec = convert_txt_to_vector(data)

                if dir_name not in lang_table:
                    lang_table.append(dir_name)
    
                logging.debug(dir_name)
                for idx, lang in enumerate(lang_table):
                    if dir_name == lang:
                        vec.append(idx)
                
                logging.debug(vec)
                collected_data.append(vec)
    logging.info(f"Detected languages: {lang_table}")
    logging.info("^")
    return collected_data, lang_table


def convert_txt_to_vector(txt: str) -> list[int]: # pure
    vec = [0 for _ in range(26)]
    for char in txt:
        in_ascii = ord(char.lower())
        if 96 < in_ascii < 123:
            vec[in_ascii - 97] += 1
    return vec


def translate_output(vector: list, lang_table: list[str]) -> str: # pure
    idx = vector.index(max(vector))
    if idx < len(lang_table):
        return lang_table[idx]
    return "unknown"


def display_hr_output(output: list[float], lang_table: list[str]):
    print(f"Prediction --> {translate_output(output, lang_table)}")
    print("Confidence:")
    for i, out in enumerate(output):
        if i < len(lang_table):
            print(f"  {lang_table[i]} - {out}")
        else:
            print(f"  unknown - {out}")


def custom_prediction(NN: NeuralNetwork):
    while True:
        txt = str(input("Paste the text here: "))
        input_vec = convert_txt_to_vector(txt)
        NN.custom_prediction(input_vec)

        if int(input("q to quit. Otherwise hit enter: ") == "q"):
            break
    

def init():
    # level = logging.DEBUG
    level = logging.INFO
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt) # filename = 'log_x.log', filemode = "w"


def main():
    init()
    data_loc = ask_for_data_loc()
    train_data, lang_table = download_data_set(data_loc)
    neural_network: NeuralNetwork = NeuralNetwork(26, [3])
    neural_network.show_arch()
    neural_network.train(train_data, lang_table)

    custom_prediction(neural_network)

    # logging.debug("##############################")
    # for example in train_data:
    #     output: list[float] = neural_network.feed_forward(example)
    #     expected_out = expected_output(example[-1], 4)
    #     full_error = calc_full_error(output, expected_out) 
    #     logging.debug(f"output: {output}  -- {translate_output(output)}; expect -- {translate_output(expected_out)}; err - {full_error}")
    

if __name__ == "__main__":
    main()
