from enum import Enum
import logging
import random


class Variables:
    data_loc = ""
    train_data = []
    predict_data = []
    number_of_features = 0


class DefaultVariables:
    # max_iterations = 10
    # level = logging.INFO
    level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt)
    # filename = 'log_k-means.log', filemode = "w"


class TypeOfRead(Enum):
    TRAINING = 0
    PREDICTING = 1
    

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1
                 ) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = step_func
        self.weights = [int(random.uniform(1, 4)) for _ in range(Variables.number_of_features)]
        self.bias = -3 


    def train(self, data_set) -> None:
        for _ in range(self.n_iters):
            #for i, x_i in enumerate(Variables.train_data):
            a = 0
            for x_i in data_set:
                # a+=1
                # if a > 47:
                    # break
                output = dot_product(x_i[:-1], self.weights) + self.bias
                prediction = self.activation_func(output)

                if prediction != x_i[-1]:
                    update = (x_i[-1] - prediction) * self.lr 
                    logging.debug("---")
                    logging.debug(self.weights)
                    logging.debug(x_i)

                    #for x, w in zip(x_i, self.weights):
                    for j in range(len(x_i) - 1):
                    # for j, x in enumerate(x_i):
                        #x_i[i] *= update
                        # x *= update
                        logging.debug(x_i[j])
                        logging.debug(f"ubdate: {update}")
                        logging.debug(x_i[j] * update)
                        self.weights[j] += x_i[j] * update

                    for i in range(len(self.weights)):
                        self.weights[i] = round(self.weights[i] , 3)

                    logging.debug("---")
                    logging.debug(self.weights)
                    logging.debug(x_i)
                    logging.debug("---")
                else:
                    logging.debug("corre")


    def predict(self, X) -> int:
        return self.activation_func(dot_product(X[:-1], self.weights))


def step_func(x) -> int:
    if x >= 0:
        logging.debug("1 ret")
        return 1
    logging.debug("0 ret")
    return 0


def dot_product(X: list, weights: list) -> int:
    result = 0
    for x, y in zip(X, weights):
        result += x * y
    logging.debug(int(result))
    return int(result)


def ask_for_data_loc():
    missing_input = True
    while missing_input:
        data_loc = int(input("For default data location type 1. Otherwise type 0: "))
        if data_loc == 1:
            Variables.data_loc = "data/iris_training.txt"
            missing_input = False
        elif data_loc == 0:
            data_loc = str(input("Enter custom data location: "))
            Variables.data_loc = data_loc
            missing_input = False
        else:
            print("Enter valid input")


def get_data(line: str, read_type: TypeOfRead):
    tmp_list: list = line.split()
    # logging.debug(tmp_list)

    # cast to tuple?
    # parsed_tmp_list = [eval(i) for i in tmp_list]
    # parsed_tmp_list = []
    # logging.debug(eval(tmp_list[1]))
    # i = 0
    # for i in range(len(tmp_list) - 1):
    # # for i in range(1):
    #     parsed_tmp_list.append(eval(tmp_list[i]))
    # logging.debug(parsed_tmp_list)
    # logging.debug(tmp_list)

    # code for dealing with clusters described using a string
    parsed_tmp_list2 = []
    for i in range(len(tmp_list) - 1):
        parsed_tmp_list2.append(eval(tmp_list[i]))
    if tmp_list[-1] == "Iris-setosa":
        parsed_tmp_list2.append(1)
    else:
        parsed_tmp_list2.append(0)


    # logging.debug(parsed_tmp_list2)
        
    if read_type == TypeOfRead.TRAINING:
        # Variables.train_data.append(parsed_tmp_list)
        Variables.train_data.append(parsed_tmp_list2)
    else:
        # Variables.predict_data.append(parsed_tmp_list)
        Variables.predict_data.append(parsed_tmp_list2)


def download_data_set(data_loc :str, read_type: TypeOfRead):
    logging.info("v")
    logging.info("downloading data set")

    with open(data_loc, "r") as f:
        for line in f:
            get_data(line, read_type)

    if read_type == TypeOfRead.TRAINING:
        Variables.number_of_features = len(Variables.train_data[0]) - 1
        logging.info(f"number of features {len(Variables.train_data[0]) - 1}")
    logging.info("^")



def train():
    download_data_set(Variables.data_loc, TypeOfRead.TRAINING)
    perceptron = Perceptron()
    perceptron.train(Variables.train_data)
    logging.debug(perceptron.weights)



def main():
    ask_for_data_loc()
    train()


if __name__ == "__main__":
    main()
