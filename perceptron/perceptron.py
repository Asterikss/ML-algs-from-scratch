"Pereptron that predicts if a flower is an Iris-setosa"
from enum import Enum
import logging
import random

class Variables:
    data_loc = ""
    train_data = []
    predict_data = []
    number_of_features = 0


class DefaultVariables:
    level = logging.INFO
    # level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt)
    # filename = 'log_x.log', filemode = "w"


class TypeOfRead(Enum):
    TRAINING = 0
    PREDICTING = 1
 

def step_func(x) -> int:
    logging.StreamHandler.terminator = "  "
    if x >= 0:
        logging.debug("  1 ret")
        logging.StreamHandler.terminator = "\n"
        return 1
    else:
        logging.debug("  0 ret")
        logging.StreamHandler.terminator = "\n"
        return 0

# 1 - Iris-setosa
class Perceptron:

    # def __init__(self, learning_rate=0.02, n_iters=3) -> None:
    def __init__(self, learning_rate=0.02, n_iters=7) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = step_func
        self.weights = [round(random.uniform(-1, 1), 2) for _ in range(Variables.number_of_features)]
        # TODO see the change
        self.bias = round(random.uniform(-1, 1), 2)


    def train(self, data_set) -> None:
        logging.info("v")
        for idx in range(self.n_iters):
            logging.debug(f"~~~~~~inter {idx + 1}")
            logging.debug(self.weights)
            logging.debug(self.bias)
            logging.debug("~~~~~~")

            n_correct = 0
            for x_i in data_set:
                output = dot_product(x_i[:-1], self.weights) + self.bias

                logging.StreamHandler.terminator = "  "
                logging.debug(f"+bias {round(output, 2)}")
                logging.StreamHandler.terminator = "\n"

                prediction = self.activation_func(output)
                logging.debug(f"acutal label: {x_i[-1]}")

                if prediction != x_i[-1]:
                    update = (x_i[-1] - prediction) * self.lr 
                    
                    logging.debug("---")
                    logging.debug(self.bias)
                    logging.debug(self.weights)
                    logging.debug(x_i)

                    for j in range(len(x_i) - 1):
                        logging.debug(x_i[j])
                        logging.debug(f"ubdate: {update}")
                        logging.debug(x_i[j] * update)
                        self.weights[j] += x_i[j] * update
                        # Multiplying the error by the input feature in the
                        # weight update rule is important because it tells us how
                        # much each input feature contributes to the error. By
                        # updating the weights proportionally to the contribution
                        # of each input feature, we can adjust the perceptron's
                        # decision boundary in the direction that reduces the
                        # error.

                    for i in range(len(self.weights)):
                        self.weights[i] = round(self.weights[i] , 3)

                    self.bias += (x_i[-1] - prediction) * self.lr
                    self.bias = round(self.bias, 2)

                    logging.debug("---")
                    logging.debug(self.bias)
                    logging.debug(self.weights)
                    logging.debug(x_i)
                    logging.debug("---")
                else:
                    n_correct+=1

            acc = round((n_correct/len(Variables.train_data)) * 100, 2)
            logging.info("~~~~~~~~~~~~~~~~")
            logging.info(f"Accuracy for {idx + 1} iteration: {acc}%")
            logging.info("~~~~~~~~~~~~~~~~")
            if acc == 100:
                logging.info("Acc 100% reached")
                break
        logging.info("^")


    def predict_data_set(self) -> None:
        logging.info("v")
        logging.info("prediciting data set")

        choice = -1
        print("For predicting the cluster from the default file (data/iris_test.txt) type 1")
        print("For custom guess (providing a vector) type 2 ")
        print("For predicting the cluster from custom file type 3")

        while choice != "0" and choice != "1" and choice != "2":
            choice = input(": ")

        if choice == "1" or choice == "3":

            if choice == "1":
                download_data_set("data/iris_test.txt", TypeOfRead.PREDICTING)
            if choice == "3":
                path = str(input("Provide path: "))
                download_data_set(path, TypeOfRead.PREDICTING)

            logging.debug(Variables.predict_data)

            n_correct = 0
            for vector in Variables.predict_data:
                prediction = self.predict(vector)
                actual_anwser = vector[-1]
                logging.info(f"prediction: {prediction} Actual label: {actual_anwser}")

                if prediction == actual_anwser:
                    n_correct+=1

            logging.info(f"Accuracy: {(n_correct/len(Variables.predict_data)) * 100}%")

        if choice == "2":
            end = False
            print(f"Enter a vector with {Variables.number_of_features} features plus it's actual label")
            print(f"If the label is unknown enter 2 there")
            while not end:
                custom_vector: list[float] = []
                for i in range(Variables.number_of_features):
                    custom_vector.append((float(input(f"Input {i+1} feature: "))))
                custom_vector.append(int(input("Input the cluster: ")))

                prediction = self.predict(custom_vector)
                actual_anwser = custom_vector[-1]

                if actual_anwser != 0 and actual_anwser != 1:
                    logging.info(f"prediction: {prediction} Actual label: unknown")
                else:
                    logging.info(f"prediction: {prediction} Actual label: {actual_anwser}")


                q = input("Type q to exit. Otherwise hit enter: ")
                if q == "q":
                    end = True

        logging.info("^")


    def predict(self, X) -> int:
        return self.activation_func(dot_product(X[:-1], self.weights) + self.bias)
        # return self.activation_func(dot_product(X[:-1], self.weights))


class State():
    # perceptron: object = None
    # perceptron = None
    perceptron = Perceptron()   
    # Can't write None here, does not work
    # this Perceptron() will be overriten and is not used


def dot_product(X: list, weights: list) -> int:
    result = 0
    for x, y in zip(X, weights):
        result += x * y
    logging.StreamHandler.terminator = "  "
    logging.debug(round(result, 2))
    logging.StreamHandler.terminator = "\n"
    return round(result, 2)


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
    State.perceptron = perceptron


def predict():
    perceptron = State.perceptron

    # missing_input = True
    # data_location = ""
    # while missing_input:
    #     data_loc = int(input("For default data location for prediciton type 1. Otherwise type 0: "))
    #     if data_loc == 1:
    #         data_location = "data/iris_test.txt"
    #         missing_input = False
    #     elif data_loc == 0:
    #         data_location = str(input("Enter custom data location: "))
    #         data_loc = data_loc
    #         missing_input = False
    #     else:
    #         print("Enter valid input")
    #
    # download_data_set(data_location, TypeOfRead.PREDICTING)
    # logging.debug(Variables.predict_data)
    perceptron.predict_data_set()


def main():
    ask_for_data_loc()
    train()
    predict()


if __name__ == "__main__":
    main()
