"Pereptron that predicts if a flower is an Iris-setosa (1)"
import logging
import random
import os


class Variables:
    data_loc = ""
    train_data = []
    predict_data = []
    number_of_features = 0


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


class Perceptron:

    def __init__(self, learning_rate=0.02, n_iters=7) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = step_func
        self.weights = [round(random.uniform(-1, 1), 2) for _ in range(Variables.number_of_features)]
        self.bias = round(random.uniform(-1, 1), 2)


    def train(self, data_set) -> None:
        logging.info("v")
        for idx in range(self.n_iters):
            logging.debug(f"~~~~~~iter {idx + 1}")
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
                    # update weights
                    update = (x_i[-1] - prediction) * self.lr 
                    
                    logging.debug("---")
                    logging.debug(self.bias)
                    logging.debug(self.weights)
                    logging.debug(x_i)

                    for j in range(len(x_i) - 1):
                        logging.debug(x_i[j])
                        logging.debug(f"update: {update}")
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


    def predict_dataset(self) -> None:
        logging.info("v")
        logging.info("prediciting data set")

        choice = -1
        print("For predicting the labels from the default file (data/iris_test.txt) type 1")
        print("For custom guess (providing a vector) type 2 ")
        print("For predicting the label from custom file type 3")

        while choice != "0" and choice != "1" and choice != "2":
            choice = input(": ")

        if choice == "1" or choice == "3":

            if choice == "1":
                dataset, number_of_features = download_dataset("data/iris_test.txt") 
                Variables.predict_data = dataset
            if choice == "3":
                path = str(input("Provide path: "))
                dataset, number_of_features = download_dataset(path)
                print(number_of_features)
                Variables.predict_data = dataset

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
                custom_vector.append(int(input("Input the label: ")))

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


def dot_product(X: list, weights: list) -> int:
    result = 0
    for x, y in zip(X, weights):
        result += x * y
    logging.StreamHandler.terminator = "  "
    logging.debug(round(result, 2))
    logging.StreamHandler.terminator = "\n"
    return round(result, 2)


def ask_for_data_loc() -> str:
    while True:
        answer = int(input("For default data location type 1. Otherwise type 0: "))
        if answer == 1:
            return "data/iris_training.txt"
        elif answer == 0:
            while True:
                data_loc = str(input("Enter custom data location (with ""): "))
                if os.path.exists(data_loc):
                    return data_loc
                print("The file does not exits")


def download_dataset(data_loc :str) -> tuple[list[list[float]], int]:
    dataset = []

    with open(data_loc, "r") as f:
        for line in f:
            splited = line.split()
            parsed_tmp_list = [eval(splited[i]) for i in range(len(splited) - 1)]
            if splited[-1] == "Iris-setosa":
                parsed_tmp_list.append(1)
            else:
                parsed_tmp_list.append(0)
            dataset.append(parsed_tmp_list)


    number_of_features = len(dataset[0]) - 1
    logging.info(f"number of features {number_of_features}")

    return dataset, number_of_features


def train(data_loc) -> Perceptron:
    dataset, number_of_features = download_dataset(data_loc)
    Variables.train_data = dataset
    Variables.number_of_features = number_of_features
    perceptron = Perceptron()
    perceptron.train(Variables.train_data)
    logging.debug(perceptron.weights)
    return perceptron


def predict(perceptron):
    perceptron.predict_dataset()


def init():
    level = logging.INFO
    # level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt) # filename = 'log_x.log', filemode = "w"


def main():
    init()
    data_loc = ask_for_data_loc()
    Variables.data_loc = data_loc
    perceptron = train(data_loc)
    predict(perceptron)


if __name__ == "__main__":
    main()
