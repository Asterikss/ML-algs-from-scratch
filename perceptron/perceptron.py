
from enum import Enum
import logging


class Variables:
    # k = 0
    data_loc = ""
    train_data = []
    # k_means = []
    number_of_features = 0
    # prev_k_means = []
    predict_data = []

class DefaultVariables:
    # max_iterations = 10
    # threshold = 0.000001
    #level = logging.INFO
    level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt)
    # filename = 'log_k-means.log', filemode = "w"


class TypeOfRead(Enum):
    TRAINING = 0
    PREDICTING = 1

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
    logging.debug(tmp_list)

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
    parsed_tmp_list2.append(tmp_list[-1])

    logging.debug(parsed_tmp_list2)
    # maby later change every chosen flower into 1 else 0
        
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


def main():
    ask_for_data_loc()
    train()


if __name__ == "__main__":
    main()
