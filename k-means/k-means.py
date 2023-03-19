# pyright: reportUnusedVariable=false
# pyright: ignore - single line (# type: ignore)
# pyright: ignore [reportUnusedVariable=] - single line
# https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670
# use pathlib later
# use parsing library. json, pydantic

# from dataclasses import dataclass
# @dataclass
## class classname(object):
# class classname(order = True, frozen = True):
#   pass

# Iris-setosa = 0
# Iris-versicolor = 1
# Iris-virginica = 2
import random
from enum import Enum
import logging

class Variables:
    k = 0
    data_loc = ""
    points = []
    k_means = []
    number_of_features = 0
    prev_k_means = []

    predict_data = []


class DefaultVariables:
    max_iterations = 10
    threshold = 0.000001
    level = logging.INFO
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt)
    # filename = 'log_k-means.log', filemode = "w"

class TypeOfRead(Enum):
    TRAINING, PREDICTING = range(2)


def is_done_iterating() -> bool:
    for i in range(Variables.k):
        dist = 0
        for j in range(Variables.number_of_features):
            dist += (Variables.prev_k_means[i][j] - Variables.k_means[i][j]) ** 2

        # print("ASDASD")
        # print(dist)
        if dist < DefaultVariables.threshold:
            print("Reached threshold")
            return True

    return False

def get_max() -> list:
    print("v")
    print(f"calculating max for {Variables.number_of_features} features")
    max: list = []

    for j in range(0,  Variables.number_of_features):
        max.append(Variables.points[0][j])
        for i in range(1 , len(Variables.points)): #is inclusinve?
            if max[j] < Variables.points[i][j]:
                max[j] = Variables.points[i][j]
    
    logging.debug(f"max list: {max}")
    print("^")
    return max


def get_min() -> list:
    print("v")
    print(f"calculating min for {Variables.number_of_features} features")
    min: list = []

    for j in range(0,  Variables.number_of_features):
        min.append(Variables.points[0][j])
        for i in range(1 , len(Variables.points)): #is inclusinve?
            if min[j] > Variables.points[i][j]:
                min[j] = Variables.points[i][j]
    
    print(f"min list: {min}")
    print("^")
    return min


def pick_random_points():
    print("v")
    print("picking centroids")
    '''
    Calc min and max for each feature. Calc the interval beetween min and max for each of them.
    Devide it by the number of means. Calc single feature in single mean by picking random value
    beetween min and min + devided interval. Calc the same feature, but for the next mean pick
    a value beetween min + devided interval and min + devided interval * 2. And so on.
    It makes in unlikely that there will be no points assigned to a given mean. Therefore no
    need to populate in artificially or trying to drop it alltogether. Plus they are more
    evenly distribuated which given reasonable data and k-value will improve results
    '''
    rand_points :list = []
    # tmp_rand_points :list = []

    max_list = get_max()
    min_list = get_min()
    #interval :list = max_list - min_list
    #intervals :list = [max_list[i] - min_list[i] for i in range(Variables.number_of_features)]
    intervals :list = [round(max_list[i] - min_list[i], 2) for i in range(Variables.number_of_features)]

    logging.debug(max_list)
    logging.debug(min_list)
    print(intervals)

    for i in range(Variables.number_of_features):
        intervals[i] = round((intervals[i] / Variables.k), 1)

    print(intervals)

    for i in range(1, Variables.k + 1):
        tmp_rand_points :list = []

        for j in range(Variables.number_of_features):
            tmp_rand_points.append(round(random.uniform(min_list[j] + (intervals[j] * (i - 1)), min_list[j] + (intervals[j] * i)), 2))

        rand_points.append(tmp_rand_points)
    
    print(f"means: {rand_points}")
    Variables.k_means = rand_points
    print("^")


def calc_means(points :list) -> list[float]:
    print("v")
    print("calculating means")
    new_single_k_mean :list[float] = [0 for _ in range(Variables.number_of_features)]

    for i in range(Variables.number_of_features):
        sum = 0.0
        mean = 0.0
        for j in range(len(points)):
            sum += points[j][i]

        mean = round(sum/len(points), 2)
        new_single_k_mean[i] = mean

    print("^")
    return new_single_k_mean


def ask_for_k_value_and_data_loc():
    missing_input = True

    while missing_input:
        k = int(input("Enter k value: "))
        if 0 < k < 7:
            Variables.k = k
            missing_input = False
        else:
            print("K beetween 1 and 6")

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

    # Variables.data_loc = "data/iris_training.txt"


def calc_euclidean_distance(a: tuple, b: tuple) -> float:
    # print("eucal")
    # print(a)
    # print(len(a))


    if len(a) -1 != len(b):
        print("tuples are of wrong size. 'a'(first param) should be longer by 1 (last arg is the answer) ")
        return -1
    dist = 0

    for i in range(0, len(b)):
        dist += (a[i] - b[i]) ** 2

    # print(f"c_e_d: {dist ** (1 / 2)}")

    #return dist ** (1 / 2)
    return round(dist, 3)


def one_full_iter():
    # print("v")
    # print("full one----@@@@@@@@@@@@")
    points_sorted = [[] for _ in range(Variables.k)]
    # tmp_dist :list[float] = [0 for _ in range(Variables.k)]
    # print(tmp_dist)

    for point in Variables.points:
        tmp_dist :list[float] = [0 for _ in range(Variables.k)]
        # print(f"first {tmp_dist}")

        for i in range(Variables.k):
            tmp_dist[i] = calc_euclidean_distance(point, Variables.k_means[i])

        # print(f"second {tmp_dist}")
        which_mean_closest = tmp_dist.index(min(tmp_dist)) if tmp_dist else -1
        # print(which_mean_closest)

        points_sorted[which_mean_closest].append(point)

    # print(points_sorted)

    # Sometimes, with few data points, there can be a situation where none
    # of the points are closest to a particualr starting mean. Therefore
    # an empty list is created. I'm populating it here with big numbers
    # so none of the future points will be closest to that mean(point).
    # Algorithm, by doing that, will suggest that their are less groups
    # than entered by a user. Puting big numbers there is unnecesarry. 
    # List can't be empty though, because I'm deviding by it's len later
    for i in range(len(points_sorted)):
        if len(points_sorted[i]) == 0:
            points_sorted[i].append([1248 for _ in range(Variables.k)])
            points_sorted[i][0].append(-1)
            print("One of the lists created as an empty list.")
            print("Now populated artificially.")
            
    # print(points_sorted)




    Variables.prev_k_means = Variables.k_means

    print(f"prev k_means: {Variables.prev_k_means}")

    new_k_means :list[list[float]] = [[] for _ in range(Variables.k)]
    # new_k_means = [0.0 for _ in range(Variables.k)]
    # new_k_means :list[float] = []
    # new_k_means :list[float] = [0 for _ in range(Variables.k)]

    for i in range(Variables.k):
        new_k_means[i] = calc_means(points_sorted[i])
        # new_k_means.append(calc_means(points_sorted[i]))
        # new_k_means[i] = 3.0

    print(f"new k_means {new_k_means}")
    
    Variables.k_means = new_k_means

    # print(calc_means([[1,4,5,6], [1,4,6,6], [2, 4, 6, 1]]))
    # print("full end----$$$$$$$$$$$$$$$")
    # print("^")


def interation_loop():
    print("v")
    print("--Start of interation loop--")
    i = 1

    #while i < DefaultVariables.max_iterations and Variables.k_means != Variables.prev_k_means:
    #while i < 4 and Variables.k_means != Variables.prev_k_means:

    #So the is_done_iterating() does not crash (no prev_k_means)
    one_full_iter()
    print(f"inter {i}")
    while i < DefaultVariables.max_iterations and not is_done_iterating():
        one_full_iter()
        i += 1
        print(f"inter {i}")

    # print(Variables.k_means)
    print("--End of interation loop--")
    print("^")


def get_data(line: str, read_type: TypeOfRead):
    # print("getting data")
    tmp_list: list = line.split()
    # tmp_list: list = line.split("   " || " ")
    # print(tmp_list)
    # print(type(tmp_list))
    # print(len(tmp_list))

    # lis = ['1', '-4', '3', '-6', '7']
    int_tmp_list = [eval(i) for i in tmp_list]
    # print("Modified list is: ", int_tmp_list)

    if read_type == TypeOfRead.TRAINING:
        Variables.points.append(int_tmp_list)
    else:
        Variables.predict_data.append(int_tmp_list)
 

def download_data_set(data_loc :str, read_type: TypeOfRead):
    print("v")
    print("downloading data det")
    # f0 = open("data.txt", "r")

    with open(data_loc, "r") as f:
        for line in f:
            get_data(line, read_type)

    # get_data("5.2   2.1   1.5   3   1", read_type)
    # get_data("5 2   3       3   1", read_type)
    # get_data("2 5   1   3   0", read_type)
    # get_data("3 2   4   3   1", read_type)
    # get_data("2 2   6   3   2", read_type)
    # get_data("2 2   6   3   2", read_type)
    #
    # get_data("3 4   2   3   0", read_type)
    # get_data("4 3   4   3   1", read_type)
    # get_data("3 2   5   3   1", read_type)


        # while f
        #   str = f.readline()
    #if Variables.number_of_features == 0:
    if read_type == TypeOfRead.TRAINING:
        Variables.number_of_features = len(Variables.points[0]) - 1
        print(f"number of features {len(Variables.points[0]) - 1}")
    print("^")


# Different way to return tuples maby
def predict_label(point :tuple) -> tuple[int, int]:
    tmp_dist :list[float] = [0 for _ in range(Variables.k)]

    for i in range(Variables.k):
        tmp_dist[i] = calc_euclidean_distance(point, Variables.k_means[i])

    logging.debug(f"second_in_label {tmp_dist}")
    which_mean_closest = tmp_dist.index(min(tmp_dist)) if tmp_dist else -1

    print(f"Prediction: Label {which_mean_closest}. Actual label: {point[-1]}")
    return which_mean_closest, point[-1]


def predict():
    print("v")

    choice = -1
    #fix proper loop later
    while choice != 0 and choice != 1 and choice != 2:
        print("For predicting data from the default file (data/iris_test.txt) type 1")
        print("For custom guess (providing a vector) type 2 ")
        print("For predicting data from custom file type 3")
        choice = int(input(": "))

    if choice == 1:
        download_data_set("data/iris_test.txt", TypeOfRead.PREDICTING)

        logging.debug(Variables.predict_data)

        predictions = [0 for _ in range(Variables.k)] 
        actual_labels = [0 for _ in range(Variables.k)] 

        for observation in Variables.predict_data:
            label :tuple = predict_label(observation)
            predictions[label[0]] += 1
            actual_labels[label[1]] += 1

        accuracy_table = [1 - round((abs(predictions[i] - actual_labels[i])/actual_labels[i]), 3) for i in range(Variables.k)] 
        total_acc = 0

        for e in accuracy_table:
            total_acc += e
        total_acc /= Variables.k

        print("predictions:")
        print(predictions)
        print("acutal labels:")
        print(actual_labels)
        print("accuracy_table: ")
        print(accuracy_table)
        print("Total accuracy: ")
        print(total_acc)
    
    print("^")

def train():
    download_data_set(Variables.data_loc, TypeOfRead.TRAINING)
    pick_random_points()
    interation_loop()


def main():
    logging.error("eeaea")
    ask_for_k_value_and_data_loc()
    train()
    predict()



if __name__ == "__main__":
    main()
