# use pathlib later

# originally in default files
# 0 = Iris-setosa
# 1 = Iris-versicolor
# 2 = Iris-virginica
import random
from enum import Enum
from dataclasses import dataclass
import logging

class Variables:
    k = 0
    data_loc = ""
    points = []
    k_means = []
    number_of_features = 0
    prev_k_means = []
    predict_data = []


@dataclass(frozen=True)
class DefaultVariables:
    max_iterations = 10
    threshold = 0.000001


class TypeOfRead(Enum):
    # TRAINING, PREDICTING = range(2) - this shows an error for some reason
    TRAINING = 0
    PREDICTING = 1


def is_done_iterating() -> bool:
    for i in range(Variables.k):
        dist = 0
        for j in range(Variables.number_of_features):
            dist += (Variables.prev_k_means[i][j] - Variables.k_means[i][j]) ** 2

        if dist < DefaultVariables.threshold:
            print("~Threshold reached~")
            return True

    return False






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




def calc_euclidean_distance(a: tuple, b: tuple) -> float:
    if len(a) -1 != len(b):
        print("tuples (points) are of wrong size")
        print("'a'(first param) should be longer by 1 (last arg is the actual cluster) ")
        return -1

    dist = 0
    for i in range(0, len(b)):
        dist += (a[i] - b[i]) ** 2

    # return dist ** (1 / 2)
    return round(dist, 3)


def predict_cluster(point :tuple) -> tuple[int, int]:
    tmp_dist :list[float] = [0 for _ in range(Variables.k)]

    for i in range(Variables.k):
        tmp_dist[i] = calc_euclidean_distance(point, Variables.k_means[i])

    logging.debug(f"second_in_label {tmp_dist}")
    which_mean_closest = tmp_dist.index(min(tmp_dist)) if tmp_dist else -1

    if point[-1] == Variables.k:
        print(f"Prediction: Cluster {which_mean_closest}. Actual cluster: unknown")
        return which_mean_closest, -1
    
    print(f"Prediction: Cluster {which_mean_closest}. Actual cluster: {point[-1]}")
    return which_mean_closest, point[-1]


def predict():
    print("v")

    choice = -1
    print("For predicting the cluster from the default file (data/iris_test.txt) type 1")
    print("For custom guess (providing a vector) type 2 ")
    print("For predicting the cluster from custom file type 3")
    #That's a terrible way probably
    while choice != "0" and choice != "1" and choice != "2":
        choice = input(": ")

    if choice == "1" or choice == "3":

        if choice == "1":
            download_data_set("data/iris_test.txt", TypeOfRead.PREDICTING)
        if choice == "3":
            path = str(input("Provide path: "))
            download_data_set(path, TypeOfRead.PREDICTING)

        logging.debug(Variables.predict_data)

        predictions = [0 for _ in range(Variables.k)] 
        actual_clusters = [0 for _ in range(Variables.k)] 

        for observation in Variables.predict_data:
            cluster_tuple :tuple = predict_cluster(observation)
            logging.debug(cluster_tuple)
            predictions[cluster_tuple[0]] += 1
            actual_clusters[cluster_tuple[1]] += 1

        accuracy_table = [1 - round((abs(predictions[i] - actual_clusters[i])/actual_clusters[i]), 3) for i in range(Variables.k)] 
        total_acc = 0

        for e in accuracy_table:
            total_acc += e
        total_acc /= Variables.k

        print("predictions:")
        print(predictions)
        print("acutal clusters:")
        print(actual_clusters)
        print("accuracy_table:")
        print(accuracy_table)
        print("Total accuracy:")
        print(total_acc)

    if choice == "2":
        end = False
        print(f"Enter a vector with {Variables.number_of_features} features plus it's actual cluster")
        print(f"If the cluster is unknown enter {Variables.k} (k) there")
        while not end:
            custom_vector: list[int] = []
            for i in range(Variables.number_of_features):
                custom_vector.append((int(input(f"Input {i+1} feature: "))))
            custom_vector.append(int(input("Input the cluster: ")))

            cluster_tuple :tuple = predict_cluster(tuple(custom_vector))
            logging.debug(cluster_tuple)
    
            q = input("Type q to exit. Otherwise hit enter: ")
            if q == "q":
                end = True
    
    print("^")


def one_full_iter(k_value: int, dataset: list[list[float]]):
    # points_sorted = [[] for _ in range(Variables.k)]
    points_sorted = [[] for _ in range(k_value)]

    for point in Variables.points:
        # tmp_dist :list[float] = [0 for _ in range(Variables.k)]
        tmp_dist :list[float] = [0 for _ in range(k_value)]

        # for i in range(Variables.k):
        for i in range(k_value):
            tmp_dist[i] = calc_euclidean_distance(point, Variables.k_means[i])

        logging.debug(tmp_dist)
        which_mean_closest = tmp_dist.index(min(tmp_dist)) if tmp_dist else -1
        logging.debug(which_mean_closest)

        points_sorted[which_mean_closest].append(point)

    # Sometimes, with few data points, there can be a situation where none
    # of the points are closest to a particualr starting mean. Therefore
    # an empty list is created. I'm populating it here with big numbers
    # so none of the future points will be closest to that mean.
    # Algorithm, by doing that, will suggest that their are less groups
    # than entered by a user. Puting big numbers there is unnecesarry. 
    # List can't be empty though, because I'm deviding by it's len later
    for i in range(len(points_sorted)):
        if len(points_sorted[i]) == 0:
            # points_sorted[i].append([1248 for _ in range(Variables.k)])
            points_sorted[i].append([1248 for _ in range(k_value)])
            points_sorted[i][0].append(-1)
            print("One of the lists created as an empty list.")
            print("Now populated artificially.")
            
    logging.debug(points_sorted)

    Variables.prev_k_means = Variables.k_means

    # new_k_means :list[list[float]] = [[] for _ in range(Variables.k)]
    new_k_means :list[list[float]] = [[] for _ in range(k_value)]

    # for i in range(Variables.k):
    for i in range(k_value):
        new_k_means[i] = calc_means(points_sorted[i])

    logging.info(f"Prev k_means: {Variables.prev_k_means}")
    logging.info(f"New k_means {new_k_means}")
    
    Variables.k_means = new_k_means


def interation_loop(k_value: int, dataset: list[list[float]], max_iterations: int):
    logging.info("v - Start of the interation loop")
    i = 1

    # So the is_done_iterating() does not crash (no prev_k_means)
    print(f"-inter {i}-")
    one_full_iter(k_value, dataset)
    # while i < DefaultVariables.max_iterations and not is_done_iterating():
    while i < max_iterations and not is_done_iterating():
        i += 1
        print(f"-inter {i}-")
        one_full_iter(k_value, dataset)

    logging.info("^ - End of the interation loop")


def get_max(number_of_features: int, dataset: list[list[float]]) -> list:
    print("v")
    # print(f"calculating max for {Variables.number_of_features} features")
    print(f"calculating max for {number_of_features} features")
    max: list = []

    # for j in range(0,  Variables.number_of_features):
    for j in range(0,  number_of_features):
        # max.append(Variables.points[0][j])
        max.append(dataset[0][j])
        # for i in range(1 , len(Variables.points)): #is inclusinve?
        for i in range(1 , len(dataset)): #is inclusinve?
            # if max[j] < Variables.points[i][j]:
            if max[j] < dataset[i][j]:
                # max[j] = Variables.points[i][j]
                max[j] = dataset[i][j]
    
    logging.debug(f"max list: {max}")
    print("^")
    return max


# compress get_max() and get_min() together later
def get_min(number_of_features: int, dataset: list[list[float]]) -> list:
    print("v")
    # print(f"calculating min for {Variables.number_of_features} features")
    print(f"calculating min for {number_of_features} features")
    min: list = []

    # for j in range(0,  Variables.number_of_features):
    for j in range(0,  number_of_features):
        # min.append(Variables.points[0][j])
        min.append(dataset[0][j])
        # for i in range(1 , len(Variables.points)): #is inclusinve?
        for i in range(1 , len(dataset)): #is inclusinve?
            # if min[j] > Variables.points[i][j]:
            if min[j] > dataset[i][j]:
                # min[j] = Variables.points[i][j]
                min[j] = dataset[i][j]
    
    logging.debug(f"min list: {min}")
    print("^")
    return min


def pick_random_points(k_value:int, number_of_features: int, dataset: list[list[float]]) -> list[list[float]]: #
    print("v")
    print("picking centroids")
    '''
    Calc min and max for each feature. Calc the interval beetween min and max for each of them.
    Devide it by the number of means. Calc single feature in single mean by picking random value
    beetween min and min + devided interval. Calc the same feature, but for the next mean - pick
    a value beetween min + devided interval and min + devided interval * 2. And so on.
    It makes in unlikely that there will be no points assigned to a given mean. Therefore no
    need to populate in artificially or trying to drop it alltogether. Plus they are more
    evenly distribuated which given reasonable data and k-value will improve results
    '''
    rand_points :list[list[float]] = []

    max_list = get_max(number_of_features, dataset)
    min_list = get_min(number_of_features, dataset)

    # intervals :list = [round(max_list[i] - min_list[i], 2) for i in range(Variables.number_of_features)]
    intervals :list = [round(max_list[i] - min_list[i], 2) for i in range(number_of_features)]

    logging.debug(f"intervals: {intervals}")

    # for i in range(Variables.number_of_features):
    for i in range(number_of_features):
        # intervals[i] = round((intervals[i] / Variables.k), 1)
        intervals[i] = round((intervals[i] / k_value), 1)

    logging.debug(f"intervals2: {intervals}")

    # for i in range(1, Variables.k + 1):
    for i in range(1, k_value + 1):
        tmp_rand_points :list[float] = []

        # for j in range(Variables.number_of_features):
        for j in range(number_of_features):
            tmp_rand_points.append(round(random.uniform(min_list[j] + (intervals[j] * (i - 1)), min_list[j] + (intervals[j] * i)), 2))

        rand_points.append(tmp_rand_points)
    
    print(f"means: {rand_points}")
    Variables.k_means = rand_points
    print("^")

    return rand_points


# maby merge this function with download_data_set()
def get_data(line: str, read_type: TypeOfRead) -> list[float]: #
    tmp_list: list = line.split()
    logging.debug(tmp_list)

    parsed_tmp_list = [eval(i) for i in tmp_list]

    # code for dealing with clusters described using a string
    # parsed_tmp_list2 = []
    # for i in range(len(tmp_list) - 1):
    #     parsed_tmp_list2.append(eval(tmp_list[i]))
    # parsed_tmp_list2.append(tmp_list[-1])
        
    if read_type == TypeOfRead.TRAINING:
        Variables.points.append(parsed_tmp_list)
        # Variables.points.append(parsed_tmp_list2)
    else:
        Variables.predict_data.append(parsed_tmp_list)
        # Variables.predict_data.append(parsed_tmp_list2)

    return parsed_tmp_list


def download_data_set(data_loc :str, read_type: TypeOfRead) -> tuple[list[list[float]], int]:  #
    logging.info("v - downloading data set")
    dataset: list[list[float]] = []
    # number_of_features = 0

    with open(data_loc, "r") as f:
        for line in f:
            # get_data(line, read_type)
            dataset.append(get_data(line, read_type))

    if read_type == TypeOfRead.TRAINING:
        Variables.number_of_features = len(Variables.points[0]) - 1
        print(f"number of features {len(Variables.points[0]) - 1}")
    logging.info("^")
    
    return dataset, len(Variables.points[0]) - 1


def train(k_value: int, data_loc: str):
    dataset, number_of_features = download_data_set(data_loc, TypeOfRead.TRAINING)
    random_points = pick_random_points(k_value, number_of_features, dataset)
    interation_loop(k_value, dataset, DefaultVariables.max_iterations)


def ask_for_k_value_and_data_loc() -> tuple[int, str]: #
    # missing_input = True

    # while missing_input:
    while True:
        k = int(input("Enter k value (int): "))
        if 1 < k < 7:
            Variables.k = k ##
            break
            # missing_input = False
        else:
            print("K must be beetween 2 and 6")

    # missing_input = True
    # while missing_input:
    while True:
        answer = int(input("For default data location type 1. Otherwise type 0: "))
        if answer == 1:
            Variables.data_loc = "data/iris_training.txt"
            data_loc = "data/iris_training.txt"
            # missing_input = False
            break
        elif answer == 0:
            custom_path = str(input("Enter custom data location: "))
            Variables.data_loc = custom_path
            data_loc = custom_path
            # missing_input = False
            break
        else:
            print("Enter valid input")

    return k, data_loc


def init():
    level = logging.INFO
    # level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt)
    # logging.basicConfig(level = level, format = fmt, filename="log-k-means.log", filemode="w")


def main():
    init()
    k_value, data_loc = ask_for_k_value_and_data_loc()
    train(k_value, data_loc)
    predict()


if __name__ == "__main__":
    main()
