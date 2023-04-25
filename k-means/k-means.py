# use pathlib later

# originally in the file
# 0 = Iris-setosa
# 1 = Iris-versicolor
# 2 = Iris-virginica
import random
from dataclasses import dataclass
import logging
import math


class IncompatiblePointsLength(Exception):
    pass


class EmptyCluster(Exception):
    pass


@dataclass(frozen=True)
class DefaultVariables:
    max_iterations = 10
    threshold = 0.000001


def calc_means(points :list, number_of_features: int) -> list[float]: # pure
    new_single_k_mean :list[float] = [0 for _ in range(number_of_features)]

    for i in range(number_of_features):
        sum = 0.0
        mean = 0.0
        for j in range(len(points)):
            sum += points[j][i]

        mean = round(sum/len(points), 2)
        new_single_k_mean[i] = mean

    return new_single_k_mean


# TODO
def predict_cluster(point: list[float], centroids: list[list[float]], k_value: int) \
        -> tuple[int, int]: # ~~pure

    tmp_dist :list[float] = [0 for _ in range(k_value)]

    for i in range(k_value):
        tmp_dist[i] = calc_euclidean_distance(point, centroids[i])

    # which_mean_closest = tmp_dist.index(min(tmp_dist)) if tmp_dist else -1
    which_mean_closest = tmp_dist.index(min(tmp_dist))

    if point[-1] == k_value:
        print(f"Prediction: Cluster {which_mean_closest}. Actual cluster: unknown")
        return which_mean_closest, -1
    
    print(f"Prediction: Cluster {which_mean_closest}. Actual cluster: {point[-1]}")

    return which_mean_closest, int(point[-1])


def predict_cluster2(point: list[float], centroids: list[list[float]]) -> int: # pure
    tmp_dist :list[float] = [0 for _ in range(len(centroids))]

    for i in range(len(centroids)):
        tmp_dist[i] = calc_euclidean_distance(point, centroids[i])

    return tmp_dist.index(min(tmp_dist)) + 1


# TODO
def predict(centroids: list[list[float]]):
    number_of_features = len(centroids[0])

    choice = -1
    print("For predicting the cluster from the default file (data/iris_test.txt) type 1")
    print("For custom guess (providing a vector) type 2 ")
    print("For predicting the cluster from custom file type 3")
    #That's a terrible way probably
    while choice != "1" and choice != "2" and choice != "3":
        choice = input(": ")

    # dataset = []
    if choice == "1" or choice == "3":
        pass

        # TODO
        # if choice == "1":
        #     dataset, number_of_features = download_data_set("data/iris_test.txt")
        #     # maby check if the number of features is the same or smth
        # elif choice == "3":
        #     path = str(input("Provide path: "))
        #     dataset, number_of_features = download_data_set(path)
        #
        # predictions = [0 for _ in range(k_value)] 
        # actual_clusters = [0 for _ in range(k_value)] 
        #
        # for observation in dataset:
        #     cluster_tuple :tuple[int, int] = predict_cluster(observation, centroids, k_value)
        #     logging.debug(cluster_tuple)
        #     predictions[cluster_tuple[0]] += 1
        #     actual_clusters[cluster_tuple[1]] += 1
        #
        # accuracy_table = [1 - round((abs(predictions[i] - actual_clusters[i])/actual_clusters[i]), 3) for i in range(k_value)] 
        # total_acc = 0
        #
        # for e in accuracy_table:
        #     total_acc += e
        # total_acc /= k_value
        #
        # print("predictions:")
        # print(predictions)
        # print("acutal clusters:")
        # print(actual_clusters)
        # print("accuracy_table:")
        # print(accuracy_table)
        # print("Total accuracy:")
        # print(total_acc)

    elif choice == "2":
        end = False
        print(f"Enter a vector with {number_of_features} features")
        while not end:
            custom_vector: list[float] = []
            for i in range(number_of_features):
                custom_vector.append((float(input(f"Input {i+1} feature: "))))

            predicted_cluster: int = predict_cluster2(custom_vector, centroids)
            print(f"Prediction: Cluster {predicted_cluster}")
    
            q = input("Type q to exit. Otherwise hit enter: ")
            if q == "q":
                end = True


def display_results(centroids: list[list[float]], dataset: list[list[float]]): # ~~pure
    k_value = len(centroids)

    points_sorted = [[] for _ in range(k_value)]

    for point in dataset:
        tmp_dist :list[float] = [0 for _ in range(k_value)]

        for i in range(k_value):
            tmp_dist[i] = calc_euclidean_distance(point, centroids[i])

        which_mean_closest: int = tmp_dist.index(min(tmp_dist))

        points_sorted[which_mean_closest].append(point)

    logging.info("Clusters with the points associated with them: ")
    for idx, centroid in enumerate(centroids):
        labels_table: list[int] = [0 for _ in range(k_value)]

        logging.info(f"\ncentroid nr {idx+1} - {centroid}:")
        for point in points_sorted[idx]:
            labels_table[point[-1]] += 1
            print(point, end=" ")
        print()

        entropy = 0
        n_of_points = len(points_sorted[idx])

        for i, n_points_with_this_label in enumerate(labels_table):
            if n_points_with_this_label != 0:
                probabiliti = n_points_with_this_label / n_of_points
                entropy += probabiliti * math.log(probabiliti, 2)

        entropy *= -1                      

        print(f"\n{labels_table}")
        print(f"Entropy: {entropy}")


def is_done_iterating(new_centroids, prev_centroids, k_value, number_of_features, \
        threshold=DefaultVariables.threshold) -> bool: # pure

    for i in range(k_value):
        dist = 0
        for j in range(number_of_features):
            dist += (prev_centroids[i][j] - new_centroids[i][j]) ** 2

        if dist < threshold:
            logging.info("~Threshold reached~")
            return True

    return False


def calc_euclidean_distance(point: list[float], centroid: list[float]) -> float: # pure
    # if len(point) -1 != len(centroid):
    #     raise IncompatiblePointsLength("Points are of wrong size. " +
    #             "First param should be longer by 1 (last arg is the centroid)")

    dist = 0
    for i in range(0, len(centroid)):
        dist += (point[i] - centroid[i]) ** 2

    return round(dist, 3)


def one_full_iter(k_value: int, dataset: list[list[float]], centroids: list[list[float]], \
        number_of_features: int) -> tuple[list[list[float]], list[list[float]]]: # pure

    points_sorted = [[] for _ in range(k_value)]
    total_distance = 0

    for point in dataset:
        tmp_dist :list[float] = [0 for _ in range(k_value)]

        for i in range(k_value):
            tmp_dist[i] = calc_euclidean_distance(point, centroids[i])

        logging.debug(tmp_dist)
        # which_mean_closest = tmp_dist.index(min(tmp_dist)) if tmp_dist else -1
        min_distance = min(tmp_dist)
        total_distance += min_distance
        # which_mean_closest = tmp_dist.index(min(tmp_dist))
        which_mean_closest = tmp_dist.index(min_distance)
        logging.debug(which_mean_closest)

        points_sorted[which_mean_closest].append(point)

    # v - Just throwing an error for now
    # Sometimes, with few data points, there can be a situation where none
    # of the points are closest to a particualr starting mean. Therefore
    # an empty list is created. I'm populating it here with big numbers
    # so none of the future points will be closest to that mean.
    # Algorithm, by doing that, will suggest that their are less groups
    # than entered by a user. Puting big numbers there is unnecesarry. 
    # List can't be empty though, because I'm deviding by it's len later
    for i in range(len(points_sorted)):
        if len(points_sorted[i]) == 0:
            raise EmptyCluster("One of the clusters is empty " +
                "(no points are closest to it). There is not enough data to" +
                " train on or value of k is too large or just bad variance." +
                " Try again. If it fails reduce the value of k or add more data")
            # points_sorted[i].append([1248 for _ in range(k_value)])
            # points_sorted[i][0].append(-1)
            # print("One of the lists created as an empty list.")
            # print("Now populated artificially.")
            
    prev_centroids = centroids

    new_centroids :list[list[float]] = [[] for _ in range(k_value)]

    for i in range(k_value):
        new_centroids[i] = calc_means(points_sorted[i], number_of_features)
    logging.info(f"Summed distances between each point and their centroid: {total_distance}")
    logging.info(f"Prev centroids: {prev_centroids}")
    logging.info(f"New centroids: {new_centroids}")

    return new_centroids, prev_centroids


def iteration_loop(k_value: int, dataset: list[list[float]], centroids: list[list[float]], \
        number_of_features: int, max_iterations: int) -> list[list[float]]: # pure

    logging.info("v - Start of the interation loop")
    i = 1

    logging.info(f"-inter {i}-")
    new_centroids, prev_centroids = one_full_iter(k_value, dataset, centroids, number_of_features)

    while i < max_iterations and not is_done_iterating(new_centroids, prev_centroids, k_value, number_of_features):
    # while i < max_iterations:
        i += 1
        logging.info(f"-inter {i}-")
        new_centroids, prev_centroids = one_full_iter(k_value, dataset, new_centroids, number_of_features)

    logging.info("^ - End of the interation loop")
    return new_centroids


def get_min_and_max(number_of_features: int, dataset: list[list[float]]) -> tuple[list[float], list[float]]: # pure
    logging.info(f"calculating min and max for {number_of_features} features")
    max: list[float] = []
    min: list[float] = []

    for j in range(0,  number_of_features):
        min.append(dataset[0][j])
        max.append(dataset[0][j])
        for i in range(1 , len(dataset)):
            if min[j] > dataset[i][j]:
                min[j] = dataset[i][j]

            if max[j] < dataset[i][j]:
                max[j] = dataset[i][j]
    
    logging.debug(f"min list: {min}, max list: {max}")
    return max, min


def pick_random_points(k_value:int, number_of_features: int, dataset: list[list[float]]) -> list[list[float]]: # pure
    logging.info("v - picking centroids")
    '''
    Calc min and max for each feature. Calc the interval beetween min and max for each of them.
    Devide it by the number of clusters. Calc single feature in a single cluster by picking random value
    beetween min and min + devided interval. Calc the same feature, but for the next cluster - pick
    a value beetween min + devided interval and min + (devided interval * 2). And so on.
    It makes in unlikely that there will be no points assigned to a given mean. Therefore no
    need to populate in artificially or to try to drop it alltogether. Plus they are more
    evenly distribuated, which given reasonable data and k-value, will improve results.
    '''
    rand_points :list[list[float]] = []

    max_list: list[float]
    min_list: list[float]
    max_list, min_list = get_min_and_max(number_of_features, dataset)

    intervals :list[float] = [round(max_list[i] - min_list[i], 2) for i in range(number_of_features)]

    logging.debug(f"intervals: {intervals}")

    for i in range(number_of_features):
        intervals[i] = round((intervals[i] / k_value), 1)

    logging.debug(f"intervals2: {intervals}")

    for i in range(1, k_value + 1):
        tmp_rand_points :list[float] = []

        for j in range(number_of_features):
            tmp_rand_points.append(round(random.uniform(min_list[j] + (intervals[j] * (i - 1)), min_list[j] + (intervals[j] * i)), 2))

        rand_points.append(tmp_rand_points)

    logging.debug(f"rand_points1: {rand_points}")

    # shuffle them (keeping the index of the feature) to be more random
    for i in range(k_value):
        for j in range(number_of_features):
            rand_one: int = random.randint(0, k_value - 1)
            if rand_one != i:
                tmp: float = rand_points[i][j]
                rand_points[i][j] = rand_points[rand_one][j] 
                rand_points[rand_one][j] = tmp

    logging.debug(f"rand_points2: {rand_points}")

    logging.info(f"initial centroids: {rand_points}")
    logging.info("^")

    return rand_points


def download_data_set(data_loc :str) -> tuple[list[list[float]], int]:  # pure
    logging.info("v - downloading data set")
    dataset: list[list[float]] = []

    with open(data_loc, "r") as f:
        for line in f:
            tmp_list: list = line.split()
            logging.debug(tmp_list)
            parsed_tmp_list = [eval(i) for i in tmp_list]
            dataset.append(parsed_tmp_list)

    number_of_features = len(dataset[0]) - 1
    logging.info(f"number of features -> {number_of_features}")
    logging.info("^")

    return dataset, number_of_features


def train(k_value: int, dataset: list[list[float]], number_of_features: int, \ 
          max_iterations: int) -> list[list[float]]: # pure

    centroids = pick_random_points(k_value, number_of_features, dataset)
    centroids =  iteration_loop(k_value, dataset, centroids, number_of_features, max_iterations)
    return centroids


def ask_for_k_value_and_data_loc() -> tuple[int, str]: # pure
    while True:
        k = int(input("Enter k value (int): "))
        if 1 < k < 7:
            break
        else:
            print("K must be beetween 2 and 6")

    while True:
        answer = int(input("For default data location type 1. Otherwise type 0: "))
        if answer == 1:
            return k, "data/iris_training.txt"
        elif answer == 0:
            return k, str(input("Enter custom data location (with ""): "))
        else:
            print("Enter valid input")


def init():
    level = logging.INFO
    # level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt)
    # logging.basicConfig(level = level, format = fmt, filename="log-k-means.log", filemode="w")


def main():
    init()
    k_value, data_loc = ask_for_k_value_and_data_loc()
    dataset, number_of_features = download_data_set(data_loc)
    centroids = train(k_value, dataset, number_of_features, DefaultVariables.max_iterations)
    display_results(centroids, dataset)
    predict(centroids)


if __name__ == "__main__":
    main()
