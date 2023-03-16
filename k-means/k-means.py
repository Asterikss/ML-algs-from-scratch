# pyright: reportUnusedVariable=false
# pyright: ignore - single line (# type: ignore)
# pyright: ignore [reportUnusedVariable=] - single line
# https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670

# Iris-setosa = 0
# Iris-versicolor = 1
# Iris-virginica = 2
import random
from enum import Enum


class Variables:
    k = 0
    data_loc = ""
    points = []
    k_means = []
    number_of_features = 0
    prev_k_means = []

    predict_data = []


class DefaultVariables:
    max_iterations = 300


class TypeOfRead(Enum):
    TRAINING = 0
    PREDICTING = 1

class DataLoc(Enum):
    DEFAULT = 0
    CUSTOM = 1

def get_max() -> list:
    print("-----")
    print(f"calculating max for {Variables.number_of_features} features")
    max: list = []
    print(f"print listy init {max}")

    for j in range(0,  Variables.number_of_features):
        max.append(Variables.points[0][j])
        print(f"print listy: {max}")
        for i in range(1 , len(Variables.points)): #is inclusinve?
            if max[j] < Variables.points[i][j]:
                max[j] = Variables.points[i][j]
    
    print(f"print listy final {max}")
    print("-----")
    return max


def get_min() -> list:
    print("-----")
    print(f"calculating min for {Variables.number_of_features} features")
    min: list = []
    print(f"print listy init {min}")

    for j in range(0,  Variables.number_of_features):
        min.append(Variables.points[0][j])
        print(f"print listy: {min}")
        for i in range(1 , len(Variables.points)): #is inclusinve?
            if min[j] > Variables.points[i][j]:
                min[j] = Variables.points[i][j]
    
    print(f"print listy final {min}")
    print("-----")
    return min


def pick_random_points():
    print("----")
    print("picking random starting points")
    rand_points :list = []
    # tmp_rand_points :list = []

    max_list = get_max()
    min_list = get_min()

    for i in range(Variables.k):
        tmp_rand_points :list = []

        for i in range(Variables.number_of_features):
            # tmp_rand_points.append(random.random(min_list[i], max_list[i]))
            tmp_rand_points.append(round(random.uniform(min_list[i], max_list[i]), 2))

        rand_points.append(tmp_rand_points)
    
    print(rand_points)
    Variables.k_means = rand_points
    print("----")


def calc_means(points :list) -> list[float]:
    print("calc means")
    new_single_k_mean :list[float] = [0 for _ in range(Variables.number_of_features)]

    for i in range(Variables.number_of_features):
        sum = 0.0
        mean = 0.0
        for j in range(len(points)):
            sum += points[j][i]

        mean = sum/len(points)
        new_single_k_mean[i] = mean

    return new_single_k_mean


def ask_for_k_value_and_data_loc():
    Variables.k = int(input("Enter k value: "))
    # TODO
    Variables.data_loc = "data/iris_training.txt"


def calc_euclidean_distance(a: tuple, b: tuple) -> float:
    print("eucal")
    # print(a)
    # print(len(a))


    if len(a) -1 != len(b):
        print("tuples are of wrong size. 'a'(first param) should be longer by 1 (last arg is the answer) ")
        return -1
    dist = 0

    for i in range(0, len(b)):
        dist += (a[i] - b[i]) ** 2

    print(f"c_e_d: {dist ** (1 / 2)}")
    return dist ** (1 / 2)


def one_full_iter():
    print("full one----@@@@@@@@@@@@")
    points_sorted = [[] for _ in range(Variables.k)]
    # tmp_dist :list[float] = [0 for _ in range(Variables.k)]
    # print(tmp_dist)

    for point in Variables.points:
        tmp_dist :list[float] = [0 for _ in range(Variables.k)]
        print(f"first {tmp_dist}")

        for i in range(Variables.k):
            tmp_dist[i] = calc_euclidean_distance(point, Variables.k_means[i])

        print(f"second {tmp_dist}")
        which_mean_closest = tmp_dist.index(min(tmp_dist)) if tmp_dist else -1
        print(which_mean_closest)

        points_sorted[which_mean_closest].append(point)

    print(points_sorted)

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
            
    print(points_sorted)




    Variables.prev_k_means = Variables.k_means

    print(f"!!k_means: {Variables.prev_k_means}")

    new_k_means :list[list[float]] = [[] for _ in range(Variables.k)]
    # new_k_means = [0.0 for _ in range(Variables.k)]
    # new_k_means :list[float] = []
    # new_k_means :list[float] = [0 for _ in range(Variables.k)]

    for i in range(Variables.k):
        new_k_means[i] = calc_means(points_sorted[i])
        # new_k_means.append(calc_means(points_sorted[i]))
        # new_k_means[i] = 3.0

    print(new_k_means)
    
    Variables.k_means = new_k_means

    # print(calc_means([[1,4,5,6], [1,4,6,6], [2, 4, 6, 1]]))
    print("full end----$$$$$$$$$$$$$$$")


def interation_loop():
    i = 0
    #while i < DefaultVariables.max_iterations and Variables.k_means != Variables.prev_k_means:
    while i < 1 and Variables.k_means != Variables.prev_k_means:
        one_full_iter()
        i += 1

    print(Variables.k_means)
    print("--End of interation loop--")


#enum would be better
def get_data(line: str, read_type: TypeOfRead):
    print("----")
    print("getting data")
    tmp_list: list = line.split()
    # tmp_list: list = line.split("   " || " ")
    print(tmp_list)
    print(type(tmp_list))
    print(len(tmp_list))

    # lis = ['1', '-4', '3', '-6', '7']
    int_tmp_list = [eval(i) for i in tmp_list]
    print("Modified list is: ", int_tmp_list)

    if read_type == TypeOfRead.TRAINING:
        Variables.points.append(int_tmp_list)
    else:
        Variables.predict_data.append(int_tmp_list)

    print("----")
 

def dowload_data_set(data_loc :str, read_type: TypeOfRead):
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
    if Variables.number_of_features == 0:
        Variables.number_of_features = len(Variables.points[0]) - 1
        print(f"asdf {len(Variables.points[0]) - 1}")


def predict_label(point :tuple) -> int:
    tmp_dist :list[float] = [0 for _ in range(Variables.k)]
    print(f"first_in_label {tmp_dist}")

    for i in range(Variables.k):
        tmp_dist[i] = calc_euclidean_distance(point, Variables.k_means[i])

    print(f"second_in_label {tmp_dist}")
    which_mean_closest = tmp_dist.index(min(tmp_dist)) if tmp_dist else -1
    print(which_mean_closest)
    return which_mean_closest


def predict():
    print(f"Begin predict")

    choice = -1
    while choice != 0 and choice != 1 and choice != 2:
        print("For predicting data from the default file (data/iris_test.txt) type 0")
        print("For custom guess (providing vector) type 1 ")
        print("For predicting data from custom file type 2")
        choice = int(input(": "))

    if choice == 0:
        dowload_data_set("data/iris_test.txt", TypeOfRead.PREDICTING)
        # with open("data/iris_test.txt", "r") as f:
        #     for line in f:
        #         print(line)
        #         get_data(line, TypeOfRead.PREDICTING)

        print(Variables.predict_data)

        label = predict_label(Variables.predict_data[0])
        label_table = [0 for _ in range(Variables.k)] 

        label_table[label] += 1

        print("lable_table:")
        print(label_table)
        
    
    print(f"End predict")

def train():
    dowload_data_set(Variables.data_loc, TypeOfRead.TRAINING)
    pick_random_points()
    interation_loop()


def main():
    ask_for_k_value_and_data_loc()
    train()
    predict()



if __name__ == "__main__":
    main()
