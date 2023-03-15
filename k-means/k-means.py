# pyright: reportUnusedVariable=false
# pyright: ignore - single line (# type: ignore)
# pyright: ignore [reportUnusedVariable=] - single line
# https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670
import random


class Variables:
    k = 0
    k_points = []
    k_medians = []
    number_of_features = 0


class DefaultVariables:
    max_iterations = 300


def get_max() -> list:
    print("-----")
    print(f"calculating max for {Variables.number_of_features} features")
    max: list = []
    print(f"print listy init {max}")

    for j in range(0,  Variables.number_of_features):
        max.append(Variables.k_points[0][j])
        print(f"print listy: {max}")
        for i in range(1 , len(Variables.k_points)): #is inclusinve?
            if max[j] < Variables.k_points[i][j]:
                max[j] = Variables.k_points[i][j]
    
    print(f"print listy final {max}")
    print("-----")
    return max


def get_min() -> list:
    print("-----")
    print(f"calculating min for {Variables.number_of_features} features")
    min: list = []
    print(f"print listy init {min}")

    for j in range(0,  Variables.number_of_features):
        min.append(Variables.k_points[0][j])
        print(f"print listy: {min}")
        for i in range(1 , len(Variables.k_points)): #is inclusinve?
            if min[j] > Variables.k_points[i][j]:
                min[j] = Variables.k_points[i][j]
    
    print(f"print listy final {min}")
    print("-----")
    return min


def pick_random_points():
    print("----")
    print("picking random starting points")
    rand_poitns :list = []
    # tmp_rand_points :list = []

    max_list = get_max()
    min_list = get_min()

    for i in range(Variables.k):
        tmp_rand_points :list = []

        for i in range(Variables.number_of_features):
            # tmp_rand_points.append(random.random(min_list[i], max_list[i]))
            tmp_rand_points.append(round(random.uniform(min_list[i], max_list[i]), 2))

        rand_poitns.append(tmp_rand_points)
    
    print(rand_poitns)
    Variables.k_medians = rand_poitns
    print("----")


def iterations():
    pass


def ask_for_k_value():
    Variables.k = int(input("Enter k value: "))


def calc_euclidean_distance(a: tuple, b: tuple) -> float:
    # print(a)
    # print(len(a))


    if len(a) != len(b):
        print("tuples are of different size")
        return -1
    dist = 0

    for i in range(0, len(a)):
        dist += (a[i] - b[i]) ** 2

    print(f"c_e_d: {dist ** (1 / 2)}")
    return dist ** (1 / 2)

def dowload_data_set():
    # f0 = open("data.txt", "r")

    # with open("venv/data.txt", "r") as f:
    #     for line in f:
    #         print("a")
    #         print(line)
    #         get_data(line)

    get_data("5 2   3   1")
    get_data("2 5   1   0")
    get_data("3 2   4   1")
    get_data("2 2   6   2")
    get_data("2 2   6   2")

        # while f
        #   str = f.readline()
    Variables.number_of_features = len(Variables.k_points[0]) - 1
    print(f"asdf {len(Variables.k_points[0]) - 1}")

def train():
    dowload_data_set()
    # get_max()
    # get_min()
    pick_random_points()
    


def get_data(line: str):
    print("----")
    print("getting data")
    tmp_list: tuple = line.split()  # still a list
    print(tmp_list)
    print(type(tmp_list))
    print(len(tmp_list))

    # lis = ['1', '-4', '3', '-6', '7']
    int_tmp_list = [eval(i) for i in tmp_list]
    print("Modified list is: ", int_tmp_list)

    Variables.k_points.append(int_tmp_list)
    print("----")
    


def main():
    # a = (1, 3, 5)
    # b = (3, 1, 4)
    # calc_euclidean_distance(a, b)
    ask_for_k_value()
    train()
    


if __name__ == "__main__":
    main()
