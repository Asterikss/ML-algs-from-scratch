# pyright: reportUnusedVariable=false
# pyright: ignore - single line (# type: ignore)
# pyright: ignore [reportUnusedVariable=] - single line
# https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670
import random


class Variables:
    k = 0
    points = []
    k_means = []
    number_of_features = 0

    prev_k_means = []


class DefaultVariables:
    max_iterations = 300


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

        # with few data points there can be a situation where one list is empty.
        # I devide by len of them later. Probably fix it there. Marked as @!@1
        # or here just go through all lists and add a dumy point
        points_sorted[which_mean_closest].append(point)

    print(points_sorted)



    Variables.prev_k_means = Variables.k_means

    new_k_means :list[float] = [0.0 for _ in range(Variables.k)]
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
    while i < 4 and Variables.k_means != Variables.prev_k_means:
        one_full_iter()

    print(Variables.k_means)


def calc_means(points :list) -> list[float]:
    print("calc means")
    new_k_means :list[float] = [0 for _ in range(Variables.k)]

    for i in range(Variables.k):
        sum = 0.0
        mean = 0.0
        for j in range(len(points)):
            sum += points[j][i]

        # @!@1
        mean = sum/len(points)
        new_k_means[i] = mean

    return new_k_means


def ask_for_k_value():
    Variables.k = int(input("Enter k value: "))


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

    # get_data("3 4   2   0")
    # get_data("4 3   4   1")
    # get_data("3 2   5   1")


        # while f
        #   str = f.readline()
    Variables.number_of_features = len(Variables.points[0]) - 1
    print(f"asdf {len(Variables.points[0]) - 1}")



def get_data(line: str):
    print("----")
    print("getting data")
    tmp_list: list = line.split()  # still a list
    print(tmp_list)
    print(type(tmp_list))
    print(len(tmp_list))

    # lis = ['1', '-4', '3', '-6', '7']
    int_tmp_list = [eval(i) for i in tmp_list]
    print("Modified list is: ", int_tmp_list)

    Variables.points.append(int_tmp_list)
    print("----")
    

def train():
    dowload_data_set()
    pick_random_points()
    interation_loop()


def main():
    ask_for_k_value()
    train()


if __name__ == "__main__":
    main()
