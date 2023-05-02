import pathlib
import logging

def ask_for_data_loc() -> pathlib.Path: # ~~pure
    while True:
        answer = int(input("For default data location type 1. Otherwise type 0: "))
        if answer == 1:
            return pathlib.Path("data/" + "iris_training.txt")
        elif answer == 0:
            while True:
                custom_path = pathlib.Path((input("Enter custom data location")))
                if custom_path.exists():
                    return custom_path
                print("Path not found")


def downlad_dataset(data_loc: pathlib.Path) -> tuple[list[list[float]], int, list[str], list[int]]: # pure
    collected_data = []
    label_tabel = []
    # ^, v - not using a dict since indexes matter
    label_occurrence_tabel = []

    #           [
    # feature_n   [min, max]
    # feature_n+1 [min, max]
    # ...
    #                      ]
    min_and_max_table: list[list[float]] = []
    firstTime = True
    

    with open(data_loc, "r", encoding="utf-8") as f:
        for line in f:
            splited: list[str] = line.split()
            # decoded2: list[float] = [eval(splited[i]) for i in range(len(splited) - 1)]
            if firstTime:
                    min_and_max_table: list[list[float]] = [[0,0] for _ in range(len(splited) - 1)]
            
            decoded: list[float] = []
            for i in range(len(splited) - 1):
                tmp_eval = eval(splited[i])
                decoded.append(tmp_eval)

                if firstTime:
                    min_and_max_table[i][0] = tmp_eval
                    min_and_max_table[i][1] = tmp_eval
                

                if not firstTime:
                    if tmp_eval < min_and_max_table[i][0]:
                        min_and_max_table[i][0] = tmp_eval
                    elif tmp_eval > min_and_max_table[i][1]:
                        min_and_max_table[i][1] = tmp_eval

            firstTime = False



            label = splited[-1]
            if label not in label_tabel:
                label_tabel.append(label)
                label_occurrence_tabel.append(0)

            index = label_tabel.index(label)
            decoded.append(index)
            label_occurrence_tabel[index] += 1
            
            collected_data.append(decoded)

    number_of_feature = len(collected_data[0]) - 1
    logging.info(f"Number of features: {number_of_feature}")
    logging.info(f"Label table: {label_tabel}")
    logging.info(f"Label occurrence table: {label_occurrence_tabel}")
    logging.info(f"Length of the dataset: {len(collected_data)}")
    logging.info(f"min and max table: {min_and_max_table}")

    return collected_data, number_of_feature, label_tabel, label_occurrence_tabel


def calc_prior_prob(label_occurrence_tabel: list[int], n_of_examples: int) -> list[float]: # pure
    prior_prob = []
    for n_of_given_label in label_occurrence_tabel:
        prior_prob.append(n_of_given_label / n_of_examples)

    logging.info(f"Prior probabilities : {prior_prob}")
    return prior_prob


def train():
    ...


def init():
    level = logging.INFO
    # level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt) # filename = 'log_x.log', filemode = "w"


def main():
    init()
    data_loc: pathlib.Path = ask_for_data_loc()
    dataset, number_of_feature, label_tabel, label_occurrence_tabel = downlad_dataset(data_loc)
    prior_probability: list[float] = calc_prior_prob(label_occurrence_tabel, len(dataset))


if __name__ == "__main__":
    main()
