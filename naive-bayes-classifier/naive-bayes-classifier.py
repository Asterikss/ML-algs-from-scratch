import pathlib
import logging
from enum import Enum


class InputType(Enum):
    FOR_BASE_MODEL = 0
    FOR_PREDICTION = 1


class NumberOfFeaturesError(Exception):
    pass


class MissmatchedLabels(Exception):
    pass


def ask_for_data_loc(input_type: InputType) -> pathlib.Path: # ~~pure
    while True:
        answer = int(input("For default data location type 1. Otherwise type 0: "))
        if answer == 1:
            if input_type == InputType.FOR_BASE_MODEL:
                # not including "/" does not work when
                # oppening a file with: with open...
                # Will look into it later
                return pathlib.Path("data/" + "iris_training.txt")
            else:
                return pathlib.Path("data/" + "iris_test.txt")
        elif answer == 0:
            while True:
                custom_path = pathlib.Path((input("Enter custom data location")))
                if custom_path.exists():
                    return custom_path
                print("Path not found")


def downlad_dataset(data_loc: pathlib.Path) -> tuple[list[list[float]], int, list[str], list[int], list[list[float]]]: # pure
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

    number_of_features = len(collected_data[0]) - 1
    logging.info(f"Number of features: {number_of_features}")
    logging.info(f"Label table: {label_tabel}")
    logging.info(f"Label occurrence table: {label_occurrence_tabel}")
    logging.info(f"Length of the dataset: {len(collected_data)}")
    logging.info(f"min and max table: {min_and_max_table}")

    return collected_data, number_of_features, label_tabel, label_occurrence_tabel, min_and_max_table


def calc_prior_prob(label_occurrence_tabel: list[int], n_of_examples: int) -> list[float]: # pure
    prior_prob = []
    for n_of_given_label in label_occurrence_tabel:
        prior_prob.append(n_of_given_label / n_of_examples)

    logging.info(f"Prior probabilities : {prior_prob}")
    return prior_prob


def train():
    ...


def bin_single_vector(vector: list[float], bins: list[list[list[float]]]) -> list[int]: # pure
    tmp_binned_example = []
    for i in range(len(vector) - 1):
        allocated_to_bin_index = -1
        for j in range(len(bins[0])):
            if vector[i] >= bins[i][j][0] and vector[i] <= bins[i][j][1]:
                allocated_to_bin_index = j

        tmp_binned_example.append(allocated_to_bin_index)

    tmp_binned_example.append(vector[-1])

    return tmp_binned_example


def bin_dataset(dataset: list[list[float]], min_and_max_table: list[list[float]], n_bins=3) -> tuple[list[list[int]], list[list[list[float]]]]: # pure
    # binned_dataset: list[list[float]] = []
    # print(binned_dataset)

    intervals_len: list[float] = [round((min_and_max_table[i][1] - min_and_max_table[i][0]) / n_bins, 2) for i in range(len(min_and_max_table))]
    logging.info(f"Intervals length: {intervals_len}")
    
    bins: list[list[list[float]]] = []

    for j in range(len(min_and_max_table)):
        bins.append([[min_and_max_table[j][0] + (intervals_len[j] * i), min_and_max_table[j][0] + (intervals_len[j] * (i + 1))] for i in range(n_bins)])

    logging.info(f"Bins: {bins}")

    # Allocate each feature in each example in the dataset to its bean (replace it with the index of the bin that it lies in for the given feature).
    # Seperate bins are created for every feature. Number of bins = n_bins * n_features
    # for example in dataset:
    #     tmp_binned_example = []
    #     for i in range(len(example) - 1):
    #         allocated_to_bin_index = -1
    #         for j in range(n_bins):
    #             if example[i] >= bins[i][j][0] and example[i] <= bins[i][j][1]:
    #                 allocated_to_bin_index = j
    #
    #         tmp_binned_example.append(allocated_to_bin_index)
    #
    #     tmp_binned_example.append(example[-1])
    #     binned_dataset.append(tmp_binned_example)


    # for example in dataset:
    #     binned_dataset.append(bin_single_vector(example, bins))

    # binned_dataset: list[list[float]] = [bin_single_vector(example, bins) for example in dataset]
    binned_dataset: list[list[int]] = [bin_single_vector(example, bins) for example in dataset]


    logging.debug(dataset)
    logging.debug("--------------")
    logging.debug(binned_dataset)

    return binned_dataset, bins


def calc_idx_most_prob_label(binned_vec: list[int], orig_label_occurrence_table: list[int],
        orig_binned_dataset: list[list[int]], n_bins: int,
        prior_prob: list[float]) -> tuple[int, int]: # pure

    # This name is funny. I know
    for_each_label_n_exmpl_with_x_label_in_same_bin: list[list[int]] = \
            [[0 for _ in range(len(binned_vec) - 1)] for _ in orig_label_occurrence_table]

    for example in orig_binned_dataset:
        label_idx: int = int(example[-1])
        
        for i in range(len(binned_vec) - 1):
            # logging.debug(label_idx, binned_vec[i], example[i])
            if binned_vec[i] == example[i]:
                for_each_label_n_exmpl_with_x_label_in_same_bin[label_idx][i] += 1


            
    logging.debug(for_each_label_n_exmpl_with_x_label_in_same_bin)
    
    probability_tabel: list[float] = [1 for _ in orig_label_occurrence_table]

    for i, tab in enumerate(for_each_label_n_exmpl_with_x_label_in_same_bin):
        for n_positives_for_feature in tab:
            if n_positives_for_feature != 0:
                probability_tabel[i] *= n_positives_for_feature / orig_label_occurrence_table[i]
            else:
                probability_tabel[i] *= (n_positives_for_feature + 1) / (orig_label_occurrence_table[i] + n_bins)

        probability_tabel[i] *= prior_prob[i]

    logging.debug(probability_tabel)

    idx_most_prob_label = probability_tabel.index(max(probability_tabel))
    idx_true_label = binned_vec[-1]

    return idx_most_prob_label, idx_true_label



def predict_dataset(new_dataset: list[list[float]], new_label_table: list[str],
        bins: list[list[list[float]]], prior_probability: list[float],
        orig_label_tabel: list[str], orig_binned_dataset: list[list[int]],
        orig_label_occurrence_table: list[int]):
    
    n_correct = 0
    print("prediction:\ttrue label:")

    for example in new_dataset:
        binned_example: list[int] = bin_single_vector(example, bins)

        idx_most_prob_label, idx_true_label = calc_idx_most_prob_label(binned_example,
            orig_label_occurrence_table, orig_binned_dataset, len(bins[0]),
            prior_probability)

        # If the second dataset has labels occurring in different
        # order, this approach will prevent issues
        pred_label = orig_label_tabel[idx_most_prob_label]
        true_label = new_label_table[idx_true_label]
       
        correct = False
        if pred_label == true_label:
            n_correct += 1
            correct = True
        
        print(f"{pred_label}\t{true_label}\tCorrect: {correct}")

        # break

    print(f"Accuracy: {n_correct/ len(new_dataset)}")


def check_compatibility(number_of_feature1, number_of_feature2, label_tabel1, label_tabel2): # pure
    if number_of_feature1 != number_of_feature2:
        raise NumberOfFeaturesError("Number of features is differ between datasets." +
                                    " If the second dataset does not hava labels, add dummy labels"+
                                    " and ignor it (Must be one of the actuall labels from first dataset)")

    for label in label_tabel2:
        if label not in label_tabel1:
            raise MissmatchedLabels("Some labels from the second "+
                                    "dataset are not present in the first one")


def custom_prediction(number_of_features: int, bins: list[list[list[float]]],
          orig_label_occurrence_table: list[int], orig_binned_dataset:
          list[list[int]], prior_prob: list[float], orig_label_tabel: list[str]):

    print("Custom prediction")
    custom_vector = []
    for i in range(number_of_features):
        custom_vector.append(float(input(f"Input {i+1} feature: ")))

    # True label - unknown
    custom_vector.append(-1)

    binned_cust_vec: list[int] = bin_single_vector(custom_vector, bins)

    idx_most_prob_label, _ = calc_idx_most_prob_label(binned_cust_vec,
                                   orig_label_occurrence_table, orig_binned_dataset,
                                   len(bins[0]), prior_prob)

    pred_label = orig_label_tabel[idx_most_prob_label]

    print(f"Prediction -> {pred_label}")


def init():
    level = logging.INFO
    # level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt) # filename = 'log_x.log', filemode = "w"


def main():
    init()
    data_loc: pathlib.Path = ask_for_data_loc(InputType.FOR_BASE_MODEL)
    dataset, number_of_features, label_tabel, label_occurrence_tabel, min_and_max_table = downlad_dataset(data_loc)
    prior_probability: list[float] = calc_prior_prob(label_occurrence_tabel, len(dataset))
    binned_dataset, bins = bin_dataset(dataset, min_and_max_table)
    
    predict_dataset_loc = ask_for_data_loc(InputType.FOR_PREDICTION)
    dataset_for_prediction, number_of_feature_pred, label_tabel_pred, _, _ = downlad_dataset(predict_dataset_loc)
    check_compatibility(number_of_features, number_of_feature_pred, label_tabel, label_tabel_pred)
    
    predict_dataset(dataset_for_prediction, label_tabel_pred, bins, prior_probability, label_tabel, binned_dataset, label_occurrence_tabel)

    custom_prediction(number_of_features, bins, label_occurrence_tabel, binned_dataset, prior_probability, label_tabel)


if __name__ == "__main__":
    main()
