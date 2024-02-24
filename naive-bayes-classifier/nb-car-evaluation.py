from collections import defaultdict
from functools import reduce as ft_reduce
from random import shuffle as rnd_shuffle
from typing import Tuple, List, Dict


def calc_individual_probs(
    train_set: List[List[str]],
    X: str,
    labels_counter_dict: Dict[int, List[str]],
    target_label_idx: int,
    decision_label_counter: Dict[str, int],
) -> List[float]:  # pure
    # [
    #   [ ... n features ]
    #   ...
    #   n target labels
    # ]
    out = [
        [0.0 for _ in range(len(X) - 1)]
        for _ in range(len(labels_counter_dict[target_label_idx]))
    ]

    for j, target_label in enumerate(labels_counter_dict[target_label_idx]):
        for i in range(len(X) - 1):
            a = sum(1 for z in train_set if z[i] == X[i] and z[-1] == target_label)
            b = decision_label_counter[target_label]
            # Laplace smoothing
            if a == 0:
                a = 1
                b += len(labels_counter_dict[i])
            out[j][i] = a / b

    return [ft_reduce(lambda x, y: x * y, l) for l in out]


def perform_prediction(
    train_set,
    test_set,
    labels_counter_dict: Dict[int, List[str]],
    prior_probs: Dict[str, float],
    decision_label_counter: Dict[str, int],
    show: bool = False,
) -> float:
    target_label_idx = len(train_set[0]) - 1
    n_correct = 0

    for X in test_set:
        probs: List[float] = calc_individual_probs(
            train_set, X, labels_counter_dict, target_label_idx, decision_label_counter
        )
        # walk through all the possible target lables and update the probs with the a static prob associated with every target lable
        for i, target_label in enumerate(labels_counter_dict[target_label_idx]):
            probs[i] *= prior_probs[target_label]

        guess = labels_counter_dict[target_label_idx][probs.index(max(probs))]
        actual = X[-1]
        if show:
            print(f"Guess: {guess} Actual lable: {actual} Verdict: {guess == actual}")

        n_correct += guess == actual * 1

    print(f"Correct guesses: {n_correct} / {len(test_set)}")
    print(f"Accuracy {n_correct/len(test_set)}")
    return n_correct / len(test_set)


def calc_prior_prob(
    decision_label_counter: Dict[str, int], len_dataset: int
) -> Dict[str, float]:  # pure
    return {l: (c / len_dataset) for l, c in decision_label_counter.items()}


def get_decision_label_counter(dataset) -> Dict[str, int]:  # pure
    decision_label_counter = defaultdict(int)
    for X in dataset:
        decision_label_counter[X[-1]] += 1
    return decision_label_counter


def split_dataset(
    dataset: List[List[str]],
) -> Tuple[List[List[str]], List[List[str]]]: # pure
    return (
        dataset[: len(dataset) * 70 // 100],
        dataset[len(dataset) * 70 // 100 :],
    )


def download_dataset(
    path: str,
) -> Tuple[List[List[str]], Dict[int, List[str]]]:
    dataset = []
    lables_counter_dict = defaultdict(list)

    with open(path, "r") as f:
        for line in f:
            l = line.strip().split(",")

            for j, x in enumerate(l):
                if x not in lables_counter_dict[j]:
                    lables_counter_dict[j].append(x)

            dataset.append(l)

    return dataset, lables_counter_dict


def main():
    dataset: List[List[str]]
    lables_counter_dict: Dict[int, List[str]]
    dataset, lables_counter_dict = download_dataset("./data/car_evaluation.data")

    total_decision_label_counter: Dict[str, int] = get_decision_label_counter(dataset)

    # prior probs could be instead calculated from the train_set every loop
    prior_probs: Dict[str, float] = calc_prior_prob(
        total_decision_label_counter, len(dataset)
    )

    n_iterations = 10
    acc_table = [0.0 for _ in range(n_iterations)]
    print("Testing model...")
    for i in range(n_iterations):
        print(f"{i+1} iteration")
        rnd_shuffle(dataset)
        train_set, test_set = split_dataset(dataset)

        decision_label_counter: Dict[str, int] = get_decision_label_counter(train_set)

        acc = perform_prediction(
            train_set,
            test_set,
            lables_counter_dict,
            prior_probs,
            decision_label_counter,
            show=False,
        )
        assert acc != 0.0
        acc_table[i] = acc
        print("~")

    print(f"Average accuracy: {sum(acc_table)/len(acc_table)}")


if __name__ == "__main__":
    main()
