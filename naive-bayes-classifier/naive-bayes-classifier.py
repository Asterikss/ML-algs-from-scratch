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


def downlad_dataset(data_loc: pathlib.Path) -> tuple[list[list[float]], int]: # pure
    collected_data = []
    label_tabel = []
    

    with open(data_loc, "r", encoding="utf-8") as f:
        for line in f:
            splited: list[str] = line.split()
            decoted: list[float] = [eval(splited[i]) for i in range(len(splited) - 1)]

            label = splited[-1]
            if label not in label_tabel:
                label_tabel.append(label)

            decoted.append(label_tabel.index(label))
            
            collected_data.append(decoted)

    number_of_feature = len(collected_data[0]) - 1
    logging.info(f"Number of features: {number_of_feature}")

    return collected_data, number_of_feature



def init():
    level = logging.INFO
    # level = logging.DEBUG
    fmt = "%(levelname)s:%(lineno)d:%(funcName)s: %(message)s"
    logging.basicConfig(level = level, format = fmt) # filename = 'log_x.log', filemode = "w"


def main():
    data_loc: pathlib.Path = ask_for_data_loc()
    dataset, number_of_feature = downlad_dataset(data_loc)
    print(dataset)
    ...


if __name__ == "__main__":
    main()
