from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

import utils
from prepare import PrepareData

# DataSet link: https://archive.ics.uci.edu/dataset/464/superconductivty+data
# DataSet doc: https://github.com/uci-ml-repo/ucimlrepo

def main():
    print("reading args")
    args = utils.read_args()
    print(args)
    
    if os.path.isfile(args.dataset_path):
        dataset = pd.read_csv(args.dataset_path)
    else:
        # fetch dataset
        print("fetch dataset")
        superconductivty_data = fetch_ucirepo(id=464)

        # data (as pandas dataframes)
        print("saving data in veriable")
        dataset = superconductivty_data.data.original
        print("saving data in file")
        dataset.to_csv("DataSet/superconductvty.csv")
    
    print(dataset)
    # Prepare the dataset
    new_data = PrepareData(dataset)
    train_data = new_data.get_train_data()
    validate_data = new_data.get_validate_data()
    test_data = new_data.get_test_data()
    print(f"len(train):{len(train_data)}; len(validate):{len(validate_data)}; len(test):{len(test_data)}; tot_samples:{len(dataset)}")

if __name__=='__main__':
    main()
