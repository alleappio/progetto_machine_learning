from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

import utils

# DataSet link: https://archive.ics.uci.edu/dataset/464/superconductivty+data
# DataSet doc: https://github.com/uci-ml-repo/ucimlrepo

def main():
    print("reading args")
    args = utils.read_args()
    print(args)
    
    if os.path.isfile(args.dataset_path):
        o = pd.read_csv(args.dataset_path)
    else:
        # fetch dataset
        print("fetch dataset")
        superconductivty_data = fetch_ucirepo(id=464)

        # data (as pandas dataframes)
        print("saving data in veriable")
        o = superconductivty_data.data.original
        print("saving data in file")
        o.to_csv("DataSet/superconductvty.csv")
    
    print(o)
if __name__=='__main__':
    main()
