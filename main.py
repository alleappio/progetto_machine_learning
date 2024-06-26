from ucimlrepo import fetch_ucirepo
import utils

# DataSet link: https://archive.ics.uci.edu/dataset/464/superconductivty+data
# DataSet doc: https://github.com/uci-ml-repo/ucimlrepo

def main():
    args = utils.read_args()
    
    # fetch dataset
    superconductivty_data = fetch_ucirepo(id=464)

    # data (as pandas dataframes)
    o = superconductivty_data.data.original
    o.to_csv("DataSet/superconductvty.csv")

if __name__=='__main__':
    main()
