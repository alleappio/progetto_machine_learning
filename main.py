from ucimlrepo import fetch_ucirepo

# fetch dataset
superconductivty_data = fetch_ucirepo(id=464)

# data (as pandas dataframes)
#X = superconductivty_data.data.features
#y = superconductivty_data.data.targets
o = superconductivty_data.data.original

print(type(o))
o.to_csv("DataSet/superconductvty.csv")
# metadata
#print(superconductivty_data.metadata)

# variable information
#print(superconductivty_data.variables)

#print(X)
#print(y)