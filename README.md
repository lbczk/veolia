# veolia

Files related to the Veolia 2016-2017 challenge data project. The goal was to predict the
failures of pipes accross a network.

utils.py:
contains some simple tools to manipulate panda dataframes.

models.py:
contains a few models (implemented in sci-kit learn)

main.py:
benchmarks some simple linear models.

the data files should be placed in a folder called `data' placed in the working directory.
they should be renamed train.csv, test.csv, and output_train.csv

python main.py
