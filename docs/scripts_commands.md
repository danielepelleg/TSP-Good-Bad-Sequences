- [Create Graphs](#create-graphs)
- [Create Dataset](#create-dataset)
- [Test Algorithms](#test-algorithms)
- [Greedy Algorithm](#greedy-algorithm)

# Create Graphs
```bash
python create_graphs.py [--t TYPE] [--s SIZE] [--g N_GRAPHS]
```
- `--t TYPE`    Type of Graphs to create ( Dataset | Test ). Default is *dataset*.
- `--s SIZE`    Step Size to add on every graph. Default is *5*.
- `--g N_GRAPHS`  Number of graphs to create. Default is *500*.

# Create Dataset
```bash
python create_dataset.py [--g N_GRAPHS] [--r N_RECORDS] [--s SEQUENCE_NODES] [--e ERROR_TARGET]
```
- `--g N_GRAPHS`        Number of graphs to use. Default is *500*.
- `--r N_RECORDS`       Number of Records to Save for each Graph. Default is *56*
- `--s SEQUENCE_NODES`  Nodes in each sequence. Default is *7*.
- `--e ERROR_TARGET`    Error target on the tour. Default is *4.0*.

# Test Algorithms
```bash
test_algorithms.py [--r N_RECORDS] [--s SEQUENCE_NODES] [--e ERROR_TARGET]
```
- `--r N_RECORDS`       Number of Records to Save for each Graph. Default is *56*
- `--s SEQUENCE_NODES`  Nodes in each sequence. Default is *7*.
- `--e ERROR_TARGET`    Error target on the tour. Default is *4.0*.

# Greedy Algorithm
```bash
python greedy_algorithm.py [--p PROBLEM] [--s SEQUENCE] [--i STARTING_NODE] [--S SEED] [--file FILE]
```
- `--p PROBLEM `       Number of the problem to use. Default is *1*.
- `--s SEQUENCE`       Number of the sequence to use to solve the problem. Default is *7*.
- `--i STARTING_NODE`  Number of the starting node of the solution. Default is *1*.
- `--S SEED`           Seed used for the choice of a random number. Default is *None*.
- `--file FILE `       Name of the file to use. By default a problem is used. File specified must be in *./TSPGraph-Test/Problems* folder.