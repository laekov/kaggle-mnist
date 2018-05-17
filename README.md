MNIST homework
===
## How to run this project
### Environment
This code depends on `PyTorch`. Current version of PyTorch requires conda installation.

### Training
Copy `train.csv` into this directory.

Run `python3 divide.py` to divide the data into training set and validation set.

Run `python3 main.py` to train the model. Generally, an epoch takes 1 hour on a single cpu.

### Test
Remove the first line of `test.csv`.

Run `python3 -test` to get the test result in `result.csv`. 
