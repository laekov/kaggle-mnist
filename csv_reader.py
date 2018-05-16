import numpy as np

def readCSV(filename, batch_size = 1, is_test = False):
    with open(filename, 'r') as f:
        batch = [], []
        while True:
            raw_line = f.readline().split(',')
            if len(raw_line) <= 1:
                break
            label = 0
            if not is_test:
                label = int(raw_line[0])
                raw_line = raw_line[1:]
            data = np.array([ int(x) for x in raw_line ]).reshape((1, 28, 28))
            batch[0].append(data)
            batch[1].append(label)
            if len(batch[0]) == batch_size:
                yield batch
                batch = [], []
        if len(batch) > 0:
            yield batch
