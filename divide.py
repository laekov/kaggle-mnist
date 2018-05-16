from random import randint


with open('train.csv', 'r') as f:
    total = 42000
    train_cnt = 2000
    tc = 0
    f.readline()
    f_train = open('div_train.csv', 'w')
    f_valid = open('div_valid.csv', 'w')
    while True:
        l = f.readline()
        if len(l) < 2:
            break
        if tc < train_cnt and randint(1, total) <= train_cnt:
            tc += 1
            f_valid.write(l)
        else:
            f_train.write(l)
    f_train.close()
    f_valid.close()
