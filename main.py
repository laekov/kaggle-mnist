import torch
import torch.nn.functional as F
import torch.optim as optim

import argparse

from model import Net
from csv_reader import readCSV

total_epoch = 100
batch_size = 64
cut_count = 6

def validate(model, device):
    total, accurate = 0, 0
    for data, tgt in readCSV('div_valid.csv', batch_size):
        d, t = torch.tensor(data).float().to(device), torch.tensor(tgt).to(device)
        accurate += (model(d).argmax(dim = 1) == t).sum().item()
        total += t.shape[0]
    acc = float(accurate) / total
    print('Accuracy = %.8lf' % acc)
    return acc

def trainEpoch(model, optim, device):
    model.train()
    i = 0
    for data, tgt in readCSV('div_train.csv', batch_size):
        optim.zero_grad()
        d, t = torch.tensor(data).float().to(device), torch.tensor(tgt).to(device)
        result = model(d)
        loss = F.nll_loss(result, t)
        loss.backward()
        optim.step()
        i += 1
        if (i + 1) % 128 == 0:
            print('Training batch=%d loss=%.5lf' % (i, loss.item()))

def test(model, device):
    with open('result.csv', 'w') as f:
        f.write('ImageId,Label\n')
        base_id = 1
        i = 0
        for data, tgt in readCSV('test.csv', batch_size, is_test = True):
            d = torch.tensor(data).float().to(device)
            result = model(d).argmax(dim = 1).numpy().tolist()
            f.write(''.join([ '%d,%d\n' % (i + base_id, v) for i, v in enumerate(result) ]))
            base_id += len(result)
            i += 1
            if i % 128 == 0:
                print('Tested %d samples' % base_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', '--test', action = 'store_true', default = False)
    parser.add_argument('--device', type = str, default = 'cpu')
    parser.add_argument('--parameter', type = str, default = 'net.pt')
    parser.add_argument('--lr', type = float, default = 0.0002)
    config = parser.parse_args()
    device = torch.device(config.device)
    net = Net().to(device)
    try: 
        net.load_state_dict(torch.load(config.parameter))
    except:
        print('state dict loading failed')
    if config.test:
        test(net, device)
    else:
        optimizer = optim.Adam(net.parameters(), lr = config.lr)
        best_acc = validate(net, device)
        cnt_decend = 0
        for count_epoch in range(total_epoch):
            trainEpoch(net, optimizer, device)
            acc = validate(net, device)
            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), 'net.pt')
                cnt_decend = 0
            else:
                cnt_decend += 1
                if cnt_decend > cut_count:
                    break
    
if __name__ == '__main__':
    main()
