# encoding: utf-8

import pickle  as pkl

if __name__ == '__main__':
    with open('../Dataset/train.pkl', 'rb') as f:
        train_dataset = pkl.load(f)

    for aitem in train_dataset:
        print(aitem[0])
        print(aitem[1])
        print()