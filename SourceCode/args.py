# encoding: utf-8

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--sou_wd_num', type=int, default=3000, help='sou_wd_num')
    parser.add_argument('--tag_wd_num', type=int, default=3000, help='tag_wd_num')
    parser.add_argument('--emd_size', type=int, default=150, help='emd_size')
    parser.add_argument('--hidden_size', type=int, default=150, help='hidden_size')
    parser.add_argument('--train', type=bool, default=False, help='train')

    parser.add_argument('--evaluate', type=bool, default=False, help='evaluete')
    parser.add_argument('--attention', type=bool, default=True, help='attention')
    parser.add_argument('--beam_search_num', type=int, default=-1, help='beam_search_num')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')

    args = parser.parse_args()
    return args