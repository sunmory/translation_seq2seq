# encoding: utf-8

import os
import numpy
import re
import random
import pickle as pkl
import tensorflow as tf
from pyhanlp import *
from tqdm import tqdm
from args import parse_args
from model import Seq2seq
from langconv import *


def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


args = parse_args()
en_dict_path = '../Dataset/en_dict.pkl'
ch_dict_path = '../Dataset/ch_dict.pkl'

with open(en_dict_path, 'rb') as f:
    en_words_dict = pkl.load(f)

with open(ch_dict_path, 'rb') as f:
    ch_words_dict = pkl.load(f)

args.sou_wd_num = len(en_words_dict.keys())
args.tag_wd_num = len(ch_words_dict.keys())

def strQ2B(ustring): # 全角转半角
    rstring = ''
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288: # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def split_words(sentence):
    seg = HanLP

    words = seg.segment(sentence)
    words = [i.word for i in words]
    return words


# def generate_dataset():
#     line_index = 0
#     dataset_list = []
#     with open('../Dataset/Bi-Education.txt', 'r', encoding='utf-8') as f:
#         aitem_list = []
#         for l in tqdm(f):
#             l = strQ2B(l.strip())
#             if line_index % 2 == 0:
#                 words = re.split(r"([\"\\()\[\],./%*、!@…?&'—^><“””~=+-:;_$ ])", l)
#                 words = [w for w in words if len(w) > 0 and len(w.strip()) > 0]
#
#                 aitem_list.append(words)
#             else:
#                 aitem_list.append(split_words(l))
#                 dataset_list.append(aitem_list)
#                 aitem_list = []
#             line_index = line_index + 1
#
#     random.shuffle(dataset_list)
#
#     with open('../Dataset/train.pkl', 'wb') as f:
#         pkl.dump(dataset_list[:int(len(dataset_list) * 0.9)], f)
#
#     with open('../Dataset/evaluate.pkl', 'wb') as f:
#         pkl.dump(dataset_list[int(len(dataset_list) * 0.9):], f)
#
#     print(len(dataset_list))


def generate_dataset():
    line_index = 0
    dataset_list = []
    with open('../Dataset/cmn.txt', 'r', encoding='utf-8') as f:
        aitem_list = []
        for l in tqdm(f):
            en, ch = l.strip().split('\t')
            en = strQ2B(en.strip())
            ch = strQ2B(ch.strip())
            ch = Traditional2Simplified(ch)
            en_words = re.split(r"([\"\\()\[\],./%*、!@…?&'—^><“””~=+-:;_$ ])", en)
            words = [w for w in en_words if len(w) > 0 and len(w.strip()) > 0]
            aitem_list.append(words)
            aitem_list.append(split_words(ch))
            dataset_list.append(aitem_list)
            aitem_list = []
            line_index = line_index + 1

    random.shuffle(dataset_list)

    with open('../Dataset/train.pkl', 'wb') as f:
        pkl.dump(dataset_list[:int(len(dataset_list) * 0.9)], f)

    with open('../Dataset/evaluate.pkl', 'wb') as f:
        pkl.dump(dataset_list[int(len(dataset_list) * 0.9):], f)

    print(len(dataset_list))


def generate_word_dict(dataset, mode='en', min_count=3):
    word_index = 4
    word_dict = dict()
    word_num_dict = dict()
    word_dict['_GO'] = 0
    word_dict['_EOS'] = 1
    word_dict['PAD'] = 2
    word_dict['UNK'] = 3

    for aitem in dataset:
        if mode == 'en':
            word_list = aitem[0]
        else:
            word_list = aitem[1]

        for word in word_list:
            if word not in word_num_dict.keys():
                word_num_dict[word] = 1
            else:
                word_num_dict[word] = word_num_dict[word] + 1

    for word in word_num_dict.keys():
        if word_num_dict[word] >= min_count:
            word_dict[word] = word_index
            word_index = word_index + 1

    if mode == 'en':
        data_path = '../Dataset/en_dict.pkl'
    else:
        data_path = '../Dataset/ch_dict.pkl'

    with open(data_path, 'wb') as f:
        pkl.dump(word_dict, f)


def generate_id2word_dict(mode='en'):
    if mode == 'en':
        word_id_dict = en_words_dict
        id_word_path = '../Dataset/en_id2word_dict.pkl'
    else:
        word_id_dict = ch_words_dict
        id_word_path = '../Dataset/ch_id2word_dict.pkl'

    id_word_dict = dict()

    for key in word_id_dict.keys():
        id_word_dict[word_id_dict[key]] = key

    with open(id_word_path, 'wb') as f:
        pkl.dump(id_word_dict, f)


def load_dataset(data_path):
    with open(data_path, 'rb') as f:
        dataset = pkl.load(f)
    return dataset


def pad_sentences(sentences_list, mode='sou'):
    if mode == 'tag':
        for sentence in sentences_list:
            sentence.append('_EOS')

    length_list = [len(sen) for sen in sentences_list]
    max_length = max(length_list)
    new_sentence_list = []

    for sentence in sentences_list:
        sentence.extend(['PAD'] * (max_length - len(sentence)))
        # print(sentence)
        new_sentence_list.append(sentence)
    return new_sentence_list, length_list


def change_ids(sentence_list, mode='sou'):
    if mode == 'sou':
        words_dict = en_words_dict
    else:
        words_dict = ch_words_dict

    sentence_ids = []
    for sentence in sentence_list:
        ids = []
        for w in sentence:
            if w in words_dict.keys():
                ids.append(words_dict[w])
            else:
                ids.append(words_dict['UNK'])
        sentence_ids.append(ids)

    return sentence_ids


def batch_genetator(mode='train'):
    if mode == 'train':
        data_path = '../Dataset/train.pkl'
    else:
        data_path = '../Dataset/evaluate.pkl'

    dataset = load_dataset(data_path)
    batch_source_dataset, batch_target_dataset = [], []

    for aitem in tqdm(dataset):
        batch_source_dataset.append(aitem[0])
        batch_target_dataset.append(aitem[1])

        if len(batch_source_dataset) == args.batch_size:
            # print(batch_source_dataset)
            #
            # print(batch_target_dataset)
            # for sentence in batch_target_dataset:
            #     print(sentence)

            sou_pad_sentences, sou_length_list = pad_sentences(batch_source_dataset, mode='sou')

            tag_pad_sentences, tag_length_list = pad_sentences(batch_target_dataset, mode='tag')

            sou_pad_sentences = change_ids(sou_pad_sentences, mode='sou')
            tag_pad_sentences = change_ids(tag_pad_sentences, mode='tag')

            yield (sou_pad_sentences, sou_length_list, tag_pad_sentences, tag_length_list)

            batch_source_dataset, batch_target_dataset = [], []


def train(epoch_num):
    args.train = True
    model = Seq2seq(args, ch_words_dict)
    batch_index, min_loss = 0, 100
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch_num):
            train_gegnator = batch_genetator(mode='train')
            try:
                while True:
                    sou_sentences_list, sou_length_list, tag_sentences_list, tag_length_list = next(train_gegnator)
                    print('sentence length{}'.format(len(sou_sentences_list[0])))

                    if len(sou_sentences_list[0]) > 90:
                        continue
                    feed_dict = {model.sequence_input: sou_sentences_list,
                                 model.sequence_length: sou_length_list,
                                 model.target_input: tag_sentences_list,
                                 model.target_length: tag_length_list}

                    loss, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict)
                    print('epoch: {}, batch index: {}, loss: {}, current min loss: {}'.format(epoch, batch_index, loss, min_loss))
                    if loss < min_loss:
                        min_loss = loss
                        print('save at epoch: {}, batch {} the loss is {}'.format(epoch, batch_index, min_loss))
                        saver.save(sess, '../model/model.ckpt')

                    batch_index = batch_index + 1
            except StopIteration as e:
                print('finish training')


def evaluate():
    args.beam_search_num = -1
    en_id2word_path = '../Dataset/en_id2word_dict.pkl'
    ch_id2word_path = '../Dataset/ch_id2word_dict.pkl'

    with open(en_id2word_path, 'rb') as f:
        en_id2word_dict = pkl.load(f)

    with open(ch_id2word_path, 'rb') as f:
        ch_id2word_dict = pkl.load(f)

    model = Seq2seq(args, ch_words_dict)
    evaluate_generator = batch_genetator(mode='eva')
    batch_index = 0

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, '../model/model.ckpt')
        try:
            while True:
                sou_sentences_list, sou_length_list, tag_sentences_list, tag_length_list = next(evaluate_generator)
                if len(sou_sentences_list[0]) > 90:
                    continue
                feed_dict = {model.sequence_input: sou_sentences_list,
                             model.sequence_length: sou_length_list,
                             model.target_input: tag_sentences_list,
                             model.target_length: tag_length_list}
                predict_ids = sess.run(model.out, feed_dict=feed_dict)
                for sentence_index in range(len(sou_sentences_list)):
                    sou_sentence = [en_id2word_dict[i] for i in sou_sentences_list[sentence_index]]

                    predict_sentence = [ch_id2word_dict[i] for i in predict_ids[sentence_index]]
                    tag_sentence = [ch_id2word_dict[i] for i in tag_sentences_list[sentence_index]]
                    print('sou_sentence: {}'.format(sou_sentence))
                    print('predict_sentence: {}'.format(predict_sentence))
                    print('tag_sentence: {}'.format(tag_sentence))
                batch_index = batch_index + 1
        except StopIteration as e:
            print('finish training')


if __name__ == '__main__':
    # generate_dataset()
    # train_dataset_path = '../Dataset/train.pkl'
    # evaluate_dataset_Path = '../Dataset/evaluate.pkl'
    #
    # train_dataset = load_dataset(train_dataset_path)
    # generate_word_dict(train_dataset, mode='en')
    #
    # generate_word_dict(train_dataset, mode='ch')

    # generate_id2word_dict(mode='en')
    # generate_id2word_dict(mode='ch')
    #
    #
    # en_dict_path = '../Dataset/en_dict.pkl'
    # with open(en_dict_path, 'rb') as f:
    #     en_dict = pkl.load(f)
    #
    # print(len(en_dict.keys()))
    #
    # for key, value in en_dict.items():
    #     print(key, value)










    # train(150)

    evaluate()