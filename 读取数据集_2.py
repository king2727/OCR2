import glob, codecs, json, os
import numpy as np

date_jpgs = glob.glob('./训练集/date/images/*.jpg')
amount_jpgs = glob.glob('./训练集/amount/images/*.jpg')

lines = codecs.open('./训练集/date/gt.json', encoding='utf-8').readlines()
lines = ''.join(lines)
date_gt = json.loads(lines.replace(',\n}', '}'))

lines = codecs.open('./训练集/amount/gt.json', encoding='utf-8').readlines()
lines = ''.join(lines)
amount_gt = json.loads(lines.replace(',\n}', '}'))

data_path = date_jpgs + amount_jpgs
date_gt.update(amount_gt)

s = ''
for x in date_gt:
    s += date_gt[x]

char_list = list(set(list(s)))
char_list = char_list

with open('dataset/vocabulary.txt', 'w') as up:
    for x in char_list:
        up.write(x + '\n')

data_path = glob.glob('dataset/images/*.jpg')
np.random.shuffle(data_path)
with open('dataset/train_list.txt', 'w') as up:
    for x in data_path[:-100]:
        up.write(f'{x}\t{date_gt[os.path.basename(x)]}\n')

with open('dataset/test_list.txt', 'w') as up:
    for x in data_path[-100:]:
        up.write(f'{x}\t{date_gt[os.path.basename(x)]}\n')