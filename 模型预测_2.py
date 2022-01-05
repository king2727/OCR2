import os
from PIL import Image
import numpy as np
import paddle

from utils.model import Model
from utils.data import process
from utils.decoder import ctc_greedy_decoder


with open('dataset/vocabulary.txt', 'r', encoding='utf-8') as f:
    vocabulary = f.readlines()

vocabulary = [v.replace('\n', '') for v in vocabulary]

save_model = 'models/'
model = Model(vocabulary, image_height=32)
model.set_state_dict(paddle.load(os.path.join(save_model, 'model.pdparams')))
model.eval()


def infer(path):
    data = process(path, img_height=32)
    data = data[np.newaxis, :]
    data = paddle.to_tensor(data, dtype='float32')
    # 执行识别
    out = model(data)
    out = paddle.transpose(out, perm=[1, 0, 2])
    out = paddle.nn.functional.softmax(out)[0]
    # 解码获取识别结果
    out_string = ctc_greedy_decoder(out, vocabulary)

    # print('预测结果：%s' % out_string)
    return out_string

if __name__ == '__main__':
    image_path = 'dataset/images/0_8bb194207a248698017a854d62c96104.jpg'
    display(Image.open(image_path))
    print(infer(image_path))

from tqdm import tqdm, tqdm_notebook

result_dict = {}
for path in tqdm(glob.glob('./测试集/date/images/*.jpg')):
    text = infer(path)
    result_dict[os.path.basename(path)] = {
        'result': text,
        'confidence': 0.9
    }

for path in tqdm(glob.glob('./测试集/amount/images/*.jpg')):
    text = infer(path)
    result_dict[os.path.basename(path)] = {
        'result': text,
        'confidence': 0.9
    }

with open('answer.json', 'w', encoding='utf-8') as up:
    json.dump(result_dict, up, ensure_ascii=False, indent=4)