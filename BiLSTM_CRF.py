"""
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-05-27 09:20:17
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-05-31 09:24:57
FilePath: /nlp_hw_3/BiLSTM_CRF.py
Description: coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn
"""

import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm


def load_vocab():
    """
    加载词典
    @return word2id: 词典
    @return tag2id: 标签字典
    """
    with open('./data/train.txt', 'r', encoding='utf-8') as f:  # 读取训练数据集
        train_content = f.read().split()  # 以空格分割，得到词语列表

    with open('./data/train_TAG.txt', 'r', encoding='utf-8') as f:  # 读取训练标签集
        train_tag = f.read().split()  # 以空格分割，得到标签列表

    print("训练数据集大小：", len(train_content))
    print("训练标签集大小：", len(train_tag))

    word2id_ = {word: i for i, word in enumerate(set(train_content))}  # 构建词典

    tag2id_ = {tag: i for i, tag in enumerate(set(train_tag))}  # 构建标签字典

    print("vocab大小:", len(word2id_))
    print("tag2id:", tag2id_)

    # word2id["<PAD>"] = len(word2id) # 填充
    # word2id["<UNK>"] = len(word2id) + 1  # 未知

    return word2id_, tag2id_


def dispose_train_data(text_file_, tag_file_):
    """
    处理训练数据
    @param text_file_: 训练数据集路径
    @param tag_file_: 训练标签集路径
    """
    with open(text_file_, 'r',
              encoding='utf-8') as f_text, open(tag_file_,
                                                'r',
                                                encoding='utf-8') as f_tag:
        text_lines = f_text.readlines()  # 读取文本数据
        tag_lines = f_tag.readlines()  # 读取标签数据

    assert len(text_lines) == len(tag_lines)  # 确保长度相等

    # # 寻找最长的句子
    # max_length = 0
    # for text_line in text_lines:
    #     text_line = text_line.strip().split(' ')
    #     if len(text_line) > max_length:
    #         max_length = len(text_line)
    # print("最长的句子长度：",max_length)

    # 处理数据
    data = []

    # data_mask = []
    for i in range(len(text_lines)):  # 一行一行处理
        text_line = text_lines[i].strip().split(' ')  # 以空格分割
        tag_line = tag_lines[i].strip().split(' ')  # 以空格分割

        assert len(text_line) == len(tag_line)  # 确保长度相等

        # assert len(text_line) == len(tag_line) # 确保长度相等

        # text_indices = [vocab[word] if word in vocab  # 在词典中，转换为对应的数字
        #                 else vocab["<UNK>"] for word in text_line] # 未知
        # tag_indices = [tag_to_ix[tag] if tag in tag_to_ix  # 在词典中，转换为数字
        #                 else tag_to_ix["O"] for tag in tag_line] # O

        # mask = [1 if word in vocab else 0 for word in text_line] # 未知的词语，mask为0

        # # 填充
        # text_indices.extend([vocab["<PAD>"]] * (max_length - len(text_indices)))
        # tag_indices.extend([tag_to_ix["O"]] * (max_length - len(tag_indices)))
        # mask.extend([0] * (max_length - len(mask)))

        # assert len(text_indices) == len(tag_indices) == len(mask) == max_length # 确保长度相等
        data.append((text_line, tag_line))  # 添加到数据集中
        # data_mask.append(mask)
    return data


torch.manual_seed(1)  # 设置随机种子


def argmax(vec):
    # 以int形式返回argmax
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    # 将句子转换为数字序列

    # idxs = [to_ix[w] for w in seq] # 转换为数字
    idxs = []
    for w in seq:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(to_ix["O"])
    return torch.tensor(idxs, dtype=torch.long)  # 转换为tensor


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # 词向量的维度
        self.hidden_dim = hidden_dim  # 隐藏层的维度
        self.vocab_size = vocab_size  # 词汇表的大小
        self.tag2id = tag2id  # 标签到ID的映射
        self.tagset_size = len(tag2id)  # 标签集合的大小

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)  # 词嵌入矩阵
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True)  # 双向LSTM

        # 将LSTM的输出映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)  # 线性层

        # 转移参数矩阵。入口i,j表示从j转移到i的参数。
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 这两句强制性的约束条件，确保我们从不进行到开始标记的转移，也从不从结束标记进行转移。
        self.transitions.data[tag2id[START_TAG], :] = -10000
        self.transitions.data[:, tag2id[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        初始化LSTM的隐藏状态
        @return: 隐藏状态
        """
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))  # 初始化隐藏状态

    def _forward_alg(self, feats):
        """
        前向算法来计算分区函数
        @param feats: 一组LSTM的输出特征
        @return: 分区函数的值
        """
        # 执行前向算法以计算分区函数
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG拥有所有的分数
        init_alphas[0][self.tag2id[START_TAG]] = 0.

        # 将其包裹在变量中，这样我们就可以得到自动的反向传播
        forward_var = init_alphas

        # 遍历整个句子
        for feat in feats:
            alphas_t = []  # 这个时间步的前向张量
            for next_tag in range(self.tagset_size):
                # 广播发射分数：无论以前的标签是什么，它都是相同的
                emit_score = feat[next_tag].view(1, -1).expand(
                    1, self.tagset_size)
                # trans_score的第i项是从i转移到next_tag的分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # next_tag_var的第i项是在我们执行对数-求和-指数之前的边(i -> next_tag)的值
                next_tag_var = forward_var + trans_score + emit_score
                # 这个标签的前向变量是所有分数的对数-求和-指数
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag2id[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        """
        得到LSTM的输出
        @param sentence: 一个词索引列表
        @return: LSTM的输出
        """
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        """
        给出所提供的标签序列的分数
        @param feats: 一个包含每个标签的发射分数的矩阵
        @param tags: 标签序列
        @return: 标签序列的分数
        """
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag2id[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag2id[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        """
        维特比算法来解码
        @param feats: 一个包含每个标签的发射分数的矩阵
        @return: 最好的路径的标签
        """
        backpointers = []

        # 在对数空间初始化viterbi变量
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag2id[START_TAG]] = 0

        # 第i步的forward_var保存了第i-1步的viterbi变量
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # 保存这一步的backpointers
            viterbivars_t = []  # 保存这一步的viterbi变量

            for next_tag in range(self.tagset_size):
                # next_tag_var[i]保存了前一步标签i的viterbi变量，以及从标签i转移到next_tag的分数
                # 我们这里不包括发射分数，因为最大值并不依赖它们（我们在下面加上）
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 现在加入发射分数，并将forward_var赋值为我们刚刚计算的一组viterbi变量
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 转移到STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2id[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 按照back pointers解码最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出开始标签（我们不想返回这个给调用者）
        start = best_path.pop()
        assert start == self.tag2id[START_TAG]  # 检查一下
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        """
        计算负对数似然损失
        @param sentence: 一个词索引的列表
        @param tags: 一个标签索引的列表
        @return: 负对数似然损失
        """
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # 不要和上面的 _forward_alg混淆。
        """
        返回最佳路径的得分以及最佳路径的标签
        @param sentence: 一个词索引的列表
        @return: 得分，路径
        """
        # 获取BiLSTM的发射分数
        lstm_feats = self._get_lstm_features(sentence)

        # 根据特征找出最佳路径。
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


def dispose_dev_data():
    print("处理dev集...")
    """
    处理验证数据
    """
    text_file = './data/dev.txt'  # 验证数据
    tag_file = './data/dev_TAG.txt'  # 验证数据对应的标签
    dev_data = dispose_train_data(text_file, tag_file)  # 处理验证数据

    # print("dev_data示例:",dev_data[150])
    return dev_data


def predict_dev_data(dev_data, epoch):
    """
    预测验证数据
    @param dev_data: 验证数据
    @param epoch: 当前轮数
    """
    print("开始验证...")
    dev_predicts = []  # 保存预测的标签
    dev_true_tags = []  # 保存真实的标签
    IS_DEV = False
    if IS_DEV:
        for piece in dev_data:  # 逐个句子进行预测
            with torch.no_grad():
                precheck_sent = prepare_sequence(piece[0], word2id)  # 将句子转换成id
                dev_predicts.append(model(precheck_sent))  # 预测
            dev_tags_piece = []  # 保存每个句子的标签
            for each in piece[1]:
                dev_tags_piece.append(tag2id[each])
            dev_tags_piece = torch.tensor(dev_tags_piece,
                                            dtype=torch.long)  # 将标签转换成id
            dev_true_tags.append(dev_tags_piece)

            # 提取出标签
        dev_predicts_tag = []  # 保存预测的标签
        for i in range(len(dev_predicts)):  # 逐个句子进行预测
            dev_predicts_tag.append(
                torch.tensor(dev_predicts[i][1], dtype=torch.long))  # 提取出标签

        with open(  f'./{EPOCH}_epoch_model/dev_true_tags.pkl',
                    'wb') as f:  # 保存真实的标签
            pickle.dump(dev_true_tags, f)
        with open(f'./{EPOCH}_epoch_model/dev_predicts_tag_{epoch}.pkl',
                    'wb') as f:  # 保存预测的标签
            pickle.dump(dev_predicts_tag, f)

    else:
        with open(f'./{EPOCH}_epoch_model/dev_predicts_tag_{epoch}.pkl',
                    'rb') as f:  # 读取预测的标签
            dev_predicts_tag = pickle.load(f)
        with open(f'./{EPOCH}_epoch_model/dev_true_tags.pkl',
                    'rb') as f:  # 读取真实的标签
            dev_true_tags = pickle.load(f)

    # print("dev_predicts_tag示例:",dev_predicts_tag[150])
    # print("dev_tags示例:",dev_true_tags[150])

    assert len(dev_predicts_tag) == len(dev_true_tags)

    return dev_predicts_tag, dev_true_tags


def evaluate_model(dev_predicts_tag, dev_true_tags):
    print("评估模型...")
    # 评估模型
    for i in range(len(dev_predicts_tag)):
        assert len(dev_predicts_tag[i]) == len(dev_true_tags[i])

    # 首先我们需要将这些结果扁平化：
    true_tags_flat = torch.cat(dev_true_tags).numpy()
    predicted_tags_flat = torch.cat(dev_predicts_tag).numpy()

    # 移除标签0
    mask = true_tags_flat != tag2id['O']
    true_tags_flat = true_tags_flat[mask]
    predicted_tags_flat = predicted_tags_flat[mask]

    # 然后我们就可以计算评估指标了：
    precision = precision_score(true_tags_flat,
                                predicted_tags_flat,
                                average='macro')
    recall = recall_score(true_tags_flat, predicted_tags_flat, average='macro')
    f1 = f1_score(true_tags_flat, predicted_tags_flat, average='macro')

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    global y
    y[0].append(precision)
    y[1].append(recall)
    y[2].append(f1)

    # 计算混淆矩阵
    cm = confusion_matrix(true_tags_flat, predicted_tags_flat)

    # 使用 seaborn 画混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # plt.show()

    plt.savefig(f'./{EPOCH}_epoch_model/confusion_matrix_{e}.png')


START_TAG = "<START>"  # 开始标签
STOP_TAG = "<STOP>"  # 结束标签
EMBEDDING_DIM = 50  # 词向量维度
HIDDEN_DIM = 60  # LSTM隐藏层维度
EPOCH = 9  # 训练轮数

y = [[], [], []]
IS_TRAIN = False
# training_data = training_data[:1000]
if IS_TRAIN:

    print("****************准备数据****************")
    word2id, tag2id = load_vocab()  # 加载字典
    tag2id[START_TAG] = len(tag2id)  # 增加开始标签
    tag2id[STOP_TAG] = len(tag2id)  # 增加结束标签
    text_file = './data/train.txt'  # 训练数据
    tag_file = './data/train_TAG.txt'  # 训练数据对应的标签
    training_data = dispose_train_data(text_file, tag_file)  # 处理训练数据

    print("data示例:", training_data[150])
    print(len(training_data[150][0]), len(training_data[150][1]))

    model = BiLSTM_CRF(len(word2id), tag2id, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # 训练前检查预测结果
    # with torch.no_grad():
    #     precheck_sent = prepare_sequence(training_data[150][0], word2id)
    #     precheck_tags = torch.tensor([tag2id[t] for t in training_data[0][1]], dtype=torch.long)
    # # print("训练前：", model(precheck_sent))

    with open(f"./{EPOCH}_epoch_model/word2id.pkl", "wb") as f:
        pickle.dump(word2id, f)
    with open(f"./{EPOCH}_epoch_model/tag2id.pkl", "wb") as f:
        pickle.dump(tag2id, f)

    print("****************开始训练****************")
    for epoch in range(EPOCH):
        print("===========================================")
        print("epoch:", epoch)
        # 随机选取10000条数据
        training_data_epoch = random.sample(training_data, 10000)

        for sentence, tags in training_data_epoch:
            # Step 1. 梯度清零，否则会累加
            model.zero_grad()

            # Step 2. 准备网络的输入，将其转化为单词索引的张量
            sentence_in = prepare_sequence(sentence, word2id)
            targets = torch.tensor([tag2id[t] for t in tags], dtype=torch.long)

            # Step 3. 计算损失
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. 计算梯度，更新参数
            loss.backward()
            optimizer.step()

        # 每训练一个epoch，保存一次模型
        with open(f'./{EPOCH}_epoch_model/model_{epoch}_epoch.pkl', 'wb') as f:
            torch.save(model, f)
        dev_data = dispose_dev_data()
        dev_predicts_tag, dev_true_tags = predict_dev_data(dev_data, epoch)
        evaluate_model(dev_predicts_tag, dev_true_tags)
        print("loss:", loss)
        print("===========================================")

else:
    with open(f"./{EPOCH}_epoch_model/word2id.pkl", "rb") as f:
        word2id = pickle.load(f)
    with open(f"./{EPOCH}_epoch_model/tag2id.pkl", "rb") as f:
        tag2id = pickle.load(f)
    model = BiLSTM_CRF(len(word2id), tag2id, EMBEDDING_DIM, HIDDEN_DIM)

IS_EVALUATE = False
if IS_EVALUATE:
    dev_data = dispose_dev_data()
    for e in range(EPOCH):
        with open(f'./{EPOCH}_epoch_model/model_{e}_epoch.pkl', 'rb') as f:
            model = torch.load(f)
        dev_predicts_tag, dev_true_tags = predict_dev_data(dev_data, e)
        evaluate_model(dev_predicts_tag, dev_true_tags)

    plt.figure()
    plt.title('evaluate')
    plt.xlabel('epoch')
    plt.ylabel('value')
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.plot(x, y[0], color='red', label='precision')
    plt.plot(x, y[1], color='blue', label='recall')
    plt.plot(x, y[2], color='green', label='f1-score')

    plt.savefig('./evaluate.png')

    with open(f"./{EPOCH}_epoch_model/tag2id.txt", "w") as f:
        f.write(str(tag2id))


def output(model_num):
    """
    处理测试数据
    """
    test_data = []
    with open('./data/test.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            if line:
                test_data.append(line)

    with open(f'./{EPOCH}_epoch_model/model_{model_num}_epoch.pkl', 'rb') as f:
        model = torch.load(f)

    with open(f"./{EPOCH}_epoch_model/word2id.pkl", "rb") as f:
        word2id = pickle.load(f)

    with open(f"./{EPOCH}_epoch_model/tag2id.pkl", "rb") as f:
        tag2id = pickle.load(f)

    print("test_data示例:", test_data[150])
    test_predicts = []
    for piece in tqdm(test_data):  # 逐个句子进行预测
        with torch.no_grad():
            precheck_sent = prepare_sequence(piece, word2id)  # 将句子转换成id
            test_predicts.append(model(precheck_sent))  # 预测

    print("test_predicts示例:", test_predicts[150])

    # 提取出标签
    test_predicts_tag = []
    for i in range(len(test_predicts)):
        test_predicts_tag.append(test_predicts[i][1])

    print("test_predicts_tag示例:", test_predicts_tag[150])

    id2tag = {v: k for k, v in tag2id.items()}
    # 将预测的标签转换成标签名
    test_predicts_tag = [[id2tag[i] for i in piece]
                         for piece in test_predicts_tag]

    with open('2020212185.txt', 'w', encoding='utf-8') as f:
        for i in range(len(test_predicts_tag)):
            for j in range(len(test_predicts_tag[i])):
                f.write(test_predicts_tag[i][j] + ' ')
            f.write('\n')


output(7)

# dev_predicts = dev_predicts.tolist()

# with open('./10_epoch_model/dev_predicts.txt', 'w') as f:
#     for i in dev_predicts:
#         f.write(str(i))
#         f.write('\n')
"""
之前的旧版实现
"""

# import torch
# import torch.nn as nn
# from torch.nn import init
# from TorchCRF import CRF

# class BiLSTM_CRF(nn.Module):

#     def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim):
#         super(BiLSTM_CRF, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.vocab_size = vocab_size
#         self.tag2id = tag2id
#         self.tagset_size = len(tag2id)

#         self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
#         # 初始化词向量矩阵，这里可以使用预训练的词向量
#         init.uniform_(self.word_embeds.weight)

#         self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
#                             num_layers=1, bidirectional=True)

#         # Maps the output of the LSTM into tag space.
#         self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

#         # CRF Layer
#         self.crf = CRF(self.tagset_size)

#     def forward(self, sentence):
#         embeds = self.word_embeds(sentence)
#         lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
#         lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
#         lstm_feats = self.hidden2tag(lstm_out)

#         return lstm_feats

#     def loss(self, sentence, tags):
#         feats = self.forward(sentence)
#         loss_value = -self.crf(feats, tags)
#         return loss_value

#     def predict(self, sentence):
#         lstm_feats = self.forward(sentence)
#         best_path = self.crf.decode(lstm_feats)
#         return best_path
