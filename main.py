import argparse
import math
import os
import random
import time
import numpy as np
import torch
import pkuseg
from tqdm import trange
import torch.nn as nn
from sklearn.metrics import f1_score

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboard import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup
from model import TextCNN
from dataloader import Vocab
from utils import batch_iter, read_corpus



def set_seed(random_seed=3344):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

# 没有去除停止词
def tokenizer(text):
    seg = pkuseg.pkuseg()
    return [word for word in seg.cut(text) if word.strip()]

def train(args, model, train_data, dev_data, vocab, dtype='CNN'):
    LOG_FILE = args.output_file
    with open(LOG_FILE, 'a') as fout:
        fout.write('\n')
        fout.write('==========='*6)
        fout.write('start training: {}'.format(dtype))
        fout.write('==========='*6)
        fout.write('\n')

    time_start = time.time()
    # if not os.path.exists(os.path.join('runs', dtype)):
    #     os.makedirs(os.path.join('runs', dtype))
    # tb_writer = SummaryWriter(os.path.join('runs', dtype))

    t_total = args.num_epoch * math.ceil(len(train_data) / args.batch_size)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps,num_training_steps=t_total)  # 选择Warmup预热学习率的方式，可以使得开始训练的几个epoches或者一些steps内学习率较小,在预热的小学习率下，模型可以慢慢趋于稳定,等模型相对稳定后再选择预先设置的学习率进行训练,使得模型收敛速度变得更快，模型效果更佳

    global_step = 0
    total_loss = 0.
    logg_loss = 0.
    val_acces = []
    best_epoch = 0
    train_epoch = trange(args.num_epoch, desc='train_epoch')
    for epoch in train_epoch:
        model.train()

        for src_sents, labels in batch_iter(train_data, args.batch_size, shuffle=True):
            src_sents = vocab.vocab.to_input_tensor(src_sents, args.device)
            criterion = nn.CrossEntropyLoss()

            global_step += 1
            optimizer.zero_grad()

            logits = model(src_sents)
            y_labels = torch.tensor(labels, device=args.device)

            loss = criterion(logits, y_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if global_step % 100 == 0:
                loss_scalar = (total_loss - logg_loss) / 100
                logg_loss = total_loss

                with open(LOG_FILE, 'a') as fout:
                    fout.write('epoch: {}, iter: {}, loss: {}, learn_rate: {}\n'.format(epoch, global_step, loss_scalar, scheduler.get_lr()[0]))
                
                print('epoch: {}, iter: {}, loss: {}, learn_rate: {}\n'.format(epoch, global_step, loss_scalar, scheduler.get_lr()[0]))

                # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                # tb_writer.add_scalar('loss', loss_scalar, global_step)
        print('Epoch', epoch, 'Training loss', total_loss / global_step)


        eval_loss, eval_result, _ = evaluate(args, criterion, model, dev_data, vocab)  # 评估模型
        with open(LOG_FILE, 'a') as fout:
            fout.write('Evaluate: epoch: {}, loss: {}, eval_result: {}\n'.format(epoch, eval_loss, eval_result))

        eval_acc = eval_result['acc']
        if len(val_acces) == 0 or eval_acc > max(val_acces):
            best_epoch = epoch
            print('best model on epoch: {}, eval_acc: {}'.format(epoch, eval_acc))
            torch.save(model.state_dict(), 'classification-best-{}.th'.format(dtype))
            val_acces.append(eval_acc)

        time_end = time.time()
        print('run model of {}, taking total {} m'.format(dtype, (time_end - time_start)/60))
        with open(LOG_FILE, 'a') as fout:
            fout.write('run model of {}, taking total {} m\n'.format(dtype, (time_end - time_start)/60))
            fout.write('best model on epoch: {}, eval_acc: {}'.format(best_epoch, max(val_acces)))


def evaluate(args, criterion, model, dev_data, vocab):
    model.eval()
    total_loss = 0.
    total_step = 0.
    preds = None
    out_labels_ids = None

    with torch.no_grad():
        for src_sents, labels in batch_iter(dev_data, args.batch_size):
            src_sents = vocab.vocab.to_input_tensor(src_sents, args.device)
            logits = model(src_sents)
            labels = torch.tensor(labels, device=args.device)
            loss = criterion(logits, labels)

            total_loss = loss.item()
            total_step += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_labels_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_labels_ids = np.append(out_labels_ids, labels.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    result = acc_and_f1(preds, out_labels_ids)
    model.train()
    print('Evaluation loss', total_loss / total_step)
    print('Evaluation result', result)
    return total_loss / total_step, result, preds

def acc_and_f1(preds, labels):
    acc = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return {
        'acc': acc, 
        'f1': f1,
        'acc_and_f1': (acc + f1) / 2
    }

def build_vocab(args):
    if not os.path.exists(args.vocab_path):
        # src_sents, labels = read_corpus(args.train_data_dir)
        if os.path.exists('cnews/cahe_train_data'):
            train_data = torch.load('cnews/cahe_train_data')
            src_sents = [e[0] for e in train_data]
            labels = [e[1] for e in train_data]
            print(len(set(labels)))
        else:
            src_sents, labels = read_corpus(args.train_data_dir)
        labels = {labels: idx for idx, labels in enumerate(labels)}
        vocab = Vocab.build(src_sents, labels, args.max_vocab_size, args.min_freq)
    else:
        vocab = Vocab.load(args.vocab_path)
    return vocab

def main():
    parse = argparse.ArgumentParser()
    
    parse.add_argument('--train_data_dir', default='cnews/cnews.train.txt', type=str)
    parse.add_argument('--dev_data_dir', default='cnews/cnews.dev.txt', type=str)
    parse.add_argument('--test_data_dir', default='cnews/cnews.test.txt', type=str)
    parse.add_argument('--output_file', default='deep_model.log', type=str)
    parse.add_argument('--do_train', default=True, action='store_true')
    parse.add_argument('--do_test', default=True, action='store_true')

    parse.add_argument('--batch_size', default=32, type=int)
    parse.add_argument('--learning_rate', default=5e-4, type=float)
    parse.add_argument('--num_epoch', default=10, type=int)
    parse.add_argument('--max_vocab_size', default=50000, type=int)
    parse.add_argument('--min_freq', default=2, type=int)
    parse.add_argument('--emb_size', default=300, type=int)
    parse.add_argument('--hidden_size', default=256, type=int)
    parse.add_argument('--dropout_rate', default=0.1, type=float)
    parse.add_argument('--warmup_steps', default=0, type=int)
    parse.add_argument('--GRAD_CLIP', default=1, type=float)
    parse.add_argument('--num_filter', default=100, type=int)

    parse.add_argument('--vocab_path', default='vocab.json', type=str)

    args = parse.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(random_seed=1996)

    if os.path.exists('cnews/cahe_train_data'):
        train_data = torch.load('cnews/cahe_train_data')
        dev_data = torch.load('cnews/cahe_dev_data')
    else:
        train_data = read_corpus(args.train_data_dir)
        train_data = [(text, labs) for text, labs in zip(*train_data)]
        torch.save(train_data, 'cnews/cahe_train_data')

        dev_data = read_corpus(args.dev_data_dir)
        dev_data = [(text, labs) for text, labs in zip(*dev_data)]
        torch.save(dev_data, 'cnews/cahe_dev_data')


    vocab = build_vocab(args)
    label_map = vocab.labels
    print(label_map)
    cnn_model = TextCNN(args.num_filter, [2, 3, 4], len(vocab.vocab), args.emb_size, len(label_map),\
                        dropout=args.dropout_rate)
    cnn_model.to(device)

    if args.do_train:        
        train(args, cnn_model, train_data, dev_data, vocab, dtype='CNN')

    if args.do_test:
        if os.path.exists('cnews/cahe_test_data'):
            dev_data = torch.load('cnews/cahe_test_data')
        else:
            test_data = read_corpus(args.test_data_dir)
            test_data = [(text, labs) for text, labs in zip(*test_data)]
            torch.save(test_data, 'cnews/cahe_test_data')

        criterion = nn.CrossEntropyLoss()
        cnn_model.load_state_dict(torch.load('classification-best-CNN.th'))
        cnn_model.to(device)
        _, cnn_result, preds = evaluate(args, criterion, cnn_model, test_data, vocab)

        with open(args.output_file, 'a') as fout:
            fout.write('\n')
            fout.write('============== test result ==============\n')
            fout.write('test model of {}, result: {}\n'.format('CNN', cnn_result))

if __name__=='__main__':
    main()

