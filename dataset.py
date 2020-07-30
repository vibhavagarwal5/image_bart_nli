import json
import logging
import pickle
import os
from itertools import chain

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset

LABEL_TOKENS_DICT = {
    'contradiction': 0,
    'neutral': 1,
    'entailment': 2
}


def get_data(data_path, data_type, no_image=False):
    # files = ['expl_1', 'labels', 's2']
    # data = {
    #     f: [line.rstrip() for line in
    #         open(os.path.join(data_path, f"{f}.{data_type}"), 'r')] for f in files
    # }
    data = {}
    data['expl'] = [line.rstrip() for line in open(
        os.path.join(data_path, f"expl_1.{data_type}"), 'r')]
    data['label'] = [line.rstrip() for line in open(
        os.path.join(data_path, f"labels.{data_type}"), 'r')]
    data['label_int'] = [
        LABEL_TOKENS_DICT[i] for i in data['label']]
    data['hypothesis'] = [line.rstrip() for line in open(
        os.path.join(data_path, f"s2.{data_type}"), 'r')]
    if no_image:
        data['premise'] = [line.rstrip() for line in open(
            os.path.join(data_path, f"s1.{data_type}"), 'r')]
    else:
        data['image_f'] = [line.rstrip() for line in open(
            os.path.join(data_path, f"images.{data_type}"), 'r')]
    return data


class InferenceDataset(Dataset):
    def __init__(self, data, tokenizer, no_image=False, no_premise=True, with_expl=True):
        self.data = data
        self.tokenizer = tokenizer
        self.no_image = no_image
        self.no_premise = no_premise
        self.with_expl = with_expl
        if not no_image:
            self.all_images_np = np.load(
                '/home/hdd1/vibhav/VE-SNLI/e-SNLI-VE/data/flickr30k_resnet101_bottom_up_img_features.npy')
            f = open(
                '/home/hdd1/vibhav/VE-SNLI/e-SNLI-VE/data/filenames_77512.json', 'r')
            self.all_image_names = json.load(f)

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, index):
        if self.no_image and not self.no_premise:
            input_seq = build_input_seq((self.data['premise'][index],
                                         self.data['hypothesis'][index]),
                                        self.tokenizer,
                                        no_premise=self.no_premise)
        else:
            input_seq = build_input_seq(self.data['hypothesis'][index],
                                        self.tokenizer, no_premise=self.no_premise)
        input_ids = torch.tensor(input_seq).long()
        label = torch.tensor(self.data['label_int'][index]).long()
        output = (input_ids, label)

        if not self.no_image:
            image = self.all_images_np[self.all_image_names.index(
                self.data['image_f'][index])]
            output = output + (image,)
        if self.with_expl:
            expl_ids = self.tokenizer(self.data['expl'][index])['input_ids']
            expl_ids = torch.tensor(expl_ids).long()
            output = output + (expl_ids,)
        return output   # input_ids, label, image, expl_ids


def build_input_seq(inp, tokenizer, no_premise=False):
    if no_premise:
        hypothesis = inp
        return tokenizer(hypothesis)['input_ids']
    else:
        premise, hypothesis = inp
        return tokenizer(premise, hypothesis)['input_ids']


def collate_fn(batch, pad_token, no_image=False, with_expl=True):
    def padding(seq, max_len, pad_token):
        padded_mask = torch.ones((len(seq), max_len)).long() * pad_token
        for i in range(len(seq)):
            padded_mask[i, :len(seq[i])] = seq[i]
        return padded_mask

    input_ids, label = [], []
    if not no_image:
        image = []
    if with_expl:
        expl_ids = []
    for i in batch:
        input_ids.append(i[0])
        label.append(i[1])
        if not no_image:
            image.append(i[2])
            if with_expl:
                expl_ids.append(i[3])
        else:
            if with_expl:
                expl_ids.append(i[2])

    if with_expl:
        max_len_inp_ids = max(len(s) for s in input_ids)
        max_len_expl_ids = max(len(s) for s in expl_ids)
        max_len = max(max_len_inp_ids, max_len_expl_ids)
        input_ids = padding(input_ids, max_len, pad_token)
        expl_ids = padding(expl_ids, max_len, pad_token)
        label = torch.tensor(label).long()
        output = (input_ids, label, expl_ids)
    else:
        max_len_inp_ids = max(len(s) for s in input_ids)
        input_ids = padding(input_ids, max_len_inp_ids, pad_token)
        label = torch.tensor(label).long()
        output = (input_ids, label)
    if not no_image:
        image = torch.tensor(image)
        input_mask = input_ids.ne(pad_token).long()
        image_mask = torch.ones((len(image), 36)).long()
        input_mask = torch.cat([image_mask, input_mask], dim=1)
        output = (image,) + output + (input_mask,)
    else:
        input_mask = input_ids.ne(pad_token).long()
        output = output + (input_mask,)

    return output   # image, input_ids, label, expl_ids, input_mask


'''main'''
if __name__ == "__main__":
    from transformers import *
    from torch.utils.data import DataLoader
    import itertools
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str,
                        default="dev", help="dev or train or test")
    parser.add_argument("--data_path", type=str,
                        default="/home/hdd1/vibhav/VE-SNLI/mycode-vesnli/dataset/e-SNLI-VE", help="Path of the dataset")
    parser.add_argument("--no_image", action="store_true",
                        help="To process image or not")
    parser.add_argument("--no_premise", action="store_true",
                        help="To process premise or not")
    parser.add_argument("--with_expl", action="store_true",
                        help="To use explanations or not")
    parser.add_argument("--to_save", action="store_true",
                        help="To save the dataset processed or not")
    parser.add_argument("--final_data_path", type=str,
                        default="/home/hdd1/vibhav/VE-SNLI/DSTC8-AVSD-vibhav/vesnli/data/lbl1_expl_out", help="Path of the folder where dataset is to be stored")
    args = parser.parse_args()

    if not args.no_image:
        args.no_premise = True

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    if not os.path.exists(args.final_data_path):
        os.mkdir(args.final_data_path)
    data = get_data(args.data_path, args.data_type, no_image=args.no_image)
    dataset = InferenceDataset(data,
                               tokenizer,
                               no_image=args.no_image,
                               no_premise=args.no_premise,
                               with_expl=args.with_expl)
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            collate_fn=lambda x: collate_fn(x,
                                                            tokenizer.pad_token_id,
                                                            no_image=args.no_image,
                                                            with_expl=args.with_expl))

    batch = next(iter(dataloader))
    if args.no_image:
        if args.with_expl:
            input_ids, label, expl_ids, input_mask = batch
            print('expl_ids', expl_ids[0])
            print('expl_ids', tokenizer.convert_ids_to_tokens(expl_ids[0]))
        else:
            input_ids, label, input_mask = batch
    else:
        if args.with_expl:
            image, input_ids, label, expl_ids, input_mask = batch
            print('expl_ids', expl_ids[0])
            print('expl_ids', tokenizer.convert_ids_to_tokens(expl_ids[0]))
        else:
            image, input_ids, label, input_mask = batch

    for i, v in enumerate(batch):
        print(i, v.shape)
    print('input_ids', input_ids[0])
    print('input_ids', tokenizer.convert_ids_to_tokens(input_ids[0]))
    print('input_mask', input_mask[0])
    print('label', label)
