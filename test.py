import copy
import json
import logging
import os
import random
import time
from argparse import ArgumentParser
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import *

import ibart
from dataset import InferenceDataset, collate_fn, get_data


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint_dir", type=str,
                        default="", help="short name of the model")
    parser.add_argument("--model_checkpoint", type=str,
                        default="", help="name of the model")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                        else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_image", action="store_true",
                        help="To process image or not")
    parser.add_argument("--no_premise", action="store_true",
                        help="To process premise or not")
    parser.add_argument("--with_expl", action="store_true",
                        help="To use explanations or not")
    parser.add_argument("--batch_size", type=int,
                        default=4, help="Batch size")

    parser.add_argument("--data_path", type=str,
                        default="/home/hdd1/vibhav/VE-SNLI/mycode-vesnli/dataset/e-SNLI-VE")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--output", type=str, default="result.json")

    parser.add_argument("--do_sample", action='store_true',
                        help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--beam_search", action='store_true',
                        help="Set to use beam search instead of sampling")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    parser.add_argument("--max_length", type=int, default=40,
                        help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=6,
                        help="Minimum length of the output utterances")
    parser.add_argument("--length_penalty", type=float,
                        default=0.3, help="length penalty")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7,
                        help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if not args.no_image:
        args.no_premise = True

    logger.info(f"Arguments: {pformat(args)}")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('Loading model params from ' + args.model_checkpoint)

    tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint_dir)
    model_config = BartConfig.from_pretrained(args.model_checkpoint_dir)
    if args.with_expl:
        model = AutoModelForSeq2SeqLM.from_config(model_config)
    else:
        if args.no_image:
            model = BartForSequenceClassification(model_config)
        else:
            model = ibart.BartForSequenceClassification(model_config)
    model.load_state_dict(torch.load(os.path.join(args.model_checkpoint_dir,
                                                  args.model_checkpoint)))
    model.to(args.device)
    model.eval()

    logging.info('Loading test data from ' + args.data_path)
    data = get_data(args.data_path, args.data_type, args.no_image)
    dataset = InferenceDataset(data,
                               tokenizer,
                               no_image=args.no_image,
                               no_premise=args.no_premise,
                               with_expl=args.with_expl)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=4,
                            collate_fn=lambda x: collate_fn(x,
                                                            tokenizer.pad_token_id,
                                                            no_image=args.no_image,
                                                            with_expl=args.with_expl))
    if args.with_expl:
        for batch in tqdm(dataloader):
            batch = tuple(input_tensor.to(args.device)
                          for input_tensor in batch)
            if args.no_image:
                if args.with_expl:
                    input_ids, label, expl_ids, input_mask = batch
                else:
                    input_ids, label, input_mask = batch
            else:
                if args.with_expl:
                    image, input_ids, label, expl_ids, input_mask = batch
                else:
                    image, input_ids, label, input_mask = batch
            output = model.generate(input_ids,
                                    num_beams=args.beam_size,
                                    max_length=args.max_length,
                                    min_length=args.min_length,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    temperature=args.temperature,
                                    do_sample=args.do_sample,
                                    length_penalty=args.length_penalty,
                                    early_stopping=True)
            input_output = list(zip(input_ids, expl_ids, output))
            for i in input_output:
                in_sent = tokenizer.decode(i[0],
                                           clean_up_tokenization_spaces=False)
                expl = tokenizer.decode(i[1],
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                out_expl = tokenizer.decode(i[2],
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
                print('PREMISE+HYPOTHESIS: ',
                      in_sent.split(tokenizer.pad_token)[0])
                print('GROUND EXPL', expl)
                print('GEN. EXPL', out_expl)
                print('--------------------------------')
    else:
        lbl_accuracy = 0
        for batch in tqdm(dataloader):
            batch = tuple(input_tensor.to(args.device)
                          for input_tensor in batch)
            if args.no_image:
                if args.with_expl:
                    input_ids, label, expl_ids, input_mask = batch
                else:
                    input_ids, label, input_mask = batch
            else:
                if args.with_expl:
                    image, input_ids, label, expl_ids, input_mask = batch
                else:
                    image, input_ids, label, input_mask = batch
            if args.no_image:
                output = model(input_ids=input_ids,
                               attention_mask=input_mask)
            else:
                output = model(input_ids=input_ids,
                               images=image,
                               attention_mask=input_mask)
            logits, _ = output
            logits = logits.argmax(dim=1)
            if not args.with_expl:
                lbl_accuracy += torch.eq(label,
                                         logits).float().sum() / len(label)
        print(lbl_accuracy / len(dataloader))
        # P+H
        # 0.9295 DEV SET
        # 0.9235 TEST SET
        # only H
        # 0.6907 DEV SET
        # 0.6927 TEST SET
        # I+H
        # 0.6940 DEV SET
        # 0.6949 TEST SET
