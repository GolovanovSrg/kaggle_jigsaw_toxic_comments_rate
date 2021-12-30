#!/usr/bin/env python3

import argparse
import json
import sys
import os

import torch

sys.path.append('../')

from gpt2 import (MODEL_INFO, get_vocab, load_gpt_weights, prepare_bpe_codes,
                  prepare_bpe_vocab, prepare_gpt_weights)
from training.model import RegressionModel
from training.datasets import ToxicTextsDataset, RangingToxicTextsDataset
from training.trainer import Trainer


def get_model_vocab(config):
    model_type = config.pop('model_type')
    vocab_dir = config.pop('vocab_dir')
    parameters_dir = config.pop('parameters_dir')

    vocab_path = os.path.join(vocab_dir, 'gpt2_bpe.vocab')
    codes_path = os.path.join(vocab_dir, 'gpt2_bpe.codes')
    prepare_bpe_vocab(vocab_path, model_type)
    prepare_bpe_codes(codes_path, model_type)
    vocab = get_vocab(vocab_path, codes_path)

    model_config = MODEL_INFO[model_type]['config']
    model_config.pop('n_embeddings')

    model = RegressionModel(n_embeddings=len(vocab),
                            padding_idx=vocab.pad_id,
                            **config,
                            **model_config)

    if parameters_dir is not None:
        parameters_path = os.path.join(parameters_dir, model_type + '_parameters.pt')
        prepare_gpt_weights(parameters_path, model_type)
        parameters = torch.load(parameters_path, map_location='cpu')
        load_gpt_weights(model.encoder, parameters, vocab.n_special_tokens)

    return model, vocab

def get_datasets(config, vocab):
    train_dataset = ToxicTextsDataset.from_file(vocab=vocab, **config['train_dataset'])
    test_dataset = RangingToxicTextsDataset.from_file(vocab=vocab, **config['test_dataset'])

    return train_dataset, test_dataset


def get_trainer(config, model):
    trainer = Trainer(model=model, **config)
    return trainer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.json', help='Path of config.')

    return parser


def main(args):
    with open(args.config_path, 'r') as json_file:
        config = json.load(json_file)

    model, vocab = get_model_vocab(config['model'])

    train_dataset, test_dataset = get_datasets(config['datasets'], vocab)
    trainer = get_trainer(config['trainer'], model)

    train_process_config = config['train_process']
    trainer.train(train_data=train_dataset,
                  test_data=test_dataset,
                  **train_process_config)


if __name__ == '__main__':
    arg_parser = get_parser()
    args = arg_parser.parse_known_args()[0]
    main(args)
