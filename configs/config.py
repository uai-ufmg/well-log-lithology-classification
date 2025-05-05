import os
from pathlib import Path
import argparse
import yaml
import numpy as np


class ConfigArgs():
    
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='xgb', help='Model name')
        parser.add_argument('--dataset', type=str, default='force', help='Dataset name')
        parser.add_argument('--seq_size', type=int, default=50, help='Sequence size (Deep Learning models)')
        parser.add_argument('--weighted', type=self.__str2bool, nargs='?', const=True, default=False, help='Use weighted training')
        parser.add_argument('--run', type=int, default=1, help='Execution number')
        
        parser.add_argument("--save_model", type=self.__str2bool, nargs='?', const=True, default=False, help="Save trained model")
        parser.add_argument("--save_dir", type=Path, default=Path("trained_models"), help="Path to save model weights")
        
        parser.add_argument("--output_dir", type=Path, default=Path("results"), help="Path to save logs")
        parser.add_argument("--plot_dir", type=Path, default=Path("plots"), help="Path to save plots")
        parser.add_argument("--config_dir", type=Path, default=Path("configs"), help="Path to load config files")
        parser.add_argument("--verbose", type=self.__str2bool, nargs='?', const=True, default=False, help="Use verbose")
        
        self.parser = parser
    
    def __str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def standardize_model_name(self, model_name: str) -> (str, str):
        """
        Given a user-provided model_name (possibly with variations/spaces/hyphens),
        return a canonical (model_name, cfg_model_file) tuple.
        Arguments:
        ---------
            - model_name (str): Raw model name string.
        Return:
        ---------
            - MODEL_MAP[key](tuple): Tuple containing (model_name, cfg_model_file).
        """
        
        # Map all possible inputs to a canonical pair:
        #   - The first item in the tuple is the canonical 'model_name'
        #   - The second item is the 'cfg_model_file' used later on
        MODEL_MAP = {
            # NB
            'naive_bayes':    ('nb', 'naive_bayes'),
            'naive bayes':    ('nb', 'naive_bayes'),
            'nb':             ('nb', 'naive_bayes'),
    
            # XGB
            'xgb':            ('xgb', 'xgboost'),
            'xgboost':        ('xgb', 'xgboost'),
    
            # RF
            'rf':             ('rf', 'random_forest'),
            'random_forest':  ('rf', 'random_forest'),
            'random forest':  ('rf', 'random_forest'),
    
            # MLP
            'mlp':            ('mlp', 'mlp'),
    
            # BiLSTM
            'bilstm':         ('bilstm', 'bilstm'),
            'bi_lstm':        ('bilstm', 'bilstm'),
            'bi-lstm':        ('bilstm', 'bilstm'),
            'bi lstm':        ('bilstm', 'bilstm'),
    
            # BiGRU
            'bigru':          ('bigru', 'bigru'),
            'bi_gru':         ('bigru', 'bigru'),
            'bi-gru':         ('bigru', 'bigru'),
            'bi gru':         ('bigru', 'bigru'),
    
            # CNN
            'cnn':            ('cnn', 'cnn'),
    
            # ResNet
            'resnet':         ('resnet', 'resnet'),
    
            # AdaBoost-Transformer
            'adaboost_transformer':  ('adaboost_transformer', 'adaboost_transformer'),
            'adaboost-transformer':  ('adaboost_transformer', 'adaboost_transformer'),
            'transformer':           ('adaboost_transformer', 'adaboost_transformer'),
            'ada_transformer':       ('adaboost_transformer', 'adaboost_transformer'),
            'ada-transformer':       ('adaboost_transformer', 'adaboost_transformer'),
    
            # HNFCL / Tri-Training
            'hnfcl':         ('hnfcl', 'hnfcl'),
            'tri_training':  ('hnfcl', 'hnfcl'),
            'tri-training':  ('hnfcl', 'hnfcl'),
        }
    
        key = model_name.strip().lower()
        if key in MODEL_MAP:
            return MODEL_MAP[key]
        else:
            raise NotImplementedError(f"Model name '{model_name}' does not exist")
    
    def parse_args(self):

        cfg = dict()        
        args = self.parser.parse_args()
        
        model_name = args.model.lower()
        dataset = args.dataset.lower()
        run = args.run

        cfg['dataset'] = dataset
        
        cfg['seq_size'] = args.seq_size
        cfg['weighted'] = args.weighted

        cfg['config_dir'] = args.config_dir
        cfg['save_model'] = args.save_model
        cfg['save_dir'] = args.save_dir
        cfg['output_dir'] = args.output_dir
        cfg['plot_dir'] = args.plot_dir
        
        cfg['verbose'] = args.verbose

        # Mapping model_name provided to a standard model_name and cfg_file_name
        model_name, cfg_model_file = self.standardize_model_name(model_name)
        cfg['model_name'] = model_name
        
        dl_models = {
            'adaboost_transformer',
            'bigru',
            'bilstm',
            'cnn',
            'resnet'
        }
    
        if model_name in dl_models:
            cfg['filename'] = f"{args.model}_{args.dataset}_{args.seq_size}_{args.run}_{args.weighted}"
            cfg['input_format'] = 'dl'
        else:
            cfg['filename'] = f"{args.model}_{args.dataset}_{args.run}_{args.weighted}"
            cfg['input_format'] = 'ml'

        # Merge data.yml
        data_file = os.path.join(cfg['config_dir'], "data.yml")
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                cfg_data = yaml.safe_load(f) or {}
                cfg.update(cfg_data)

        # Merge {dataset}.yml
        dataset_file = os.path.join(cfg['config_dir'], f"{dataset}.yml")
        if os.path.exists(dataset_file):
            with open(dataset_file, 'r') as f:
                cfg_dataset = yaml.safe_load(f) or {}
                cfg.update(cfg_dataset)

        # Merge the {model}.yml
        model_file = os.path.join(cfg['config_dir'], f"{cfg_model_file}.yml")
        if os.path.exists(model_file):
            with open(model_file, 'r') as f:
                cfg_model = yaml.safe_load(f) or {}
                cfg.update(cfg_model)
        
        return cfg
