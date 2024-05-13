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
    
    def parse_args(self):
        
        args = self.parser.parse_args()
        
        model_name = args.model
        dataset = args.dataset
        seq_size = args.seq_size
        weighted = args.weighted
        config_dir = args.config_dir
        
        save_model = args.save_model
        save_dir = args.save_dir
        
        output_dir = args.output_dir
        verbose = args.verbose
        run = args.run
        
        cfg = dict()
        
        cfg['seq_size'] = seq_size
        cfg['weighted'] = weighted

        cfg['config_dir'] = config_dir
        cfg['save_model'] = save_model
        cfg['save_dir'] = save_dir
        cfg['output_dir'] = output_dir
        
        cfg['verbose'] = verbose
        
        model_name = model_name.lower()
        if model_name == 'naive_bayes' or model_name == 'nb' or model_name == 'naive bayes':
            model_name = 'nb'
            cfg_model_file = 'naive_bayes'
        elif model_name == 'xgb' or model_name == 'xgboost':
            model_name = 'xgb'
            cfg_model_file = 'xgboost'
        elif model_name == 'rf' or model_name == 'random_forest' or model_name == 'random forest':
            model_name = 'rf'
            cfg_model_file = 'random_forest'
        elif model_name == 'mlp':
            model_name = 'mlp'
            cfg_model_file = 'mlp'
        elif model_name == 'bilstm' or model_name == 'bi_lstm' or model_name == 'bi-lstm' or model_name == 'bi lstm':
            model_name = 'bilstm'
            cfg_model_file = 'bilstm'
        elif model_name == 'bigru' or model_name == 'bi_gru' or model_name == 'bi-gru' or model_name == 'bi gru':
            model_name = 'bigru'
            cfg_model_file = 'bigru'
        elif model_name == 'cnn':
            model_name = 'cnn'
            cfg_model_file = 'cnn'
        else:
            raise NotImplementedError('Model name does not exist')

        cfg['model_name'] = model_name
        cfg['dataset'] = dataset.lower()
        
        if model_name == 'cnn' or model_name == 'bigru'or model_name == 'bilstm':
            cfg['filename'] = f'{args.model}_{args.dataset}_{args.seq_size}_{args.run}'
            cfg['input_format'] = 'dl'
        else:
            cfg['filename'] = f'{args.model}_{args.dataset}_{args.run}'
            cfg['input_format'] = 'ml'

        with open(os.path.join(cfg['config_dir'], f'data.yml'), 'r') as file:
            cfg_data = yaml.safe_load(file)
            cfg = {**cfg,**cfg_data}

        with open(os.path.join(cfg['config_dir'], f'{dataset}.yml'), 'r') as file:
            cfg_dataset = yaml.safe_load(file)
            cfg = {**cfg,**cfg_dataset}

        with open(os.path.join(cfg['config_dir'], f'{cfg_model_file}.yml'), 'r') as file:
            cfg_model = yaml.safe_load(file)
            cfg = {**cfg,**cfg_model}
        
        return cfg
