import torch
import argparse
import warnings
import pandas as pd

from src.pytorch.label_encoding import get_encoding
from src.pytorch.get_folds import get_folds
from src.pytorch.vocab import Vocabulary
from src.pytorch.trainer import perform_for_fold
from torch.utils.tensorboard import SummaryWriter
from config.pytorch.text_config import Config

from src.tensorflow.trainer import perform_for_fold_tf

import nltk
import pickle
 
def seed_everything():

    warnings.filterwarnings("ignore", category=UserWarning)
    torch.manual_seed(Config.globals_['seed'])
    torch.cuda.manual_seed(Config.globals_['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def argp():

    parser = argparse.ArgumentParser(description='Script to train pytorch script on text data')
    parser.add_argument('target',
                        help='Enter the columns name to be treated as target')
    parser.add_argument('framework',
                        help='Enter the framework you want to train on i.e., tf or pt')
    args = parser.parse_args()
    return (args.target, args.framework)

if __name__=="__main__":
    nltk.download('stopwords')
    
    seed_everything()
    column_name, framework=argp()

    assert column_name in ['action', 'object', 'location']
    assert framework in ['pt', 'tf']

    df = pd.read_csv(f"{Config.globals_['base_dir']}/train_data.csv")
    vocab_list = Vocabulary()
    vocab_list.build_vocablary(df.transcription.values)

    with open('Vocablary', 'wb') as config_vocab_file:
        pickle.dump(vocab_list, config_vocab_file)

    df = get_encoding(df, column_name)
    df =  get_folds(df, target=column_name)

    if framework=="pt":
        print(f"Training on {Config.globals_['device']}")
        writer = SummaryWriter(f"runs/PyTorch/Experiment_{column_name}_BS_{Config.loader['train']['batch_size']}_LR_{Config.optimizer['params']['lr']}")

        perform_for_fold(df = df, target_name=column_name, vocab_list=vocab_list, save_file=Config.globals_['version'], writer=writer)

    if framework=="tf":
        perform_for_fold_tf(df = df, target_name=column_name, vocab_list=vocab_list, save_file=Config.globals_['version'])
    