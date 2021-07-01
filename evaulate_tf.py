import pickle
from numpy.lib.function_base import append
import pandas as pd
import argparse

from torch.utils import data
from torch.utils.data import dataset
from src.tensorflow.model import TfModel

from config.tensorflow.text_config import Config
from src.pytorch.label_encoding import get_encoding

from src.tensorflow.dataloader import DatasetGeneratorTF
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

def argp():

    parser = argparse.ArgumentParser(description='Script to train pytorch script on text data')
    parser.add_argument('path',
                        help='Enter the path of csv')
    parser.add_argument('target',
                        help='Enter the columns name to be treated as target')

    args = parser.parse_args()
    return args.target, args.path

def vocab():
    with open('Vocablary', 'rb') as config_vocab_file:
        vocab_list = pickle.load(config_vocab_file)
    return vocab_list

def perform_for_fold_test(df, target, vocab_list, model):

    valid_X = df.transcription.values
    valid_Y = df[target].values

    prediction=list()

    dataset_valid = DatasetGeneratorTF(x=valid_X, vocab_list=vocab_list, Is_Train=False).data()
    dataset_valid = dataset_valid.batch(Config.globals_['batch_size'], drop_remainder=False)

    temp=model.predict(dataset_valid)

    for each_pred in temp:
        prediction.append(np.argmax(each_pred))

    
    return prediction, valid_Y

if __name__=="__main__":

    target, path=argp()
    if target != "all":
      column_list = [target]
    else:
      column_list = ["action", "object", "location"]
    
    for column_name in column_list:

        valid_data = pd.read_csv(path)
        valid_data = get_encoding(valid_data, column_name)

        vocab_list = vocab()

        n_classes=len(list(set(valid_data[column_name])))

        model = TfModel(input=vocab_list.__len__(), n_classes=n_classes).get_model()
        model.load_weights(f"{Config.globals_['model_save_dir']}/{target}_{Config.globals_['version']}.ckpt")

        pred_0, true_0 = perform_for_fold_test(valid_data, target, vocab_list, model)
        
        print(f1_score(true_0, pred_0, average='micro'))
        print(classification_report(true_0, pred_0))