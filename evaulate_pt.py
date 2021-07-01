import argparse
import torch
import torch.nn as nn
import pandas as pd

from src.pytorch.label_encoding import get_encoding
from config.pytorch.text_config import Config
from src.pytorch.dataset import DatasetGenerator, Collate

from src.pytorch.model import Model

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import pickle
 
def vocab():
    with open('Vocablary', 'rb') as config_vocab_file:
        vocab_list = pickle.load(config_vocab_file)
    return vocab_list

def argp():

    parser = argparse.ArgumentParser(description='Script to train pytorch script on text data')
    parser.add_argument('path',
                        help='Enter the path of csv')
    parser.add_argument('target',
                        help='Enter the columns name to be treated as target')

    args = parser.parse_args()
    return args.target, args.path

class EngineTest:

  def __init__(self, model):
    
    self.predictions = list()
    self.model = model 
  
  def fit(self, validation_loader):
    for _, x_val in enumerate(validation_loader):
      with torch.no_grad():
        
        temp = list()
        x_val = x_val.to(Config.globals_['device'], dtype=torch.long)         
        
        pred = self.model(x_val)
        y_pred = nn.functional.softmax(pred, dim=1).data.cpu().numpy()

        for each_pred in y_pred:
          temp.append(each_pred.argmax())

        self.predictions.extend(temp)
    return self.predictions

def perform_for_fold_test(df, target, vocab_list, model):

  valid_X = df.transcription.values
  valid_Y = df[target].values

  test_dataset= DatasetGenerator(transcription=valid_X, vocab_list=vocab_list, Is_Train=False)
        
  test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            collate_fn=Collate(vocab_list.string_to_index['<PAD>'], Is_Train=False),
            batch_size=32,
            num_workers=2,
            shuffle=False,
            pin_memory=False,
        ) 
  
  engine = EngineTest(model)
  pred = engine.fit(test_loader)
  return pred, valid_Y

def load_model(path,vocab_list, n_classes, dev):
    m = Model(vocab_list.__len__(), Config.model['embedding_layer']['size'], 
            Config.model['lstm']['hidden_layer'], 
            Config.model['lstm']['n_layers'], n_classes).to(Config.globals_['device'])   

    m.load_state_dict(torch.load(path, map_location=torch.device(dev))["model_state_dict"])
    m.eval()
    return m

if __name__=="__main__":

    target, path=argp()
    if target != "all":
      column_list = [target]
    else:
      column_list = ["action", "object", "location"]
    
    for column_name in column_list:
      valid_data = pd.read_csv(f"{path}")
      valid_data = get_encoding(valid_data, column_name)

      vocab_list = vocab()

      n_classes=len(list(set(valid_data[column_name])))
      model = load_model(f"{Config.globals_['model_save_dir']}/{column_name}_{Config.globals_['version']}.pt", vocab_list, n_classes, Config.globals_['device'])
      
      pred_0, true_0=perform_for_fold_test(df=valid_data, target=column_name, vocab_list=vocab_list, model=model)


      print(f1_score(true_0, pred_0, average='micro'))
      print(classification_report(true_0, pred_0))