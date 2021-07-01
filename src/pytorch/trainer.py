import torch
from src.pytorch.model import Model
from src.pytorch.engine import Engine
from src.pytorch.dataset import DatasetGenerator, Collate

from config.pytorch.text_config import Config

def perform_for_fold(df, target_name, vocab_list, save_file,writer, load_weights_path=None):
  
  train_X = df[df["k-fold"] != Config.globals_['use_fold']].transcription.values
  train_Y = df[df["k-fold"] != Config.globals_['use_fold']][target_name].values

  valid_X = df[df["k-fold"] == Config.globals_['use_fold']].transcription.values
  valid_Y = df[df["k-fold"] == Config.globals_['use_fold']][target_name].values

  train_dataset = DatasetGenerator(transcription=train_X, target=train_Y, vocab_list=vocab_list, Is_Train=True)
  valid_dataset = DatasetGenerator(transcription=valid_X, target=valid_Y, vocab_list=vocab_list, Is_Train=True)

  train_loader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=Collate(vocab_list.string_to_index['<PAD>'], Is_Train=True),
            batch_size=Config.loader['train']['batch_size'],
            pin_memory=Config.loader['train']['pin_memory'],
            drop_last=Config.loader['train']['drop_last'],
            num_workers=Config.loader['train']['num_workers'],
            shuffle=Config.loader['train']['shuffle']
        )
        
  validation_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            collate_fn=Collate(vocab_list.string_to_index['<PAD>'], Is_Train=True),
            batch_size=Config.loader['valid']['batch_size'],
            pin_memory=Config.loader['valid']['pin_memory'],
            drop_last=Config.loader['valid']['drop_last'],
            num_workers=Config.loader['valid']['num_workers'],
            shuffle=Config.loader['valid']['shuffle']
        ) 
  
  model = Model(vocab_list.__len__(), Config.model['embedding_layer']['size'], 
                Config.model['lstm']['hidden_layer'], 
                Config.model['lstm']['n_layers'], len(list(set(df[target_name])))).to(Config.globals_['device'])

  if load_weights_path is not None:
    model.load_state_dict(torch.load(Config.globals_['model_save_dir']+ f"{save_file}.pt")["model_state_dict"]) 
    print("Weight Loaded")

  save_file = f'{target_name}_{save_file}'

  engine = Engine(model, save_file, writer)
  engine.fit(train_loader, validation_loader)