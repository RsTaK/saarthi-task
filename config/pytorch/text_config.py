import torch

class Config:
    """This class has all the parameters that we require to set"""
    
    globals_ = {
        'seed' : 42,
        'device' : "cuda" if torch.cuda.is_available() else "cpu",
        'num_epochs' : 5,
        'key_dir' : 'label_keys',
        'base_dir' : 'data',
        'model_save_dir' : 'model/pt',
        'version' : 'ver_1',
        'use_fold' : 0
    }
    
    ##### Vocablary #####
    vocablary = {
        'threshold' : 5
    }
  
    ##### EarlyStopping #####
    es = {
        'patience' : 5
    }

    ###### MODEL #######
    model = {
        'embedding_layer' : {
            'size' : 128
        },

        'lstm' : {
            'hidden_layer' : 256,
            'n_layers' : 1,
        }
    }

    #### Optimizer ####
    optimizer = {
        'name' : 'AdamW',
        'params' : {
            'lr' : 0.001,
        }
    }
    
    ##### Loss #####
    loss = {
        'name' : 'CrossEntropyLoss',
    }   
    
    ##### SPLIT ######
    split = {
        'params' : {
            'n_splits' : 5,
            'random_state' : 42,
            'shuffle' : True,
        }
    }
    
    ###### LOADER #######
    loader = {
        'train' : {
            'batch_size' : 32,
            'shuffle' : True,
            'num_workers' : 2,
            'pin_memory' : True,
            'drop_last' : True,
        },
        'valid' : {
            'batch_size' : 32,
            'shuffle' : False,
            'num_workers' : 2,
            'pin_memory' : True,
            'drop_last' : False,            
        }
    }