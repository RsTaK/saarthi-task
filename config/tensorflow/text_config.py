
class Config:
    """This class has all the parameters that we require to set"""
    
    globals_ = {
        'seed' : 42,
        'num_epochs' : 5,
        'model_save_dir' : r'model\tf',
        'version' : 'ver_1',
        'batch_size' : 32,
        'use_fold' : 0
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
        }
    }

    ##### Loss #####
    loss = {
        'name' : 'CategoricalCrossentropy',
    }   
    

    #### Optimizer ####
    optimizer = {
        'name' : 'Adam',
        'params' : {
            'learning_rate' : 0.001,
        }
    }