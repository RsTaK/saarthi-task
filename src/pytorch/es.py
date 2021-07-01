import torch
from config.pytorch.text_config import Config

class EarlyStopping:
  def __init__(self, patience):
    self.patience = patience
    self.best_score = 0
  
  def update(self, score, epoch, save_file, model):
    model.eval() 
    flag=0

    if not self.best_score:
      self.best_score = score
      print('Saving model with best val as {}'.format(self.best_score))
        
      torch.save(
                  {'model_state_dict': model.state_dict(),'best_score': self.best_score, 'epoch': epoch},  
                  f"{Config.globals_['model_save_dir']}/{save_file}.pt"
                  )
  
    if score <= self.best_score:   
      self.best_score = score
      patience = Config.es['patience'] 
      print('Imporved model with best val as {}'.format(self.best_score))   
      torch.save(
                  {'model_state_dict': model.state_dict(),'best_score': self.best_score, 'epoch': epoch},  
                  f"{Config.globals_['model_save_dir']}/{save_file}.pt"
                  )

    else:
        self.patience -= 1
        print('Patience Reduced')
        if self.patience == 0:
            print('Early stopping. Best Val score: {:.3f}'.format(self.best_score))
            flag=1
    return flag