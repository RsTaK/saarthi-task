import time
import torch

from tqdm import tqdm

from src.pytorch.meter import AverageMeter
from src.pytorch.es import EarlyStopping
from config.pytorch.text_config import Config

class Engine:
    
    def __init__(self, model , save_file, writer):

        self.writer = writer
        self.step = 0
        self.model=model
        self.save_file = save_file
    
        self.criterion = self.model.loss().to(Config.globals_['device'])
        self.optimizer = self.model.opt(self.model)
        self.es = EarlyStopping(Config.es['patience'])
        
    def fit(self, train_loader, validation_loader):

      for epoch in range(Config.globals_['num_epochs']):

          print("Training Started...")
          t=time.time()
          summary_loss = self.train_one_epoch(train_loader)
          print('Train : Epoch {:03}: | Summary Loss: {:.3f} | Training time: {}'.format(epoch,summary_loss.avg,time.time() - t))

          # Plot things to tensorboard

          t=time.time()
          print("Validation Started...")
          summary_loss = self.validation(validation_loader)

          print('Valid : Epoch {:03}: | Summary Loss: {:.3f} | Training time: {}'.format(epoch,summary_loss.avg,time.time() - t))


          self.step += 1

          flag = self.es.update(summary_loss.avg, epoch, self.save_file, self.model)
          self.writer.add_hparams({"lr": Config.optimizer['params']['lr'], "bsize": Config.loader['train']['batch_size']},
          {"loss": summary_loss.avg})
                    


          if flag:
            break

    def validation(self, val_loader):
    
      self.model.eval()
      summary_loss = AverageMeter()

      t = time.time()

      for steps,(text, targets) in enumerate(tqdm(val_loader)):

          with torch.no_grad():
              text = text.to(Config.globals_['device'], dtype=torch.long)           
              targets = targets.to(Config.globals_['device'], dtype=torch.long)

              batch_size = text.shape[0]    

              outputs = self.model(text)

              loss = self.criterion(outputs, targets)
              self.writer.add_scalar("Validation loss", loss, global_step=self.step)
              summary_loss.update(loss.detach().item(), batch_size)

      return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()

        t = time.time()

        for steps,(text, targets) in enumerate(tqdm(train_loader)):

            text = text.to(Config.globals_['device'], dtype=torch.long)          
            targets = targets.to(Config.globals_['device'], dtype=torch.long)

            batch_size = text.shape[0]     

            self.optimizer.zero_grad()

            outputs = self.model(text)

            loss = self.criterion(outputs, targets)
            self.writer.add_scalar("Training loss", loss, global_step=self.step)

            loss.backward()
            self.optimizer.step()      

            summary_loss.update(loss.detach().item(), batch_size)
        return summary_loss