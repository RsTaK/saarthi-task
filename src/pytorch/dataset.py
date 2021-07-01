import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class DatasetGenerator(Dataset):
    def __init__(self, transcription, vocab_list, target=None, Is_Train=True):
        self.vocab_list = vocab_list
        self.transcription = transcription
        self.target = target
        self.Is_Train = Is_Train
        
    def __getitem__(self, index):
        
        transcription = self.transcription[index]
        
        numeralised_transcription = [self.vocab_list.string_to_index['<SOS>']] + self.get_vector(transcription) + [self.vocab_list.string_to_index['<EOS>']] #Appending start and end token

        if self.Is_Train:
          target = self.target[index]
          return torch.tensor(numeralised_transcription), torch.tensor(target)
        else:
          return torch.tensor(numeralised_transcription)

    def __len__(self):
        return len(self.transcription)

    def get_vector(self, transcription):
      return ([self.vocab_list.string_to_index[each_word] if each_word in self.vocab_list.string_to_index else self.vocab_list.string_to_index['<UNK>'] for each_word in transcription.split()]) #UNK token if word not in vocab

class Collate:
    def __init__(self, pad_idx, Is_Train=True):
        self.pad_idx = pad_idx
        self.Is_Train = Is_Train

    def __call__(self, batch):

        if self.Is_Train:
          transcription = [item[0] for item in batch] 
          transcription = pad_sequence(transcription, batch_first=True, padding_value=self.pad_idx) # Padding so that sequence length is consistant in a batch
          targets = [item[1].unsqueeze(0) for item in batch]
          targets = torch.cat(targets, dim=0)
          
          return transcription, targets
        else:
          transcription = [item for item in batch] 
          transcription = pad_sequence(transcription, batch_first=True, padding_value=self.pad_idx) # Padding so that sequence length is consistant in a batch
 
          return transcription