import nltk
from nltk.corpus import stopwords
from config.pytorch.text_config import Config

class Vocabulary:

  def __init__(self):
    self.index_to_string = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
    self.string_to_index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    self.frequency = Config.vocablary['threshold']
    self.stop_words = set(stopwords.words('english'))
    self.excluded = ['down', 'more', 'off', 'on', 'up']
  
  def __len__(self):
    return len(self.index_to_string)

  def clean_text(self, text):
    text = text.lower()
    text = text.replace('?', '')
    text = text.replace('.', '')
    text = text.replace("'", '')
    text = text.replace('"', '')
    text = text.replace(',', '')
    text = text.replace('’', '')
    return text

  def build_vocablary(self, text_list):
    self.frequencies = {}
    self.removed = {}
    idx = 4

    for each_sentence in text_list:
      each_sentence = self.clean_text(each_sentence)

      for each_word in each_sentence.split(" "):

        if each_word in self.stop_words and each_word not in self.excluded:

          if each_word not in self.removed:
            self.removed[each_word] = 1
          else:
            self.removed[each_word] += 1

        else:
          if each_word not in self.frequencies:
            self.frequencies[each_word] = 1
          else:
            self.frequencies[each_word] += 1

          if self.frequencies[each_word] == self.frequency:
            self.string_to_index[each_word] = idx
            self.index_to_string[idx] = each_word
            idx += 1