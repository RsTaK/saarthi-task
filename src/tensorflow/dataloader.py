
import numpy as np
import tensorflow as tf

class DatasetGeneratorTF:
    def __init__(self, x, vocab_list, y=None, Is_Train=True):
        self.x = x
        self.y = y
        self.Is_train=Is_Train
        self.vocab_list = vocab_list 

    def get_vector(self, transcription):
        return ([self.vocab_list.string_to_index[each_word] if each_word in self.vocab_list.string_to_index else self.vocab_list.string_to_index['<UNK>'] for each_word in transcription.split()]) #UNK token if word not in vocab

    def get_numeralise(self):

        temp = list()
        for each in self.x:
            transcription = self.vocab_list.clean_text(each)
            numeralised_transcription = [self.vocab_list.string_to_index['<SOS>']] + self.get_vector(transcription) + [self.vocab_list.string_to_index['<EOS>']] #Appending start and end token
            temp.append(numeralised_transcription)
        return temp

    def data(self):

        temp_ = self.get_numeralise()
        temp_ =tf.keras.preprocessing.sequence.pad_sequences(temp_, padding='post')
        
        if self.Is_train:
            labels = np.zeros((len(self.x), len(list(set(self.y)))))
            labels[np.arange((len(self.x))), self.y] = 1

            dataset = tf.data.Dataset.from_tensor_slices((temp_, labels))
            return dataset
        
        else:
            dataset = tf.data.Dataset.from_tensor_slices((temp_))
            return dataset