import json
import numpy as np
from sklearn import preprocessing 
from numpyencoder import NumpyEncoder
from config.pytorch.text_config import Config

class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def get_encoding(df, target):

  encoder = preprocessing.LabelEncoder() 
  encoder.fit(df[target])
  key = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
  df[target]= encoder.transform(df[target])

  with open(f"{Config.globals_['key_dir']}/{target}_key.json", 'w') as f:
      json.dump(key, f, cls=NumpyEncoder) # Dumping Label_Key to be refered later 
      print(f'Encoded for {target}')

  return df