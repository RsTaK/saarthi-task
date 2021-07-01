from sklearn import model_selection
from config.pytorch.text_config import Config

def clean_text(text):
  text = text.lower()
  text = text.replace('?', '')
  text = text.replace('.', '')
  text = text.replace("'", '')
  text = text.replace('"', '')
  text = text.replace(',', '')
  text = text.replace('’', '')
  return text

def get_folds(df, target):

  df = df.sample(frac=1).reset_index(drop=True)
  df.transcription = df.transcription.apply(lambda x: clean_text(x))
  params = Config.split['params']
  kf = model_selection.StratifiedKFold(**params)

  df["k-fold"] = -1

  for fold, (tra_, val_) in enumerate(kf.split(X = df, y = df[target].values)):
    df.loc[val_,"k-fold"] = fold
    print(f'Fold No : {fold}, Training label count : {len(tra_)}, Validation label count : {len(val_)}')
  df.to_csv(f"{Config.globals_['key_dir']}/dataset_fold.csv", index=False)
  return df