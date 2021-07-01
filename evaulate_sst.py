from tqdm import tqdm
import librosa
from datasets import Dataset, load_metric

import torch
import pandas as pd

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from config.sst.config import Config

import argparse

def argp():
    parser = argparse.ArgumentParser(description='Script to train pytorch script on text data')
    parser.add_argument('path',
                        help='Enter the path of csv')
    parser.add_argument('wav_path',
                        help='Enter the base_dir of Wav files')
    args = parser.parse_args()
    return args.path, args.wav_path


def audio_feature(df, wav_path):
  audio = list()
  duration = list()
  for each in tqdm(df.path):
    speech_array, sampling_rate = librosa.load(f"{wav_path}/{each}", sr=8000)
    d = librosa.get_duration(y=speech_array, sr=sampling_rate)
    audio.append(speech_array)
    duration.append(d)
  return audio, duration
  
def get_test(path):

  test_data = pd.read_csv(f"{path}")
  test_ = test_data.copy()

  test_data["transcription"] = test_data["transcription"].str.replace("[\’\'\,\.\?]",'').str.lower()

  audio, duration = audio_feature(test_data, path)
  test_data["data"] = audio
  test_data["duration"] = duration

  test_data=test_data.drop(['path', 'action', 'object', 'location'], axis=1)

  test_data = Dataset.from_pandas(test_data)

  return test_data, test_

def speech_file_to_array_fn(batch):
	batch["sentence"] = batch["transcription"]
	batch["speech"] = batch["data"]
	return batch

def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=8000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(Config.globals_['device'])).logits
    
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch

if __name__=="__main__":
  path, wav_path = argp()

  processor = Wav2Vec2Processor.from_pretrained(Config.globals_['processor'])
  model = Wav2Vec2ForCTC.from_pretrained(Config.globals_['model'])
  test_data, test_ = get_test(path, wav_path)
  test_data = test_data.map(speech_file_to_array_fn)

  model = model.to(Config.globals_['device'])

  wer_metric = load_metric("wer")
  result = test_data.map(evaluate, batched=True, batch_size=32)
  print("WER: {:4f}".format(wer_metric.compute(predictions=result["pred_strings"], references=result["sentence"])))

  test_["SST"] = result["pred_strings"]
  test_.to_csv(Config.globals_['output_path'], index=False)