import argparse
import json

import numpy as np

from evaulate_pt import *
from evaulate_pt import Config as cg_pt
from evaulate_tf import DatasetGeneratorTF, TfModel
from evaulate_tf import Config as cg
from evaulate_sst import *

def argp():

    parser = argparse.ArgumentParser(description='Script to train pytorch script on text data')
    parser.add_argument('text',
                        help='Enter the text you want to check on')
    parser.add_argument('audio',
                        help='Pass audio else Enter None')

    parser.add_argument('fm',
                        help='Pytorch or Tensorflow model for inference')

    args = parser.parse_args()
    return args.text, args.audio, args.fm

if __name__=="__main__":
    text, audio, fm = argp()

    column_list = {"action":6, "object":14, "location":4}
    vocab_list = vocab()

    if fm=="pt":
        for column_name in column_list.keys():
            model = load_model(f"{cg_pt.globals_['model_save_dir']}/{column_name}_{cg_pt.globals_['version']}.pt",vocab_list, column_list[column_name], 'cpu')
            model.to('cpu')

            test_dataset = DatasetGenerator(transcription=[text], vocab_list=vocab_list, Is_Train=False)
            test_loader = torch.utils.data.DataLoader(
                        test_dataset, 
                        collate_fn=Collate(vocab_list.string_to_index['<PAD>'], Is_Train=False),
                        batch_size=1,
                        num_workers=1,
                        shuffle=False,
                        pin_memory=False,
                    ) 
            
            engine = EngineTest(model)
            pred = engine.fit(test_loader)
            with open(f'label_keys\{column_name}_key.json', 'r') as f:
                action = json.load(f)
                action = {k: v for k, v in enumerate(action)}  
                print(f'{column_name}: {action[pred[0]]}')
            
      
    if fm=="tf":
        for column_name in column_list.keys():
            model = TfModel(input=vocab_list.__len__(), n_classes=column_list[column_name]).get_model()
            model.load_weights(f"{cg.globals_['model_save_dir']}/{column_name}_{cg.globals_['version']}.ckpt")
            
            dataset_valid = DatasetGeneratorTF(x=[text], vocab_list=vocab_list, Is_Train=False).data()
            dataset_valid = dataset_valid.batch(1, drop_remainder=False)

            temp=model.predict(dataset_valid)
            temp=np.argmax(temp)

            with open(f'label_keys\{column_name}_key.json', 'r') as f:
                action = json.load(f)
                action = {k: v for k, v in enumerate(action)}  
                print(f'{column_name}: {action[temp]}')

    if audio:

        processor = Wav2Vec2Processor.from_pretrained(Config.globals_['processor'])
        model = Wav2Vec2ForCTC.from_pretrained(Config.globals_['model'])

        speech_array, sampling_rate = librosa.load(audio, sr=8000)
        inputs = processor(speech_array, sampling_rate=8000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to('cpu')).logits   

        pred_ids = torch.argmax(logits, dim=-1)
        pred = processor.batch_decode(pred_ids)

        print(f'Statement: {pred[0]}')


