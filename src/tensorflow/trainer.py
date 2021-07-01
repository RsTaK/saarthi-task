import os
import tensorflow as tf

from src.tensorflow.model import TfModel
from src.tensorflow.dataloader import DatasetGeneratorTF

from config.tensorflow.text_config import Config

from src.tensorflow.gpu_names import get_available_gpus

def perform_for_fold_tf(df, target_name, vocab_list, save_file):
    gpu_names = get_available_gpus()
    if not gpu_names:
        print(f"No GPU Available")

    train_X = df[df["k-fold"] !=Config.globals_['use_fold']].transcription.values
    train_Y = df[df["k-fold"] != Config.globals_['use_fold']][target_name].values

    valid_X = df[df["k-fold"] == Config.globals_['use_fold']].transcription.values
    valid_Y = df[df["k-fold"] == Config.globals_['use_fold']][target_name].values

    dataset_train = DatasetGeneratorTF(x=train_X, y=train_Y, vocab_list=vocab_list).data()
    dataset_valid = DatasetGeneratorTF(x=valid_X, y=valid_Y, vocab_list=vocab_list).data()
 
    dataset_train = dataset_train.shuffle(len(train_X)).batch(Config.globals_['batch_size'], drop_remainder=True)
    dataset_valid = dataset_valid.shuffle(len(valid_X)).batch(Config.globals_['batch_size'], drop_remainder=False)

    model = TfModel(input=vocab_list.__len__(), n_classes=len(list(set(train_Y))))

    m=model.get_model()

    metric = model.metric()
    opt = model.optimizer()
    loss = model.get_loss()
    
    es = model.EarlyStoppeing()
    cp_callback = model.save_model(target_name, save_file)
    tensorboard = model.get_tensorboard(target_name)

    m.compile(optimizer=opt, loss=loss, metrics=[metric])
    m.fit(dataset_train, validation_data=dataset_valid, epochs=Config.globals_['num_epochs'], callbacks=[tensorboard, es, cp_callback])


