import tensorflow as tf
from tensorflow.keras import layers, Model

from config.tensorflow.text_config import Config

class TfModel:
    def __init__(self, input, n_classes):
        self.input = input
        self.n_classes = n_classes

    def get_model(self):
        encoder_input = layers.Input(shape=(None,))
        encoder_embedded = layers.Embedding(input_dim=self.input, output_dim=Config.model['embedding_layer']['size'])(
            encoder_input
        )

        _, state_h, _ = layers.LSTM(Config.model['lstm']['hidden_layer'], return_state=True)(
            encoder_embedded
        )

        output = layers.Dense(self.n_classes, activation='softmax')(state_h)

        model = Model([encoder_input], output)
        return model
    
    def optimizer(self):
        return getattr(tf.keras.optimizers, Config.optimizer['name'])(**Config.optimizer['params'])

    def get_loss(self):
        return getattr(tf.keras.losses, Config.loss['name'])()

    def metric(self):
        return tf.keras.metrics.CategoricalAccuracy('accuracy')

    def EarlyStoppeing(self):
        return tf.keras.callbacks.EarlyStopping(monitor='loss', patience=Config.es['patience'])

    def save_model(self, target_name, save_file):
        return tf.keras.callbacks.ModelCheckpoint(filepath=f"{Config.globals_['model_save_dir']}/{target_name}_{save_file}.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

    def get_tensorboard(self, target_name):
        log_dir = f"runs/Tensorflow/Experiment_{target_name}_BS_{Config.globals_['batch_size']}_LR_{Config.optimizer['params']['learning_rate']}" 
        return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

