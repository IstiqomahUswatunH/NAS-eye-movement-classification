import os
import warnings
import pandas as pd
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Conv1D, MaxPooling1D, BatchNormalization, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, Flatten
from keras.callbacks import EarlyStopping

from CONSTANTS import *
from f1_score import F1_score

class SearchSpace(object):
    """Define search space and encode architectures

    layer_values: value for num of filter and num of neurons (units) for layer
    act_func: sigmoid, tanh, and relu. Softmax for the last layer
    
    vocab_dict: menghasilkan kemungkinan parameter lapisan untuk sebuah jaringan saraf
    layer_params: berisi pasangan (layer_values, act_funcs) untuk setiap kombinasi yg mungkin
    layer_id: berisi ID untuk setiap kombinasi tersebut
    vocab: membuat vocab dg layer_params dan layer_id
    
    encode_sequence: mengonversi sequence value menjadi urutan ID yg sesuai dg vocab
    decode_sequence: mengonversi sequence ID menjadi sequence value kembali
    
    Args:
        target_classes (int): 4 label with fixation (1), saccade (2), smooth pursuit (3), noise(4)
    """
    def __init__(self, target_classes):

        self.target_classes = target_classes
        self.vocab = self.vocab_dict()

    def vocab_dict(self):
        # for fully connected layer
        layer_values = [8, 16, 32, 64]
        act_funcs = ['tanh', 'relu', 'elu']
        
        # initialize lists for keys and values of vocabulary
        layer_params = []
        layer_id = []
        
        #--------------------------------start creation of vocab from which controller will create a sequence----------------#
        for i in range(len(layer_values)):
            for j in range(len(act_funcs)):
                layer_params.append((layer_values[i], act_funcs[j]))
                layer_id.append(len(act_funcs) * i + j + 1)
         
        vocab = dict(zip(layer_id, layer_params))   #tuple 
        
        # add final layer
        if self.target_classes == 2:
            vocab[len(vocab) + 1] = (self.target_classes - 1, 'sigmoid')
        else:
            vocab[len(vocab) + 1] = (self.target_classes, 'softmax')
        #print ("apa itu vocab?", vocab)
        return vocab

    #----------------------------------------------create search space-----------------------------------------------------------#
    
    # encode a sequence of configurations into a sequence of IDs
    def encode_sequence(self, sequence):
        keys = list(self.vocab.keys())          #mengambil ID
        values = list(self.vocab.values())      #mengambil semua nilai layer_params 
        encoded_sequence = []
        for value in sequence:
            encoded_sequence.append(keys[values.index(value)])
        return encoded_sequence
    
    # decode a sequence of IDs into a sequence of configurations
    def decode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])
        return decoded_sequence


class ModelGenerator(SearchSpace, F1_score):
    """generator to take these sequences and convert them into models that can be trained and evaluated
    

    Args:
        SearchSpace (_type_): _description_
    """
    def __init__(self):

        self.target_classes = TARGET_CLASSES
        self.optimizer = OPTIMIZER
        self.learning_rate = LEARNING_RATE
        self.dropout = DROPOUT
        self.loss_func = LOSS_FUNCTION
        self.one_shot = ONE_SHOT
        self.f1_score = F1_score()
        self.metrics = ['accuracy', self.f1_score.f1_FIX, self.f1_score.f1_SACC, self.f1_score.f1_SP, self.f1_score.f1_macro]


        super().__init__(TARGET_CLASSES)

        #-------------------------Initializing shared weights dictionary for one-shot training-------------------------#
        if self.one_shot:
            # path to shared weights dictionary
            self.weights_file = 'LOGS/shared_weights.pkl'
            # open an empty dataframe to store shared weights
            self.shared_weights = pd.DataFrame({'bigram_id': [], 'weights': []})
            #pickle file to store shared weights
            if not os.path.exists(self.weights_file):
                print("Initializing shared weights dictionary...")
                self.shared_weights.to_pickle(self.weights_file)
    
    # create a keras model given a sequence of configurations (sequences and input data shape)
    def create_model(self, sequence, input_shape, dropout_rate=0.3):
      #  print("apa itu mlp_input_shape?", mlp_input_shape)
        #decode sequence to get node and activation function for each layer
        layer_configs = self.decode_sequence(sequence)
        model = Sequential()
        
        for i, layer_conf in enumerate(layer_configs): 
            if i==0: # first layer
                # conv1d-1
                model.add(Conv1D(filters=layer_conf[0], kernel_size=3, padding="same", input_shape= input_shape, name="conv1d")) # kernel size follow michael's paper
                model.add(BatchNormalization(axis=-1))
                model.add(Activation(activation=layer_conf[1]))
                model.add(TimeDistributed(Flatten()))
            elif i==1: # second layer
                model.add(Bidirectional(LSTM(units=layer_conf[0], return_sequences=True), name="bilstm"))
            elif i==2: # third layer
                model.add(TimeDistributed(Dense(units=layer_conf[0], activation=layer_conf[1])))
            
            # create conv1d block
            # i is index of layer_conf
            # layer_conf[0] = layer_calue
            # layer_conf[1] = activation function
        # print(model.summary())
        return model

    def compile_model(self, model):
        model.compile(loss=self.loss_func, 
                      optimizer=self.optimizer, 
                      metrics=self.metrics)
        return model

    def update_weights(self, model):
        layer_configs = ['input']
        
        for layer in model.layers:
        # Periksa jenis lapisan berdasarkan namanya
        # layer: conv1d, batch_norm, activation, flatten, droput, dense, bilstm, dense
            if 'conv1d' in layer.name:
                layer_configs.append(('conv1d', layer.get_config()))
            elif 'batch_normalization' in layer.name:
                layer_configs.append(('batch_normalization', layer.get_config()))
            elif 'activation' in layer.name:
                layer_configs.append(('activation', layer.get_config()))
            elif 'time_distributed' in layer.name:
                layer_configs.append(('time_distributed', layer.get_config()))
            #elif 'dropout' in layer.name:
             #   layer_configs.append(('dropout', layer.get_config()))
            elif 'bilstm' in layer.name:
                layer_configs.append(('bilstm', layer.get_config()))
                
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
            
        j = 0
        for i, layer in enumerate(model.layers):
           # if 'conv1d' not in layer.name and 'dropout' not in layer.name and 'bilstm' not in layer.name:
            if 'conv1d' not in layer.name and 'bilstm' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                bigram_ids = self.shared_weights['bigram_id'].values
                search_index = []
                for i in range(len(bigram_ids)):
                    #print(f"update_weights: len of config_ids: {len(config_ids)}, len of bigram_ids: {len(bigram_ids)}")
                    if config_ids[j] == bigram_ids[i]:
                       # print(f"update_weights config_ids[j] == bigram_ids[i]: len of config_ids: {len(config_ids[j])}, len of bigram_ids: {len(bigram_ids[i])}")
                        search_index.append(i)
                if len(search_index) == 0:
                    new_row = pd.DataFrame({'bigram_id': [config_ids[j]],
                                                                      'weights': [layer.get_weights()]})
                    self.shared_weights = pd.concat([self.shared_weights, new_row], ignore_index=True)
                else:
                    self.shared_weights.at[search_index[0], 'weights'] = layer.get_weights()
                j += 1
        self.shared_weights.to_pickle(self.weights_file)

    def set_model_weights(self, model):
        layer_configs = []
        for layer in model.layers:
            if 'conv1d' in layer.name:
                layer_configs.append(('conv1d', layer.get_config()))
            elif 'batch_normalization' in layer.name:
                layer_configs.append(('batch_normalization', layer.get_config()))
            elif 'activation' in layer.name:
                layer_configs.append(('activation', layer.get_config()))
            elif 'time_distributed' in layer.name:
                layer_configs.append(('time_distributed', layer.get_config()))
            #elif 'dropout' in layer.name:
             #   layer_configs.append(('dropout', layer.get_config()))
            elif 'bilstm' in layer.name:
                layer_configs.append(('bilstm', layer.get_config()))
        
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
            
        j = 0
        for i, layer in enumerate(model.layers):
            #if 'dropout' not in layer.name:
            #if 'conv1d' not in layer.name and 'dropout' not in layer.name and 'bilstm' not in layer.name:
            if 'conv1d' not in layer.name and 'bilstm' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                bigram_ids = self.shared_weights['bigram_id'].values
                search_index = []
                for i in range(len(bigram_ids)):
                    #print(f"set_model_weights: len of config_ids: {len(config_ids)}, len of bigram_ids: {len(bigram_ids)}")
                    if config_ids[j] == bigram_ids[i]:
                        # check len of configs_idx and bigram_ids
                       # print(f"set_model_weights config_ids[j] == bigram_ids[i]: len of config_ids: {len(config_ids[j])}, len of bigram_ids: {len(bigram_ids[i])}")
                        search_index.append(i)
                if len(search_index) > 0:
                    print("Transferring weights for layer:", config_ids[j])
                    layer.set_weights(self.shared_weights['weights'].values[search_index[0]])
                j += 1
    
    def train_model(self, model, x_data, y_data, nb_epochs, nb_batch_size=32, validation_split=0.0543847722637661, callbacks=None): 
       # print ("x dari train_model:", x_data.shape)
        #print ("y dari train_model:", y_data.shape)
        callback_early_stop = EarlyStopping(monitor='val_f1_macro',
                                            patience=2)
        f1_score = F1_score()
        if self.one_shot:
            self.set_model_weights(model)
            history = model.fit(x_data,
                                y_data,
                                epochs=nb_epochs,
                                batch_size=nb_batch_size,
                                validation_split=validation_split,
                                callbacks=[f1_score, callback_early_stop],
                                verbose=0)
            
            self.update_weights(model)
            
        else:
            history = model.fit(x_data,
                                y_data,
                                epochs=nb_epochs,
                                batch_size=nb_batch_size,
                                validation_split=validation_split,
                                callbacks=[f1_score, callback_early_stop],
                                verbose=0)
        return history
