import os
import numpy as np
from keras import optimizers
from keras.layers import Dense, LSTM
from keras.models import Model
from keras.engine.input_layer import Input
from keras_preprocessing.sequence import pad_sequences  #untuk mengisi (padding) atau memangkas (truncating) sequence

from model_generator_101 import SearchSpace

from CONSTANTS import *


class Controller(SearchSpace):

    def __init__(self):

        self.max_len = MAX_ARCHITECTURE_LENGTH
        self.controller_lstm_dim = CONTROLLER_LSTM_DIM
        self.controller_optimizer = CONTROLLER_OPTIMIZER
        self.controller_lr = CONTROLLER_LEARNING_RATE
        self.controller_decay = CONTROLLER_DECAY
        self.controller_momentum = CONTROLLER_MOMENTUM
        self.use_predictor = CONTROLLER_USE_PREDICTOR

        # fie path of controller weights to be stoted at
        self.controller_weights = 'LOGS/controller_weights.h5'

        # init a list for storing sequence data that has been tested. So, no repeated the same sequence
        self.seq_data = []

        # inhereting from the search space
        super().__init__(TARGET_CLASSES)

        # number of classes for the controller (+1 for padding)
        self.controller_classes = len(self.vocab) + 1

    #for take sampling architecture seq
    def sample_architecture_sequences(self, model, number_of_samples):   #controller_model and sample per controller epoch
        # define values needed for sampling 
        final_layer_id = len(self.vocab) 
        #bilstm_id = final_layer_id - 1   
        vocab_idx = [0] + list(self.vocab.keys()) # dihitung dari 0
        samples = [] # for store the generated architecture sequence
        
        print("GENERATING ARCHITECTURE SAMPLES...")
        print('------------------------------------------------------')
        
        # while number of architectures sampled is less than required
        while len(samples) < number_of_samples:
            seed = [] # initialise the empty list for architecture sequence
            
            # while len of generated sequence is less than maximum architecture length
            while len(seed) < self.max_len:
                                
                # pad seq for corectly shaped input for controller model
                sequence = pad_sequences([seed], maxlen=self.max_len - 1, padding='pre') 
                sequence = sequence.reshape(1, 1, self.max_len - 1)
                
                # given the previous elements, get softmax distribution for the next element
                if self.use_predictor:
                    (probab, _) = model.predict(sequence)
                else:
                    probab = model.predict(sequence)
                probab = probab[0][0]                
                
                # sample the next element randomly given the probability of next elements (the softmax distribution)
                next = np.random.choice(vocab_idx, size=1, p=probab)[0]

                if next == final_layer_id and len(seed) == 0:
                    continue # if the first element is final layer, skip it and continue to next iteration
                
                if next == final_layer_id and len(seed) == 1: #conditional, follow max_len
                    continue
                
                if next == final_layer_id and len(seed) == self.max_len - 1:
                    seed.append(next)
                    break # if the next element is final layer, break the while loop
                
                if len(seed) == self.max_len - 1:
                    seed.append(final_layer_id)
                    break
                
                if not next == 0:
                    seed.append(next) 
                    
            if seed not in self.seq_data: # seed = [(8, relu), (16, tanh), (4, softmax)]
                samples.append(seed)      # sample = content of seed but < number of samples/SAMPLES_PER_CONTROLLER_EPOCH
                self.seq_data.append(seed) # seq_data = storage of all samples list
        return samples

    # simple LSTM controller
    def control_model(self, controller_input_shape, controller_batch_size):
        # input layer
        main_input = Input(shape=controller_input_shape, name='main_input')
        #print ("debug incompatible:", controller_input_shape)
        #print("debug incompatible shape:", controller_input_shape.shape)
        x = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)
        model = Model(inputs=[main_input], outputs=[main_output])
        return model

    # train controller
    def train_control_model(self, model, x_data, y_data, loss_func, controller_batch_size, nb_epochs):
        # get opt timzer for training
        if self.controller_optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=self.controller_lr, 
                                   decay=self.controller_decay, 
                                   momentum=self.controller_momentum, 
                                   clipnorm=1.0)
        else:
            optim = getattr(optimizers, self.controller_optimizer)(learning_rate=self.controller_lr, 
                                                                   decay=self.controller_decay, 
                                                                   clipnorm=1.0)
        # compile model depending on loss func and optimzer provided
        model.compile(optimizer=optim, 
                      loss={'main_output': loss_func}, run_eagerly=True)
                    # if use predictor, add mse loss for the predictor output
                    # loss={'main_output': loss_func, 'predictor_output': 'mse'}, 
                    # loss_weights={'main_output': 1, 'predictor_output': 1})
        
        # load controller weights if exists
        if os.path.exists(self.controller_weights):
            model.load_weights(self.controller_weights)
        
        # train the controller
        print("TRAINING CONTROLLER...")
        model.fit({'main_input': x_data},
                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes)},
                  epochs=nb_epochs,
                  batch_size=controller_batch_size,
                  verbose=0)
        # save controller weights
        model.save_weights(self.controller_weights)

    # for get architecture's validation f1 score 
    def hybrid_control_model(self, controller_input_shape, controller_batch_size):
        #main_input = Input(shape=controller_input_shape, batch_shape=controller_batch_size, name='main_input')         #AKU GANTI
        main_input = Input(shape=controller_input_shape, name='main_input')
        #print ("debug incompatible:", controller_input_shape)
        x = LSTM(self.controller_lstm_dim, return_sequences=True)(main_input)
        predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(x)
        main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)
        model = Model(inputs=[main_input], outputs=[main_output, predictor_output])  #outputs berupa multi-output model 
        return model

    def train_hybrid_model(self, model, x_data, y_data, pred_target, loss_func, controller_batch_size, nb_epochs):
        if self.controller_optimizer == 'sgd':
            optim = optimizers.SGD(learning_rate=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, clipnorm=1.0)
        else:
            optim = getattr(optimizers, self.controller_optimizer)(learning_rate=self.controller_lr, decay=self.controller_decay, clipnorm=1.0)
        model.compile(optimizer=optim,
                      loss={'main_output': loss_func, 'predictor_output': 'mse'},    #evaluating the difference between predicted and actual values in a machine learning model
                      loss_weights={'main_output': 1, 'predictor_output': 1})        #to control existence loss_func of each output
        if os.path.exists(self.controller_weights):
            model.load_weights(self.controller_weights)
        print("TRAINING CONTROLLER...")
        model.fit({'main_input': x_data},
                  {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes),
                   'predictor_output': np.array(pred_target).reshape(len(pred_target), 1, 1)},
                  epochs=nb_epochs,
                  batch_size=controller_batch_size,
                  verbose=0)
        model.save_weights(self.controller_weights)

    
    def get_predicted_accuracies_hybrid_model(self, model, seqs):  #model: model controller, seqs: list of architecture sequence
        pred_accuracies = []
        for seq in seqs:
            control_sequences = pad_sequences([seq], maxlen=self.max_len, padding='post', truncating='post')  #for make sure that seq memiliki length < max_len
            xc = control_sequences[:, :-1].reshape(len(control_sequences), 1, self.max_len - 1)
            (_, pred_accuracy) = [x[0][0] for x in model.predict(xc)]
            pred_accuracies.append(pred_accuracy[0])
        return pred_accuracies
