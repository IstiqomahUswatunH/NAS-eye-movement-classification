import pickle
import keras.backend as K
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import wandb
import tensorflow as tf
from CONSTANTS import *
from controller import Controller
from model_generator_101 import ModelGenerator

from utils import *


class MODEL(Controller):

    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.target_classes = TARGET_CLASSES
        self.controller_sampling_epochs = CONTROLLER_SAMPLING_EPOCHS
        self.samples_per_controller_epoch = SAMPLES_PER_CONTROLLER_EPOCH
        self.controller_train_epochs = CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = ARCHITECTURE_TRAINING_EPOCHS
        self.controller_loss_alpha = CONTROLLER_LOSS_ALPHA

        self.data = []
        self.nas_data_log = 'LOGS/nas_data.pkl'
        clean_log()

        super().__init__()

        self.model_generator = ModelGenerator()

        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (1, MAX_ARCHITECTURE_LENGTH - 1)
        if self.use_predictor:
            self.controller_model = self.hybrid_control_model(self.controller_input_shape, self.controller_batch_size)
        else:
            self.controller_model = self.control_model(self.controller_input_shape, self.controller_batch_size)

    def create_architecture(self, sequence):
        if self.target_classes == 2:
            self.model_generator.loss_func = 'binary_crossentropy'
        else:
            self.model_generator.loss_func = 'categorical_crossentropy'
       
        model = self.model_generator.create_model(sequence, input_shape=np.shape(self.x[0]))
        # print("np.shape dari x index 0:", np.shape(self.x[0]))
        model = self.model_generator.compile_model(model)
        return model

    # train the generated model
    def train_architecture(self, model):
        x = self.x
        y = self.y
        # train the model
        history = self.model_generator.train_model(model, x, y, self.architecture_train_epochs)
        wandb_val_f1_macro = history.history['val_f1_macro'][-1]
        wandb_val_f1_FIX = history.history['val_f1_FIX'][-1]
        wandb_val_f1_SACC = history.history['val_f1_SACC'][-1]
        wandb_val_f1_SP = history.history['val_f1_SP'][-1]
        #wandb_val_f1_NOISE = history.history['val_f1_NOISE'][-1]
            
        wandb.log({'val_f1_macro': wandb_val_f1_macro, 'val_f1_FIX': wandb_val_f1_FIX, 'val_f1_SACC': wandb_val_f1_SACC, 'val_f1_SP': wandb_val_f1_SP})

        return history

    # stroing the training metrics
    def append_model_metrics(self, sequence, history, pred_accuracy=None):
        # if the models are trained only for a single epoch
        if len(history.history['val_f1_macro']) == 1:
            # if an accuracy predictor is used
            if pred_accuracy:
                self.data.append([sequence,
                                  history.history['val_f1_macro'][0],
                                  pred_accuracy])
            # if no accuracy predictor is used
            else:
                self.data.append([sequence,
                                  history.history['val_f1_macro'][0]])
            print('val f1 score macro: ', history.history['val_f1_macro'][0])
        else:
            val_average_f1 = np.ma.average(history.history['val_f1_macro'],
                                    weights=np.arange(1, len(history.history['val_f1_macro']) + 1),
                                    axis=-1)
            if pred_accuracy:
                self.data.append([sequence,
                                  val_average_f1,
                                  pred_accuracy])
            else:
                self.data.append([sequence,
                                  val_average_f1])
            print('average of val f1 score macro: ', val_average_f1)

    # preparing data for controller
    def prepare_controller_data(self, sequences):
        # pad generated sequences to maximum length
        controller_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        
        xc = controller_sequences[:, :-1].reshape(len(controller_sequences), 1, self.max_len - 1)
        yc = to_categorical(controller_sequences[:, -1], self.controller_classes)
        val_acc_target = [item[1] for item in self.data]
        return xc, yc, val_acc_target

    def get_discounted_reward(self, rewards):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards)):
            running_add = 0.
            exp = 0.
            for r in rewards[t:]:
                running_add += self.controller_loss_alpha**exp * r
                exp += 1
            discounted_r[t] = running_add
        discounted_r = (discounted_r - discounted_r.mean()) / discounted_r.std()
        return discounted_r

    def custom_loss(self, target, output):
        baseline = 0.5
        reward = np.array([item[1] - baseline for item in self.data[-self.samples_per_controller_epoch:]]).reshape(
            self.samples_per_controller_epoch, 1)
        discounted_reward = self.get_discounted_reward(reward)
        loss = - K.log(output) * discounted_reward[:, None]
        return loss

    def train_controller(self, model, x, y, pred_accuracy=None):
        if self.use_predictor:
            self.train_hybrid_model(model,
                                    x,
                                    y,
                                    pred_accuracy,
                                    self.custom_loss,
                                    len(self.data),
                                    self.controller_train_epochs)
        else:
            self.train_control_model(model,
                                     x,
                                     y,
                                     self.custom_loss,
                                     len(self.data),
                                     self.controller_train_epochs)

    def search(self):
        for controller_epoch in range(self.controller_sampling_epochs):
            print('------------------------------------------------------------------')
            print('                       CONTROLLER EPOCH: {}'.format(controller_epoch))
            print('------------------------------------------------------------------')
            sequences = self.sample_architecture_sequences(self.controller_model, self.samples_per_controller_epoch)
            if self.use_predictor:
                pred_accuracies = self.get_predicted_accuracies_hybrid_model(self.controller_model, sequences)
            for i, sequence in enumerate(sequences):
                print('Architecture: ', self.decode_sequence(sequence))
                model = self.create_architecture(sequence)
                history = self.train_architecture(model)
                if self.use_predictor:
                    self.append_model_metrics(sequence, history, pred_accuracies[i])
                else:
                    self.append_model_metrics(sequence, history)
                print('------------------------------------------------------')
            xc, yc, val_acc_target = self.prepare_controller_data(sequences)
            self.train_controller(self.controller_model,
                                  xc,
                                  yc,
                                  val_acc_target[-self.samples_per_controller_epoch:])
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
        log_event()
        return self.data
