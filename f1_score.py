from keras import backend as K
import numpy as np
from keras.callbacks import Callback

class F1_score(Callback):
    
    @staticmethod
    def categorical_f1_score_for_class(y_true, y_pred, class_i, dtype=None):  #class_i is index of label
        pred_labels = K.argmax(y_pred, axis=-1)
        tp = K.sum(y_true[:, :, class_i] * K.cast(K.equal(pred_labels, class_i), 'float32' if not dtype else dtype))
        all_p_detections = K.sum(K.cast(K.equal(pred_labels, class_i), 'float32' if not dtype else dtype))
        all_p_true = K.sum(y_true[:, :, class_i])
    
        precision = tp / (all_p_detections + K.epsilon())
        recall = tp / (all_p_true + K.epsilon())
        f_score = 2 * precision * recall / (precision + recall + K.epsilon())
        return f_score

    @staticmethod
    def f1_FIX(y_true, y_pred, dtype=None):
        return F1_score.categorical_f1_score_for_class(y_true, y_pred, 0, dtype)
    
    @staticmethod
    def f1_SACC(y_true, y_pred, dtype=None):
        return F1_score.categorical_f1_score_for_class(y_true, y_pred, 1, dtype)

    @staticmethod
    def f1_SP(y_true, y_pred, dtype=None):
        return F1_score.categorical_f1_score_for_class(y_true, y_pred, 2, dtype)
    
    @staticmethod
    def f1_NOISE(y_true, y_pred, dtype=None):
        return F1_score.categorical_f1_score_for_class(y_true, y_pred, 3, dtype)
    
    # create function for calculate f1 score macro
    @staticmethod
    def f1_macro(y_true, y_pred, dtype=None):
        f1_FIX_score = F1_score.f1_FIX(y_true, y_pred, dtype)
        f1_SACC_score = F1_score.f1_SACC(y_true, y_pred, dtype)
        f1_SP_score = F1_score.f1_SP(y_true, y_pred, dtype)
        f1_NOISE_score = F1_score.f1_NOISE(y_true, y_pred, dtype)
        #print("cek class_i f1 score")
        #print("y_true", y_true)
        #print("y_pred", y_pred)
        f1_macro = (f1_FIX_score + f1_SACC_score + f1_SP_score + f1_NOISE_score) / 4
        return f1_macro