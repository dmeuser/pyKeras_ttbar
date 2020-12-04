import keras.backend as kb

def customLoss(y_true,y_pred):
    return kb.mean(kb.sum(kb.square(y_true-y_pred)*kb.square(y_true-y_pred)))
