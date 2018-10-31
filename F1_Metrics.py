from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
class F1_Metrics(Callback):
    def on_train_begin(self, logs={}):
        if not ('f1' in self.params['metrics']):
            self.params['metrics'].append('f1')
    def on_epoch_end(self, epoch, logs={}):

        logs['f1'] = float('-inf')
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        if (self.validation_data):
            logs['f1'] = f1_score(val_targ, val_predict, average='macro')
           
def get_callbacks(filepath, patience=5):
    early_stopping = EarlyStopping(monitor="val_f1", patience=patience, mode="max")
    model_checkpoint = ModelCheckpoint(filepath=filepath, monitor="val_f1", mode="max", verbose=1, save_best_only=True)
    return [early_stopping, model_checkpoint]   
    
    
_callbacks = get_callbacks(checkpoint_path)
metrics = F1_Metrics()
_callbacks.append(metrics)
#usage:model.fit(train_content, train_label, epochs=30, batch_size=128, callbacks=_callbacks, validation_data=(valid_content, valid_label), verbose=1)
    
