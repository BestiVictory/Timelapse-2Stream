import keras
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy 
from keras.callbacks import Callback

class Histories1_1(Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0])
        yp = []

        for i in range(0, len(y_pred)):
            yp.append(y_pred[i][0])

        yt = []

        for x in self.validation_data[1]:
            yt.append(x[0])
        auc = roc_auc_score(yt, yp)
        self.aucs.append(auc)


        val_predict = (numpy.asarray(self.model.predict(
        self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average=None)
        _val_recall = recall_score(val_targ, val_predict, average=None)
        _val_precision = precision_score(val_targ, val_predict, average=None)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)






        print ('val-loss',logs.get('loss'), ' val-auc: ',auc,)
        print ('\n')
        print (' val-f-measure: ',_val_f1, ' val-recall: ',_val_recall, ' val-precision: ',_val_precision,)
        print ('\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return



class Histories3_1(Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0:3])
        yp = []

        for i in range(0, len(y_pred)):
            yp.append(y_pred[i][0])

        yt = []

        for x in self.validation_data[3]:
            yt.append(x[0])
        auc = roc_auc_score(yt, yp)
        self.aucs.append(auc)


        val_predict = (numpy.asarray(self.model.predict(self.validation_data[0:3]))).round()
        val_targ = self.validation_data[3]
        _val_f1 = f1_score(val_targ, val_predict, average=None)
        _val_recall = recall_score(val_targ, val_predict, average=None)
        _val_precision = precision_score(val_targ, val_predict, average=None)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)



        print ('val-loss',logs.get('loss'), ' val-auc: ',auc,)
        print ('\n')
        print (' val-f-measure: ',_val_f1, ' val-recall: ',_val_recall, ' val-precision: ',_val_precision,)
        print ('\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return






class Histories1_2(Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0])



        yp1 = []
        for i in range(0, len(y_pred[0])):
            yp1.append(y_pred[0][i][0])
        yt1 = []
        for x in self.validation_data[1]:
            yt1.append(x[0])
        auc1 = roc_auc_score(yt1, yp1)





        
        #val_predict = (numpy.asarray(self.model.predict(self.validation_data[0]))).round()

        val_predict1 = (numpy.asarray(y_pred[0])).round()
        val_targ1 = self.validation_data[1]
        #print val_targ
        _val_f1 = f1_score(val_targ1, val_predict1, average='weighted')
        _val_recall1 = recall_score(val_targ1, val_predict1, average='weighted')
        _val_precision1 = precision_score(val_targ1, val_predict1, average='weighted')
        print (val_predict1)
        print ('\n')
        print (val_targ1)
        print ('\n')
        val_predict2 = (numpy.asarray(y_pred[1])).round()
        val_targ2 = self.validation_data[2]
        print (val_predict2)
        print ('\n')
        print (val_targ2)
        auc2 = roc_auc_score(val_targ2, val_predict2, average='weighted')
        _val_f2 = f1_score(val_targ2, val_predict2, average='weighted')
        _val_recall2 = recall_score(val_targ2, val_predict2, average='weighted')
        _val_precision2 = precision_score(val_targ2, val_predict2, average='weighted')

        print ('val-loss',logs.get('loss'),' val-auc1: ',auc1,)
        print ('\n')
        print (' val-f-measure1: ',_val_f1, ' val-recall1: ',_val_recall1, ' val-precision1: ',_val_precision1,)
        print ('\n')
        print (' val-auc2: ',auc2,)
        print ('\n')
        print (' val-f-measure2: ',_val_f2, ' val-recall1: ',_val_recall2, ' val-precision1: ',_val_precision2,)
        print ('\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return



class Histories3_2(Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0:3])
        yp = []

        for i in range(0, len(y_pred[0])):
            yp.append(y_pred[0][i][0])

        yt = []

        for x in self.validation_data[3]:
            yt.append(x[0])
        auc = roc_auc_score(yt, yp)
        self.aucs.append(auc)


        #val_predict = (numpy.asarray(self.model.predict(self.validation_data[0:3]))).round()
        val_predict = (numpy.asarray(y_pred[0])).round()
        val_targ = self.validation_data[3]
        _val_f1 = f1_score(val_targ, val_predict, average=None)
        _val_recall = recall_score(val_targ, val_predict, average=None)
        _val_precision = precision_score(val_targ, val_predict, average=None)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)



        print ('val-loss',logs.get('loss'), ' val-auc: ',auc,)
        print ('\n')
        print (' val-f-measure: ',_val_f1, ' val-recall: ',_val_recall, ' val-precision: ',_val_precision,)
        print ('\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


