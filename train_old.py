"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data_old import DataSet
import time
import os.path
import my_callbacks
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'
def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=5000):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data_old', 'checkpoints', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True,save_weights_only=True)
    
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data_old', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5000)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data_old', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.8) // batch_size
    if load_to_memory:
        # Get data.
        if data_type=='multitype':
            X1,X2, y = data.get_all_sequences_in_memory_2stream('train', data_type)
            X1_test,X2_test, y_test = data.get_all_sequences_in_memory_2stream('test', data_type)
        elif data_type == '3stream':
            X1,X2,X3, y1,y2 = data.get_all_sequences_in_memory_2stream('train', data_type)
            X1_test,X2_test,X3_test, y1_test,y2_test = data.get_all_sequences_in_memory_2stream('test', data_type)
            #X1,X2,X3,y,y1,y2,X1_test,X2_test,X3_test, y1_test,y2_test=[],[],[],[],[],[],[],[],[],[],[]
        elif data_type in ['trajectory','point']:
            X1, y1,y2 = data.get_all_sequences_in_memory_2stream('train', data_type)
            X1_test, y1_test,y2_test = data.get_all_sequences_in_memory_2stream('test', data_type)
            #X1,X2,X3,y,y1,y2,X1_test,X2_test,X3_test, y1_test,y2_test=[],[],[],[],[],[],[],[],[],[],[]
    else:
        # Get generators.
        if data_type in ['trajectory','point']:
            generator = data.frame_generator_point(batch_size, 'train', data_type)
            val_generator = data.frame_generator_point(batch_size, 'test', data_type)
        elif data_type=='multitype':
            generator = data.frame_generator_2stream(batch_size, 'train', data_type)
            val_generator = data.frame_generator_2stream(batch_size, 'test', data_type)
        else:
            generator = data.frame_generator(batch_size, 'train', data_type)
            val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)
    print(model)
    print(len(data.classes))
    # Fit!
    if load_to_memory:
        # Use standard fit.
        if data_type=='multitype':
            rm.model.fit(
                [X1,X2],
                y,
                batch_size=batch_size,
                validation_data=([X1_test,X2_test], y_test),
                verbose=1,
                callbacks=[tb, early_stopper, csv_logger],
                epochs=nb_epoch)
        elif data_type=='3stream':
            histories3_2 = my_callbacks.Histories3_2()
            rm.model.fit(
                [X1,X2,X3],
                [y1,y2],
                batch_size=batch_size,
                validation_data=([X1_test,X2_test,X3_test], [y1_test,y2_test]),
                verbose=1,
                callbacks=[histories3_2, tb, early_stopper, csv_logger, checkpointer],
                epochs=nb_epoch)
        elif data_type in ['trajectory','point']:
            histories1_2 = my_callbacks.Histories1_2()
            rm.model.fit(
                X1,
                [y1,y2],
                batch_size=batch_size,
                validation_data=(X1_test, [y1_test,y2_test]),
                verbose=1,shuffle=True,
                callbacks=[histories1_2, tb, early_stopper, csv_logger, checkpointer],
                epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=10,
            #use_multiprocessing=True,
            workers=4)


def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, mlp, 2stream,3stream, point,trajectory
    model = '2stream'
    saved_model = None  # None or weights file
    class_limit = 2  # int, can be 1-101 or None
    seq_length = 20
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 10
    nb_epoch = 2000

    # Chose images or features and image shape based on network.

    if model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    elif model in ['2stream']:
        data_type = 'multitype'
        image_shape = None
    elif model in ['point']:
        data_type = 'point'
        image_shape = None
    elif model in ['trajectory']:
        data_type = 'trajectory'
        image_shape = None
    elif model in ['3stream']:
        data_type = '3stream'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
