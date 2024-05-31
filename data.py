"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from processor import process_image
from keras.utils import to_categorical


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():

    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3)):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences')
        self.max_frames = 99999999999999999999  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()
        #self.contentclasses = self.get_contentclasses()


        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data():
        """Load our data from file."""
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            #if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
            if int(item[3]) >= self.seq_length \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            #print(item)
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            #return classes[:self.class_limit]
            return classes
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))
    #label_hot = label_encoded
        #print (label_hot)
        assert len(label_hot) == len(self.classes)

        return label_hot

    def get_contentclasses(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        contentclasses = []
        for item in self.data:
            if item[4] not in contentclasses:
                contentclasses.append(item[4])

        # Sort them.
        contentclasses = sorted(contentclasses)
        #print (contentclasses)

        return contentclasses

    def get_contentclass_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.contentclasses.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.contentclasses))


        #assert len(label_hot) == len(self.classes)

        return label_hot


    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test


    def get_all_sequences_in_memory(self, train_test, data_type):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        X, y1, y2 = [], [],[]
        random.shuffle(data)
        for row in data:
            #print (row[0])
            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise

            X.append(sequence)
            y1.append(self.get_class_one_hot(row[1]))
            y2.append(self.get_contentclass_one_hot(row[4]))
        return np.array(X), np.array(y1), np.array(y2)

    def get_all_sequences_in_memory_2stream(self, train_test, data_type):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))


        if data_type == 'multitype':
            X1,X2, y = [], [], []
            random.shuffle(data)
            for row in data:
                sequence = self.get_extracted_sequence('features', row)
                tra = self.get_trajectory(data_type, row)
                if sequence is None:
                    raise ValueError("Can't find sequence. Did you generate them?")
                        
            
                X1.append(sequence)
                X2.append(tra)
                y.append(self.get_class_one_hot(row[1]))

            return np.array(X1), np.array(X2), np.array(y)

        elif data_type == '3stream':
            X1,X2,X3, y1, y2 = [], [], [], [], []
            random.shuffle(data)
            for row in data:
                sequence = self.get_extracted_sequence('features', row)
                tra = self.get_trajectory(data_type, row)
                pcl = self.get_point(data_type, row)
                if sequence is None:
                            raise ValueError("Can't find sequence. Did you generate them?")

                X1.append(sequence)
                X2.append(pcl)
                X3.append(tra)
                y1.append(self.get_class_one_hot(row[1]))
                print (row[0])
                print (row[1])
                print (row[2])
                print (row[3])
                print (row[4])
                y2.append(self.get_contentclass_one_hot(row[4]))
            
            return np.array(X1), np.array(X2), np.array(X3),np.array(y1),np.array(y2)

        elif data_type == 'trajectory':
            X, y1,y2 = [], [],[]
            random.shuffle(data)
            for row in data:
                tra = self.get_trajectory(data_type, row)

                X.append(tra)
                y1.append(self.get_class_one_hot(row[1]))
                y2.append(self.get_contentclass_one_hot(row[4]))            
            #print (np.array(X).shape)
            return np.array(X),np.array(y1),np.array(y2)

        elif data_type == 'point':
            X, y1,y2 = [], [],[]
            random.shuffle(data)
            for row in data:
                pcl = self.get_point(data_type, row)

                X.append(pcl)
                y1.append(self.get_class_one_hot(row[1]))
                y2.append(self.get_contentclass_one_hot(row[4]))             
            #print (np.array(X).shape)
            return np.array(X),np.array(y1),np.array(y2)

    @threadsafe_generator
    def frame_generator(self, batch_size, train_test, data_type):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)
                    #print(sequence)
                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield (np.array(X), np.array(y))
            
    @threadsafe_generator
    def frame_generator_2stream(self, batch_size, train_test, data_type):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X1,X2, y = [], [],[]

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None
                tra = None
                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type is "multitype":
                   # Get the sequence from disk.
                    sequence = self.get_extracted_sequence('features', sample)
                    tra = self.get_trajectory(data_type, sample)
                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")

                else:
                    raise ValueError("data_type is wrong")

                X1.append(sequence)
                X2.append(tra)
                y.append(self.get_class_one_hot(sample[1]))

            yield ([np.array(X1), np.array(X2)], np.array(y))
    def frame_generator_point(self, batch_size, train_test, data_type):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
   
                tra = None
                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type is "point":
                   # Get the sequence from disk.
                    
                    tra = self.get_trajectory(data_type, sample)
                    
                else:
                    raise ValueError("data_type is wrong")

                
                X.append(tra)
        #print np.array(X1)
                y.append(self.get_class_one_hot(sample[1]))

            yield (np.array(X), np.array(y))
    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
            '-' + data_type + '.npy')
        #print ('get_sequence')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_trajectory(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = os.path.join('data', 'photo', sample[0], sample[1], sample[2], 'inter.txt')
        txtdata = np.loadtxt(path)
        #print ('get_trajectory')
        if os.path.isfile(path):
            #return np.delete(txtdata, [0,4,5,6,7], axis=1)
            return np.delete(txtdata, 0, axis=1)
        else:
            return path

    def get_point(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = os.path.join('data', 'photo', sample[0], sample[1], sample[2], 'inter_downsample.txt')
        txtdata = np.loadtxt(path)
        #print ('get_point')
        if os.path.isfile(path):
            return txtdata
        else:
            return path

    def get_frames_by_filename(self, filename, data_type):
        """Given a filename for one of our samples, return the data
        the model needs to make predictions."""
        # First, find the sample row.
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)

        if data_type == "images":
            # Get and resample frames.
            frames = self.get_frames_for_sample(sample)
            frames = self.rescale_list(frames, self.seq_length)
            # Build the image sequence
            sequence = self.build_image_sequence(frames)
        else:
            # Get the sequence from disk.
            sequence = self.get_extracted_sequence(data_type, sample)

            if sequence is None:
                raise ValueError("Can't find sequence. Did you generate them?")

        return sequence

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = os.path.join('data', 'photo', sample[0], sample[1], sample[2],'img')
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, '*jpg')))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split(os.path.sep)
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(self.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
