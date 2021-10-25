import tensorflow
from sklearn.metrics import confusion_matrix
import numpy
import matplotlib.pyplot as plt

from util import make_confusion_matrix
from util import accumulate_data
from util import aggregate_data
from util import process_data
from util import get_gesture_data
from util import truncate_data
from util import train_test_split

from models import create_lstm_model


def main():
    # data = accumulate_data( '../tev-gestures-dataset/',
    #                           target='./gesture_data.csv' )

    # data = aggregate_data( './gesture_data.csv',
    #                           target='./aggregated_gesture_data.csv',
    #                           dir_path=None )

    # data = process_data( './aggregated_gesture_data.csv',
    #                      target='./processed_gesture_data.csv', dir_path=None )

    # Load in the pre-processed data.
    data = get_gesture_data( './processed_gesture_data.csv' )
    data = truncate_data( data, length=19 )

    X_train, y_train, X_test, y_test = train_test_split( data,
                                                         test_subjects=[6, 7] )

    lstm_naive_model = create_lstm_model( num_classes=20, dropout=0.8, units=64,
                                         data_length=19 )
    lstm_naive_model.fit( X_train, y_train, epochs=250 )

    score = lstm_naive_model.evaluate( X_test, y_test )
    print( "test loss, test acc: ", score )

if __name__ == "__main__":
    main()
