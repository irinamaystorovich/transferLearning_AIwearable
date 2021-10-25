import tensorflow
from sklearn.metrics import confusion_matrix
import numpy as np
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
    # Define simulation parameters.
    data_length = 26
    epochs = 150

    # data = accumulate_data( '../tev-gestures-dataset/',
    #                           target='./gesture_data.csv' )

    # data = aggregate_data( './gesture_data.csv',
    #                           target='./aggregated_gesture_data.csv',
    #                           dir_path=None )

    # data = process_data( './aggregated_gesture_data.csv',
    #                      target='./processed_gesture_data.csv', dir_path=None )

    # Load in the pre-processed data.
    data = get_gesture_data( './processed_gesture_data.csv' )
    data = truncate_data( data, length=data_length )

    X_train, y_train, X_test, y_test = train_test_split( data,
                                                         test_subjects=[6, 7] )

    lstm_naive_model = create_lstm_model( num_classes=20, dropout=0.8, units=64,
                                         data_length=data_length )
    lstm_naive_model.fit( X_train, y_train, epochs=epochs )

    score = lstm_naive_model.evaluate( X_test, y_test )
    print( "test loss, test acc: ", score )
    cf_matrix = confusion_matrix( y_test, np.argmax( lstm_naive_model.predict( X_test ), axis=1 ) )
    make_confusion_matrix( cf_matrix, categories=[ '01', '02', '03', '04', '05',
                                                   '06', '07', '08', '09', '10',
                                                   '11', '12', '13', '14', '15',
                                                   '16', '17', '18', '19', '20' ],
                           figsize=[8,8])

if __name__ == "__main__":
    main()
