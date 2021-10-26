from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import numpy as np

from util import make_confusion_matrix
from util import accumulate_data
from util import aggregate_data
from util import process_data
from util import get_gesture_data
from util import truncate_data
from util import train_test_split
from util import print_results_table
from util import save_results

from models import create_lstm_model


def main():
    # Define simulation parameters.
    data_length = 26        # Length to truncate/pad samples to.
    epochs = 80             # Length of training.
    num_iterations = 50     # Number of iterations of fit/test.
    user_selections = []    # List to hold tuple of user splits (train/test/val)

    # Define naive model parameters.
    dropout_rate = 0.8
    lstm_units = 64


    data = get_data( data_length=data_length )

    cf_matrix_true = np.array( [] )
    cf_matrix_pred = np.array( [] )
    scores = []

    for i in range( num_iterations ):
        # Select the training, test, and validation subjects.
        # Get random ordering of subjects.
        subject_list = np.random.permutation( 8 )

        # Select the second from last as validation and last user as test.
        train_subjects = subject_list[ :-2 ].tolist()
        test_subjects = subject_list[ -2:-1 ].tolist()
        val_subjects = [ subject_list[ -1 ] ]

        user_selections.append( ( train_subjects, val_subjects, test_subjects ) )

        print( f"============================================================\n"
               f"Iteration {i+1}:\n"
               f"    Train Subjects:      {user_selections[ i ][ 0 ]}\n"
               f"    Validation Subjects: {user_selections[ i ][ 1 ]}\n"
               f"    Test Subjects:       {user_selections[ i ][ 2 ]}\n" )

        # Split the data into training, testing, and validation data and labels.
        X_train, y_train, \
        X_test, y_test,\
        X_val, y_val = train_test_split( data,
                                         train_subjects=train_subjects,
                                         test_subjects=test_subjects,
                                         val_subjects=val_subjects )
        validation_data = (X_val, y_val) if X_val is not None else None

        # Create the naive LSTM model without dropout.
        # early_stopping = EarlyStopping( monitor='val_loss' )
        lstm_naive_model = create_lstm_model( num_classes=20,
                                              dropout=dropout_rate,
                                              units=lstm_units,
                                              data_length=data_length )

        # Train the model on the training data.
        lstm_naive_model.fit( X_train, y_train, epochs=epochs,
                              validation_data=validation_data )

        # Test the model to see how well we did.
        score = lstm_naive_model.evaluate( X_test, y_test )
        scores.append( score )

        # Generate data for confusion matrix.
        cf_matrix_true = np.hstack( ( cf_matrix_true, y_test ) )
        cf_matrix_pred = np.hstack( ( cf_matrix_pred, np.argmax( lstm_naive_model.predict( X_test ), axis=1 ) ) )

        # Logging output for current iteration.
        print( "test loss, test acc: ", score )

    # Generate the confusion matrix.
    # cf_matrix = confusion_matrix( y_test, np.argmax( lstm_naive_model.predict( X_test ), axis=1 ) )
    cf_matrix = confusion_matrix( cf_matrix_true, cf_matrix_pred )


    # Print the results for each simulation run in a tabular format.
    print_results_table( scores, user_selections, cf_matrix )


    # Save results.
    save_results( scores, user_selections, cf_matrix, epochs,
                  filename=f'results_{epochs}-epochs_{num_iterations}-iterations', 
                  loc='./results/', filetype=0 )


    # Plot the confusion matrix.
    make_confusion_matrix( cf_matrix, categories=[ '01', '02', '03', '04', '05',
                                                   '06', '07', '08', '09', '10',
                                                   '11', '12', '13', '14', '15',
                                                   '16', '17', '18', '19', '20' ],
                           figsize=[8,8])


def get_data( data_length ):
    """
    Process and read in the gesture data.

    :param data_length: Desired length to truncate the accelerometer data to. If
    None, do not truncate/pad data.

    :return: Gesture data
    """

    ############################################################################
    #### Below are examples of how to process the data.
    # Accumulate data from original text files into a single (pandas dataframe).
    # Data frame columns are user, gesture, iteration (attempt number), millis,
    # nano, timestamp, accel0, accel1, accel2.
    # Each row is a separate sample from the accelerometer.
    # data = accumulate_data( '../tev-gestures-dataset/',
    #                           target='./gesture_data.csv' )

    # Take the data from accumulate_data and aggregate the iterations so that
    # each row is a single gesture attempt (iteration). Removes the millis,
    # nano, and timestamp.
    # data = aggregate_data( './gesture_data.csv',
    #                           target='./aggregated_gesture_data.csv',
    #                           dir_path=None )

    # After accumulating the data, scale it so that each accelerometer axis has
    # a zero mean and unit variance. This scaling is done per gesture attempt
    # and per axis (you can test a couple of samples to verify that the mean is
    # approximately zero and the variance is approximately 1).
    # data = process_data( './aggregated_gesture_data.csv',
    #                      target='./processed_gesture_data.csv', dir_path=None )
    ############################################################################


    # Load in the pre-processed data.
    data = get_gesture_data( './processed_gesture_data.csv' )

    # Truncate the data as desired. Comment out to test non-truncated data.
    # Make sure your model can handle variable length data!
    data = truncate_data( data, length=data_length )

    return data


if __name__ == "__main__":
    main()
