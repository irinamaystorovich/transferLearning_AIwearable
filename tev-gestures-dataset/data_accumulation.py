import pandas as pd
import numpy as np
import os
from ast import literal_eval

if __name__ == "__main__":
    data_folder = './'
    users = 8
    gestures = 20
    directory_list = np.empty( ( users, gestures ), dtype=object )

    for i in range( 1, users+1 ):
        path = f"{data_folder}/U0{i}/"

        directories = []
        for filename in os.listdir( path ):
            f = os.path.join( path, filename )
            if os.path.isdir( f ):
                directories.append( f"{f}" )
                # print( f"    '{path}' '{filename}'\n" )
        directories = np.sort( np.array( directories ) )
        # print( np.sort( np.array( directories ) ) )
        directory_list[ i - 1 ] = directories
    print( directory_list )

    df = pd.DataFrame( columns=[ 'user', 'gesture', 'iteration',
                                 'millis', 'nano', 'timestamp',
                                 'accel0', 'accel1', 'accel2' ] )

    total_iterations = 0
    total_samples = 0
    for user in range( users ):
        print( f"User: {user}" )

        for gesture in range( gestures ):
            print( f"    Gesture: {gesture}" )

            files = []
            for filename in os.listdir( directory_list[ user ][ gesture ] ):
                path = os.path.join( directory_list[ user ][ gesture ], filename )
                if os.path.isfile( path ):
                    files.append( path )

            iterations = 0
            files = np.sort( np.array( files ) )
            for path in files:
                # print( f"        Path: {path}" )
                samples = 0
                for line in open( path ):
                    data = line.split()
                    data = [
                        user, gesture, iterations,
                        int( data[ 0 ] ), int( data[ 1 ] ), int( data[ 2 ] ),
                        float( data[ 3 ] ), float( data[ 4 ] ), float( data[ 5 ] )
                    ]
                    # print( data )
                    df.loc[ len( df.index ) ] = data
                    samples += 1
                total_samples += samples
                iterations += 1
            # print( f"        Iterations: {iterations}" )
            total_iterations += iterations

    print( f"{total_samples = }\n"
     		f"{total_iterations = }\n\n"
     		f"{total_samples / total_iterations = }" )

    exit(0)

    convert_dict = { 'user' : 'int', 'gesture' : 'int', 'iteration' : 'int',
                     'millis' : 'int', 'nano' : 'int', 'timestamp' : 'int' }
    df = df.astype( convert_dict )
    print( df )
    print( df.describe() )

    df.to_csv( 'gesture_data.csv', index=False )
