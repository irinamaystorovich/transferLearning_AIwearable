#%% md

#Dataset: https://tev.fbk.eu/technologies/smartwatch-gestures-dataset

#Research Paper: https://www.eurasip.org/Proceedings/Eusipco/Eusipco2014/HTML/papers/1569922319.pdf

#%% md

#<h2>Import dataset to get started<h2>

#%%
# import numpy as np
# import pandas as pd
# df=pd.read_csv('gesture_data.csv')
# print("data read into dataframe'")
# #%% md
# #<h2>Profile Data<h2>
# #%%
# samples = pd.unique(df['sample'])
# iteration = pd.unique(df['iteration'])
# gesture = pd.unique(df['gesture'])
# user = pd.unique(df['user'])
# dat=np.zeros((len(user),len(gesture),len(iteration),len(samples),3))
# data=[]
# for i in pd.unique(df['user']):
#     ges=[]
#     for j in pd.unique(df['gesture']):
#         its =[]
#         for k in pd.unique(df['iteration']):
#             its.append(df[(df['user'] == i) & (df['gesture'] == j) & (df['iteration'] == k)])
#             d=df[(df['user'] == i) & (df['gesture'] == j) & (df['iteration'] == k)][['accel0','accel1','accel2']].to_numpy()
#             d_sh = np.shape(d)
#             dat[i,j,k,:d_sh[0]]=d
#         ges.append(its)
#     data.append(ges)
#
#
# #%%

import numpy as np
from csv import writer
import os.path
from sklearn.metrics import confusion_matrix

def append_list_as_row(file_name, list_of_elem,new=False):
    # Open file in append mode or write
    mo='a'
    if new is True:
        mo = 'w'
    with open(file_name, mo, newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

dirname = os.path.abspath(os.curdir)

s_f = "AI_Art3.csv"
if os.path.isfile(s_f):
    print("Found file")
else:
    print("File not found,Creating profile file")
    heads = ['Model','Data Split','User test', 'User Val', 'Init Classes']
    for i in range(20):
        for j in range(20):
                heads.append('Class ' + str(i) + ':'+str(j))
    append_list_as_row(s_f, heads, new=True)

da=np.load('gesture.npz')
data=da[da.files[0]]
print("data read in using numpy file")

# #%% md
#
# <h2>Profile Data and divide into train test sets into <h2>

#%% train test data splits
for i in range(15):
    data_sh = np.shape(data)
    val_ts_user = np.random.choice(range(data_sh[0]),2,replace=False)
    ts_user = val_ts_user[0]
    val_user = val_ts_user[1:]
    print('Users randomly selected for test isolation')
    print(ts_user)
    print('Users randomly selected for validation')
    print(ts_user)
    train_set = data.copy()
    test_set = data[ts_user]
    val_set = data[val_user]
    train_set = np.delete(train_set,val_ts_user,axis=0)
    if len(np.shape(val_set))<5:
        val_set = np.expand_dims(val_set,axis=0)
    if len(np.shape(test_set))<5:
        test_set = np.expand_dims(test_set,axis=0)
    #%% split up gestures for transfer learning

    n=15
    init_gest = np.sort(np.random.choice(range(data_sh[2]),n,replace=False))
    print('Gestures randomly selected for initial training')
    print(init_gest)
    init_train_set = train_set[:,init_gest].copy()
    init_val_set = val_set[:,init_gest].copy()
    init_test_set = test_set[:,init_gest].copy()
    init_ts_sh=np.shape(init_test_set)
    init_val_sh=np.shape(init_val_set)
    init_tr_sh=np.shape(init_train_set)

    def product(list):
        p =1
        for i in list:
            p *= i
        return p

    init_train_set=np.reshape(init_train_set,(product(init_tr_sh[:-2]),init_tr_sh[-2],init_tr_sh[-1]))
    init_val_set=np.reshape(init_val_set,(product(init_val_sh[:-2]),init_val_sh[-2],init_val_sh[-1]))
    init_test_set=np.reshape(init_test_set,(product(init_ts_sh[:-2]),init_ts_sh[-2],init_ts_sh[-1]))

    #check for zero array samples and remove
    init_train_set= init_train_set[np.nonzero(np.sum(init_train_set[:,:-1],axis=(1,2)))[0]]
    init_val_set= init_val_set[np.nonzero(np.sum(init_val_set[:,:-1],axis=(1,2)))[0]]
    init_test_set= init_test_set[np.nonzero(np.sum(init_test_set[:,:-1],axis=(1,2)))[0]]

    np.random.shuffle(init_train_set)
    np.random.shuffle(init_val_set)
    np.random.shuffle(init_test_set)

    ts_sh=np.shape(test_set)
    tr_sh=np.shape(train_set)
    val_sh=np.shape(val_set)
    train_set=np.reshape(train_set,(product(tr_sh[:-2]),tr_sh[-2],tr_sh[-1]))
    val_set=np.reshape(val_set,(product(val_sh[:-2]),val_sh[-2],val_sh[-1]))
    test_set=np.reshape(test_set,(product(ts_sh[:-2]),ts_sh[-2],ts_sh[-1]))
    train_set = train_set[np.nonzero(np.sum(train_set[:,:-1],axis=(1,2)))[0]]
    val_set = train_set[np.nonzero(np.sum(val_set[:,:-1],axis=(1,2)))[0]]
    test_set = test_set[np.nonzero(np.sum(test_set[:,:-1],axis=(1,2)))[0]]
    np.random.shuffle(train_set)
    np.random.shuffle(test_set)


    init_train_set_x = init_train_set[:,:-1]
    init_val_set_x = init_val_set[:,:-1]
    init_test_set_x = init_test_set[:,:-1]
    init_train_set_y = init_train_set[:,-1,1]
    init_val_set_y = init_val_set[:,-1,1]
    init_test_set_y = init_test_set[:,-1,1]

    #%%

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import Dense, Convolution2D, MaxPool2D, Flatten, Input, Dropout
    from tensorflow.keras.layers import BatchNormalization
    #import system_query as sq
    from tensorflow.keras.models import Model
    from ttictoc import Timer

    ####Neural network nlayer
    def network_2d(X_train,Y_train,layers=4):
        sh_in = np.shape(X_train)
        sh_out = np.shape(Y_train)
        inputs_cnn = Input(shape=(sh_in[1],sh_in[2],1), name='inputs_cnn')
        conv1_n = Convolution2D(filters=128, kernel_size=(6,1), activation='relu', input_shape=(sh_in[1],sh_in[2],1))(inputs_cnn)
        batch1_n = BatchNormalization()(conv1_n)
        pool1_n = MaxPool2D(pool_size=(3), strides=(2), padding="same")(batch1_n)
        for i in range(1,layers):
            if i > 2:
                f_S = 64
            else:
                f_S = 32
            conv1_n = Convolution2D(f_S, (3,1), activation='relu')(pool1_n)
            batch1_n = BatchNormalization()(conv1_n)
            pool1_n = MaxPool2D(pool_size=(2), strides=(2), padding="same")(batch1_n)
        flatten = Flatten()(pool1_n)
        dense_end1 = Dense(64, activation='relu')(flatten)
        dense_end2 = Dense(32, activation='relu')(dense_end1)
        main_output = Dense(sh_out[1], activation='softmax', name='main_output')(dense_end2)
        model = Model(inputs=inputs_cnn, outputs=main_output)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005, decay=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_network_2d(model,X_train, y_train, X_test, y_test,n_train,b_size=1024,f_path=[],pt=False):
        if f_path==[]:
            callbacks = [EarlyStopping(monitor='val_loss', patience=100)]
        #else:
          #Qsys = sq.query_all()
            #callbacks = [EarlyStopping(monitor='val_loss', patience=75),
            #             ModelCheckpoint(filepath=f_path+Qsys['host'], monitor='val_loss', save_best_only=True)]
        t = Timer()
        t.start()
        history = model.fit(X_train, y_train, epochs=n_train, callbacks=callbacks, batch_size=b_size,
                        validation_data=(X_test, y_test))
        elapsed = t.stop()
        if pt:
            print(n_train,'Epoch time:', elapsed)
        return model, history

    def model_conf(mod, Mod, xtr, ytr, xv, yv, xts, yts, ts_us, val_us, init_g):
        dat = [Mod, 'train', str(ts_us), str(val_us), str(init_g)]
        m_y_pred = np.argmax(mod.predict(xtr), axis=1)
        m_conf = confusion_matrix(ytr, m_y_pred)
        #print("model conf matrix:", m_conf)
        m_conf=m_conf.flatten().tolist()
        value=dat+m_conf
        append_list_as_row(s_f, value, new=False)

        m_y_pred = np.argmax(mod.predict(xv), axis=1)
        m_conf = confusion_matrix(yv, m_y_pred)
        #print("model conf matrix:", m_conf)
        dat[1] = 'val'
        m_conf = m_conf.flatten().tolist()
        value = dat + m_conf
        append_list_as_row(s_f, value, new=False)

        m_y_pred = np.argmax(mod.predict(xts), axis=1)
        m_conf = confusion_matrix(yts, m_y_pred)
        #print("model conf matrix:\n", m_conf)
        dat[1] = 'test'
        m_conf = m_conf.flatten().tolist()
        value = dat + m_conf
        append_list_as_row(s_f, value, new=False)

    init_U = np.unique(init_train_set_y)

    for i in range(len(init_U)):
        init_train_set_y[init_train_set_y == init_U[i]] = i
        init_val_set_y[init_val_set_y == init_U[i]] = i
        init_test_set_y[init_test_set_y == init_U[i]] = i

    y_tr =tf.keras.utils.to_categorical(init_train_set_y, dtype='float32')
    y_v =tf.keras.utils.to_categorical(init_val_set_y, dtype='float32')
    y_ts =tf.keras.utils.to_categorical(init_test_set_y, dtype='float32')

    n_train = 500
    model = network_2d(init_train_set_x,y_tr,4)

    train_network_2d(model,init_train_set_x,y_tr, init_val_set_x,y_v,n_train)

    print("Evaluate initial transfer model 500 epochs with all layers trainable on less classes on test data")
    results = model.evaluate(init_test_set_x, y_ts, batch_size=1024)
    print("test loss, test acc:", results)
    model_conf(mod=model, Mod='init_model', xtr=init_train_set_x , ytr=init_train_set_y, xv=init_val_set_x, yv=init_val_set_y, xts=init_test_set_x, yts=init_test_set_y, ts_us=ts_user, val_us=val_user, init_g=init_gest)


    print('Initial model trained now popping final layer and replacing with full class')

    from tensorflow.keras.models import Model
    from tensorflow.keras.models import Sequential

    train_set_x = train_set[:,:-1]
    val_set_x = val_set[:,:-1]
    test_set_x = test_set[:,:-1]
    train_set_y = train_set[:,-1,1]
    val_set_y = val_set[:,-1,1]
    test_set_y = test_set[:,-1,1]
    y_train =tf.keras.utils.to_categorical(train_set_y, dtype='float32')
    y_val =tf.keras.utils.to_categorical(val_set_y, dtype='float32')
    y_test =tf.keras.utils.to_categorical(test_set_y, dtype='float32')
    sh_out = np.shape(y_test)
    f_model = Sequential()
    f_model.add(model)
    f_model.layers.pop()
    for layer in f_model.layers:
             layer.trainable=False
    f_model.add(Dense(sh_out[1], activation='softmax', name='main_output'))
    f_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005, decay=1e-5), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    f_model.summary()
    ft_model =tf.keras.models.clone_model(f_model)
    n_train = 500
    train_network_2d(f_model,train_set_x,y_train, val_set_x,y_val,n_train)
    print("Evaluate transfer model after 500 epochs training with only final layer trainable on test data")
    results = f_model.evaluate(test_set_x, y_test, batch_size=1024)
    print("test loss, test acc:", results)
    model_conf(mod=f_model, Mod='Fl Model', xtr=train_set_x , ytr=train_set_y, xv=val_set_x, yv=val_set_y, xts=test_set_x, yts=test_set_y, ts_us=ts_user, val_us=val_user, init_g=init_gest)

    for layer in ft_model.layers:
        layer.trainable = True

    ft_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005, decay=1e-5), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_network_2d(ft_model,train_set_x,y_train, val_set_x,y_val,n_train)
    print("Evaluate transfer model after 500 epochs training with all layer trainable on test data")
    results = ft_model.evaluate(test_set_x, y_test, batch_size=1024)
    print("test loss, test acc:", results)
    model_conf(mod=ft_model, Mod='Ft Model', xtr=train_set_x , ytr=train_set_y, xv=val_set_x, yv=val_set_y, xts=test_set_x, yts=test_set_y, ts_us=ts_user, val_us=val_user, init_g=init_gest)

    for layer in f_model.layers:
             layer.trainable=True

    n_train = 50
    train_network_2d(f_model,train_set_x,y_train, val_set_x,y_val,n_train)
    print("Evaluate transfer model after 50 more epochs training with all layers trainable now on test data")
    results = f_model.evaluate(test_set_x, y_test, batch_size=1024)
    print("test loss, test acc:", results)
    model_conf(mod=f_model, Mod='Fl-ft Model', xtr=train_set_x , ytr=train_set_y, xv=val_set_x, yv=val_set_y, xts=test_set_x, yts=test_set_y, ts_us=ts_user, val_us=val_user, init_g=init_gest)


    model_naive = network_2d(train_set_x,y_train,4)
    n_train = 250
    train_network_2d(model_naive,train_set_x,y_train, val_set_x,y_val,n_train)
    print("Evaluate naive model on after 250 epochs test data")
    results = model_naive.evaluate(test_set_x, y_test, batch_size=1024)
    print("test loss, test acc:", results)
    model_conf(mod=model_naive, Mod='Naive Model', xtr=train_set_x , ytr=train_set_y, xv=val_set_x, yv=val_set_y, xts=test_set_x, yts=test_set_y, ts_us=ts_user, val_us=val_user, init_g=init_gest)
