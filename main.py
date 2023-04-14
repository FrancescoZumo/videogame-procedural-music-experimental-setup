import numpy as np
import time
import tensorflow as tf
import os
import src.live_plot as live_plot
import src.utils as utils
import pandas as pd


if __name__ == '__main__':
    model_name = '3D_CNN_pat100_lr1e-05'
    model = tf.keras.models.load_model('models/' + model_name + '_checkpoint')
    # model.summary()
    # convert model to tensorflowlite

    # Convert the model (TODO not working right now)
    #converter = tf.lite.TFLiteConverter.from_saved_model('models/' + model_name + '_checkpoint') # path to the SavedModel directory
    # model = converter.convert()

    # Save the model.
    #with open('model.tflite', 'wb') as f:
    #    f.write(tflite_model)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    size = 100
    x_vec = np.linspace(0,size,size+1)[0:-1]
    y_valence_vec = np.zeros(len(x_vec))
    y_arousal_vec = np.zeros(len(x_vec))
    line1 = []
    line2 = []
    va_plot = []

    current_va_path = 'output\\current_va.csv'

    loop_time = time.time()

    debug = False
    gpu_scheduling = True
    use_fixed_predictions = True
    fixed_predictions = [[1, 0]]

    while True:
        print('waiting for music generation...')
        while os.path.isfile(current_va_path) and gpu_scheduling:
            time.sleep(1)
        print('VA estimation started!')

        if not use_fixed_predictions:

            # get n frames from screen at desired fps
            frames = utils.stream_screen(fps=6, n_of_frames=6, debug=debug)

            # convert frames to tensors
            # print(type(frames[0]))
            for i, frame in enumerate(frames):
                frames[i] = utils.format_frames(frame)
            # print(frames[0].shape)

            frames = np.array(frames)[..., [2, 1, 0]]
            
            # reduce dimensionality in case of single frame
            if frames.shape[0] == 1:
                frames = frames[0, :, :, :]
            frames_ds = tf.data.Dataset.from_tensors(frames)
            frames_ds = frames_ds.batch(1)

            print("Generate predictions")
            predictions = model.predict(frames_ds)

        else:
            predictions = fixed_predictions


        #y_valence_vec[-1] = predictions[0][0]
        #y_arousal_vec[-1] = predictions[0][1]
        #lines = live_plot.live_plotter(x_vec, [y_valence_vec, y_arousal_vec], [line1, line2])
        #line1, line2 = lines[0], lines[1] 
        #va_plot = live_plot.va_2d_plot(predictions[0][0], predictions[0][1], va_plot)
        
        y_valence_vec = np.append(y_valence_vec[1:],0.0)
        y_arousal_vec = np.append(y_arousal_vec[1:],0.0)

        print("Valence: {}, Arousal: {}".format(predictions[0][0], predictions[0][1]))
        print("elapsed time: {} seconds".format(time.time() - loop_time))

        pred = pd.DataFrame({'valence': [predictions[0][0]], 'arousal': [predictions[0][1]]})
        pred.to_csv('output\\current_va.csv')
        loop_time = time.time()

        print('VA estimation completed...')


