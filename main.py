import numpy as np
import time
import tensorflow as tf
import os
import src.live_plot as live_plot
import src.utils as utils
import pandas as pd
import datetime


if __name__ == '__main__':
    model_name = '3D_CNN_pat100_lr1e-05'
    model = tf.keras.models.load_model('models/' + model_name + '_checkpoint')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #size = 100
    #x_vec = np.linspace(0,size,size+1)[0:-1]
    #y_valence_vec = np.zeros(len(x_vec))
    #y_arousal_vec = np.zeros(len(x_vec))
    #line1 = []
    #line2 = []
    #va_plot = []

    current_va_path = 'output\\current_va.csv'

    loop_time = time.time()

    debug = False
    gpu_scheduling = False
    prediction_modes = {
        0: 'fixed',
        1: 'from_file',
        2: 'screen_streaming'
    }
    prediction_choice = prediction_modes[1]
    fixed_predictions = [[0, 0]]
    videos_folder = 'C:\\Users\\franc\\PycharmProjects\\videogame-procedural-music\\VA_real_time\\videos\\'
    video_max_duration = 180

    videogame_choice = 'no_mans_sky'
    videos_folder = videos_folder + videogame_choice + '\\'


    available_videos = os.listdir(videos_folder)

    for video in available_videos:

        # skip some videos:
        if video in []:
            print('skipping ' + video)
            continue

        video_filename = video[:len(video)-4]
        video_filename_path = videos_folder + video_filename + '.mp4'
        count = 0
        fps = 6
        n_of_frames = 6
        predictions = []
        # define video lenght for analysis
        if prediction_choice == prediction_modes[1]:
            video_duration = utils.get_video_duration(video_filename_path)
            video_duration = np.min([video_max_duration, video_duration])

        while True:
            print('waiting for music generation...')
            while os.path.isfile(current_va_path) and gpu_scheduling:
                time.sleep(1)
            print('VA estimation started!')

            if prediction_choice == prediction_modes[0]:
                predictions = fixed_predictions
            
            elif prediction_choice == prediction_modes[1] or prediction_choice == prediction_modes[2]:
                
                if prediction_choice == prediction_modes[1]:
                    print('reading file: ', video_filename, ' from second: ', datetime.timedelta(seconds=count/fps), ' for ', n_of_frames/fps, 'seconds')
                    frames, count = utils.extract_frames_from_file(video_filename_path, fps, n_of_frames, count)

                elif prediction_choice == prediction_modes[2]:
                    # get n frames from screen at desired fps
                    frames = utils.stream_screen(fps=fps, n_of_frames=n_of_frames, debug=debug)

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
                predictions.append(model.predict(frames_ds)[0])
            else:
                print("predicion choice not valid! ")
                break

            #y_valence_vec[-1] = predictions[0][0]
            #y_arousal_vec[-1] = predictions[0][1]
            #lines = live_plot.live_plotter(x_vec, [y_valence_vec, y_arousal_vec], [line1, line2])
            #line1, line2 = lines[0], lines[1] 
            #va_plot = live_plot.va_2d_plot(predictions[0][0], predictions[0][1], va_plot)
            #y_valence_vec = np.append(y_valence_vec[1:],0.0)
            #y_arousal_vec = np.append(y_arousal_vec[1:],0.0)

            print(" - Valence: {}, Arousal: {}".format(predictions[-1][0], predictions[-1][1]))
            print("elapsed time: {} seconds".format(time.time() - loop_time))

            # saving complete prediction history
            pred = pd.DataFrame({'valence': [p[0] for p in predictions], 'arousal': [p[1] for p in predictions]})        
            pred.to_csv('output\\current_va.csv')
            loop_time = time.time()

            print('VA estimation completed...')

            if round(count + n_of_frames)/fps >= (video_duration - 1):
                print("video file analysis completed!")
                break
        
        # rename final file
        os.system('move /Y output\\current_va.csv output\\' + video_filename + '.csv')

