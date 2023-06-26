
import cv2
import numpy as np
import pyautogui
import time
import tensorflow as tf
import pandas as pd
import os
from dtaidistance import dtw
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statistics
from math import sqrt

SCREEN_SIZE = tuple(pyautogui.size())

def format_frames(input_frame, output_size=(224, 224)):
    """
    Pad and resize an image from a video.

    Args:
      input_frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
    """
    converted_frame = tf.image.convert_image_dtype(input_frame, tf.float32)
    resized_frame = tf.image.resize_with_pad(converted_frame, *output_size)
    return resized_frame

def get_video_duration(video_file_path):
    data = cv2.VideoCapture(video_file_path)
    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)
    
    # calculate duration of the video
    seconds = round(frames / fps)
    return seconds

def extract_frames_from_file(video_file_path, fps, n_of_frames, count):
    frames = []
    vidcap = cv2.VideoCapture(video_file_path)
    for _ in range(0, n_of_frames):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,((count + len(frames))* np.rint(1000/fps)))    # added this line 
        success, frame = vidcap.read()
        #print('Read frame number ', len(frames) ,' : ', success)
        frames.append(frame)
    count += len(frames)
    return frames, count


def stream_screen(fps, n_of_frames, debug=True):

    screenshot_time = time.time()
    result = []

    if debug:
        out = cv2.VideoWriter(
            'filename.mp4',
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            fps,
            SCREEN_SIZE
        )

    for screenshot in range(n_of_frames):
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        result.append(screenshot)
        if debug:
            out.write(screenshot) #TODO to be removed, just for debugging

        #uncomment to show screen
        #if cv2.waitKey(1) == ord('q'):
        #    cv2.destroyAllWindows()
        #    break
        # get desired frame rate
        time.sleep(max((1 / fps) - (time.time() - screenshot_time), 0))
        if debug:
            print('delay between frames: {}'.format(time.time() - screenshot_time))
            print('desired delay at {} fps: {}'.format(fps, 1 / fps))
        screenshot_time = time.time()
    
    return result


# TODO for now not working
def handler(signum, frame):
    res = input("\nCtrl-c was pressed. Do you really want to exit? [y/n]:  ")
    if res == 'y':
        exit(1)

def generate_random_walk_timeseries(step_n):
    # Define parameters for the walk
    dims = 1
    #step_n = 10000
    step_set = [-1, 0, 1]
    origin = np.zeros((1,dims))
    
    # Simulate steps in 1D
    step_shape = (step_n,dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    path = np.concatenate([origin, steps]).cumsum(0)
    start = path[:1]
    stop = path[-1:]

    return path

def process_annotation(name, feat, annotator):
    curr_path = "output\\annotations\\"+name+"_"+feat+"_"+annotator+".csv"
    if not os.path.isfile(curr_path):
        print("file does not esist: \n", curr_path)
        return None
    curr_annotation = pd.read_csv("output\\annotations\\"+name+"_"+feat+"_"+annotator+".csv")
    print("current annotation number of samples: ", curr_annotation.shape[0])
    # adding intermediate points every 0.2 seconds (annotation time window)
    processed_annotation = pd.DataFrame(columns=['time','current_value','total_value'])
    ann_time_window = 0.2
    time_axis = list(np.linspace(0, 30, int(30/(ann_time_window) + 1)))

    # generating final dataframe
    for row in curr_annotation.itertuples():
        if row.time not in time_axis:
            time_axis.append(row.time)
    time_axis.sort()

    # complete current value column
    processed_annotation['time'] = time_axis
    for idx, t in enumerate(processed_annotation['time'].values):
        if t in set(curr_annotation['time']):
            curr_ann_idx = curr_annotation.index[curr_annotation['time'] == t].tolist()[0]
            processed_annotation.at[idx, 'current_value'] = curr_annotation.at[curr_ann_idx, 'current_value']
        else:
            processed_annotation.at[idx, 'current_value'] = 0

    # complete total value column

    prev_val = 0
    for idx in range(processed_annotation.shape[0]):
        processed_annotation.at[idx, "total_value"] = prev_val + processed_annotation.at[idx, "current_value"]
        prev_val = processed_annotation.at[idx, "total_value"]
    return processed_annotation.values.tolist()


def compute_metrics(name, feat, annotator, resampling_frequency, normalization = None, plot_series=False):
    available_normalizations = [None, 'minmax', 'zscore']
    if normalization not in available_normalizations:
        print("normalization choice not valid: ", normalization, " is not in ", available_normalizations)
        quit()

    curr_annotation_df_list = process_annotation(name, feat, annotator)
    if curr_annotation_df_list is None:
        print("file does not exist")
        quit()

    curr_annotation_df = pd.DataFrame(curr_annotation_df_list, columns=['time','current_value','total_value'])

    
    if curr_annotation_df['total_value'].std() != 0 and normalization is not None:
        if normalization == 'zscore':
            # z-score normalization
            curr_annotation_df['total_value'] = (curr_annotation_df['total_value'] - curr_annotation_df['total_value'].mean()) / curr_annotation_df['total_value'].std()
        elif normalization == 'minmax':
            # min max normalization (range 0-1)
            curr_annotation_df['total_value'] = (curr_annotation_df['total_value'] - curr_annotation_df['total_value'].min())/(curr_annotation_df['total_value'].max() - curr_annotation_df['total_value'].min())
        pass

    curr_prediction_df = pd.read_csv("output\\"+name+".csv", names=['time', 'valence', 'arousal'], header=None)
    curr_prediction_df = curr_prediction_df.tail(-1)
    curr_prediction_df = curr_prediction_df.astype('float64')
    if normalization == 'zscore':
        # z-score normalization
        curr_prediction_df[feat] = (curr_prediction_df[feat] - curr_prediction_df[feat].mean()) / curr_prediction_df[feat].std()
    elif normalization == 'minmax':
        # min max normalization (range 0-1)
        curr_prediction_df[feat] = (curr_prediction_df[feat] - curr_prediction_df[feat].min()) / (curr_prediction_df[feat].max() - curr_prediction_df[feat].min())

    curr_rand_walk = generate_random_walk_timeseries(step_n=curr_prediction_df.shape[0])
    curr_rand_walk_df = pd.DataFrame(curr_rand_walk, columns=['total_value'])
    if normalization == 'zscore':
        # z-score normalization
        curr_rand_walk_df['total_value'] = (curr_rand_walk_df['total_value'] - curr_rand_walk_df['total_value'].mean()) / curr_rand_walk_df['total_value'].std()
    elif normalization == 'minmax':
        # min max normalization (range 0-1)
        curr_rand_walk_df['total_value'] = (curr_rand_walk_df['total_value'] - curr_rand_walk_df['total_value'].min()) / (curr_rand_walk_df['total_value'].max() - curr_rand_walk_df['total_value'].min())

    # resample to common resampling_frequency for comparison

    curr_annotation_df['time'] = pd.to_timedelta(curr_annotation_df['time'],'s')
    curr_annotation_df = curr_annotation_df.set_index(curr_annotation_df['time'])['total_value'].resample(resampling_frequency).ffill()


    curr_prediction_df['time'] = pd.to_timedelta(curr_prediction_df['time'],'s')
    curr_prediction_df = curr_prediction_df.set_index(curr_prediction_df['time'])[feat].resample(resampling_frequency).ffill()


    curr_rand_walk_df['time'] = pd.to_timedelta(range(0, 31),'s')
    curr_rand_walk_df = curr_rand_walk_df.set_index(curr_rand_walk_df['time'])['total_value'].resample(resampling_frequency).ffill()

    if plot_series:
        p = plt.plot(curr_annotation_df, label='annotator ' + annotator)
        p = plt.plot(curr_prediction_df, label='prediction')
        p = plt.plot(curr_rand_walk_df, label='random_walk')
        plt.legend()
        plt.show()
    
    # compute metrics
    distance_pred = dtw.distance(curr_annotation_df[0:curr_prediction_df.shape[0]], curr_prediction_df)
    distance_rand_walk = dtw.distance(curr_annotation_df[0:curr_prediction_df.shape[0]], curr_rand_walk_df[0:curr_prediction_df.shape[0]])

    RMSE_pred = sqrt(mean_squared_error(curr_annotation_df[0:curr_prediction_df.shape[0]], curr_prediction_df))
    RMSE_rand_walk = sqrt(mean_squared_error(curr_annotation_df[0:curr_prediction_df.shape[0]], curr_rand_walk_df[0:curr_prediction_df.shape[0]]))

    return [distance_pred, distance_rand_walk], [RMSE_pred, RMSE_rand_walk]


def plot_confidence_interval(x, values, z=1.96, color='#2187bb', label='', symbol = 'o',horizontal_line_width=0.1):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, symbol, color=color, label=label)

    return mean, confidence_interval


