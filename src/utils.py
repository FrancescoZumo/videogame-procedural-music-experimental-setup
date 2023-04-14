
import cv2
import numpy as np
import pyautogui
import time
import tensorflow as tf

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
