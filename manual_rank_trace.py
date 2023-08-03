import numpy as np
import src.live_plot as live_plot
import keyboard  # using module keyboard
import time
import pandas as pd
import time
import os
import sys
import subprocess

# plot for realtime params
size = 500
x_vec = np.linspace(0,size,size+1)[0:-1]
y_valence_vec = np.zeros(len(x_vec))
line1 = []
va_plot = []
current_value = 0

first_interaction=True
total_value = 0
annotations = []
total_values = []
current_feature = sys.argv[1]
filename = sys.argv[2]
total_time = 0
total_times = []
timer = None

c=0
save_path = 'output\\annotations\\' + filename + '_' + current_feature + '_annotator' + str(c) +'.csv'
while os.path.isfile(save_path):
    c+=1
    save_path = 'output\\annotations\\' + filename + '_' + current_feature + '_annotator' + str(c) +'.csv'
    
def countdown(t):
    
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
      
    print('Video starts now!')

loop_time = time.time()
while True:  # making a loop
    try:  # used try, so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('up') or keyboard.is_pressed('down'):  # if key 'q' is pressed 
            if keyboard.is_pressed('up'):
                current_value = 1
            else:
                current_value = -1
            
            
    except:
        
        continue  # if user pressed a key other than the given key the loop will break

    # real time plot of predictions , without filtering inputs
    if not first_interaction:
        total_value+=current_value
    
    y_valence_vec[-1] = total_value
    lines = live_plot.live_plotter(x_vec, [y_valence_vec], [line1])
    line1 = lines[0]
    y_valence_vec = np.append(y_valence_vec[1:],0.0)


    # update csv
    # here we filter input in order to prevent false multiple keypressed
    time_from_previous_input = time.time() - loop_time

    if current_value!= 0 and time_from_previous_input > 0.2:
        
        if first_interaction:
            countdown(5)
            total_time = 0
            current_value = 0
            first_interaction = False
            total_times = []
            p = subprocess.Popen(["C:\\Program Files\\VideoLAN\\VLC\\vlc.exe","videos\\current_experiment\\" + filename + ".mp4"])
            time.sleep(0.8)
            loop_time = time.time()
            continue
        else:
            total_time += time_from_previous_input
        total_times.append(total_time)
        annotations.append(current_value)
        total_values.append(total_value)
        pred = pd.DataFrame({"current_value": [a for a in annotations], "total_value": [v for v in total_values], "time": [t for t in total_times]})        
        pred.to_csv(save_path)
        loop_time = time.time()

    #reset current value
    current_value = 0
    if total_time + time.time() - loop_time >= 30:
        break
    
    time.sleep(0.01)
