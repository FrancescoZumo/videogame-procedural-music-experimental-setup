import matplotlib.pyplot as plt
import numpy as np

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

def live_plotter(x_value,y_data,lines,identifier='',pause_time=0.01):
    if lines[0]==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(7.5,3))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        for i, line in enumerate(lines):
            lines[i], = ax.plot(x_value,y_data[i],'-',alpha=0.8)        
        #update plot label/title
        plt.ylabel('values')
        plt.legend(['valence', 'arousal'], loc='upper left')
        plt.title('Timeseries: {}'.format(identifier))
        plt.show()
    for i, line in enumerate(lines):
        # after the figure, axis, and line are created, we only need to update the y-data
        line.set_ydata(y_data[i])
        # adjust limits if new data goes beyond bounds
        if np.min(y_data[i])<=line.axes.get_ylim()[0] or np.max(y_data[i])>=line.axes.get_ylim()[1]:
            plt.ylim([np.min(y_data)-np.std(y_data),np.max(y_data)+np.std(y_data)])
        # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
        plt.pause(pause_time)
        lines[i] = line
    
    # return line so we can update it again in the next iteration
    return lines

def va_2d_plot(x_value,y_value,line1,identifier='',pause_time=0.1):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_value,y_value,'-o',alpha=0.8)        
        #update plot label/title
        plt.ylabel('Arousal')
        plt.ylabel('Valence')
        plt.title('2D Plot: {}'.format(identifier))
                # adjust limits if new data goes beyond bounds
        plt.ylim([-1, 1])
        plt.xlim([-1, 1])
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y_value)
    plt.ylim([-1, 1])
    plt.xlim([-1, 1])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1