import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import src.utils as utils
import os
import numpy as np
from scipy.stats import sem

# PART 1: Process questionnaire results
questionnaire_df = pd.read_csv('subjective_evaluation\\sample_results.csv')

print("generating questionnaire results")

questions = [
    'Gameplay match', 
    'Emotion match', 
    'Immersion', 
    'Preference'
]
possible_answers = ['a', 'b', 'c', 'd']
question_results= pd.DataFrame(columns=["question", "answer"])

for q_index, question in enumerate(questions):

    curr_q__results = [0, 0, 0, 0]

    for row in questionnaire_df.itertuples():
        sorting = row[1]

        # obtain letter from vote on current question
        video_number_voted = row[2 + q_index]
        correspondent_letter = sorting[video_number_voted - 1]
        converted_answer = possible_answers.index(correspondent_letter)
        # update count current letter
        curr_q__results[converted_answer] += 1
    

    print(questions[q_index] + " Results:\n", curr_q__results)
    

    for index, answer in enumerate(possible_answers):
        for i in range(curr_q__results[index]):
            new_row = pd.DataFrame({"question": [question], "answer": [answer]})
            question_results = pd.concat([new_row,question_results.loc[:]]).reset_index(drop=True)

question_results['question'] = pd.Categorical(question_results['question'], questions)
sns.set(font_scale=1.7)
#sns.set(font="Times new roman")
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(13, 7))
sns.histplot(data=question_results, x="question", hue="answer", hue_order = possible_answers, multiple="dodge", shrink=.4, ax=ax)
ax.axes.set_title("Questionnaire Results",fontsize=20)
sns.move_legend(ax, "upper left",  bbox_to_anchor=(1, 1))
#plt.show()
fig.savefig("subjective_evaluation\\questionnaire_hist.pdf") 

#PART 2: process timeseries
plt.clf()
print("analyzing timeseries")

resampling_frequency = '100ms'
normalization = 'zscore'

available_features =['valence', 'arousal']
categories = {
    'a': 0, 
    'b': 1, 
    'c': 2, 
    'd': 3
}

annotation_files = os.listdir('output\\annotations')


#TODO: create function 

# for each video + feature combination, get all annotators and generate mean series
annotations_for_curr_video = []
for file in annotation_files:
    [name, feat, annotator] = file.split('_')
    if len(annotations_for_curr_video) == 0:
        annotations_for_curr_video.append(file)
        curr_pattern = name + feat
        continue
    if name + feat == curr_pattern:
        annotations_for_curr_video.append(file)
        continue
    # se arrivo fin qui, vuol dire che ho cambiato pattern, quindi generate file medio e poi aggiornare curr_pattern e file
    list_of_timeseries = []
    for n, nth_file_for_curr_video in enumerate(annotations_for_curr_video):
        [saved_name, saved_feat, saved_annotator] = nth_file_for_curr_video.split('_')
        saved_annotator = saved_annotator[0:len(saved_annotator)-4]
        curr_annotation_list = utils.process_annotation(saved_name, saved_feat, saved_annotator)
        curr_annotation_df = pd.DataFrame(curr_annotation_list, columns=['time','current_value','total_value'])
        # resample to common resampling_frequency for comparison
        curr_annotation_df['time'] = pd.to_timedelta(curr_annotation_df['time'],'s')
        curr_annotation_df = curr_annotation_df.set_index(curr_annotation_df['time'])['total_value'].resample(resampling_frequency).ffill()
        # store final series
        list_of_timeseries.append(curr_annotation_df.tolist())

        if n == 0:
            final_df = pd.DataFrame(curr_annotation_df.tolist(), columns=['total_value_' + str(n)])
            final_df = final_df.set_index(curr_annotation_df.index)
            continue
        final_df['total_value_' + str(n)] = curr_annotation_df.tolist()

    #TODO: converti utilizzo lista con utilizzo dataframe
    list_of_timeseries.append([])
    for t in range(len(list_of_timeseries[0])):
        ith_values = [list_of_timeseries[timeseries_idx][t] for timeseries_idx in range(len(list_of_timeseries) -1)]
        list_of_timeseries[-1].append(np.mean(ith_values))
    
    final_df = pd.DataFrame(list_of_timeseries[-1], columns=['total_value'])
    final_df = final_df.set_index(curr_annotation_df.index)

    # update variables for new file annotation list
    annotations_for_curr_video = []
    annotations_for_curr_video.append(file)
    curr_pattern = name + feat


# end of function TODO

plot_series = False
for chosen_feature in available_features:

    all_pred_distances = [[],[],[],[]]
    all_rand_distances = [[],[],[],[]]
    all_pred_RMSEs = [[],[],[],[]]
    all_rand_RMSEs = [[],[],[],[]]
    pred_distances_mean = []
    rand_distances_mean = []
    pred_RMSEs_mean = []
    rand_RMSEs_mean = []
    pred_distances_SEM = []
    rand_distances_SEM = []
    pred_RMSEs_SEM = []
    rand_RMSEs_SEM = []

    for file in annotation_files:
        [name, feat, annotator] = file.split('_')

        if feat != chosen_feature:
            continue 

        annotator = annotator[0:len(annotator)-4]
        curr_category = name[1]
        print("current file: ", file, " category: ", curr_category)
        distances, RMSEs = utils.compute_metrics(name, feat, annotator, resampling_frequency, normalization=normalization, plot_series=plot_series)


        print("distance pred vs annotation: ", distances[0], "\ndistance pred vs rand walk: ", distances[1])
        print("RMSE pred vs annotation: ", RMSEs[0], "\nRMSE pred vs rand walk: ", RMSEs[1], "\n")

        all_pred_distances[categories[curr_category]].append(distances[0])
        all_rand_distances[categories[curr_category]].append(distances[1])
        all_pred_RMSEs[categories[curr_category]].append(RMSEs[0])
        all_rand_RMSEs[categories[curr_category]].append(RMSEs[1])




    # calculate average values

    for category in categories:
        pred_distances_SEM.append(sem(all_pred_distances[categories[category]]))
        pred_distances_mean.append(np.mean(all_pred_distances[categories[category]]))
        rand_distances_SEM.append(sem(all_rand_distances[categories[category]]))
        rand_distances_mean.append(np.mean(all_rand_distances[categories[category]]))
        pred_RMSEs_SEM.append(sem(all_pred_RMSEs[categories[category]]))
        pred_RMSEs_mean.append(np.mean(all_pred_RMSEs[categories[category]]))
        rand_RMSEs_SEM.append(sem(all_rand_RMSEs[categories[category]]))
        rand_RMSEs_mean.append(np.mean(all_rand_RMSEs[categories[category]]))
        

    print('Affective Dimension: ', chosen_feature)
    print('prediction distances mean: ', pred_distances_mean)
    print('prediction distances SEM: ', pred_distances_SEM)
    print('rand_walk distances mean: ', rand_distances_mean)
    print('rand_walk distances SEM: ', rand_distances_SEM)
    print('prediction RMSEs mean : ', pred_RMSEs_mean)
    print('prediction RMSEs SEM: ', pred_RMSEs_SEM)
    print('rand_walk RMSEs mean: ', rand_RMSEs_mean)
    print('rand_walk RMSEs SEM: ', rand_RMSEs_SEM)

    plt.xticks([1, 2, 3, 4], ['Conditioned', 'Unconditioned', 'Original', 'None'])
    plt.title('Distance Confidence Intervals for ' + chosen_feature + '\n Normalization: ' + normalization)
    utils.plot_confidence_interval(1, all_pred_distances[0], color='#2187bb',label='prediction')
    utils.plot_confidence_interval(2, all_pred_distances[1], color='#2187bb',label='prediction')
    utils.plot_confidence_interval(3, all_pred_distances[2], color='#2187bb',label='prediction')
    utils.plot_confidence_interval(4, all_pred_distances[3], color='#2187bb',label='prediction')
    utils.plot_confidence_interval(1.1, all_rand_distances[0], color='#ff2187',label='rand walk', symbol='x')
    utils.plot_confidence_interval(2.1, all_rand_distances[1], color='#ff2187',label='rand walk', symbol='x')
    utils.plot_confidence_interval(3.1, all_rand_distances[2], color='#ff2187',label='rand walk', symbol='x')
    utils.plot_confidence_interval(4.1, all_rand_distances[3], color='#ff2187',label='rand walk', symbol='x')
    #plt.show()
    fig.savefig("subjective_evaluation\\distances_"+chosen_feature+"_"+normalization+".pdf")
    plt.clf()

#TODO : mediare annotazioni di stesso video fatte da persone diverse?
