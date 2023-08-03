import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import src.utils as utils
import os
import numpy as np
from scipy.stats import sem
from scipy.stats import normaltest
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from math import isnan
from contextlib import redirect_stdout
import itertools



# PART 1: Process questionnaire results
questionnaire_df = pd.read_csv('subjective_evaluation\\questionnaire_results.csv')
n_participants = questionnaire_df.shape[0]
print("generating questionnaire results")

questions = [
    'Gameplay match', 
    'Emotion match', 
    'Immersion', 
    'Preference'
]
possible_answers = ['a', 'b', 'c', 'd']
question_results= pd.DataFrame(columns=["Question", "answer"])

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
            new_row = pd.DataFrame({"Question": [question], "answer": [answer]})
            question_results = pd.concat([new_row,question_results.loc[:]]).reset_index(drop=True)

question_results['Question'] = pd.Categorical(question_results['Question'], questions)
question_results = question_results.sort_values(by='answer', ascending=True)

question_results['answer'] = question_results['answer'].replace({'a':'Conditioned', 'b':'Unconditioned','c':'Original', 'd':'None'})
sns.set(font_scale=1.8)
#sns.set(font="Times new roman")
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(13, 7.5))
sns.histplot(data=question_results, x="Question", hue="answer", multiple="dodge", shrink=.4, ax=ax)
#ax.axes.set_title("Questionnaire Results",fontsize=20)
ax.yaxis.set_ticks(np.arange(0, n_participants, n_participants/4), ['0%', '25%', '50%', '75%'])
sns.move_legend(ax, "upper left",  bbox_to_anchor=(0.7, 1))

#plt.show()
fig.savefig("subjective_evaluation\\questionnaire_hist.pdf") 

#PART 2: process timeseries
plt.clf()
print("analyzing timeseries")

resampling_frequency = '100ms'
normalization = 'zscore'
detrend = False
plot_series = False
subfolder = None
filter_files = True
threshold_percentile = 80

output_stats_file = "subjective_evaluation\\subfolder_" + str(subfolder)+"_normalization_"+str(normalization)+".txt"
output_plot_filenames = output_stats_file[22:len(output_stats_file)-4]
print("redirecting output stats to output file: ", output_stats_file)

if os.path.isfile(output_stats_file):
    os.system("del /q " + output_stats_file)

available_features =['valence', 'arousal']
categories = {
    'a': 0, 
    'b': 1, 
    'c': 2, 
    'd': 3
}

annotation_files = os.listdir('output\\annotations')

output_dir = 'output\\annotations\\mean'

# generate average annotation for each file and dimension: 40 videos * len([valence, arousal]) = 80
utils.generate_mean_annotations(annotation_files, output_dir, resampling_frequency, overwrite=False)

if subfolder is not None:
    files_to_use = os.listdir('output\\annotations' + '\\' + subfolder)
else:
    files_to_use = os.listdir('output\\annotations')


files_to_use_filtered = []
for file in files_to_use:

    if file[-4:len(file)] != '.csv':
        continue

    if subfolder is not None:
        df = pd.read_csv('output\\annotations' + '\\' + subfolder + '\\' + file)
    else:
        df = pd.read_csv('output\\annotations' + '\\'+ file)

    # remove files with only zeros
    if df['current_value'].std() == 0 and df['current_value'].mean() == 0:
        print("skipping file with values: ", df['current_value'])
    else:
        files_to_use_filtered.append(file)
        
if filter_files:
    print("using only: ", len(files_to_use_filtered), "files ")
else:
    files_to_use_filtered = files_to_use

ann_stats = utils.get_annotations_stats(files_to_use_filtered, subfolder = subfolder)
print("annptation stats: \n", ann_stats)

pred_stats = utils.get_pred_stats(['valence', 'arousal'])
#pred_stats_val = utils.get_pred_stats(['valence'])
#pred_stats_aro = utils.get_pred_stats(['arousal'])
print("prediction stats: \n", pred_stats)

all_pred_distances_byfeat = []
all_pred_RMSEs_byfeat = []
all_pred_PCORRs_byfeat = []


for i_feat, chosen_feature in enumerate(available_features):

    all_pred_distances = [[],[],[],[]]
    all_pred_RMSEs = [[],[],[],[]]
    all_pred_PCORRs = [[],[],[],[]]
    pred_distances_mean = []
    pred_RMSEs_mean = []
    pred_PCORRs_mean = []
    pred_distances_SEM = []
    pred_RMSEs_SEM = []
    pred_PCORRs_SEM = []
    
    for i, file in enumerate(files_to_use_filtered):

        if file[-4:len(file)] != '.csv':
            continue
        [name, feat, annotator] = file.split('_')

        if feat != chosen_feature:
            continue 

        annotator = annotator[0:len(annotator)-4]
        curr_category = name[1]

        print("current file: ", file, " category: ", curr_category)
        distances, RMSEs, PCORRs = utils.compute_metrics(name, feat, annotator, ann_stats,
                                                         pred_stats, resampling_frequency,
                                                         subfolder=subfolder, normalization=normalization, detrend=detrend, 
                                                         threshold_percentile=threshold_percentile, plot_series=plot_series)


        print("distance pred vs annotation: ", distances[0])
        print("RMSE pred vs annotation: ", RMSEs[0])
        print("PCORR pred vs annotation: ", PCORRs[0])
        
            
        all_pred_distances[categories[curr_category]].append(distances[0])
        all_pred_RMSEs[categories[curr_category]].append(RMSEs[0])
        if isnan(PCORRs[0][0]) or isnan(PCORRs[0][0]):
            print("this file has nan pcorr: ", file)
            continue
        all_pred_PCORRs[categories[curr_category]].append(PCORRs[0][0])


    # calculate average values

    for curr_category in categories:
        pred_distances_SEM.append(sem(all_pred_distances[categories[curr_category]]))
        pred_distances_mean.append(np.mean(all_pred_distances[categories[curr_category]]))
        pred_RMSEs_SEM.append(sem(all_pred_RMSEs[categories[curr_category]]))
        pred_RMSEs_mean.append(np.mean(all_pred_RMSEs[categories[curr_category]]))
        pred_PCORRs_mean.append(np.nanmean(all_pred_PCORRs[categories[curr_category]]))
        pred_PCORRs_SEM.append(sem(all_pred_PCORRs[categories[curr_category]], nan_policy='omit'))

        if os.path.isfile(output_stats_file):
            with open(output_stats_file, 'r') as f:
                log = f.read()
        else:
            log = ""
        with open(output_stats_file, 'w') as f:
            with redirect_stdout(f):
                print(log)
                print("\ntest assumption of normality, dimension: "+chosen_feature+", category: ", curr_category)
                print("DISTANCE")
                statistic, pvalue = normaltest(all_pred_distances[categories[curr_category]])
                if pvalue < 0.05:
                    print("the null hypothesis can be rejected")
                else:
                    print("the null hypothesis CANNOT be rejected")
                    print(pvalue)
                print("RMSE")
                statistic, pvalue = normaltest(all_pred_RMSEs[categories[curr_category]])
                if pvalue < 0.05:
                    print("the null hypothesis can be rejected")
                else:
                    print("the null hypothesis CANNOT be rejected")
                    print(pvalue) 
                print("PCORR")
                statistic, pvalue = normaltest(all_pred_PCORRs[categories[curr_category]])
                if pvalue < 0.05:
                    print("the null hypothesis can be rejected")
                else:
                    print("the null hypothesis CANNOT be rejected")
                    print(pvalue) 
                
    with open(output_stats_file, 'r') as f:
        log = f.read()
    with open(output_stats_file, 'w') as f:
        with redirect_stdout(f):
            print(log)
            print('RESULTS: Affective Dimension: ', chosen_feature)
            print('prediction distances mean: ', ' & '.join(str(round(x, 3)) for x in pred_distances_mean))
            print('prediction distances SEM: ', ' & '.join(str(round(x, 3)) for x in pred_distances_SEM))
            print('prediction RMSEs mean : ', ' & '.join(str(round(x, 3)) for x in pred_RMSEs_mean))
            print('prediction RMSEs SEM: ', ' & '.join(str(round(x, 3)) for x in pred_RMSEs_SEM))
            print('prediction PCORRs mean : ', ' & '.join(str(round(x, 3)) for x in pred_PCORRs_mean))
            print('prediction PCORRs SEM: ', ' & '.join(str(round(x, 3)) for x in pred_PCORRs_SEM))
            # ANOVA
            print("\nnow performing ANOVA for dimension: "+chosen_feature+" among all categories (a,b,c,d)\n")
            anova = f_oneway(all_pred_distances[0],all_pred_distances[1],all_pred_distances[2],all_pred_distances[3])
            print('one-way ANOVA test for Distance: ', anova)
            if anova[1] <= 0.05:
                #create DataFrame to hold data
                df = pd.DataFrame({
                    'score': all_pred_distances[0] + all_pred_distances[1] + all_pred_distances[2] + all_pred_distances[3],
                    'group': np.array(['a'] * len(all_pred_distances[0]) + ['b'] * len(all_pred_distances[1]) + 
                                    ['c'] * len(all_pred_distances[2]) + ['d'] * len(all_pred_distances[3]))
                    }) 
                # perform Tukey's test
                tukey = pairwise_tukeyhsd(endog=df['score'],
                                        groups=df['group'],
                                        alpha=0.05)
                #display results
                print(tukey)
            
            anova = f_oneway(all_pred_RMSEs[0],all_pred_RMSEs[1],all_pred_RMSEs[2],all_pred_RMSEs[3])
            print('one-way ANOVA test for RMSE: ', anova)
            if anova[1] <= 0.05:
                #create DataFrame to hold data
                df = pd.DataFrame({
                    'score': all_pred_RMSEs[0] + all_pred_RMSEs[1] + all_pred_RMSEs[2] + all_pred_RMSEs[3],
                    'group': np.array(['a'] * len(all_pred_RMSEs[0]) + ['b'] * len(all_pred_RMSEs[1]) +
                                    ['c'] * len(all_pred_RMSEs[2]) + ['d'] * len(all_pred_RMSEs[3]))
                    }) 
                # perform Tukey's test
                tukey = pairwise_tukeyhsd(endog=df['score'],
                                        groups=df['group'],
                                        alpha=0.05)
                #display results
                print(tukey)

            anova = f_oneway(all_pred_PCORRs[0],all_pred_PCORRs[1],all_pred_PCORRs[2],all_pred_PCORRs[3])
            print('one-way ANOVA test for PCORR: ', anova)
            if anova[1] <= 0.05:
                #create DataFrame to hold data
                df = pd.DataFrame({
                    'score': all_pred_PCORRs[0] + all_pred_PCORRs[1] + all_pred_PCORRs[2] + all_pred_PCORRs[3],
                    'group': np.array(['a'] * len(all_pred_PCORRs[0]) + ['b'] * len(all_pred_PCORRs[1]) +
                                    ['c'] * len(all_pred_PCORRs[2]) + ['d'] * len(all_pred_PCORRs[3]))
                    }) 
                # perform Tukey's test
                tukey = pairwise_tukeyhsd(endog=df['score'],
                                        groups=df['group'],
                                        alpha=0.05)
                #display results
                print(tukey)
            
    # merge all data for this feature
    all_pred_distances_byfeat.append([all_pred_distances[0], all_pred_distances[1], all_pred_distances[2], all_pred_distances[3]])
    all_pred_RMSEs_byfeat.append([all_pred_RMSEs[0], all_pred_RMSEs[1], all_pred_RMSEs[2], all_pred_RMSEs[3]])
    all_pred_PCORRs_byfeat.append([all_pred_PCORRs[0], all_pred_PCORRs[1], all_pred_PCORRs[2], all_pred_PCORRs[3]])
    

    # plot distances
    plt.xticks([1, 2, 3, 4], ['Conditioned', 'Unconditioned', 'Original', 'None'])
    plt.title('Distance Confidence Intervals for ' + chosen_feature + '\n Normalization: ' + str(normalization))
    utils.plot_confidence_interval(1, all_pred_distances[0], color='#2187bb',label='prediction')
    utils.plot_confidence_interval(2, all_pred_distances[1], color='#2187bb',label='prediction')
    utils.plot_confidence_interval(3, all_pred_distances[2], color='#2187bb',label='prediction')
    utils.plot_confidence_interval(4, all_pred_distances[3], color='#2187bb',label='prediction')
    #plt.show()

    fig.savefig("subjective_evaluation\\distances_"+chosen_feature+"_"+output_plot_filenames+".pdf")
    plt.clf()

    # plot RMSEs
    plt.xticks([1, 2, 3, 4], ['Conditioned', 'Unconditioned', 'Original', 'None'])
    plt.title('RMSE Confidence Intervals for ' + chosen_feature + '\n Normalization: ' + str(normalization))
    utils.plot_confidence_interval(1, all_pred_RMSEs[0], color='#2187bb',label='prediction')
    utils.plot_confidence_interval(2, all_pred_RMSEs[1], color='#2187bb',label='prediction')
    utils.plot_confidence_interval(3, all_pred_RMSEs[2], color='#2187bb',label='prediction')
    utils.plot_confidence_interval(4, all_pred_RMSEs[3], color='#2187bb',label='prediction')
    #plt.show()

    fig.savefig("subjective_evaluation\\RMSEs_"+chosen_feature+"_"+output_plot_filenames+".pdf")
    plt.clf()


with open(output_stats_file, 'r') as f:
    log = f.read()
with open(output_stats_file, 'w') as f:
    with redirect_stdout(f):

        print(log)

        print("\n\nPERFORMING STATS for Dimensions MERGED\n")

        # stats for categories across all dimensions
        pred_distances_mean = []
        pred_RMSEs_mean = []
        pred_PCORRs_mean = []
        pred_distances_SEM = []
        pred_RMSEs_SEM = []
        pred_PCORRs_SEM = []


        for i in range(0,4):
            # mean
            pred_distances_mean.append(np.mean(all_pred_distances_byfeat[0][i] + all_pred_distances_byfeat[1][i]))
            pred_RMSEs_mean.append(np.mean(all_pred_RMSEs_byfeat[0][i] + all_pred_RMSEs_byfeat[1][i]))
            pred_PCORRs_mean.append(np.mean(all_pred_PCORRs_byfeat[0][i] + all_pred_PCORRs_byfeat[1][i]))
            # sem
            pred_distances_SEM.append(sem(all_pred_distances_byfeat[0][i] + all_pred_distances_byfeat[1][i]))
            pred_RMSEs_SEM.append(sem(all_pred_RMSEs_byfeat[0][i] + all_pred_RMSEs_byfeat[1][i]))
            pred_PCORRs_SEM.append(sem(all_pred_PCORRs_byfeat[0][i] + all_pred_PCORRs_byfeat[1][i]))

            print("\ntest assumption of normality, dimensions merged, category: ", i)
            print("DISTANCE")
            statistic, pvalue = normaltest(all_pred_distances_byfeat[0][i] + all_pred_distances_byfeat[1][i])
            if pvalue < 0.05:
                print("the null hypothesis can be rejected")
            else:
                print("the null hypothesis CANNOT be rejected")
                print(pvalue)
            print("RMSE")
            statistic, pvalue = normaltest(all_pred_RMSEs_byfeat[0][i] + all_pred_RMSEs_byfeat[1][i])
            if pvalue < 0.05:
                print("the null hypothesis can be rejected")
            else:
                print("the null hypothesis CANNOT be rejected")
                print(pvalue) 
            print("PCORR")
            statistic, pvalue = normaltest(all_pred_PCORRs_byfeat[0][i] + all_pred_PCORRs_byfeat[1][i])
            if pvalue < 0.05:
                print("the null hypothesis can be rejected")
            else:
                print("the null hypothesis CANNOT be rejected")
                print(pvalue)

        print('\n\nRESULTS: Affective Dimensions merged')
        print('prediction distances mean: ', ' & '.join(str(round(x, 3)) for x in pred_distances_mean))
        print('prediction distances SEM: ', ' & '.join(str(round(x, 3)) for x in pred_distances_SEM))
        print('prediction RMSEs mean : ', ' & '.join(str(round(x, 3)) for x in pred_RMSEs_mean))
        print('prediction RMSEs SEM: ', ' & '.join(str(round(x, 3)) for x in pred_RMSEs_SEM))
        print('prediction PCORRs mean : ', ' & '.join(str(round(x, 3)) for x in pred_PCORRs_mean))
        print('prediction PCORRs SEM: ', ' & '.join(str(round(x, 3)) for x in pred_PCORRs_SEM))
        # ANOVA
        print("\nnow performing ANOVA comparing all categories (a,b,c,d), dimensions merged\n")
        anova = f_oneway(all_pred_distances_byfeat[0][0] + all_pred_distances_byfeat[1][0],
                         all_pred_distances_byfeat[0][1] + all_pred_distances_byfeat[1][1],
                         all_pred_distances_byfeat[0][2] + all_pred_distances_byfeat[1][2],
                         all_pred_distances_byfeat[0][3] + all_pred_distances_byfeat[1][3]
                         )
        print('one-way ANOVA test for Distance: ', anova)
        if anova[1] <= 0.05:
            #create DataFrame to hold data
            df = pd.DataFrame({
                'score': all_pred_distances_byfeat[0][0] + all_pred_distances_byfeat[1][0]+
                         all_pred_distances_byfeat[0][1] + all_pred_distances_byfeat[1][1]+
                         all_pred_distances_byfeat[0][2] + all_pred_distances_byfeat[1][2]+
                         all_pred_distances_byfeat[0][3] + all_pred_distances_byfeat[1][3]
                         ,
                'group': np.array(['a'] * len(all_pred_distances_byfeat[0][0] + all_pred_distances_byfeat[1][0]) + 
                                  ['b'] * len(all_pred_distances_byfeat[0][1] + all_pred_distances_byfeat[1][1]) +
                                  ['c'] * len(all_pred_distances_byfeat[0][2] + all_pred_distances_byfeat[1][2]) + 
                                  ['d'] * len(all_pred_distances_byfeat[0][3] + all_pred_distances_byfeat[1][3])
                                  )
                }) 
            # perform Tukey's test
            tukey = pairwise_tukeyhsd(endog=df['score'],
                                    groups=df['group'],
                                    alpha=0.05)
            #display results
            print(tukey)
        anova = f_oneway(all_pred_RMSEs_byfeat[0][0] + all_pred_RMSEs_byfeat[1][0],
                         all_pred_RMSEs_byfeat[0][1] + all_pred_RMSEs_byfeat[1][1],
                         all_pred_RMSEs_byfeat[0][2] + all_pred_RMSEs_byfeat[1][2],
                         all_pred_RMSEs_byfeat[0][3] + all_pred_RMSEs_byfeat[1][3]
                         )
        print('one-way ANOVA test for RMSE: ', anova)
        if anova[1] <= 0.05:
            #create DataFrame to hold data
            df = pd.DataFrame({
                'score': all_pred_RMSEs_byfeat[0][0] + all_pred_RMSEs_byfeat[1][0]+
                         all_pred_RMSEs_byfeat[0][1] + all_pred_RMSEs_byfeat[1][1]+
                         all_pred_RMSEs_byfeat[0][2] + all_pred_RMSEs_byfeat[1][2]+
                         all_pred_RMSEs_byfeat[0][3] + all_pred_RMSEs_byfeat[1][3]
                         ,
                'group': np.array(['a'] * len(all_pred_RMSEs_byfeat[0][0] + all_pred_RMSEs_byfeat[1][0]) + 
                                  ['b'] * len(all_pred_RMSEs_byfeat[0][1] + all_pred_RMSEs_byfeat[1][1]) +
                                  ['c'] * len(all_pred_RMSEs_byfeat[0][2] + all_pred_RMSEs_byfeat[1][2]) + 
                                  ['d'] * len(all_pred_RMSEs_byfeat[0][3] + all_pred_RMSEs_byfeat[1][3])
                                  )
                }) 
            # perform Tukey's test
            tukey = pairwise_tukeyhsd(endog=df['score'],
                                    groups=df['group'],
                                    alpha=0.05)
            #display results
            print(tukey)
        anova = f_oneway(all_pred_PCORRs_byfeat[0][0] + all_pred_PCORRs_byfeat[1][0],
                         all_pred_PCORRs_byfeat[0][1] + all_pred_PCORRs_byfeat[1][1],
                         all_pred_PCORRs_byfeat[0][2] + all_pred_PCORRs_byfeat[1][2],
                         all_pred_PCORRs_byfeat[0][3] + all_pred_PCORRs_byfeat[1][3]
                         )
        print('one-way ANOVA test for PCORR: ', anova)
        if anova[1] <= 0.05:
            #create DataFrame to hold data
            df = pd.DataFrame({
                'score': all_pred_PCORRs_byfeat[0][0] + all_pred_PCORRs_byfeat[1][0]+
                         all_pred_PCORRs_byfeat[0][1] + all_pred_PCORRs_byfeat[1][1]+
                         all_pred_PCORRs_byfeat[0][2] + all_pred_PCORRs_byfeat[1][2]+
                         all_pred_PCORRs_byfeat[0][3] + all_pred_PCORRs_byfeat[1][3]
                         ,
                'group': np.array(['a'] * len(all_pred_PCORRs_byfeat[0][0] + all_pred_PCORRs_byfeat[1][0]) + 
                                  ['b'] * len(all_pred_PCORRs_byfeat[0][1] + all_pred_PCORRs_byfeat[1][1]) +
                                  ['c'] * len(all_pred_PCORRs_byfeat[0][2] + all_pred_PCORRs_byfeat[1][2]) + 
                                  ['d'] * len(all_pred_PCORRs_byfeat[0][3] + all_pred_PCORRs_byfeat[1][3])
                                  )
                }) 
            # perform Tukey's test
            tukey = pairwise_tukeyhsd(endog=df['score'],
                                    groups=df['group'],
                                    alpha=0.05)
            #display results
            print(tukey)
        
        # TODO: turkey if necessary

        
        # plot distance and rmse grouped by category (a,b,c,d), all dimensions merged

        # plot distances
        plt.xticks([1, 2, 3, 4], ['Conditioned', 'Unconditioned', 'Original', 'None'])
        plt.title('Distance Confidence Intervals\n Normalization: ' + str(normalization))
        utils.plot_confidence_interval(1, all_pred_distances_byfeat[0][0] + all_pred_distances_byfeat[1][0], color='#2187bb',label='prediction')
        utils.plot_confidence_interval(2, all_pred_distances_byfeat[0][1] + all_pred_distances_byfeat[1][1], color='#2187bb',label='prediction')
        utils.plot_confidence_interval(3, all_pred_distances_byfeat[0][2] + all_pred_distances_byfeat[1][2], color='#2187bb',label='prediction')
        utils.plot_confidence_interval(4, all_pred_distances_byfeat[0][3] + all_pred_distances_byfeat[1][3], color='#2187bb',label='prediction')
        #plt.show()

        fig.savefig("subjective_evaluation\\distances_alldims_"+output_plot_filenames+".pdf")
        plt.clf()

        # plot RMSEs
        plt.xticks([1, 2, 3, 4], ['Conditioned', 'Unconditioned', 'Original', 'None'])
        plt.title('RMSE Confidence Intervals\n Normalization: ' + str(normalization))
        utils.plot_confidence_interval(1, all_pred_RMSEs_byfeat[0][0] + all_pred_RMSEs_byfeat[1][0], color='#2187bb',label='prediction')
        utils.plot_confidence_interval(2, all_pred_RMSEs_byfeat[0][1] + all_pred_RMSEs_byfeat[1][1], color='#2187bb',label='prediction')
        utils.plot_confidence_interval(3, all_pred_RMSEs_byfeat[0][2] + all_pred_RMSEs_byfeat[1][2], color='#2187bb',label='prediction')
        utils.plot_confidence_interval(4, all_pred_RMSEs_byfeat[0][3] + all_pred_RMSEs_byfeat[1][3], color='#2187bb',label='prediction')
        #plt.show()

        fig.savefig("subjective_evaluation\\RMSEs_alldims_"+output_plot_filenames+".pdf")
        plt.clf()




        # PERFERM STATs TESTS

        print("\n\nPERFORMING STATS: VALENCE vs AROUSAL, all categories merged")
        print("VALENCE normal test:")
        print("Distance: ", normaltest(list(itertools.chain.from_iterable(all_pred_distances_byfeat[0]))))
        print("RMSE: ", normaltest(list(itertools.chain.from_iterable(all_pred_RMSEs_byfeat[0]))))
        print("AROUSAL normal test:")
        print("Distance: ", normaltest(list(itertools.chain.from_iterable(all_pred_distances_byfeat[1]))))
        print("RMSE: ", normaltest(list(itertools.chain.from_iterable(all_pred_RMSEs_byfeat[1]))))

        # ANOVA per gruppo valence e arousal
        all_pred_distances_val = list(itertools.chain.from_iterable(all_pred_distances_byfeat[0]))
        all_pred_distances_aro = list(itertools.chain.from_iterable(all_pred_distances_byfeat[1]))
        print("\nnow performing ANOVA\n")
        anova = f_oneway(all_pred_distances_val,all_pred_distances_aro)
        print('one-way ANOVA test for DISTANCE: ', anova)
        if anova[1] <= 0.05:
            #create DataFrame to hold data
            df = pd.DataFrame({
                'score': all_pred_distances_val + all_pred_distances_aro,
                'group': np.array(['valence'] * len(all_pred_distances_val) + ['arousal'] * len(all_pred_distances_aro))
                }) 
            # perform Tukey's test
            tukey = pairwise_tukeyhsd(endog=df['score'],
                                    groups=df['group'],
                                    alpha=0.05)
            #display results
            print(tukey)

        all_pred_RMSEs_val = list(itertools.chain.from_iterable(all_pred_RMSEs_byfeat[0]))
        all_pred_RMSEs_aro = list(itertools.chain.from_iterable(all_pred_RMSEs_byfeat[1]))
        anova = f_oneway(all_pred_RMSEs_val, all_pred_RMSEs_aro)
        print('one-way ANOVA test for RMSE: ', anova)
        if anova[1] <= 0.05:
            #create DataFrame to hold data
            df = pd.DataFrame({
                'score': all_pred_RMSEs_val + all_pred_RMSEs_aro,
                'group': np.array(['valence'] * len(all_pred_RMSEs_val) + ['arousal'] * len(all_pred_RMSEs_aro))
                }) 
            # perform Tukey's test
            tukey = pairwise_tukeyhsd(endog=df['score'],
                                    groups=df['group'],
                                    alpha=0.05)
            #display results
            print(tukey)

        all_pred_PCORRs_val = list(itertools.chain.from_iterable(all_pred_PCORRs_byfeat[0]))
        all_pred_PCORRs_aro = list(itertools.chain.from_iterable(all_pred_PCORRs_byfeat[1]))
        anova = f_oneway(all_pred_PCORRs_val, all_pred_PCORRs_aro)
        print('one-way ANOVA test for PCORR: ', anova)  
        if anova[1] <= 0.05:
            #create DataFrame to hold data
            df = pd.DataFrame({
                'score': all_pred_PCORRs_val + all_pred_PCORRs_aro,
                'group': np.array(['valence'] * len(all_pred_PCORRs_val) + ['arousal'] * len(all_pred_PCORRs_aro))
                }) 
            # perform Tukey's test
            tukey = pairwise_tukeyhsd(endog=df['score'],
                                    groups=df['group'],
                                    alpha=0.05)
            #display results
            print(tukey)