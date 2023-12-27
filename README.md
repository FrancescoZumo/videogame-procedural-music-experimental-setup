# Experimental setup and results evaluation for videogame generative music system

Code written for my master's thesis: Procedural music generation for videogames conditioned through video emotion recognition. 

Complementary repositories, part of the same project:
-  [valence-arousal-video-estimation](https://github.com/FrancescoZumo/valence-arousal-video-estimation)
-  [midi-emotion-primer-continuation](https://github.com/FrancescoZumo/midi-emotion-primer-continuation)
-  videogame-procedural-music-experimental-setup (current repository)

More details can be found in the [thesis manuscript](https://www.politesi.polimi.it/handle/10589/210809) and in the [published paper](https://ieeexplore.ieee.org/document/10335439).

## Description
This folder contains the code written for the experimental evaluation of the thesis, in particular:
- `main.py` generates valence-arousal time series for any input `*.mp4` video. requires a tensorflow trained model performing valence-arousal detection
- `manual_rank_trace.py` provides a simple interface based on RankTrace functionalities for performing the emotion annotation task
- `process_experiment_results.py` compares the annotations collected from participants with the model predictions, generating figures and a `.txt` summarizing the statistical analysis of the results

The experimental procedure was based on a similar work proposed by Plut et al. : https://dl.acm.org/doi/abs/10.1145/3555858.3555947

The videos used for conducting our experimental evaluation are available at this [link](https://www.youtube.com/playlist?list=PLw4IxoxadimNV2-oapulrengf79XCcXNM)

## Usage: Time series generation

First of all, a tensorflow trained model must be put in `models/model_folder`. 

Then, you must put all the videos you want to analyze at the following path `videos/videogame_name/*`, where `videogame_name` is the subfolder containing all videos from the same source.

Finally, you can run `main.py`, that for each video will return a `.csv` file inside the folder `output`.

## Usage: emotion annotation interface

This code uses `VLC` media player in order to reproduce the current video.

For beginning the annotation task, the script must be run with the following CLI arguments: `current_affective_dimension current_video_filename`.

For example: `python manual_rank_trace.py valence na1`

## Usage: experimental results analysis
Apart from the user annotations, which are obtained locally, we download and insert inside `questionnaire_results.csv` all answers to the questionnaire, implemented with Google Form: https://forms.gle/bHahSJvXsnDHw6pd7. 

Our code automatically sorts the randomized order of the videos for each participant.

Once all the user annotations have been collected, simply run `process_experiment_results.py`, which returns a `.txt` summary with all the statistical tests performed, as well as some few useful `.pdf` plots.
