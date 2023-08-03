# experimental setup and results evaluation for videogame generative music system

Code written for my master's thesis: Procedural music generation for videogames conditioned through video emotion recognition. 

This folder contains the code written for the experimental evaluation of the thesis, in particular:
- `main.py` generates valence-arousal time series for any input `*.mp4` video. requires a tensorflow trained model performing valence-arousal detection
- `manual_rank_trace.py` provides a simple interface based on RankTrace functionalities for performing the emotion annotation task
- `process_experiment_results.py` compares the annotations collected from participants with the model predictions, generating figures and a `.txt` summarizing the statistical analysis of the results 

More details can be found in the thesis manuscript (TODO link once article/thesis is published)

## Usage

### Time series generation

First of all, a tensorflow trained model must be put in `models/model_folder`. 

Then, you must put all the videos you want to analyze at the following path `videos/videogame_name/*`, where `videogame_name` is the subfolder containing all videos from the same source.

Finally, you can run `main.py`, that for each video will return a `.csv` file inside the folder `output`.

### emotion annotation interface

This code uses `VLC` media player in order to reproduce the current video.

For beginning the annotation task, the script must be run with the following CLI arguments: `current_affective_dimension current_video_filename`.

For example: `python manual_rank_trace.py valence na1`

### analyze annotations collected and obtain results

In order to 
