# experimental setup and results evaluation for videogame generative music system

Code written for my master's thesis: Procedural music generation for videogames conditioned through video emotion recognition. 

This folder contains the code written for the experimental evaluation of the thesis, in particular:
- `main.py` generates valence-arousal time series for any input `*.mp4` video. requires a tensorflow trained model performing valence-arousal detection
- `manual_rank_trace.py` provides a simple interface based on RankTrace functionalities for performing the emotion annotation task
- `process_experiment_results.py` compares the annotations collected from participants with the model predictions, generating figures and a `.txt` summarizing the statistical analysis of the results 

More details can be found in the thesis manuscript (TODO link once article/thesis is published)
