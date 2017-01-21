Repository for Hw1 of CSE 517

To run the grid search, run
`source grid_search.sh`
It will start run grid search from the hyperparameter sets in `hp_update.ini`
The base hyperparameter is in `hp.ini`. The mechanism is it will first parse the hyperparameter from `hp.ini` and then update the hyperparameters from `hp_update.ini` in each sections. So if there are five sections, it will loop for five times. The `hp_update.ini.template` is basically the updated hyperparameters for me to do grid search.

To run get the final result, run
`python final_result.py`
It will output the perplexity by updating the base hyperparameter sets from `hp_final.ini`.

**Be sure to run `pip install -r requirements.txt` to install the required packages**

**Be sure to copy `hp_update.ini.template`, `hp_final.ini.template` and `hp.ini.template` to `hp_update.ini`, `hp_final.ini` and `hp.ini` if any of each are missing**
