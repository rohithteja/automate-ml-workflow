# Automate ML Workflow
Code relating to the article How To Automate and Simplify Your Machine Learning Experiment Workflow
Link to article: https://towardsdatascience.com/how-to-automate-and-simplify-your-machine-learning-experiment-workflow-2be19f4aaa6

## How to use this repo:
1. To run all experiments use: `bash loop.sh`
2. To run individual experiments use: `python argparse_automate.py --dim_red_type "pca" --n_comp 10 --classifier "svc"`

The results (accuracy scores) are saved in a text file in nested folders titled with the same arguments used to run the code.
