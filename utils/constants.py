BLOCK2ID = {
    'Hyperparameters':1,
    'Approach':2, 
    'Dataset':3, 
    'Baselines':4, 
    'has research problem':5, 
    'Experiments':6, 
    'Experimental setup':7, 
    'Ablation analysis':8, 
    'Tasks':9, 
    'Code':10, 
    'Model':11, 
    'Results':12
}

ID2BLOCK = {
    1:'Hyperparameters',
    2:'Approach', 
    3:'Dataset', 
    4:'Baselines', 
    5:'has research problem', 
    6:'Experiments', 
    7:'Experimental setup', 
    8:'Ablation analysis', 
    9:'Tasks', 
    10:'Code', 
    11:'Model', 
    12:'Results'
}

FILENAME2BLOCK = {
    'hyperparameters.txt': 'Hyperparameters',
    'approach.txt': 'Approach', 
    'dataset.txt': 'Dataset', 
    'baselines.txt': 'Baselines', 
    'research-problem.txt': 'has research problem', 
    'experiments.txt': 'Experiments', 
    'experimental-setup.txt': 'Experimental setup', 
    'ablation-analysis.txt': 'Ablation analysis', 
    'tasks.txt': 'Tasks', 
    'code.txt': 'Code', 
    'model.txt': 'Model', 
    'results.txt': 'Results'
}

BLOCK2FILENAME = {
    'Hyperparameters': 'hyperparameters.txt',
    'Approach': 'approach.txt', 
    'Dataset': 'dataset.txt', 
    'Baselines': 'baselines.txt', 
    'has research problem': 'research-problem.txt', 
    'Experiments': 'experiments.txt', 
    'Experimental setup': 'experimental-setup.txt', 
    'Ablation analysis': 'ablation-analysis.txt', 
    'Tasks': 'tasks.txt', 
    'Code': 'code.txt', 
    'Model': 'model.txt', 
    'Results': 'results.txt' 
}

BLOCK_BACK_NAMES = set([
    'Model',
    'Baselines',
    'Results',
    'Experiments',
    'Ablation analysis',
    'Experimental setup',
    'Approach',
    'Hyperparameters',
    'Tasks',
    'Dataset'
])

BLOCK_MID_NAMES = {
    'Code',
    'has research problem'
}

NER_LABEL2ID = {
    'O': 0,
    'B':1,
    'I':2
}

NER_ID2LABEL = {
    0:'O',
    1:'B',
    2:'I'
}