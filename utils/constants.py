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
    'Result':12
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
    12:'Result'
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
    'results.txt': 'Result'
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

LABEL2ID = {
    'ablation-analysis':1,
    'code':2,
    'model':3,
    'research-problem':4,
    'results':5,
    'dataset':6,
    'hyperparameters':7,
    'experiments':8,
    'baselines':9,
    'experimental-setup':10,
    'approach':11
}

ID2LABEL = {
    1:'ablation-analysis',
    2:'code',
    3:'model',
    4:'research-problem',
    5:'results',
    6:'dataset',
    7:'hyperparameters',
    8:'experiments',
    9:'baselines',
    10:'experimental-setup',
    11:'approach'
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