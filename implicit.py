import pandas as pd
from simpletransformers.classification import ClassificationModel
import sklearn
import numpy as np


# read the input files
df = pd.read_csv("offenseval-training-implicit.csv", sep='\t')


df['subtask_a'] = (df['subtask_a'] == 'OFF').astype(int)

# process the input data
df = pd.DataFrame({
        'text': '[CLS] ' + df['tweet'].replace(r'\n', ' ', regex= True),
        'label': df['subtask_a']
    })

print(df.head())

# shuffle the samples
df = df.sample(frac=1).reset_index(drop=True)

# split the data into 10 equal parts
dfs = np.array_split(df, 10)

# results of each experiment
results = []

# ten cross validation
for k in range(0, 10):
    
    train = [dfs[i] for i in range(0, len(dfs)) if i != k]
    train_ = pd.concat(train, sort=False)
    eval_ = dfs[k]

    # Create a TransformerModel
    model = ClassificationModel(
                    'roberta', 
                    'roberta-large', 
                        num_labels=2,
                            args=(
                                        {'overwrite_output_dir': True,
                                        'fp16': False,
                                        'num_train_epochs': 2,  
                                        'reprocess_input_data': False,
                                        "learning_rate": 1e-5,                                       
                                        "train_batch_size": 64,
                                        "eval_batch_size": 64,
                                        "weight_decay": 0,
                                        "evaluate_during_training_verbose": True,
                                        "evaluate_during_training": True,
                                        "do_lower_case": False,
                                        "n_gpu": 2, # can be 1 if you have enough memory
                                        })
                                )
    # Train the model
    model.train_model(train_, eval_df=eval_)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_, acc=sklearn.metrics.accuracy_score)
    results.append(result)

print(results)

# save the results
outputs = open("outputs-implicit.txt", "w")
outputs.write(str(results))






