import glob
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
    
    print('Generating results table...')
    
    # Table header
    table = \
'''\\begin{table}[tbp]
\\centering
\\caption{Mean reward of each algorithm over the 10 environment variations for all 3 settings}
\\resizebox{\\linewidth}{!}{
\\begin{tabular}{|c|c|c|c|c|c|}
\\hline
\\textbf{Setting} & \\textbf{Algorithm} & \\textbf{Mean} & \\textbf{SD} & \\textbf{Max} & \\textbf{Range} \\\\ 
& & $\\times 10^{5}$ & $\\times 10^{5}$ & $\\times 10^{5}$ & $\\times 10^{5}$ \\\\ \\hline'''

    # Gather training data
    training_data = []
    training_logs = glob.glob("./training_logs/*/*")
    for log in training_logs:
        experiment_info = log.split('/')[2].split('_')
        algorithm = experiment_info[0]
        st = int(experiment_info[1][3:])

        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == 'rollout/ep_rew_mean':
                    training_data.append({
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value
                    })
    train_df = pd.DataFrame(training_data)
    
    # Add setting A results to table
    table += '\n\\multirow{6}{*}{A} '
    for algorithm in sorted(train_df['algorithm'].unique()):
        rewards = train_df[train_df['algorithm'] == algorithm]['reward'] / 1e5
        alg_mean = rewards.mean()
        alg_std = rewards.std()
        alg_max = rewards.max()
        alg_range = alg_max - rewards.min()
        
        table += f'\n& {algorithm} '
        table += f'& \\textbf{{{alg_mean:.3f}}} ' if alg_max >= 0.9 else fr'& {alg_mean:.3f} '
        table += f'& {alg_std:.3f} '
        table += f'& \\textbf{{{alg_max:.3f}}} ' if alg_max >= 0.9 else fr'& {alg_max:.3f} '
        table += f'& {alg_range:.3f} \\\\ \\cline{{2-6}} '
    table += '\\hline'
        
    # Gather tuning data
    tuning_data = []
    tuning_logs = glob.glob("./tuning_logs/*/*")
    for log in tuning_logs:
        experiment_info = log.split('/')[2].split('_')
        algorithm = experiment_info[0]
        st = int(experiment_info[1][3:])
        trial = int(experiment_info[2])

        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == 'rollout/ep_rew_mean':
                    tuning_data.append({
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value,
                        'trial': trial
                    })
    tune_df = pd.DataFrame(tuning_data)
    
    # Add setting B results to table
    table += '\n\\multirow{6}{*}{B} '
    for algorithm in sorted(tune_df['algorithm'].unique()):
        rewards = tune_df[tune_df['algorithm'] == algorithm]['reward'] / 1e5
        alg_mean = rewards.mean()
        alg_std = rewards.std()
        alg_max = rewards.max()
        alg_range = alg_max - rewards.min()
        
        table += f'\n& {algorithm} '
        table += f'& \\textbf{{{alg_mean:.3f}}} ' if alg_max >= 0.9 else fr'& {alg_mean:.3f} '
        table += f'& {alg_std:.3f} '
        table += f'& \\textbf{{{alg_max:.3f}}} ' if alg_max >= 0.9 else fr'& {alg_max:.3f} '
        table += f'& {alg_range:.3f} \\\\ \\cline{{2-6}} '
    table += '\\hline'
        
    # Gather transfer data
    transfer_data = []
    transfer_logs = glob.glob("./transfer_logs/*/*")
    for log in transfer_logs:
        experiment_info = log.split('/')[2].split('_')
        algorithm = experiment_info[0]
        st = int(experiment_info[2][2:])

        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == 'rollout/ep_rew_mean':
                    transfer_data.append({
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value
                    })
    transfer_df = pd.DataFrame(transfer_data)
    
    # Add setting C results to table
    table += '\n\\multirow{6}{*}{C} '
    for algorithm in sorted(transfer_df['algorithm'].unique()):
        rewards = transfer_df[transfer_df['algorithm'] == algorithm]['reward'] / 1e5
        alg_mean = rewards.mean()
        alg_std = rewards.std()
        alg_max = rewards.max()
        alg_range = alg_max - rewards.min()
        
        table += f'\n& {algorithm} '
        table += f'& \\textbf{{{alg_mean:.3f}}} ' if alg_max >= 0.9 else fr'& {alg_mean:.3f} '
        table += f'& {alg_std:.3f} '
        table += f'& \\textbf{{{alg_max:.3f}}} ' if alg_max >= 0.9 else fr'& {alg_max:.3f} '
        table += f'& {alg_range:.3f} \\\\ \\cline{{2-6}} '
    table += '\\hline'
        
    # Table footer
    table += \
'''\n\\end{tabular}}
\\label{tab:alg_analysis}
\\end{table}'''

    # Save table to file
    with open('tables/results_table.tex', 'w') as file:
        file.write(table)
