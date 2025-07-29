import glob
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import argparse

def plot_setting_a():
    print('Plotting Setting A figure...')
    
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

    # Plot Setting A figure
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,8))
    seaborn.lineplot(data=train_df, x='step', y='reward', hue='algorithm')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1)).set_title('')
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    plt.grid()
    plt.tight_layout()
    plt.savefig('plotting/plots/setting_a.png')
    
def plot_setting_b():
    print('Plotting Setting B figures...')
    
    # Gather training data
    training_data = []
    training_logs = glob.glob("./training_logs/*/*")
    for log in training_logs:
        experiment_info = log.split('/')[2].split('_')
        algorithm = experiment_info[0]
        st = int(experiment_info[1][3:])
        
        # TODO: Run tuning experiments for all 10 environment variations
        if st != 1:
            continue

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
        
    # Best trial info from slurm logs
    best_trials = {
        'A2C': 19,
        'PPO': 11,
        'TRPO': 6,
        'DQN': 14,
        'ARS': 2,
        'RecurrentPPO': 18,
    }
    
    # Gather tuning data
    tuning_data = []
    tuning_logs = glob.glob("./tuning_logs/*/*")
    for log in tuning_logs:
        experiment_info = log.split('/')[2].split('_')
        algorithm = experiment_info[0]
        st = int(experiment_info[1][3:])
        trial = int(experiment_info[2])
        
        if trial != best_trials[algorithm]:
            continue

        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == 'rollout/ep_rew_mean':
                    tuning_data.append({
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value
                    })
    tune_df = pd.DataFrame(tuning_data)
    
    # Plot Setting B figures
    plt.rcParams.update({'font.size': 22})
    for algorithm in tune_df['algorithm'].unique():
        plt.figure(figsize=(10,8))
        plt.plot('step', 'reward', data=train_df[(train_df['algorithm'] == algorithm) & (train_df['step'] <= 1e6)], label='default')
        plt.plot('step', 'reward', data=tune_df[tune_df['algorithm'] == algorithm], label='best')
        plt.xlabel('step')
        plt.ylabel('reward')
        plt.ticklabel_format(style='sci', scilimits=(0,0))
        plt.grid()
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(f'plotting/plots/setting_b_{algorithm}.png')
        
def plot_setting_c():
    print('Plotting Setting C figures...')
    
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
                        'type': 'non-transfer',
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value
                    })
    train_df = pd.DataFrame(training_data)
    
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
                        'type': 'transfer',
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value
                    })
    transfer_df = pd.DataFrame(transfer_data)
    
    all_df = pd.concat([train_df, transfer_df])
    
    # Plot Setting C figures
    plt.rcParams.update({'font.size': 22})
    for algorithm in all_df['algorithm'].unique():
        plt.figure(figsize=(10,8))
        seaborn.lineplot(data=all_df[all_df['algorithm'] == algorithm], x='step', y='reward', hue='type')
        plt.legend(loc='upper left', bbox_to_anchor=(0, 1)).set_title('')
        plt.ticklabel_format(style='sci', scilimits=(0,0))
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'plotting/plots/setting_c_{algorithm}.png')
    
if __name__ == '__main__':
    
    tf.get_logger().setLevel('INFO')
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-a', action='store_true', help='Plot results for Setting A')
    parser.add_argument('-b', action='store_true', help='Plot results for Setting B')
    parser.add_argument('-c', action='store_true', help='Plot results for Setting C')
    
    args = parser.parse_args()
    
    plot_all = not any(vars(args).values())

    if args.a or plot_all:
        plot_setting_a()
    if args.b or plot_all:
        plot_setting_b()
    if args.c or plot_all:
        plot_setting_c()
