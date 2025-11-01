import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

def train_val_test_split(ids, train_ratio = 0.80,
                        val_ratio = 0.10,
                        test_ratio = 0.10,
                        rng_seed=0):
    
    ids_train, ids_test = train_test_split(ids, 
                                        test_size=1 - train_ratio, 
                                        shuffle=True, 
                                        random_state=rng_seed)
    ids_val, ids_test = train_test_split(ids_test,
                                        test_size=test_ratio/(test_ratio + val_ratio), 
                                        shuffle=True, 
                                        random_state=rng_seed)
    
    return ids_train, ids_val, ids_test

def group_train_val_test_split(ids: np.ndarray, 
                                groups: np.ndarray,
                                train_ratio = 0.80,
                                val_ratio = 0.10,
                                test_ratio = 0.10,
                                rng_seed=0):
    ids = np.array(ids)
    gss1 = GroupShuffleSplit(n_splits=1, 
                            train_size=train_ratio, 
                            test_size=test_ratio + val_ratio, 
                            random_state=rng_seed)
    train_idx, holdout_idx = next(gss1.split(X=ids, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, 
                            train_size=val_ratio/(test_ratio + val_ratio), 
                            test_size=test_ratio/(test_ratio + val_ratio), 
                            random_state=rng_seed)
    val_idx, test_idx = next(gss2.split(X=ids[holdout_idx], groups=groups[holdout_idx]))
    val_idx = holdout_idx[val_idx]
    test_idx = holdout_idx[test_idx]

    return train_idx, val_idx, test_idx                        

def plot_training_stats(training_stats, show=False, save=False, save_dir=None):
    ## Plot training stats
    if isinstance(training_stats, dict):
        for i, (key, values) in enumerate(training_stats.items()):
            if key == 'epoch':
                continue
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

            if isinstance(values, dict) and 'train' in values and 'val' in values:
                train_values = values['train']
                val_values = values['val']
                ax.plot(training_stats['epoch'], train_values, label=f'train {key}')
                ax.plot(training_stats['epoch'], val_values, label=f'val {key}')
                ax.set_ylabel(key)
            elif key == 'lr':
                ax.plot(training_stats['epoch'], values, label='learning rate')
                ax.set_ylabel('learning rate')
                ax.set_yscale('log')  # Log scale for learning rate
            else:
                ax.plot(training_stats['epoch'], values, label=key)
                ax.set_ylabel(key)
            ax.set_xlabel('Epoch')
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(f"{save_dir}/{key}.png", dpi=300)
            if show:
                plt.show()
    
    elif isinstance(training_stats, pd.DataFrame):
        # Columns: epoch,train_loss,val_loss,train_<metric1>,val_<metric1>,...,train_<metricN>,val_<metricN>,lr
        for column in training_stats.columns:
            if column == 'epoch' or column.startswith('val_'): # Skip val columns as they are handled with train columns
                continue

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

            if column.startswith('train_'):
                metric_name = column[len('train_'):]
                val_column = f'val_{metric_name}'
                ax.plot(training_stats['epoch'], training_stats[column], label=f'train {metric_name}')
                if val_column in training_stats.columns:
                    ax.plot(training_stats['epoch'], training_stats[val_column], label=f'val {metric_name}')
                ax.set_ylabel(metric_name)
            elif column == 'lr':
                ax.plot(training_stats['epoch'], training_stats[column], label='learning rate')
                ax.set_ylabel('learning rate')
                ax.set_yscale('log')  # Log scale for learning rate
            else:
                ax.plot(training_stats['epoch'], training_stats[column], label=column)
                ax.set_ylabel(column)
            ax.set_xlabel('Epoch')
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(f"{save_dir}/{column}.png", dpi=300)
            if show:
                plt.show()
            