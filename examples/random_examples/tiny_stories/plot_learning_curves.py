import matplotlib.pyplot as plt
from syne_tune.experiments import load_experiment

e = load_experiment('wrapper-2024-03-06-14-56-46-562', local_path='/Users/kleiaaro/experiments/tiny-stories-checkpoints/syne-tune')

df = e.results
print(df.keys())

for trial_id in df.trial_id.unique():
    df_trial = df[df['trial_id'] == trial_id]
    plt.plot(df_trial['eval_loss'])
    worker_time = df_trial['st_worker_time'].max()
    print(f'{trial_id}: worker time={worker_time}')
plt.show()