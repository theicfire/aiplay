#!/usr/bin/env python
# coding: utf-8


# sudo apt-get update
# !sudo apt install python-opengl
# !sudo apt install ffmpeg
# !sudo apt install xvfb
# get_ipython().system('pip3 install pyvirtualdisplay')
# get_ipython().system('pip install gym[box2d]')
# get_ipython().system('pip install stable-baselines3[extra]')
# get_ipython().system('pip install huggingface_sb3')
# get_ipython().system('pip install pyglet')
# get_ipython().system('pip install ale-py==0.7.4 # To overcome an issue with gym (https://github.com/DLR-RM/stable-baselines3/issues/875)')

# get_ipython().system('pip install pickle5')
# get_ipython().system('pip install wandb')

# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import gym

from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

# We create our environment with gym.make("<name_of_the_environment>")
env = gym.make("LunarLander-v2")
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation


print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action


# Create the environment
env = make_vec_env('LunarLander-v2', n_envs=16)


config = {
    "policy_type": "blah",
    "total_timesteps": 99,
    "env_name": "ppo-LunarLander-v2",
}

run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1,
    tensorboard_log=f"runs/{run.id}")

model.learn(total_timesteps=500000,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ))

model_name = "ppo-LunarLander-v2"
model.save(model_name) # Ooo, puts the weights inside of a zip file
run.finish()


# ### Step 7: Evaluate the agent 📈
# - Now that our Lunar Lander agent is trained 🚀, we need to **check its performance**.
# - Stable-Baselines3 provides a method to do that: `evaluate_policy`.

# eval_env = gym.make("LunarLander-v2")
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
