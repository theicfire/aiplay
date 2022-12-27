#!/usr/bin/env python
# coding: utf-8

from pyvirtualdisplay import Display
import gym

# To log to our Hugging Face account to be able to upload models to the Hub.
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# We create our environment with gym.make("<name_of_the_environment>")
env = gym.make("LunarLander-v2")
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample())  # Get a random observation


print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())  # Take a random action


# Create the environment
env = make_vec_env('LunarLander-v2', n_envs=16)


config = {
    'env_name': 'ppo-LunarLander-v2',
    'total_timesteps': 500000,
    'batch_size': 32,
    'gamma': 0.999,
}

run = wandb.init(
    project='sb3-2',
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

model = PPO(
    policy='MlpPolicy',
    env=env,
    n_steps=1024,
    batch_size=config['batch_size'],
    n_epochs=4,
    gamma=config['gamma'],
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
    seed=8557,
    tensorboard_log=f"runs/{run.id}")

model.learn(total_timesteps=config['total_timesteps'],
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
))

model_name = "ppo-LunarLander-v2"
model.save(model_name)  # Ooo, puts the weights inside of a zip file
run.finish()


# ### Step 7: Evaluate the agent 📈

# eval_env = gym.make("LunarLander-v2")
# mean_reward, std_reward = evaluate_policy(model, eval_env,
# n_eval_episodes=10, deterministic=True)
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
