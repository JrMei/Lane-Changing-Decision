import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import numpy as np
 
env = gym.make("highway-fast-v0")
model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=7e-4,
              buffer_size=100000,
              learning_starts=10000,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="highway_dqn/")

# 自定义回调函数以记录碰撞率、安全性和平均速度
class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomTensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_collisions = []
        self.episode_safety = []
        self.episode_speeds = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        episode_rewards = []
        episode_collisions = 0
        episode_speeds = []

        for info in self.locals['infos']:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                episode_collisions += int(info.get('crashed', False))
                if 'speed' in info:
                    episode_speeds.append(info['speed'])

        if episode_rewards:
            episode_reward = np.mean(episode_rewards)
            self.episode_rewards.append(episode_reward)

            episode_safety = 1.0 - episode_collisions / len(self.locals['infos'])
            self.episode_safety.append(episode_safety)

            average_speed = np.mean(episode_speeds) if episode_speeds else 0
            self.episode_speeds.append(average_speed)

            # 写入TensorBoard
            if self.model.logger:
                self.model.logger.record('rollout/episode_reward', episode_reward)
                self.model.logger.record('rollout/episode_collisions', episode_collisions)
                self.model.logger.record('rollout/episode_safety', episode_safety)
                self.model.logger.record('rollout/average_speed', average_speed)


# 设置回调函数以保存最佳模型并记录TensorBoard数据
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./highway_ddpg/', name_prefix='best_model')
custom_callback = CustomTensorboardCallback()

# 训练模型
model.learn(total_timesteps=200000, callback=[checkpoint_callback, custom_callback])

model.save("highway_dqn/model")
 
# 模型运行
env = gym.make("highway-fast-v0", render_mode='rgb_array')
model = DQN.load("highway_dqn/model")
while True:
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
 
 
 