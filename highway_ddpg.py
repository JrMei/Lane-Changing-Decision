import gymnasium as gym
import highway_env
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import tensorflow as tf
import logging

# 调整日志记录级别以屏蔽警告信息
logging.getLogger().setLevel(logging.ERROR)

# 离散到连续动作空间的封装类
class DiscreteToContinuousWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscreteToContinuousWrapper, self).__init__(env)
        self.action_space = Box(low=-1.0, high=1.0, shape=(env.action_space.n,), dtype=np.float32)

    def action(self, action):
        return np.argmax(action)

# 包装环境以记录每个步骤的动作
class ActionLoggerWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ActionLoggerWrapper, self).__init__(env)
        self.last_action = None

    def step(self, action):
        self.last_action = action
        return self.env.step(action)

# 创建并配置环境
env = gym.make("highway-v0")
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 50,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True
    },
    "vehicles_count": 50,
    "duration": 40,
    "destination": "offroad",
    "lanes_count": 4,
    "controlled_vehicles": 1,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "policy_frequency": 2,
    "reward_speed_range": [60, 90],
    "collision_reward": -3,
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "lane_change_reward": 0.2, # 变道奖励
    "safety_reward": 0.5  # 安全奖励
}

env.configure(config)
env = DiscreteToContinuousWrapper(env)
env = ActionLoggerWrapper(env)

# 确保观察空间与模型期望的输入形状一致
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(env.config["vehicles_count"] * len(env.config["observation"]["features"]),), dtype=np.float32)

    def observation(self, obs):
        return obs.flatten()

env = ObservationWrapper(env)

# 自定义奖励函数
def custom_reward_function(env, action):
    reward = 0
    controlled_vehicle = env.road.vehicles[0]
    speed = controlled_vehicle.speed
    lane_index = controlled_vehicle.lane_index[2]
    target_lane_index = 1
    target_speed = np.mean(env.config["reward_speed_range"])

    # 根据速度给予奖励
    if env.config["reward_speed_range"][0] < speed < env.config["reward_speed_range"][1]:
        reward += env.config["high_speed_reward"]
    else:
        reward -= env.config["high_speed_reward"] * abs(speed - target_speed) / target_speed

    # 根据车道位置给予奖励
    if lane_index == target_lane_index:
        reward += env.config["right_lane_reward"]
    else:
        reward -= env.config["right_lane_reward"]

    # 根据变道给予奖励，考虑与他车的距离
    if action is not None and not np.array_equal(action, controlled_vehicle.lane_index[2]):
        reward += env.config["lane_change_reward"]

        # 确保变道安全，检查与前车和旁车的距离
        front_vehicle, _ = env.road.neighbour_vehicles(controlled_vehicle)
        left_vehicle, _ = env.road.neighbour_vehicles(controlled_vehicle)
        right_vehicle, _ = env.road.neighbour_vehicles(controlled_vehicle)

        # 计算与前车的距离
        if front_vehicle:
            front_distance = front_vehicle.position[0] - controlled_vehicle.position[0]
            if front_distance < 10:  # 距离过近，惩罚
                reward -= env.config["lane_change_reward"]
            else:  # 距离合适，增加奖励
                reward += env.config["lane_change_reward"] * (1 - front_distance / 10)  # 根据距离调整奖励

        # 计算与旁车的距离
        if left_vehicle:
            left_distance = controlled_vehicle.position[1] - left_vehicle.position[1]
            if abs(left_distance) < 2:  # 距离过近，惩罚
                reward -= env.config["lane_change_reward"]
            else:  # 距离合适，增加奖励
                reward += env.config["lane_change_reward"] * (1 - abs(left_distance) / 2)  # 根据距离调整奖励

        if right_vehicle:
            right_distance = right_vehicle.position[1] - controlled_vehicle.position[1]
            if abs(right_distance) < 2:  # 距离过近，惩罚
                reward -= env.config["lane_change_reward"]
            else:  # 距离合适，增加奖励
                reward += env.config["lane_change_reward"] * (1 - abs(right_distance) / 2)  # 根据距离调整奖励

    # 碰撞检测
    if controlled_vehicle.crashed:
        reward += env.config["collision_reward"]

    # 安全性奖励
    if not controlled_vehicle.crashed:
        reward += env.config["safety_reward"]

    return reward


# 包装环境以添加
# 包装环境以添加自定义奖励函数
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)

    def reward(self, reward):
        # 确保ActionLoggerWrapper在env的包装链中
        action = self.env.last_action if hasattr(self.env, 'last_action') else None
        custom_reward = custom_reward_function(self.env, action)
        return reward + custom_reward

env = RewardWrapper(env)

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

# 增加动作噪声以促进探索
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 初始化DDPG模型
model = DDPG('MlpPolicy', env,
             policy_kwargs=dict(net_arch=[256, 256]),
             learning_rate=5e-4,
             buffer_size=100000,
             learning_starts=10000,
             batch_size=100,
             gamma=0.99,
             train_freq=(1, 'step'),
             gradient_steps=-1,  # 每次调用train()时进行的梯度步数
             tau=0.005,
             action_noise=action_noise,
             verbose=1,
             tensorboard_log="highway_ddpg/")

# 设置回调函数以保存最佳模型并记录TensorBoard数据
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./highway_ddpg/', name_prefix='best_model')
custom_callback = CustomTensorboardCallback()

# 训练模型
model.learn(total_timesteps=200000, callback=[checkpoint_callback, custom_callback])

# 保存最终模型
model.save("highway_ddpg/final_model")

# 加载并测试保存的模型
env = gym.make("highway-v0", render_mode='rgb_array')
env.configure(config)
env = DiscreteToContinuousWrapper(env)
env = ActionLoggerWrapper(env)
env = ObservationWrapper(env)
env = RewardWrapper(env)
model = DDPG.load("highway_ddpg/final_model")

# 测试训练好的模型
num_episodes = 10
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
