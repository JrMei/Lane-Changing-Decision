import gymnasium as gym
import highway_env
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback

class DiscreteToContinuousWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscreteToContinuousWrapper, self).__init__(env)
        self.action_space = Box(low=-1.0, high=1.0, shape=(env.action_space.n,), dtype=np.float32)

    def action(self, action):
        return np.argmax(action)

# 创建并配置环境
env = gym.make("merge-v0")
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 20,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True
    },
    "vehicles_count": 20,
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
    "initial_configurations": {
        "controlled_vehicle": {
            "type": "highway_env.vehicle.controlled.DeterministicVehicle",
            "initial_state": {
                "position": [110.,   14.5],  # 设置在匝道上的初始位置，y为负数表示匝道
                "velocity": 15,        # 设置初始速度
                "heading": 0           # 设置初始方向
            }
        }
    }
}

env.configure(config)
env = DiscreteToContinuousWrapper(env)

# 确保观察空间与模型期望的输入形状一致
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(env.config["vehicles_count"] * len(env.config["observation"]["features"]),), dtype=np.float32)

    def observation(self, obs):
        return obs.flatten()

env = ObservationWrapper(env)

# 初始化TD3模型
model = TD3('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=100,
            gamma=0.99,
            train_freq=(1, 'episode'),
            gradient_steps=-1,  # 每次调用train()时进行的梯度步数
            tau=0.005,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            verbose=1,
            tensorboard_log="merge_td3/")

# 设置回调函数以保存最佳模型
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./merge_td3/',
                                         name_prefix='best_model')

# 打印所有车辆的初始位置
def print_initial_positions(env):
    print("Initial vehicle positions:")
    for vehicle in env.road.vehicles:
        print(f"Vehicle ID {id(vehicle)}: Position {vehicle.position}")

# 重置环境并打印初始位置
obs, info = env.reset()
print_initial_positions(env)

# 训练模型
model.learn(total_timesteps=20000, callback=checkpoint_callback)

# 保存最终模型
model.save("merge_td3/final_model")

# 加载并测试保存的模型
env = gym.make("merge-v0", render_mode='rgb_array')
env.configure(config)
env = DiscreteToContinuousWrapper(env)
env = ObservationWrapper(env)
model = TD3.load("merge_td3/final_model")

# 重置环境并打印初始位置
obs, info = env.reset()
print_initial_positions(env)

num_episodes = 10
for episode in range(num_episodes):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
