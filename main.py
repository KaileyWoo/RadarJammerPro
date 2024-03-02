from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from myEnv import RadarEnv, JammerEnv
from os import path, makedirs
import multiprocessing

total_timesteps = int(2e8)
seed = 1


def train_radar(radar_shared_data, jammer_shared_data):
    # 保存日志路径
    date = "2024-02-24"
    models_save_dir = "./radar/models/" + date + "/"
    monitor_dir = "./radar/logs/" + date + "/monitor/"
    tensorboard_log_dir = "./radar/logs/" + date + "/tensorboard/"
    if not path.exists(monitor_dir):
        makedirs(monitor_dir)
    if not path.exists(tensorboard_log_dir):
        makedirs(tensorboard_log_dir)

    # 创建雷达智能体环境
    env = RadarEnv(radar_shared_data, jammer_shared_data)
    env = Monitor(env, monitor_dir)
    env = DummyVecEnv([lambda: env])

    # 创建 PPO 模型
    model = SAC('MlpPolicy', env, device='cuda', gamma=0, tensorboard_log=tensorboard_log_dir, seed=seed)

    print("雷达智能体日志保存路径：" + tensorboard_log_dir)
    print("开始训练雷达~")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(models_save_dir+"ppo_radar.zip")


def train_jammer(radar_shared_data, jammer_shared_data):
    # 保存日志路径
    date = "2024-02-24"
    models_save_dir = "./jammer/models/" + date + "/"
    monitor_dir = "./jammer/logs/" + date + "/monitor/"
    tensorboard_log_dir = "./jammer/logs/" + date + "/tensorboard/"
    if not path.exists(monitor_dir):
        makedirs(monitor_dir)
    if not path.exists(tensorboard_log_dir):
        makedirs(tensorboard_log_dir)

    # 创建雷达智能体环境
    env = JammerEnv(radar_shared_data, jammer_shared_data)
    env = Monitor(env, monitor_dir)
    env = DummyVecEnv([lambda: env])

    model = SAC('MlpPolicy', env, verbose=0, device='cuda', tensorboard_log=tensorboard_log_dir, seed=seed)

    print("干扰机智能体日志保存路径：" + tensorboard_log_dir)
    print("开始训练干扰机~")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(models_save_dir + "ppo_jammer.zip")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 创建共享数据结构
    manager = multiprocessing.Manager()
    radar_data = manager.Queue()
    jammer_data = manager.Queue()

    # 创建两个子进程
    process1 = multiprocessing.Process(target=train_radar, args=(radar_data, jammer_data))
    process2 = multiprocessing.Process(target=train_jammer, args=(radar_data, jammer_data))

    # 启动子进程
    process1.start()
    process2.start()

    # 其他操作

    # 等待子进程结束
    process1.join()
    process2.join()


