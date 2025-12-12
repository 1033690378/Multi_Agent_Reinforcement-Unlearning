# -*- coding: utf-8 -*-
# @Author: 宁静致远
# @File: multi_agent_unlearning.py
# @Software: PyCharm
# @Datetime: 2025/6/12 14:30
import random
import numpy as np
import torch
import gym
from gym import spaces
from collections import deque, defaultdict
import json
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

# 解决某些库重复加载的问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300
# 当前文件夹下有自定义.ttc字体文件
font_path = os.path.join(os.getcwd(), "msyh.ttc")
# 加载.ttc字体（需指定具体字体名称，可通过fontManager.findfont()查看）
# 先将字体添加到字体管理器
fontManager.addfont(font_path)
# 查找.ttc中包含的字体名称（可选步骤，用于确认字体名称）
# 遍历字体管理器查找刚添加的字体
for font in fontManager.ttflist:
    if font_path in font.fname:
        print(f"发现.ttc中的字体: {font.name}")  # 输出字体名称，例如"SimSun"
# 手动指定.ttc中的字体名称（根据实际输出的名称修改）
# 例如，如果.ttc包含"Microsoft YaHei"字体
font_name = "Microsoft YaHei"  # 替换为实际的字体名称
# 设置全局字体
plt.rcParams["font.family"] = font_name
# 设置中文字体，解决中文乱码
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 创建保存文件夹
save_folder = 'multi_trained_unlearning_data/grid_world'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
if not os.path.exists('models'):
    os.makedirs('models')

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# 改进的经验回放缓冲区类，支持多种遗忘策略
class ImprovedReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        # 记录每个经验所属的地图ID
        self.map_ids = deque(maxlen=capacity)
        # 按地图ID分组存储经验
        self.map_experiences = defaultdict(deque)
        # 经验优先级
        self.priorities = deque(maxlen=capacity)
        # 优先级指数
        self.alpha = 0.6
        # 新增：记录经验重要性权重
        self.importance_weights = deque(maxlen=capacity)
        # 新增：记录经验时间戳
        self.timestamps = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, map_id, priority=1.0, importance=1.0, timestamp=0):
        # 检查状态维度，如果维度小于2才进行扩展
        if state.ndim < 2:
            state = np.expand_dims(state, 0)
        if next_state.ndim < 2:
            next_state = np.expand_dims(next_state, 0)
        # 确保状态和下一个状态的形状一致
        assert state.shape == next_state.shape, f"状态形状不匹配: {state.shape} vs {next_state.shape}"

        # 将经验数据添加到缓冲区
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.map_ids.append(map_id)
        self.map_experiences[map_id].append(experience)
        self.priorities.append(priority ** self.alpha)
        self.importance_weights.append(importance)
        self.timestamps.append(timestamp)

    def sample(self, batch_size, beta=0.4, map_id=None, exclude_forgotten=True, forgotten_maps=None):
        # 基于优先级的采样
        priorities = np.array(self.priorities)

        # 如果指定了地图ID，只从该地图采样
        if map_id is not None:
            map_indices = [i for i, mid in enumerate(self.map_ids) if mid == map_id]
            if not map_indices:
                return None, None, None, None, None, None, None
            probs = priorities[map_indices] / priorities[map_indices].sum()
            indices = np.random.choice(map_indices, min(batch_size, len(map_indices)), p=probs)
        else:
            # 排除已遗忘地图的经验（如果启用）
            valid_indices = list(range(len(self.buffer)))
            if exclude_forgotten and forgotten_maps is not None:
                valid_indices = [i for i in valid_indices if self.map_ids[i] not in forgotten_maps]

            if not valid_indices:
                return None, None, None, None, None, None, None

            probs = priorities[valid_indices] / priorities[valid_indices].sum()
            indices = np.random.choice(valid_indices, batch_size, p=probs)

        if len(indices) < batch_size:
            # 不足时补充采样，确保数量正确
            remaining = batch_size - len(indices)
            additional_indices = np.random.choice(indices, remaining, p=probs if len(probs) > 0 else None)
            indices = np.concatenate([indices, additional_indices])

        # 确保权重长度与索引长度一致
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs) ** (-beta)
        weights /= weights.max()
        # 调整权重长度以匹配索引
        if len(weights) < len(indices):
            # 重复权重以匹配索引长度
            weights = np.tile(weights, (len(indices) // len(weights) + 1))[:len(indices)]
        elif len(weights) > len(indices):
            weights = weights[:len(indices)]

        state, action, reward, next_state, done = zip(*samples)

        # 去掉多余的维度
        state = [np.squeeze(s) for s in state]
        next_state = [np.squeeze(ns) for ns in next_state]
        # 检查采样的状态数组形状是否一致
        first_shape = state[0].shape
        for s in state:
            assert s.shape == first_shape, f"采样状态形状不匹配: {s.shape} != {first_shape}"
        for ns in next_state:
            assert ns.shape == first_shape, f"采样下一状态形状不匹配: {ns.shape} != {first_shape}"

        state = np.stack(state)
        next_state = np.stack(next_state)
        return state, action, reward, next_state, done, indices, weights

    def update_priorities(self, indices, td_errors):
        # 更新经验优先级
        for i, idx in enumerate(indices):
            self.priorities[idx] = (abs(td_errors[i]) + 1e-6) ** self.alpha

    # 部分遗忘机制 - 降低指定地图经验的重要性而非完全删除
    def partial_forget_map(self, map_id, importance_factor=0.1):
        if map_id not in self.map_experiences:
            return 0

        affected_count = 0
        for i, (exp, mid) in enumerate(zip(self.buffer, self.map_ids)):
            if mid == map_id:
                self.importance_weights[i] *= importance_factor
                affected_count += 1

        return affected_count

    # 渐进式遗忘 - 按比例删除指定地图的经验
    def gradual_forget_map(self, map_id, forget_ratio=0.5):
        if map_id not in self.map_experiences:
            return 0

        map_indices = [i for i, mid in enumerate(self.map_ids) if mid == map_id]
        num_to_forget = int(len(map_indices) * forget_ratio)
        if num_to_forget == 0:
            return 0

        # 按优先级排序，优先删除低优先级经验
        prioritized_indices = sorted(map_indices, key=lambda x: self.priorities[x])
        indices_to_remove = set(prioritized_indices[:num_to_forget])

        new_buffer = []
        new_map_ids = []
        new_priorities = []
        new_importance = []
        new_timestamps = []
        new_map_experiences = defaultdict(deque)

        for i, (exp, mid, prio, imp, ts) in enumerate(
                zip(self.buffer, self.map_ids, self.priorities, self.importance_weights, self.timestamps)):
            if mid != map_id or i not in indices_to_remove:
                new_buffer.append(exp)
                new_map_ids.append(mid)
                new_priorities.append(prio)
                new_importance.append(imp)
                new_timestamps.append(ts)
                new_map_experiences[mid].append(exp)

        self.buffer = deque(new_buffer, maxlen=self.capacity)
        self.map_ids = deque(new_map_ids, maxlen=self.capacity)
        self.priorities = deque(new_priorities, maxlen=self.capacity)
        self.importance_weights = deque(new_importance, maxlen=self.capacity)
        self.timestamps = deque(new_timestamps, maxlen=self.capacity)
        self.map_experiences = new_map_experiences

        return num_to_forget

    # 保留原有的完全遗忘机制
    def complete_forget_map(self, map_id):
        if map_id not in self.map_experiences:
            return 0

        # 记录要遗忘的经验数量
        all_experiences = list(self.map_experiences[map_id])
        forgotten_count = len(all_experiences)

        # 从主缓冲区中移除该地图的所有经验
        new_buffer = []
        new_map_ids = []
        new_priorities = []
        new_importance = []
        new_timestamps = []
        new_map_experiences = defaultdict(deque)

        # 重新构建缓冲区（仅保留非目标地图的经验）
        for exp, mid, prio, imp, ts in zip(self.buffer, self.map_ids, self.priorities, self.importance_weights,
                                           self.timestamps):
            if mid != map_id:
                new_buffer.append(exp)
                new_map_ids.append(mid)
                new_priorities.append(prio)
                new_importance.append(imp)
                new_timestamps.append(ts)
                new_map_experiences[mid].append(exp)

        # 更新缓冲区
        self.buffer = deque(new_buffer, maxlen=self.capacity)
        self.map_ids = deque(new_map_ids, maxlen=self.capacity)
        self.priorities = deque(new_priorities, maxlen=self.capacity)
        self.importance_weights = deque(new_importance, maxlen=self.capacity)
        self.timestamps = deque(new_timestamps, maxlen=self.capacity)
        self.map_experiences = new_map_experiences

        return forgotten_count

    def __len__(self):
        return len(self.buffer)


# 带知识蒸馏的深度Q网络
class DistilledDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DistilledDQN, self).__init__()
        # 主网络
        self.lin1 = nn.Linear(input_dim, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, output_dim)

        # 蒸馏辅助网络
        self.distill_lin1 = nn.Linear(input_dim, 128)
        self.distill_lin2 = nn.Linear(128, 64)
        self.distill_lin3 = nn.Linear(64, output_dim)

        # 特征提取器（用于知识蒸馏）
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        # 主网络前向传播
        x1 = torch.relu(self.lin1(x))
        x1 = torch.relu(self.lin2(x1))
        x1 = self.lin3(x1)
        return x1

    def distill_forward(self, x):
        # 蒸馏网络前向传播
        x2 = torch.relu(self.distill_lin1(x))
        x2 = torch.relu(self.distill_lin2(x2))
        x2 = self.distill_lin3(x2)
        return x2

    def get_features(self, x):
        # 获取特征用于蒸馏
        return self.feature_extractor(x)


# 协作式2D环境
class Cooperative2DEnvironment(gym.Env):
    def __init__(self, size=10, n_ob=10):
        super(Cooperative2DEnvironment, self).__init__()

        # 环境的大小
        self.size = size
        # 障碍物的数量
        self.n_ob = n_ob
        # 定义动作空间，包含4个离散动作
        self.action_space = spaces.Discrete(4)
        # 定义观察空间，形状为(10,)，取值范围为-1到size
        self.observation_space = spaces.Box(low=-1, high=self.size, shape=(10,), dtype=np.int32)
        # 智能体A当前状态
        self.state_a = None
        # 智能体B当前状态
        self.state_b = None
        # 障碍物列表
        self.obstacles = None
        # 目标位置
        self.target = None
        # 存储所有地图数据的列表
        self.maps = []
        # 记录是否碰撞
        self.collision_occurred = False

    def save_maps(self, filename):
        # 将地图数据保存到JSON文件中
        with open(filename, 'w') as f:
            maps = [[list(obstacle) for obstacle in map_data["obstacles"]] + [list(map_data["target"])] for map_data in
                    self.maps]
            json.dump(maps, f)

    def load_maps(self, filename):
        # 从JSON文件中加载地图数据
        with open(filename, 'r') as f:
            maps = json.load(f)
            self.maps = [{"obstacles": [tuple(obstacle) for obstacle in map_data[:-1]], "target": tuple(map_data[-1])}
                         for map_data in maps]

    def step(self, action_a, action_b):
        # 获取当前状态的坐标
        x_a, y_a = self.state_a
        x_b, y_b = self.state_b

        # 保存旧状态
        old_state_a = self.state_a
        old_state_b = self.state_b

        # 根据动作更新智能体A状态
        if action_a == 0:  # 向上
            y_a = min(y_a + 1, self.size - 1)
        elif action_a == 1:  # 向下
            y_a = max(y_a - 1, 0)
        elif action_a == 2:  # 向左
            x_a = max(x_a - 1, 0)
        elif action_a == 3:  # 向右
            x_a = min(x_a + 1, self.size - 1)

        # 根据动作更新智能体B状态
        if action_b == 0:  # 向上
            y_b = min(y_b + 1, self.size - 1)
        elif action_b == 1:  # 向下
            y_b = max(y_b - 1, 0)
        elif action_b == 2:  # 向左
            x_b = max(x_b - 1, 0)
        elif action_b == 3:  # 向右
            x_b = min(x_b + 1, self.size - 1)

        # 检查智能体之间是否碰撞
        self.collision_occurred = (x_a, y_a) == (x_b, y_b)

        # 检查是否撞到障碍物
        a_hit_obstacle = (x_a, y_a) in self.obstacles
        b_hit_obstacle = (x_b, y_b) in self.obstacles

        # 碰撞或撞障碍物时恢复状态
        if self.collision_occurred or a_hit_obstacle:
            x_a, y_a = old_state_a
        if self.collision_occurred or b_hit_obstacle:
            x_b, y_b = old_state_b

        # 更新当前状态
        self.state_a = (x_a, y_a)
        self.state_b = (x_b, y_b)

        # 判断是否到达目标位置
        a_done = self.state_a == self.target
        b_done = self.state_b == self.target
        # 只要有一个智能体到达目标，任务就完成
        done = a_done or b_done

        # 协作奖励机制
        # 基础移动惩罚
        reward_a = -1
        reward_b = -1

        # 到达目标奖励
        if a_done:
            reward_a += 100
        if b_done:
            reward_b += 100

        # 碰撞惩罚
        if self.collision_occurred:
            reward_a -= 20
            reward_b -= 20

        # 障碍物惩罚
        if a_hit_obstacle:
            reward_a -= 10
        if b_hit_obstacle:
            reward_b -= 10

        # 协作奖励：如果一个智能体靠近目标，另一个也会获得部分奖励
        a_dist = abs(x_a - self.target[0]) + abs(y_a - self.target[1])
        b_dist = abs(x_b - self.target[0]) + abs(y_b - self.target[1])
        min_dist = min(a_dist, b_dist)

        # 给予靠近目标的团队奖励
        if min_dist < 3:
            reward_a += (3 - min_dist) * 5
            reward_b += (3 - min_dist) * 5

        # 获取周围环境观察
        state_a = self.get_surrounding_cells(self.state_a)
        state_a = np.insert(state_a, 0, self.state_a)

        state_b = self.get_surrounding_cells(self.state_b)
        state_b = np.insert(state_b, 0, self.state_b)

        # 返回各自的观察、奖励和是否完成
        return state_a, state_b, reward_a, reward_b, done, {}

    def reset(self, map_index=None, game_type="grid_world"):
        if map_index is None:
            # 如果没有指定地图索引，创建新的地图
            if game_type == "grid_world":
                # 初始化两个智能体在不同位置
                self.state_a = (int(self.size / 4), self.size - 1)
                self.state_b = (int(self.size * 3 / 4), self.size - 1)
                # 初始化目标位置为中间顶部
                self.target = (int(self.size / 2), 0)

                # 随机生成障碍物
                self.obstacles = [(np.random.randint(1, self.size - 1), np.random.randint(1, self.size - 1)) for _ in
                                  range(self.n_ob)]
                # 添加边界作为障碍物
                self.obstacles += [(i, -1) for i in range(self.size + 2)]
                self.obstacles += [(i, self.size) for i in range(self.size + 2)]
                self.obstacles += [(-1, i) for i in range(-1, self.size + 1)]
                self.obstacles += [(self.size, i) for i in range(-1, self.size + 1)]

                # 保存地图数据
                map_data = {"obstacles": self.obstacles, "target": self.target}
                self.maps.append(map_data)
        else:
            # 如果指定了地图索引，加载对应的地图
            map_data = self.maps[map_index]
            # 初始化两个智能体在不同位置
            self.state_a = (int(self.size / 4), self.size - 1)
            self.state_b = (int(self.size * 3 / 4), self.size - 1)
            # 设置目标位置
            self.target = map_data['target']
            # 设置障碍物
            self.obstacles = map_data['obstacles']

        # 重置碰撞状态
        self.collision_occurred = False

        # 获取周围环境观察
        state_a = self.get_surrounding_cells(self.state_a)
        state_a = np.insert(state_a, 0, self.state_a)

        state_b = self.get_surrounding_cells(self.state_b)
        state_b = np.insert(state_b, 0, self.state_b)

        return state_a, state_b

    def get_surrounding_cells(self, pos):
        # 初始化周围单元格信息为全0数组
        surrounding = np.full(8, 0)
        # 获取当前位置的坐标
        x, y = pos
        # 定义8个方向
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        for i, (dx, dy) in enumerate(directions):
            # 计算新的坐标
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                if (new_x, new_y) == self.state_a or (new_x, new_y) == self.state_b:
                    # 如果是另一个智能体，标记为4
                    surrounding[i] = 4
                elif (new_x, new_y) == self.target:
                    # 如果是目标位置，标记为2
                    surrounding[i] = 2
                elif (new_x, new_y) in self.obstacles:
                    # 如果是障碍物，标记为3
                    surrounding[i] = 3
            else:
                # 如果超出边界，标记为3
                surrounding[i] = 3
        return surrounding

    def render(self, mode='human'):
        # 渲染环境
        for y in range(self.size + 1, -2, -1):
            for x in range(-1, self.size + 1):
                if self.state_a == (x, y):
                    # 绘制智能体A
                    print('A', end='')
                elif self.state_b == (x, y):
                    # 绘制智能体B
                    print('B', end='')
                elif self.target == (x, y):
                    # 绘制目标位置
                    print('T', end='')
                elif (x, y) in self.obstacles:
                    # 绘制障碍物
                    print('#', end='')
                else:
                    print(' ', end='')
            print()


# 带遗忘增强的DQN智能体
class EnhancedDQNAgent():
    def __init__(self, state_dim, action_dim, replay_buffer, agent_id):
        # 选择设备，如果有GPU则使用GPU，否则使用CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 动作空间的维度
        self.action_dim = action_dim
        # 折扣因子
        self.gamma = 0.99
        # 初始化DQN模型
        self.model = DistilledDQN(state_dim, action_dim).to(self.device)
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # 经验回放缓冲区
        self.replay_buffer = replay_buffer
        # 损失函数，使用均方误差损失
        self.loss_fn = nn.MSELoss(reduction='mean')
        # 用于区分不同智能体
        self.agent_id = agent_id

        # 知识蒸馏参数
        self.distillation_weight = 0.5
        # 教师模型，用于知识蒸馏
        self.teacher_model = None

        # 遗忘后恢复参数
        self.restore_factor = 0.3  # 恢复学习率因子
        self.is_recovering = False  # 是否处于遗忘后恢复阶段
        self.recovery_progress = 0.0  # 恢复进度 (0-1)
        self.recovery_phase = 0  # 恢复阶段 (0: 调整, 1: 巩固, 2: 优化)
        self.recovery_episodes = 0  # 恢复阶段已进行的轮数
        self.total_recovery_steps = 0  # 总恢复步数目标
        self.pretrained_params = None  # 遗忘前的模型参数
        self.critical_layers = ['lin3']  # 关键层，恢复时重点保护
        self.protection_factor = 0.8  # 关键层保护因子

        # 记录已遗忘的地图ID
        self.forgotten_map_ids = set()

        # 遗忘策略参数
        self.forget_strategy = "gradual"  # 可选: "complete", "partial", "gradual"
        self.forget_ratio = 0.7  # 渐进式遗忘比例
        self.importance_factor = 0.1  # 部分遗忘的重要性因子
        self.current_map_id = 0  # 当前地图ID

        # 恢复检查点
        self.recovery_checkpoints = []

    def set_teacher_model(self, teacher_model):
        # 设置教师模型用于知识蒸馏
        self.teacher_model = teacher_model

    def save_pretrained_params(self):
        # 保存当前模型参数作为遗忘前的参考
        self.pretrained_params = {k: v.clone() for k, v in self.model.state_dict().items()}

    def save_recovery_checkpoint(self, step):
        """保存恢复过程中的检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'recovery_progress': self.recovery_progress,
            'recovery_phase': self.recovery_phase,
            'step': step
        }
        self.recovery_checkpoints.append(checkpoint)
        # 只保留最近的3个检查点
        if len(self.recovery_checkpoints) > 3:
            self.recovery_checkpoints.pop(0)
        return checkpoint

    def load_best_checkpoint(self):
        """加载表现最好的检查点"""
        if not self.recovery_checkpoints:
            return False
        # 简单策略：选择最后一个检查点（可根据实际情况修改）
        best_checkpoint = self.recovery_checkpoints[-1]
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(best_checkpoint['optimizer_state_dict'])
        self.recovery_progress = best_checkpoint['recovery_progress']
        self.recovery_phase = best_checkpoint['recovery_phase']
        return True

    def gradient_unlearning(self, map_id, steps=10, batch_size=32):
        if map_id not in self.replay_buffer.map_experiences:
            return 0.0

        total_unlearn_loss = 0.0
        self.model.train()

        for _ in range(steps):
            # 只从要遗忘的地图采样
            state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(
                batch_size, map_id=map_id)

            if state is None:
                break

            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            action = torch.LongTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            done = torch.FloatTensor(done).to(self.device)

            # 计算当前Q值
            q_values = self.model(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            # 用教师模型的Q值作为目标（而非自身），实现知识移除
            with torch.no_grad():
                target_q = self.teacher_model(next_state).max(1)[0]
                expected_q = reward + self.gamma * target_q * (1 - done)

            # 梯度修正：最大化与目标的差异（反向损失）
            unlearn_loss = -self.loss_fn(q_value, expected_q)

            self.optimizer.zero_grad()
            unlearn_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_unlearn_loss += unlearn_loss.item()

        return total_unlearn_loss / steps if steps > 0 else 0.0

    def update(self, batch_size, other_agent=None, use_importance=True):
        if len(self.replay_buffer) < batch_size:
            return 0.0

        # 采样带优先级的经验，从缓冲区中采样一批数据
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(
            batch_size, forgotten_maps=self.forgotten_map_ids)

        if state is None:
            return 0.0

        # 将数据转换为PyTorch张量并移动到指定设备
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        assert len(indices) == len(weights), f"索引数量 {len(indices)} 与权重数量 {len(weights)} 不匹配"

        # 如果启用重要性权重，将其应用于损失计算
        if use_importance:
            # 确保索引和权重长度一致（以较短的为准）
            min_len = min(len(indices), len(weights))
            valid_indices = indices[:min_len]
            valid_weights = weights[:min_len]
            imp_weights = torch.FloatTensor([self.replay_buffer.importance_weights[i] for i in valid_indices]).to(
                self.device)
            weights = valid_weights * imp_weights

        assert state.shape[
                   1] == self.model.lin1.in_features, f"输入状态维度不匹配: {state.shape[1]} != {self.model.lin1.in_features}"

        # 计算当前状态的Q值
        q_values = self.model(state)
        # 计算下一个状态的Q值
        next_q_values = self.model(next_state)

        # 获取当前动作的Q值
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # 获取下一个状态的最大Q值
        next_q_value = next_q_values.max(1)[0]
        # 计算期望的Q值
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        # 计算TD误差用于更新优先级
        td_errors = q_value - expected_q_value.detach()
        self.replay_buffer.update_priorities(indices, td_errors.cpu().detach().numpy())

        # 基础Q学习损失
        loss = (weights * self.loss_fn(q_value, expected_q_value.detach())).mean()

        # 知识蒸馏损失
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_q = self.teacher_model(state)
            student_distill_q = self.model.distill_forward(state)
            distill_loss = F.kl_div(
                F.log_softmax(student_distill_q, dim=1),
                F.softmax(teacher_q, dim=1),
                reduction='batchmean'
            )
            loss = (1 - self.distillation_weight) * loss + self.distillation_weight * distill_loss

        # 协作正则化：如果存在其他智能体，增加特征一致性损失
        if other_agent is not None:
            with torch.no_grad():
                other_features = other_agent.model.get_features(state)
            self_features = self.model.get_features(state)
            cooperation_loss = F.mse_loss(self_features, other_features) * 0.1
            loss += cooperation_loss

        # 遗忘后恢复阶段：分阶段应用恢复损失
        recovery_loss = 0.0
        if self.is_recovering and self.pretrained_params is not None:
            current_params = self.model.state_dict()

            # 阶段0：快速调整，恢复主要知识
            if self.recovery_phase == 0:
                for name in current_params:
                    if name in self.pretrained_params:
                        # 关键层给予更高的保护权重
                        weight = self.protection_factor if any(layer in name for layer in self.critical_layers) else 0.3
                        recovery_loss += F.mse_loss(current_params[name], self.pretrained_params[name]) * weight

            # 阶段1：巩固，平衡新旧知识
            elif self.recovery_phase == 1:
                for name in current_params:
                    if name in self.pretrained_params:
                        weight = self.protection_factor * 0.7 if any(
                            layer in name for layer in self.critical_layers) else 0.2
                        recovery_loss += F.mse_loss(current_params[name], self.pretrained_params[name]) * weight

            # 阶段2：优化，微调适应
            elif self.recovery_phase == 2:
                for name in current_params:
                    if name in self.pretrained_params:
                        weight = self.protection_factor * 0.4 if any(
                            layer in name for layer in self.critical_layers) else 0.1
                        recovery_loss += F.mse_loss(current_params[name], self.pretrained_params[name]) * weight

            # 根据恢复进度动态调整恢复损失权重
            recovery_weight = max(0.01, 0.1 * (1 - self.recovery_progress))
            loss += recovery_loss * recovery_weight

        # 动态调整学习率
        if self.is_recovering:
            # 阶段0：较低学习率，稳定调整
            if self.recovery_phase == 0:
                current_lr = 1e-3 * self.restore_factor * 0.5
            # 阶段1：中等学习率，巩固知识
            elif self.recovery_phase == 1:
                current_lr = 1e-3 * self.restore_factor * 0.8
            # 阶段2：正常学习率，优化适应
            else:
                current_lr = 1e-3 * self.restore_factor * 1.0

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

        # 清空优化器的梯度
        self.optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # 更新模型参数
        self.optimizer.step()

        # 恢复阶段更新进度
        if self.is_recovering:
            self.recovery_episodes += 1
            self.recovery_progress = min(1.0, self.recovery_episodes / self.total_recovery_steps)

            # 检查是否需要切换阶段
            if self.recovery_progress >= 0.33 and self.recovery_phase == 0:
                self.recovery_phase = 1
                print(
                    f"智能体 {'A' if self.agent_id == 0 else 'B'} 恢复阶段切换到巩固阶段 (进度: {self.recovery_progress:.2f})")
                self.save_recovery_checkpoint(self.recovery_episodes)
            elif self.recovery_progress >= 0.66 and self.recovery_phase == 1:
                self.recovery_phase = 2
                print(
                    f"智能体 {'A' if self.agent_id == 0 else 'B'} 恢复阶段切换到优化阶段 (进度: {self.recovery_progress:.2f})")
                self.save_recovery_checkpoint(self.recovery_episodes)
            elif self.recovery_progress >= 1.0:
                self.end_recovery_phase()
                print(f"智能体 {'A' if self.agent_id == 0 else 'B'} 恢复完成!")

        return loss.item()

    def get_action(self, state, epsilon=0.1):
        # 以epsilon的概率随机选择动作
        if random.random() < epsilon:
            return random.randrange(self.action_dim), epsilon
        # 将状态转换为PyTorch张量并添加一个维度
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # 计算Q值
        with torch.no_grad():
            q_value = self.model.forward(state)
        # 获取最大Q值对应的动作
        action = q_value.max(1)[1].data[0].item()
        return action, epsilon

    def start_recovery_phase(self, total_steps):
        # 开始遗忘后的恢复阶段
        self.is_recovering = True
        self.recovery_progress = 0.0
        self.recovery_phase = 0
        self.recovery_episodes = 0
        self.total_recovery_steps = total_steps
        self.recovery_checkpoints = []  # 重置检查点
        print(f"智能体 {'A' if self.agent_id == 0 else 'B'} 开始恢复阶段，总步数: {total_steps}")

    def end_recovery_phase(self):
        # 结束恢复阶段
        self.is_recovering = False
        self.recovery_progress = 0.0
        self.recovery_phase = 0
        self.recovery_episodes = 0
        self.total_recovery_steps = 0
        # 恢复正常学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 1e-3

    # 改进：支持多种遗忘策略
    def forget_map(self, map_id):
        # 遗忘指定地图的经验
        if map_id in self.forgotten_map_ids:
            print(f"智能体 {'A' if self.agent_id == 0 else 'B'} 已遗忘地图 {map_id}，无需重复遗忘")
            return 0, 0.0

        forgotten_count = 0
        unlearn_loss = 0.0

        # 根据选择的策略执行不同的遗忘方法
        # 完整遗忘：直接移除所有该地图的经验
        if self.forget_strategy == "complete":
            forgotten_count = self.replay_buffer.complete_forget_map(map_id)
        # 部分遗忘：移除部分该地图的经验
        elif self.forget_strategy == "partial":
            forgotten_count = self.replay_buffer.partial_forget_map(map_id, self.importance_factor)
        # 梯度修正：通过梯度修正来遗忘部分经验
        elif self.forget_strategy == "gradual":
            forgotten_count = self.replay_buffer.gradual_forget_map(map_id, self.forget_ratio)

        # 无论采用哪种策略，都进行梯度修正以增强遗忘效果
        unlearn_loss = self.gradient_unlearning(map_id)

        self.forgotten_map_ids.add(map_id)
        return forgotten_count, unlearn_loss


# 经验共享函数：两个智能体之间共享指定比例的经验数据
def share_experiences(agent_a, agent_b, share_ratio=0.3, map_id=None):
    if len(agent_a.replay_buffer) > 0 and len(agent_b.replay_buffer) > 0:
        a_valid_experiences = [exp for exp, mid in zip(agent_a.replay_buffer.buffer, agent_a.replay_buffer.map_ids)
                               if mid not in agent_a.forgotten_map_ids or map_id == mid]
        b_valid_experiences = [exp for exp, mid in zip(agent_b.replay_buffer.buffer, agent_b.replay_buffer.map_ids)
                               if mid not in agent_b.forgotten_map_ids or map_id == mid]

        if a_valid_experiences and b_valid_experiences:
            a_samples = random.sample(a_valid_experiences,
                                      min(int(len(a_valid_experiences) * share_ratio), len(a_valid_experiences)))
            b_samples = random.sample(b_valid_experiences,
                                      min(int(len(b_valid_experiences) * share_ratio), len(b_valid_experiences)))

            current_timestamp = max(
                list(agent_a.replay_buffer.timestamps) + [0]) + 1 if agent_a.replay_buffer.timestamps else 1
            for state, action, reward, next_state, done in a_samples:
                state = np.squeeze(state) if state.ndim > 1 else state
                next_state = np.squeeze(next_state) if next_state.ndim > 1 else next_state
                agent_b.replay_buffer.push(
                    state, action, reward, next_state, done,
                    agent_b.current_map_id,
                    timestamp=current_timestamp
                )

            current_timestamp = max(
                list(agent_b.replay_buffer.timestamps) + [0]) + 1 if agent_b.replay_buffer.timestamps else 1
            for state, action, reward, next_state, done in b_samples:
                state = np.squeeze(state) if state.ndim > 1 else state
                next_state = np.squeeze(next_state) if next_state.ndim > 1 else next_state
                agent_a.replay_buffer.push(
                    state, action, reward, next_state, done,
                    agent_a.current_map_id,
                    timestamp=current_timestamp
                )


# 模型同步函数：使两个智能体的网络参数保持一致态调整同步比例
def sync_model_params(agent_a, agent_b, sync_ratio=0.1, map_id=None):
    # 根据当前地图动态调整同步比例
    if map_id in agent_a.forgotten_map_ids or map_id in agent_b.forgotten_map_ids:
        # 对于要遗忘的地图，降低同步比例
        sync_ratio *= 0.5

    params_a = agent_a.model.state_dict()
    params_b = agent_b.model.state_dict()

    for name in params_a:
        # 关键层参数同步比例较低，避免过度干扰
        if 'lin3' in name:  # 输出层
            layer_sync_ratio = sync_ratio * 0.3
        else:  # 其他层
            layer_sync_ratio = sync_ratio

        params_a[name] = (1 - layer_sync_ratio) * params_a[name] + layer_sync_ratio * params_b[name]
        params_b[name] = (1 - layer_sync_ratio) * params_b[name] + layer_sync_ratio * params_a[name]

    agent_a.model.load_state_dict(params_a)
    agent_b.model.load_state_dict(params_b)


# 恢复评估函数：评估恢复效果
def evaluate_recovery(agent_a, agent_b, env, map_ids, episodes_per_map=5):
    """评估智能体在指定地图上的恢复效果"""
    a_rewards = []
    b_rewards = []
    success_rates = []

    agent_a.model.eval()
    agent_b.model.eval()

    for map_id in map_ids:
        total_reward_a = 0
        total_reward_b = 0
        success_count = 0

        for _ in range(episodes_per_map):
            state_a, state_b = env.reset(map_index=map_id)
            done = False
            steps = 0
            reward_a_sum = 0
            reward_b_sum = 0

            while not done and steps < 500:
                action_a, _ = agent_a.get_action(state_a, epsilon=0.05)  # 评估时降低探索率
                action_b, _ = agent_b.get_action(state_b, epsilon=0.05)

                next_state_a, next_state_b, reward_a, reward_b, done, _ = env.step(action_a, action_b)

                reward_a_sum += reward_a
                reward_b_sum += reward_b
                state_a, state_b = next_state_a, next_state_b
                steps += 1

                if done:
                    success_count += 1

            total_reward_a += reward_a_sum
            total_reward_b += reward_b_sum

        avg_reward_a = total_reward_a / episodes_per_map
        avg_reward_b = total_reward_b / episodes_per_map
        success_rate = success_count / episodes_per_map

        a_rewards.append(avg_reward_a)
        b_rewards.append(avg_reward_b)
        success_rates.append(success_rate)

        print(
            f"地图 {map_id + 1} 恢复评估: A平均奖励 {avg_reward_a:.2f}, B平均奖励 {avg_reward_b:.2f}, 成功率 {success_rate:.2f}")

    agent_a.model.train()
    agent_b.model.train()

    return {
        'a_avg_reward': np.mean(a_rewards),
        'b_avg_reward': np.mean(b_rewards),
        'avg_success_rate': np.mean(success_rates)
    }


# 主训练函数
def train_cooperative_agents(total_maps=20, max_episodes_per_map=100, max_steps_per_episode=500, save_freq=10,
                             forget_min=3, forget_max=5, num_episodes=2000):
    """
    训练两个智能体在同一地图中协作完成任务

    参数:
        total_maps: 总地图数量
        max_episodes_per_map: 每个地图的最大训练轮数
        max_steps_per_episode: 每个episode的最大步数
        save_freq: 训练信息保存频率
        forget_min: 最少遗忘地图数量
        forget_max: 最多遗忘地图数量
    """
    # 初始化环境
    env = Cooperative2DEnvironment(10, 10)

    # # 确保地图数据目录存在
    # if not os.path.exists("map_data/grid_world"):
    #     os.makedirs("map_data/grid_world")
    #
    # # 尝试加载地图数据，如果不存在则创建新的
    # try:
    #     env.load_maps(f"map_data/grid_world/maps.json")
    # except:
    #     # 创建新地图并保存
    #     for _ in range(total_maps):
    #         env.reset()  # 创建新地图
    #     env.save_maps(f"map_data/grid_world/maps.json")

    # 加载地图数据
    env.load_maps(f"map_data/grid_world/maps.json")

    # 获取环境状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 初始化改进的经验回放缓冲区
    buffer_a = ImprovedReplayBuffer(5000)
    buffer_b = ImprovedReplayBuffer(5000)

    # 初始化增强型智能体
    agent_a = EnhancedDQNAgent(state_dim, action_dim, buffer_a, agent_id=0)
    agent_b = EnhancedDQNAgent(state_dim, action_dim, buffer_b, agent_id=1)

    # 设置不同智能体的遗忘策略（可按需调整）
    # 可选: "complete", "partial", "gradual"
    agent_a.forget_strategy = "complete"  # 完全遗忘
    agent_b.forget_strategy = "partial"  # 部分遗忘

    # agent_a.forget_strategy = "complete"  # 完全遗忘
    # agent_b.forget_strategy = "gradual"  # 渐进式遗忘

    # agent_a.forget_strategy = "gradual"  # 渐进式遗忘
    # agent_b.forget_strategy = "partial"  # 部分遗忘

    # agent_a.forget_strategy = "partial"  # 部分遗忘
    # agent_b.forget_strategy = "gradual"  # 渐进式遗忘

    agent_a.set_teacher_model(agent_b.model)
    agent_b.set_teacher_model(agent_a.model)

    # 随机选择要遗忘的地图ID（从1-19中选择3-5张）并按顺序排序
    num_maps_to_forget = random.randint(forget_min, forget_max)
    possible_maps = list(range(1, 20))
    forget_map_ids = random.sample(possible_maps, num_maps_to_forget)
    forget_map_ids.sort()  # 按序号从小到大排序
    print(f"将按顺序遗忘 {num_maps_to_forget} 张地图: {forget_map_ids}")

    # 配置遗忘参数
    agent_a.current_map_id = 0
    agent_b.current_map_id = 0

    # 记录训练统计信息
    agent_a_stats = {'episode_rewards': [], 'episode_steps': [], 'losses': [], 'unlearn_losses': []}
    agent_b_stats = {'episode_rewards': [], 'episode_steps': [], 'losses': [], 'unlearn_losses': []}
    ab_stats = {'team_rewards': [], 'success_rate': [], 'recovery_metrics': []}

    # 探索率参数
    epsilon = 1.0
    min_epsilon = 0.01
    # 探索率衰减率
    # epsilon_decay = 0.995
    # 指数衰减参数计算：2000轮后从1衰减到0.01
    # 指数衰减公式: epsilon = e^(k*episode)，其中k = ln(0.01)/num_episodes
    epsilon_decay = np.log(0.01) / num_episodes

    # 批量大小
    batch_size = 64
    # 记录已遗忘的地图数量
    forgotten_count = 0
    # 单张地图遗忘后的恢复轮数（根据遗忘地图数量动态调整）
    recovery_episodes_per_map = 50 * num_maps_to_forget
    # 当前总训练轮数
    total_episode = 0
    # 记录被选中遗忘的智能体ID，随机选择一个智能体进行遗忘
    # agent_to_forget = random.choice([0, 1])
    # 0表示A，1表示B
    agent_to_forget = 0
    print(f"选择智能体 {'A' if agent_to_forget == 0 else 'B'} 进行遗忘操作")

    # 训练前保存初始模型参数，用于遗忘后恢复
    agent_a.save_pretrained_params()
    agent_b.save_pretrained_params()

    # 记录当前训练步数（用于时间戳）
    global_step = 0

    # 遍历所有地图
    for map_id in range(total_maps):
        print(f"\n开始训练地图 {map_id + 1}/{total_maps}")

        # 每张地图开始时重置成功次数
        map_success_count = 0
        # 更新智能体当前地图ID
        agent_a.current_map_id = map_id
        agent_b.current_map_id = map_id

        # 在当前地图上训练
        for episode in range(max_episodes_per_map):
            state_a, state_b = env.reset(map_index=map_id)
            done = False
            rewards_a = 0
            rewards_b = 0
            steps = 0
            total_loss_a = 0.0
            total_loss_b = 0.0

            while not done and steps < max_steps_per_episode:
                global_step += 1  # 增加全局步数
                # 智能体选择动作
                action_a, _ = agent_a.get_action(state_a, epsilon)
                action_b, _ = agent_b.get_action(state_b, epsilon)

                # 执行动作
                next_state_a, next_state_b, reward_a, reward_b, done, _ = env.step(action_a, action_b)

                # 计算经验优先级（基于奖励和是否完成）
                priority_a = abs(reward_a) + (10.0 if done else 1.0)
                priority_b = abs(reward_b) + (10.0 if done else 1.0)

                # 存储经验，包含时间戳
                buffer_a.push(state_a, action_a, reward_a, next_state_a, done, map_id, priority_a,
                              timestamp=global_step)
                buffer_b.push(state_b, action_b, reward_b, next_state_b, done, map_id, priority_b,
                              timestamp=global_step)

                # 更新网络，传入其他智能体进行协作学习
                loss_a = agent_a.update(batch_size, other_agent=agent_b)
                loss_b = agent_b.update(batch_size, other_agent=agent_a)

                if loss_a > 0:
                    total_loss_a += loss_a
                if loss_b > 0:
                    total_loss_b += loss_b

                # 更新状态和奖励
                state_a, state_b = next_state_a, next_state_b
                rewards_a += reward_a
                rewards_b += reward_b
                steps += 1

                # 每10步进行一次经验共享
                if steps % 10 == 0:
                    share_experiences(agent_a, agent_b, share_ratio=0.2, map_id=map_id)

            # 每5个episode同步一次模型参数
            if episode % 5 == 0:
                sync_model_params(agent_a, agent_b, sync_ratio=0.05, map_id=map_id)

            # 增加总训练轮数
            total_episode += 1
            # 衰减探索率，这里等于是每轮训练结束后衰减一次
            epsilon = max(min_epsilon, np.exp(epsilon_decay * total_episode))

            # 记录统计信息
            agent_a_stats['episode_rewards'].append(rewards_a)
            agent_a_stats['episode_steps'].append(steps)
            agent_a_stats['losses'].append(total_loss_a / (steps + 1e-6))

            agent_b_stats['episode_rewards'].append(rewards_b)
            agent_b_stats['episode_steps'].append(steps)
            agent_b_stats['losses'].append(total_loss_b / (steps + 1e-6))

            ab_stats['team_rewards'].append(rewards_a + rewards_b)

            # 仅记录当前地图的成功次数
            if done and steps < max_steps_per_episode - 1:
                map_success_count += 1

            # 成功率 = 当前地图成功次数 / 当前地图已训练回合数（维度统一）
            current_success_rate = map_success_count / (episode + 1)
            ab_stats['success_rate'].append(current_success_rate)

            # 打印训练信息
            if episode % save_freq == 0:
                print(f"地图 {map_id + 1}, 回合 {episode + 1}/{max_episodes_per_map}, "
                      f"A奖励: {rewards_a:.2f}, B奖励: {rewards_b:.2f}, "
                      f"A损失: {total_loss_a / (steps + 1e-6):.4f}, B损失: {total_loss_b / (steps + 1e-6):.4f}, "
                      f"成功率: {current_success_rate:.2f}, 探索率: {epsilon:.4f}")

        # 检查是否需要执行遗忘操作
        if map_id + 1 in forget_map_ids and forgotten_count < num_maps_to_forget:
            print(f"\n对地图 {map_id + 1} 执行遗忘操作...")

            # 选择要执行遗忘的智能体
            if agent_to_forget == 0:  # 遗忘智能体A的指定地图经验
                forgotten, unlearn_loss = agent_a.forget_map(map_id)
                # 执行当前地图的遗忘操作
                print(f"智能体 A 遗忘了第 {map_id + 1} 张地图的 {forgotten} 条经验，遗忘损失: {unlearn_loss:.4f}")
                agent_a_stats['unlearn_losses'].append(unlearn_loss)
            else:  # 遗忘智能体B的指定地图经验
                # 执行当前地图的遗忘操作
                forgotten, unlearn_loss = agent_b.forget_map(map_id)
                print(f"智能体 B 遗忘了第{map_id + 1}张地图的 {forgotten} 条经验，遗忘损失: {unlearn_loss:.4f}")
                agent_b_stats['unlearn_losses'].append(unlearn_loss)

            print(f"遗忘了 {forgotten} 条经验，遗忘损失: {unlearn_loss:.4f}")
            forgotten_count += 1

            # 遗忘后开始恢复阶段
            print(f"开始恢复阶段，共 {recovery_episodes_per_map} 轮...")
            if agent_to_forget == 0:
                agent_a.start_recovery_phase(recovery_episodes_per_map)
            else:
                agent_b.start_recovery_phase(recovery_episodes_per_map)

            # 执行恢复训练
            recovery_episode = 0
            while (agent_a.is_recovering or agent_b.is_recovering) and recovery_episode < recovery_episodes_per_map:
                # 选择非遗忘地图进行恢复训练
                recovery_map_ids = [m for m in range(total_maps) if m not in forget_map_ids[:forgotten_count]]
                if not recovery_map_ids:
                    recovery_map_id = random.choice(range(total_maps))
                else:
                    recovery_map_id = random.choice(recovery_map_ids)

                state_a, state_b = env.reset(map_index=recovery_map_id)
                done = False
                rewards_a = 0
                rewards_b = 0
                steps = 0
                total_loss_a = 0.0
                total_loss_b = 0.0

                while not done and steps < max_steps_per_episode:
                    global_step += 1
                    # 选择动作（降低探索率促进利用）
                    action_a, _ = agent_a.get_action(state_a, max(0.05, epsilon * 0.5))
                    action_b, _ = agent_b.get_action(state_b, max(0.05, epsilon * 0.5))

                    # 执行动作
                    next_state_a, next_state_b, reward_a, reward_b, done, _ = env.step(action_a, action_b)

                    # 存储经验
                    priority_a = abs(reward_a) + (10.0 if done else 1.0)
                    priority_b = abs(reward_b) + (10.0 if done else 1.0)
                    buffer_a.push(state_a, action_a, reward_a, next_state_a, done, recovery_map_id, priority_a,
                                  timestamp=global_step)
                    buffer_b.push(state_b, action_b, reward_b, next_state_b, done, recovery_map_id, priority_b,
                                  timestamp=global_step)

                    # 更新网络
                    loss_a = agent_a.update(batch_size, other_agent=agent_b)
                    loss_b = agent_b.update(batch_size, other_agent=agent_a)

                    if loss_a > 0:
                        total_loss_a += loss_a
                    if loss_b > 0:
                        total_loss_b += loss_b

                    # 更新状态
                    state_a, state_b = next_state_a, next_state_b
                    rewards_a += reward_a
                    rewards_b += reward_b
                    steps += 1

                # 记录恢复阶段的统计信息
                agent_a_stats['episode_rewards'].append(rewards_a)
                agent_a_stats['episode_steps'].append(steps)
                agent_a_stats['losses'].append(total_loss_a / (steps + 1e-6))

                agent_b_stats['episode_rewards'].append(rewards_b)
                agent_b_stats['episode_steps'].append(steps)
                agent_b_stats['losses'].append(total_loss_b / (steps + 1e-6))

                ab_stats['team_rewards'].append(rewards_a + rewards_b)

                recovery_episode += 1
                if recovery_episode % (recovery_episodes_per_map // 5) == 0:
                    print(f"恢复进度: {recovery_episode}/{recovery_episodes_per_map}, "
                          f"A奖励: {rewards_a:.2f}, B奖励: {rewards_b:.2f}")

            # 强制结束恢复阶段（如果尚未结束）
            if agent_a.is_recovering:
                agent_a.end_recovery_phase()
            if agent_b.is_recovering:
                agent_b.end_recovery_phase()

            # 恢复后评估
            eval_maps = [m for m in range(total_maps) if m not in forget_map_ids[:forgotten_count]]
            if eval_maps:
                eval_result = evaluate_recovery(agent_a, agent_b, env, eval_maps[:5])  # 评估前5个非遗忘地图
                ab_stats['recovery_metrics'].append(eval_result)
                print(
                    f"恢复后评估: 平均奖励 A: {eval_result['a_avg_reward']:.2f}, B: {eval_result['b_avg_reward']:.2f}, "
                    f"成功率: {eval_result['avg_success_rate']:.2f}")

    # 保存最终模型
    torch.save(agent_a.model.state_dict(), f"models/agent_a_final.pth")
    torch.save(agent_b.model.state_dict(), f"models/agent_b_final.pth")
    print("模型已保存")

    # 保存统计数据
    with open(f"{save_folder}/agent_a_stats.pkl", "wb") as f:
        pickle.dump(agent_a_stats, f)
    with open(f"{save_folder}/agent_b_stats.pkl", "wb") as f:
        pickle.dump(agent_b_stats, f)
    with open(f"{save_folder}/ab_stats.pkl", "wb") as f:
        pickle.dump(ab_stats, f)

    # 返回训练统计信息
    return agent_a_stats, agent_b_stats, ab_stats, forget_map_ids


# 评估训练好的智能体在所有地图上的性能
def evaluate_agents(env, agent_a, agent_b, total_maps=20, eval_episodes=5, max_steps_per_episode=1000, forget_maps=None):
    print("\n开始评估智能体性能...")
    if forget_maps is not None:
        print(f"需要特别关注的遗忘地图: {forget_maps}")

    total_rewards_a = 0
    total_rewards_b = 0
    success_count = 0
    total_episodes = 0

    # 单独记录遗忘地图的评估结果
    forget_maps_rewards_a = []
    forget_maps_rewards_b = []
    forget_maps_success = 0
    forget_maps_total = 0

    # 评估所有地图
    for map_id in range(total_maps):
        # 每张地图评估多轮
        for _ in range(eval_episodes):
            # 重置环境到当前地图
            state_a, state_b = env.reset(map_index=map_id)
            done = False
            rewards_a = 0
            rewards_b = 0
            steps = 0

            # 执行评估
            while not done and steps < max_steps_per_episode:
                with torch.no_grad():
                    # 评估时禁用探索（epsilon=0，仅用当前策略）
                    action_a, _ = agent_a.get_action(state_a, epsilon=0.05)
                    action_b, _ = agent_b.get_action(state_b, epsilon=0.05)

                # 执行动作并获取反馈
                next_state_a, next_state_b, reward_a, reward_b, done, _ = env.step(action_a, action_b)

                # 更新状态、奖励和步数
                state_a = next_state_a
                state_b = next_state_b
                rewards_a += reward_a
                rewards_b += reward_b
                steps += 1

            total_rewards_a += rewards_a
            total_rewards_b += rewards_b
            total_episodes += 1

            # 记录遗忘地图的评估结果
            if forget_maps is not None and (map_id + 1) in forget_maps:
                forget_maps_rewards_a.append(rewards_a)
                forget_maps_rewards_b.append(rewards_b)
                forget_maps_total += 1
                if done and steps < max_steps_per_episode:
                    forget_maps_success += 1

            if done and steps < max_steps_per_episode:
                success_count += 1

    # 计算平均奖励和成功率
    avg_reward_a = total_rewards_a / total_episodes
    avg_reward_b = total_rewards_b / total_episodes
    overall_success = success_count / total_episodes * 100

    print(f"\n评估结果:")
    print(f"平均奖励 (A): {avg_reward_a:.2f}")
    print(f"平均奖励 (B): {avg_reward_b:.2f}")
    print(f"总体成功率: {overall_success:.2f}%")

    # 打印遗忘地图的评估结果
    if forget_maps is not None and forget_maps_total > 0:
        avg_forget_a = np.mean(forget_maps_rewards_a) if forget_maps_rewards_a else 0
        avg_forget_b = np.mean(forget_maps_rewards_b) if forget_maps_rewards_b else 0
        forget_success = forget_maps_success / forget_maps_total * 100 if forget_maps_total > 0 else 0
        print(f"\n遗忘地图评估结果:")
        print(f"平均奖励 (A): {avg_forget_a:.2f}")
        print(f"平均奖励 (B): {avg_forget_b:.2f}")
        print(f"遗忘地图成功率: {forget_success:.2f}%")

    # return {
    #     'avg_reward_a': avg_reward_a,
    #     'avg_reward_b': avg_reward_b,
    #     'overall_success': overall_success,
    #     'forget_avg_a': avg_forget_a if forget_maps else None,
    #     'forget_avg_b': avg_forget_b if forget_maps else None,
    #     'forget_success': forget_success if forget_maps else None
    # }
    return avg_reward_a, avg_reward_b, overall_success, forget_success


# 绘制训练过程结果对比图
def plot_training_results(agent_a_stats, agent_b_stats, ab_stats, forget_map_ids):
    total_episodes = len(agent_a_stats['episode_rewards'])
    episodes = np.arange(1, total_episodes + 1)

    # 绘制Rewards对比图
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, agent_a_stats['episode_rewards'], label='Agent A 奖励值', alpha=0.7)
    plt.plot(episodes, agent_b_stats['episode_rewards'], label='Agent B 奖励值', alpha=0.7)
    plt.plot(episodes, ab_stats['team_rewards'], label='团队总奖励', alpha=0.7)
    plt.xlabel('训练轮次')
    plt.ylabel('奖励值')
    plt.title(f'智能体奖励对比 (遗忘地图: {forget_map_ids})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('models', 'rewards_comparison.png'))
    plt.show()

    # 绘制Steps图
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, agent_a_stats['episode_steps'], label='Agent A 步数', alpha=0.7)
    plt.plot(episodes, agent_b_stats['episode_steps'], label='Agent B 步数', alpha=0.7)
    plt.xlabel('训练轮次')
    plt.ylabel('步数')
    plt.title(f'每轮训练步数变化 (遗忘地图: {forget_map_ids})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('models', 'steps_plot.png'))
    plt.show()

    # 绘制损失图
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, agent_a_stats['losses'], label='Agent A 损失值', alpha=0.7)
    plt.plot(episodes, agent_b_stats['losses'], label='Agent B 损失值', alpha=0.7)
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title(f'智能体损失对比 (遗忘地图: {forget_map_ids})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('models', 'losses_comparison.png'))
    plt.show()

    # 确保x和y数据长度一致（取较短的那个）
    min_length = min(len(ab_stats['success_rate']), len(episodes))
    episodes_trimmed = episodes[:min_length]
    success_rate_trimmed = ab_stats['success_rate'][:min_length]
    # 绘制成功率图
    plt.figure(figsize=(12, 6))
    # plt.plot(episodes, ab_stats['success_rate'], label='成功率')
    plt.plot(episodes_trimmed, success_rate_trimmed, label='成功率')  # 使用修剪后的数据
    plt.xlabel('训练轮次')
    plt.ylabel('成功率')
    plt.title(f'任务成功率变化 (遗忘地图: {forget_map_ids})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('models', 'success_rate.png'))
    plt.show()


# 主函数
if __name__ == "__main__":
    # 获取当前时间
    print(f"开始时间: {datetime.datetime.now()}")

    # 训练智能体
    print("开始训练双智能体协作系统...")
    agent_a_stats, agent_b_stats, ab_stats, forget_maps = train_cooperative_agents(
        total_maps=20,
        max_episodes_per_map=100,
        max_steps_per_episode=1000,
        save_freq=10,
        num_episodes=2000,
    )

    # 加载环境和智能体进行评估
    eval_env = Cooperative2DEnvironment(10, 10)
    eval_env.load_maps(f"map_data/grid_world/maps.json")

    # 初始化评估用的智能体
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.n
    buffer_a = ImprovedReplayBuffer(5000)
    buffer_b = ImprovedReplayBuffer(5000)
    agent_a = EnhancedDQNAgent(state_dim, action_dim, buffer_a, agent_id=0)
    agent_b = EnhancedDQNAgent(state_dim, action_dim, buffer_b, agent_id=1)

    agent_a.model.load_state_dict(torch.load(os.path.join('models/12', "agent_a_final.pth")))
    agent_b.model.load_state_dict(torch.load(os.path.join('models/12', "agent_b_final.pth")))

    # forget_maps = [1, 4, 8, 9, 19]
    # forget_maps = [9, 12, 14, 18]
    evaluate_agents(eval_env, agent_a, agent_b, forget_maps=forget_maps)

    # 打印训练统计信息
    print("\n训练统计信息:")
    print(f"总训练轮次: {len(agent_a_stats['episode_rewards'])}")
    print(f"遗忘的地图ID: {forget_maps}")
    print(f"Agent A 平均奖励: {np.mean(agent_a_stats['episode_rewards']):.2f}")
    print(f"Agent B 平均奖励: {np.mean(agent_b_stats['episode_rewards']):.2f}")
    print(f"平均团队奖励: {np.mean(ab_stats['team_rewards']):.2f}")
    print(f"最终评估成功率: {ab_stats['success_rate'][-1] * 100:.2f}%")

    # 绘制训练过程结果对比图最终评估成功率
    plot_training_results(agent_a_stats, agent_b_stats, ab_stats, forget_maps)

    # 获取当前时间
    print(f"结束时间: {datetime.datetime.now()}")
