import numpy as np
import pandas as pd
import random
import heapq
from tqdm import tqdm
from utils import *


class GWOController():
    def __init__(self):
        self.GWO: GWO = None
    
    def use_model_1(self):
        self.GWO = GWO(
            particle_count = 9,
            dimension = 2,
            group_ratio = [0.3, 0.3, 0.4],
            parameter_space = [[-100, 100], [-100, 100]]
        )
        self.GWO.init_data()
    
    def run(self, times):
        for _ in tqdm(range(times)):
            self.GWO.iterate_once()
        print(self.GWO.get_best())
        
        
    @animate_decorat
    def show(self):
        return self.GWO.frame_list

class Singleton(type):
    _instance = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            cls._instance[cls] = super().__call__(*args, **kwargs)
        return cls._instance[cls]
    
    @classmethod
    def get_gwo(cls):
        return cls._instance.get(GWO)
    
    
class GWO(metaclass = Singleton):
    def __init__(self, particle_count: int, dimension: int, group_ratio: list[float, 3], parameter_space: list[list[float, 2]]):
        self.particle_count: int = particle_count  # 总狼数 
        self.dimension: int = dimension  # 维度
        self.group_ratio: list[float, 3] = group_ratio  # 三类狼数量比例
        self.parameter_space: list[list[float, 2], dimension] = parameter_space  # 解空间
        
        self.alpha: HeadWolfA = None  # alpha 头狼对象
        self.beta: HeadWolfB = None  # beta 头狼对象
        self.detla: HeadWolfD = None  # detla 头狼对象
        self.head_list = [self.alpha, self.beta, self.detla]
        
        self.group: list[CommonWolf] = []  # 全部狼
        self.alpha_group = []  # alpha 类狼
        self.beta_group = []  # beta 类狼
        self.detla_group = []  # detla 类狼
        
        self.frame_list = []  # 动画[动画帧[[点[x,y], ...],[头狼坐标[x,y], ...]], ...]
        
    def init_data(self):
        self.group = [CommonWolf(self.dimension, self.parameter_space) for _ in range(self.particle_count)]
        self.alpha_group, self.beta_group, self.detla_group = split_by_ratio(self.group_ratio, self.group)
        
        for w in self.alpha_group:
            w.layer = 0
            
        for w in self.beta_group:
            w.layer = 1
            
        for w in self.detla_group:
            w.layer = 2
    
    
    def iterate_once(self):
        self.choose_head_wolf()
        
        points = []
        heads = []
        for i, w in enumerate(self.group):
            w.update()
            points.append([w.vec[0][0], w.vec[1][0]])
        for head in self.head_list:
            heads.append([head.vec[0][0], head.vec[1][0]])
        self.frame_list.append([points, heads])
    
    
    def choose_head_wolf(self):
        if self.alpha is not None:
            self.group.append(CommonWolf(self.alpha))
            self.group.append(CommonWolf(self.beta))
            self.group.append(CommonWolf(self.detla))
            self.group.remove(self.alpha)
            self.group.remove(self.beta)
            self.group.remove(self.detla)
        
        self.group = [e for e in self.group if e is not None]
        
        living_value_list = [w.living_value for w in self.group]
        
        max_index_tuple_list: list[tuple[int, 2], 3] = heapq.nlargest(3, enumerate(living_value_list), key=lambda x: x[1])
        self.alpha: HeadWolfA = self.group[max_index_tuple_list[0][0]]  # alpha 头狼对象
        self.beta: HeadWolfB = self.group[max_index_tuple_list[1][0]]  # beta 头狼对象
        self.detla: HeadWolfD = self.group[max_index_tuple_list[2][0]]  # detla 头狼对象
        self.head_list = [self.alpha, self.beta, self.detla]
    
    
    def get_best(self):
        return (self.alpha.vec, self.alpha.living_value)
    




class BaseWolf():
    def __init__(self):
        self.gwo = Singleton.get_gwo()
    
    def update(self):
        self.update_func()
        self.living_value = living_value(self.vec)
    
    def update_func(self):
        pass
    
    def __str__(self):
        return f'layer:{self.layer}, vec:{self.vec}\n'


class HeadWolf(BaseWolf):
    def __init__(self, wolf: BaseWolf):
        super().__init__()
        self.vec = wolf.vec.copy()
        self.dimension = wolf.dimension
        self.parameter_space = wolf.parameter_space.copy()
        self.living_value = wolf.living_value
        self.layer = wolf.layer
        self.update_args = None
        del wolf
        
    def update_func(self):
        pass


class HeadWolfA(HeadWolf):
    def update_func(self):
        delta_vec = self.gwo.delta.vec
        beta_vec = self.gwo.beta.vec
        self.vec = delta_vec + random.uniform(0,2) * (beta_vec - delta_vec)
        
        
class HeadWolfB(HeadWolf):
    def update_func(self, alpha_vec, delta_vec):
        delta_vec = self.gwo.delta.vec
        alpha_vec = self.gwo.alpha.vec
        self.vec = delta_vec + random.uniform(0,2) * (alpha_vec - delta_vec)
        
        
class HeadWolfD(HeadWolf):
    def update_func(self, alpha_vec, beta_vec):
        alpha_vec = self.gwo.alpha.vec
        beta_vec = self.gwo.beta.vec
        self.vec = self.vec + random.uniform(0,2) * (alpha_vec + beta_vec - 2 * self.vec)
        


class CommonWolf(BaseWolf):
    def __init__(self, dimension_or_headwolf: int|HeadWolf, parameter_space: list[list[float, 2]]|None = None):
        super().__init__()
        if isinstance(dimension_or_headwolf, int):  # 初始化坐标，新生成狼
            self.dimension = dimension_or_headwolf
            self.vec = []
            self.parameter_space: list[list[float, 2], self.dimension] = parameter_space  # 解空间
            self.living_value = None
            self.layer = None
            #--------------------------
            for min_r, max_r in self.parameter_space:
                self.vec.append([random.uniform(min_r, max_r)])
            self.vec = np.array(self.vec)
            self.living_value = living_value(self.vec)
        
        else:  # 将头狼还原回普通狼
            wolf = dimension_or_headwolf
            self.vec = wolf.vec.copy()
            self.dimension = wolf.dimension
            self.parameter_space = wolf.parameter_space.copy()
            self.living_value = wolf.living_value
            self.layer = wolf.layer
            del wolf
    
    def update_func(self):
        head_vec = self.gwo.head_list[self.layer].vec
        self.vec = head_vec + random.uniform(0,2) * abs(random.random() * head_vec - self.vec)
        
        