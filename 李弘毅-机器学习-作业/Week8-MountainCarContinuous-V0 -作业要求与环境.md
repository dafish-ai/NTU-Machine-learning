Week-8 作业: 小车爬坡

## 作业简介：
![](assets/markdown-img-paste-20190216093324979.png)
MountainCarContinuous-V0
汽车位于一条轨道上，位于两个“山脉”之间。 目标是在右边开山; 然而，汽车的发动机强度不足以在一次通过中攀登山峰。 因此，成功的唯一途径是来回驾驶以增强动力。 如果你花费更少的精力来达到目标呢？请发挥你的聪明才智解决该问题。

>
这个问题首先由Andrew Moore在他的博士论文[Moore90]中描述。 在这里是连续版本。[Moore90]A Moore, Efficient Memory-Based Learning for Robot Control, PhD thesis, University of Cambridge, 1990.

---
备注：环境安装部分
## 安装gym环境

### 安装 gym
在 MacOS 和 Linux 系统下, 安装 gym 很方便, 首先确定你是 python 2.7 或者 python 3.5 版本. 然后在你的 terminal 中复制下面这些. 但是 gym 暂时还不完全支持 Windows, 不过有些虚拟环境已经的到了支持, 想立杆子那个已经支持了. 所以接下来要说的安装方法只有 MacOS 和 Linux 的. Windows 用户的安装方式应该也差不多, 如果 Windows 用户遇到了问题, 欢迎在留言区分享解决的方法.

#### python 2.7
```python
 pip install gym
```

#### python 3.5
```python
pip3 install gym
```

如果没有报错, 恭喜你, 这样你就装好了 gym 的最基本款, 可以开始玩以下游戏啦:

algorithmic,
toy_text,
classic_control (这个需要 pyglet 模块)
如果在安装中遇到问题. 可能是缺少了一些必要模块, 可以使用下面语句来安装这些模块(安装时间可能有点久):

#### MacOS:
```python
brew install cmake boost boost-python sdl2 swig wget
```

#### Ubuntu 14.04:
```python
  apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```
如果想要玩 gym 提供的全套游戏, 下面这几句就能满足你:

#### python 2.7, 复制下面
```python
  pip install gym[all]
```
#### python 3.5, 复制下面
```python
  pip3 install gym[all]
```

---

### 小车爬坡细节
[openai源查看：Github](https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py)

### 调用方法

```python
import gym
env = gym.make('MountainCarContinuous-v0') # 调用环境
env = env.unwrapped # 不做这个会有很多限制

print(env.action_space) # 查看这个环境中可用的 action 有多少个
print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
print(env.observation_space.high)   # 查看 observation 最高取值
print(env.observation_space.low)    # 查看 observation 最低取值

# 请补充算法部分
# ...algorithm

```

接下来的任务就是请在上述代码尾部完成对应的算法
