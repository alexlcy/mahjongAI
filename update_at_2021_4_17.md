存在的问题：
1.监督学习的结果并不好
2.RL training的learning rate目前比较大，需要调整。
3.exploration method需要改进

进展：
问题1：（本周进展）
    1.每个玩家每个action都会更新，避免了例如玩家1回合出牌后，我们的agent的feature还未更新的问题
    2.feature的history存储顺序调换 [[t-4],[t-3],[t-2],[t-1],[t]] -> [[t],[t-1],[t-2],[t-3],[t-4]]
    3.查找文献是否有相关类似迁移操作
    4.steal和discard顺序交换

问题2：
    1.RL的training部分需要experience buffer，buffer的信息来自于Rule agent和DL agent，因此需要debug DL后重新training
    2.目前有在仅使用Rule agent的experience to train RL

问题3：
    已完成部分方法的代码部分，需要最后整合



Update at 2021/4/17

