## 训练数据结构

训练数据为digital typhoon(全集为512*512灰度图)，内部结构如下:

root/
    197830(代表1978年的第三十号台风)/
    197901/
    ...
    202224/
    202225/

然后对其中一号台风进行拆解，得到如下详细结构:

202225/
    2022121000-202225-HMW8-1.png
    2022121001-202225-HMW8-1.png
    ...

也就是{year}{month}{date}{time}-{这里的部分不用管}.png（时间从00-23）按照文件系统的默认排序来说，是顺序的，所以不用太担心。

然后我们开始根据数据结构去确定处理方案：

对digital typhoon全集遍历，构建t-1,t,t+1的组合，然后拿这个组合生成光流法的速度矢量场。

构建的json如下:

[
  [
    "./data/digital_typhoon/199605/1996070500-199605-GMS5-1.png",
    "./data/digital_typhoon/199605/1996070501-199605-GMS5-1.png",
    "./data/digital_typhoon/199605/1996070502-199605-GMS5-1.png"
  ],
  [
    "./data/digital_typhoon/199605/1996070501-199605-GMS5-1.png",
    "./data/digital_typhoon/199605/1996070502-199605-GMS5-1.png",
    "./data/digital_typhoon/199605/1996070503-199605-GMS5-1.png"
  ],
  [
    "./data/digital_typhoon/199605/1996070502-199605-GMS5-1.png",
    "./data/digital_typhoon/199605/1996070503-199605-GMS5-1.png",
    "./data/digital_typhoon/199605/1996070504-199605-GMS5-1.png"
  ],
  ...
]

然后我们要先预处理三元组数据。在GPU推理的过程中，我们接着来完善训练流程：

目前的数据结构：data/digital_typhoon下保留着和先前所述一致的数据结构，json是根据现有数据构建的三元组，也对每个三元组都提取出了光流数据，光流数据被储存在outputs/flow当中。

更新一下，现在光流已经提取完毕了，接下来准备完成dataset和train
