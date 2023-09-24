# **DiffusionDet: Diffusion Model for Object Detection**

## Abstract

DiffusionDet(扩散检测器)——一个新的框架 将目标检测模拟为一个从噪声框到目标框的去噪扩散的过程。

在训练阶段，目标框从真实标签扩散到随机分布的，模型就是要学习逆转这一噪声的过程

在推理过程，模型会循序渐进地把一组随机生成的框精细化成输出结果。

优点：灵活性比较好 使得框的动态数量和迭代评估成为可能

实验证明比之前著名的检测器表现都要好，比如在CrowdHu-man数据集上，DiffusionDet在评估时用更多的框和迭代步骤，获得了5.3 AP and 4.8 AP的增益。

## 1. Introduction

> DiffusionDet 的灵活性体现在以下几个方面：
>
> - **动态数量的框**：DiffusionDet 不需要预先设定输出框的数量，而是根据输入图像和目标的分布，自适应地生成合适数量的框。这样可以避免固定数量的框带来的限制和冗余。
> - **迭代评估**：DiffusionDet 可以在任意迭代步骤中输出检测结果，而不需要等待最终的收敛。这样可以根据不同的任务需求，灵活地调整检测精度和速度的平衡。
> - **通用性**：DiffusionDet 可以与任何基于 Transformer 的主干网络结合，如 ViT、Swin Transformer 等，从而提高检测性能。DiffusionDet 也可以适用于多种目标检测任务，如人脸检测、行人检测等。
>
> DETR
>
>  它是一种基于Transformer的端到端的目标检测模型，不需要使用锚框或者非极大值抑制等前后处理操作.它使用一个Transformer解码器作为区域提议网络（RPN），并且引入了可学习的物体查询（object queries），作为解码器的输入，来生成固定数量的区域提议框。
>
> Sparse R-CNN
>
> 它使用一个Transformer解码器作为区域提议网络（RPN），并且引入了可学习的物体查询（object queries），作为解码器的输入，来生成固定数量的区域提议框。它使用了两阶段的检测框架。它使用一个动态卷积网络（Dynamic Convolution Network）来实现区域提议框和区域特征之间的交互，而DETR使用一个Transformer解码器来实现物体查询和全局特征图之间的交互；
>
> 它使用一个迭代优化过程（Iterative Optimization Process）来逐步改善区域提议框和区域特征，而DETR使用一个Transformer解码器输出层来直接输出检测结果

简单介绍了目标检测，说明了现在大多数的检测都是通过对候选目标进行位置回归和分类，例如滑动窗口、锚框、区域建议和参考点。最近， DETR提出了一个可学习的目标询问模型，去除了需要手动设计的组件，将注意力放在问题询问上的检测范式。

文章提出了一种从噪声框一步步精细化到目标框的方法，这种方法没有需要优化的的参数，也不需要启发式的目标和可学习的询问。它进一步简化了目标的候选和推动了检测流水线的前进和发展

![image-20230904150706404](D:/CODing/pics/image-20230904150706404.png)

和图像去噪类似

本文的作者提出了DiffusionDet，用一个diffusion model通过将检测作为关于图片中心位置和尺寸大小的生成式任务。

在训练阶段，方差调度控制的高斯噪声添加到真值盒中，获得噪声盒。然后这些噪声盒送入backbone (ResNet和Swin Transformer)提取ROI。最后，这些ROI送入检测器，用来训练和预测没有噪声的真实框

在推理阶段，DiffusionDet通过反向学习扩散过程(将噪声先验分布到学习分布到边界框)产生边界框

（1）灵活的盒子数量 ：利用随机盒子作为候选对象，解偶了训练和评估阶段（2）可迭代的评估 ：由于扩散模型的迭代去噪特性，扩散网络可以以迭代的方式重复使用整个检测头，进一步提高其性能

DiffusionDet的灵活性使它在检测不同场景下的对象时具有很大的优势，例如，稀疏或拥挤，而不需要额外的微调。在COCO  CrowdHuman数据集下表现很好

![image-20230904163459475](D:/CODing/pics/image-20230904163459475.png)

主要贡献：

- 将目标检测定义为一个生成式去噪过程，这是据我们所知的第一个将扩散模型应用于目标检测的研究
- 解偶了训练和评估阶段，用于灵活的边界框设置和可迭代的评估
- DiffusionDet和目前成熟的检测器相比取得了很好的性能

## 2. Related Work

- **Object detection**

- **Diffusion model**

  从随机分布的样本出发，通过逐步去噪的过程恢复数据样本

- **Diffusion model for perception tasks**

## 3. Approach

### 3.1 前言

- **Object detection**

- **Diffusion model.**

  是一种启发于非平衡态热力学的模型。

  The forward noise process is defined as

  ![image-20230904170400834](D:/CODing/pics/image-20230904170400834.png)

​		 Zo上加高斯噪音变成Zt

​		 推到过程是从Zt经过模型用迭代 的方式推倒到Zo

![image-20230904171027626](D:/CODing/pics/image-20230904171027626.png)

### 3.2  **Architecture**

​		我们建议将整个模型分为两部分，图像编码器和检测解码器，前者只运行一次提取深度特征表示从原始输入图像x，后者把这个深层功能作为条件，而不是原始图像，逐步完善盒子Zt。

- Image encoder

  ​		将原始图像作为输入，提取高层次特征。用卷积神经网络ResNet和Swin Transformer作为backbone实现DiffusionDet。金字塔特征网络FPN用来生成多尺度的特征图谱提供给backbone

- Detection decoder

  ​		从Sparse R-CNN [91]中借用。输入为Image encoder生成的一系列用来裁剪特征图的ROI建议框，将ROI特征送入检测头来获得边界框回归和分类结果。对于DiffusionDet，这些建议框在训练阶段是从标签盒中分散出来的，并在评估阶段直接从高斯分布中采样。
  
  ​		检测器有6个阶段组成。和Sparse R-CNN的区别有：（1）DiffusionDet从随机框开始，但是Sparse R-CNN用固定的已经学习过的建议框进行学习(2)Sparse R-CNN将建议框及其特征作为输入，但是DiffusionDet只需要建议框(3)DiffusionDet可以重用检测头在迭代的方式下评估和参数共享，每一层这么做的时候是通过timestep embedding(时间步长嵌入)来特定插入进扩散过程，但Sparse R-CNN在前向推导过程中只能用一次

### 3.3 Training

​		在训练过程中，我们首先构造从真实框到噪声盒、框的扩散过程，然后训练模型来逆转这个过程

![image-20230906084008279](D:/CODing/pics/image-20230906084008279.png)

- **Ground truth boxes padding.**

  填充真实框的个数，使其达到数量Ntrain。多种填充策略， 重复框、拼接随机框或和图像大小相同的框。拼接随机框的效果最好。

- **Box corruption.** 

  在真实框中加入高斯噪声，噪声尺度由Alpha t 控制，对不同时间步长下t，采用单调递减余弦调度( the monotonically decreasing cosine schedule)。Notably, 信噪比(signal-to-noise)对模型影响比较大。4.4节会讨论

> ​			"信噪比"（Signal-to-Noise Ratio，SNR）是一个用于衡量目标信号与噪声之间相对强度的指标。信号是你想要测量或检测的真实信号，而噪声是来自各种干扰源的随机干扰或误差。一般来说 高信噪比是更好的

- **Training losses.**

  检测器将噪声化的框作为输入，预测其中的物体类别和边界框坐标。我们通过==最优传输分配方法==(an optimal transport assignment method)选择成本最小的前k个预测，为每ground truth分配多个预测

### 3.4 Inference

​		推理过程是从噪声到目标框的去噪采样过程。从高斯分布采样过的框开始，模型逐步细化其预测。

![image-20230906225837351](D:/CODing/pics/image-20230906225837351.png)

- **Sampling step.**

  > DDIM:去噪扩散隐式模型
  >
  > Jiaming Song, Chenlin Meng, and Stefano Ermon. Denois
  >
  > ing diffusion implicit models. In *International Conference*
  >
  > *on Learning Representations*, 2021. 3, 5

  每一次采样步骤后，将随机框或者预测过的框送入检测器的decoder,用DDIM来预测类别和框坐标，并将结果传入到下一次采样步骤。如果不用DDIM会带来极大的负面效果。 4.4节会讨论。

- **Box renewal.**

  每次采样过后，预测的框分为两类：期望的和不期望的。期望的包含正确位于相应对象上的，==不期望的预测==是任意分布的。不会将这些不期望的预测框输入到下一个采样，而是先将低于特定阈值的框过滤掉，然后将剩余框和==从高斯分布中抽样的新的随机框==拼接起来作为框的更新操作。

- **Flexible usage.** 

  由于随机盒子的设计，我们可以用任意数量的随机盒子和迭代次数来评估DifussionDet，这不需要等于训练阶段。作为比较，以前的方法[10,91,115]在训练和评估过程中依赖于相同数量的处理盒，它们的检测解码器在正向传递中只使用一次。

### 3.5 Discussion

我们对DiffusionDet与以前的多级探测器进行了比较分析。虽然DiffusionDet在其头部内也采用了六级结构，但其区别的特点是，DiffusionDet可以多次重复使用整个头部，以实现进一步的性能提高。然而，在以往的工作中，在大多数情况下，这些工作并不能通过重用检测头来提高性能，或者只能获得有限的性能提高。更详细的结果见第4.4节。

## 4.  Experiments

我们首先展示了DiffusionDet的灵活性。然后在COCO和CrowdHuman数据集上比较了DiffusionDet和现有的著名的检测器。最后做了DiffusionDet的消融实验。

- **COCO**

  数据集包含2017训练集中的约118K个训练图像和val2017训练集中的5K验证图像。总共有80个对象类别。我们报告了超过多个IoU阈值（AP）、阈值0.5（AP50）和0.75（AP75）的box平均精度

- **LVIS v1.0** 

  是一个大词汇量的目标检测和实例分割数据集，它有100K的训练图像和20K的验证图像。LVIS与COCO共享相同的源图像，而其注释捕获了1203个类别中的long-tailed分布。我们在LVIS评价中采用MS-COCO风格的盒度量AP、AP50和AP75。对于LVIS，训练计划分别为210K、250K和270K。

- **CrowdHuman**

  是一个覆盖各种人群场景的大型数据集。它有15K个训练图像和4.4K个验证图像，其中包括总共470K个人类实例和每张图像22.6人。根据之前的设置，我们采用评估指标作为IoU阈值0.5下的AP。

### 4.1 **Implementation Details.**

- backone: ResNet和Swin 分别用ImageNet-1K and ImageNet-21K初始化
- detection decoder:  initialized with Xavier init
- optimizer:  AdamW with the initial learning rate as 2*.*5 *×* 10*−*5 and the weight decay as 10*−*4

- 所有模型都在8个gpu上使用16的mini-batch size进行训练

- 默认的训练计划是450K次迭代，在350K次迭代和420K次迭代时，学习率除以10
- 数据增强：随机水平翻转，调整输入图像大小的比例抖动和随机裁剪

在推理阶段，我们报告了扩散det在不同设置下的性能，这些设置是不同数量的随机盒子和迭代步骤的组合。每个采样步骤的预测由NMS(非极大值抑制)集成在一起，得到最终的预测。

### 4.2 **Main Properties**

DiffusionDet 的主要性质在于==对所有推理情况的一次训练==.一旦模型被训练好，它就可以用于更改推理中的方框的数量和迭代步骤的数量，如图3和表1所示。因此，我们可以将一个扩散网络部署到多个场景中，并在不重新训练网络的情况下获得一个期望的速度-精度的权衡

- **Dynamic number of boxes**

  比较DiffusionDet和DETER在不同数量的框和询问下的AP准确率。随着框数量的增多，DiffusionDet性能逐步提升，DETER性能降低。

  ![image-20230907165040480](D:/CODing/pics/image-20230907165040480.png)

- **Iterative evaluation.**

  将迭代步数从1增加到8，进一步研究了我们提出的方法的性能，相应的结果如图3b所示。我们的研究结果表明，使用100、300和500个随机盒子的扩散网络模型随着迭代次数的增加，表现出一致的性能改进

  ![image-20230907165049360](D:/CODing/pics/image-20230907165049360.png)

- **Zero-shot transferring.**

  为了进一步验证泛化的有效性，我们在CrowdHuman数据集上对共同训练的模型进行了评估，没有进行任何额外的微调。我们提出的方法，即DiffusionDet，通过增加评估盒或迭代步骤的数量显示了显著的优势。

  ![image-20230907165023788](D:/CODing/pics/image-20230907165023788.png)

### 4.3 **Benchmarking on Detection Datasets**

- 在COCO数据集上，将DiffusionDet和其他检测器进行比较。

​		我们的DifusionDet（1@300）采用单一迭代步骤和300个评估框，通过ResNet-50主干达到45.8，大大超过了一些成熟的方法。此外，随着主干尺寸的增大，DifusionDet表现出稳定的改善。当使用ImageNet-21k预训练的Swin-Base[60]作为主干时,DifusionDet得到52.5 AP,优于强base-line，如 Cascade R-CNN和Sparse R-CNN。

​		我们目前的模型仍然落后于一些开发良好的工作，如DINO [108]，因为它使用了一些更先进的组件，如可变形的注意力[115]，更宽的检测头。其中一些技术与扩散技术是正交的，我们将探索将这些技术合并到我们当前的管道中，以供进一步改进。

- Experimental results on LVIS 

  > "Federated Loss" 的设计考虑到了分布式、去中心化的特点，因此通常需要与传统的集中式机器学习中的损失函数有所不同。它可以是一种加权平均损失，其中不同设备的损失可能根据其数据分布、设备重要性或其他因素进行加权

   reproduce Faster R-CNN and Cascade R-CNN based on detectron2。引入 federated loss来增强这些模型的表现。由于LVIS中的图像是以[34]的联邦方式进行注释的，因此负类别是稀疏注释的，这恶化了训练梯度，特别是对于罕见的类

​		   DifusionDet获得了显著的收益使用更多的评估步骤，与小的和大的骨干。此外，我们注       	意到，迭代评估与COCO相比，对LVIS带来了更多的收益

### 4.4 **Ablation Study**

我们对COCO进行了消融实验，详细研究了DifusionDet。所有实验都使用以FPN为backbone的ResNet-50和300个随机盒子进行推理，没有进一步的说明。

- **Signal scaling.** 

  The signal scaling factor 控制着扩散过程的信噪比（SNR）

  ![image-20230907175322986](D:/CODing/pics/image-20230907175322986.png)

- **GT boxes padding strategy.** 

​		对真实框不同的填充策略对模型性能的影响

![image-20230907175429813](D:/CODing/pics/image-20230907175429813.png)

- **Sampling strategy.** 

  ![image-20230907175533037](D:/CODing/pics/image-20230907175533037.png)

- **Matching between** *N*train **and** *N**eval***.

  ​         首先，无论DiffusionDet使用多少个随机盒子进行训练，精度都会随着Neval的增加而稳步提高，直到大约2000个随机盒子的饱和点。第二，当Ntrain和Neval相互匹配时，往往表现得更好

![image-20230907181153960](D:/CODing/pics/image-20230907181153960.png)

- **Running time** **vs****. accuracy**.

  我们研究了DiffusionDet在多个设置下的运行时间，这是在一个单一的NVIDIA A100 GPU上评估的，小批量大小为1。

  300个评估框，单次迭代，FPS DiffusionDet和Sparse RCNN相当，在准确率上优于Sparse RCNN。但两次迭代和一次相比，Sparse RCNN的准确率下降，DiffusionDet提升。Sparse R-CNN stage=12,性能下降。在1000个框时，DiffusionDet准确率明显提升。

![image-20230909152806014](D:/CODing/pics/image-20230909152806014.png)

DDIM是本文的扩散策略。我们相信，一个更先进的扩散策略可能会潜在地解决扩散器速度性能下降的问题，我们计划在未来的工作中探索这个问题。

- **Random Seed** 

探究推理阶段开始时随即框进行输入的随机种子对模型的影响。我们用10个不同的随机种子来评估每个模型实例，以衡量性能的分布，大多数评价结果分布接近45.7 AP。说明DiffusionDet对随机盒具有鲁棒性，并能产生可靠的结果。

### 4.5 Full-tuning on CrowdHuman

在CrowdHuman数据集上的全调优。

![image-20230909154642992](D:/CODing/pics/image-20230909154642992.png)

我们发现，与以前的方法相比，扩散网络方法取得了优越的性能。进一步将方框增加到3000个和迭代步骤都可以带来性能的提高。

## 5. Conclusion

在这项工作中，我们提出了一种新的检测范式，扩散det，通过将目标检测视为一个从噪声盒子到物体盒子的去噪扩散过程。我们的噪声到盒子的管道有几个吸引人的特性，包括盒子的动态数量和迭代评估，使我们能够使用相同的网络参数来进行灵活的评估，而不需要重新训练模型。在标准检测基准上的实验表明，与稳定的探测器相比，扩散器取得了良好的性能。

## B. Additional Experiments

### B.1 Dynamic Number of Boxes

![image-20230909155539187](D:/CODing/pics/image-20230909155539187.png)

### B.2  Iterative Evaluation

我们比较了扩散det的渐进细化特性与以前的一些方法，如DETR [10]、可变形的DETR [115]和稀疏RCNN [91]。这四种模型都有6个级联阶段作为检测解码器。细化是指前6个阶段的输出，并作为未来6个阶段的输入。
