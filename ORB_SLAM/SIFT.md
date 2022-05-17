# 0. SIFT

SIFT(Scale-Invariant Feature Transform)自1999年由David Lowe提出以后被广泛的应用于CV的各种领域：图像识别，图像检索，3D重建等等，可谓无人不知无人不晓。其应用太广泛了，而且Lowe还申请了专利，以至于想商业用的人都很担心。它的步骤可以主要分两步：1）特征点检出 keypoint localisation，2）特征点描述 feature description。

**特征点检出**主要是用了DoG，就是把图像做不同程度的高斯模糊blur，平滑的区域或点肯定变化不大，而纹理复杂的比如边缘，点，角之类区域肯定变化很大，这样变化很大的点就是特征点。当然为了找到足够的点，还需要把图像放大缩小几倍(Image Pyramids)来重复这个步骤找特征点。其实DoG并不是Lowe提出的，很久以前就有了，读过SIFT专利的人都知道，SIFT的专利里面也不包括这部分。可代替特征点检出还有很多其他方法如MSER等。

**特征点描述**就是一个简单版的HOG，即以检出的特征点为中心选16x16的区域作为local patch，这个区域又可以均分为4x4个子区域，每个子区域中各个像素的梯度都可以分到8个bin里面，这样就得到了4x4x8=128维的特征向量。特征点检出以后还需要一个很重要的步骤就是[归一化](https://www.zhihu.com/search?q=%E5%BD%92%E4%B8%80%E5%8C%96&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2222476595%22%7D)，计算这个patch的主方向，然后根据这个主方向把patch旋转到特定方向，这样计算的特征就有了 **方向不变性** ，也需要根据patch各像素梯度大小把patch缩放到一定的尺度，这样特征就有了 **尺度不变性** 。

很多人都直接把这两步统称为SIFT，其实确切来说SIFT是指第二步的特征，而SIFT features或SIFT descriptor的说法应该比较准确。


SIFT是由UBC（university of British Column）的教授**David Lowe** 于1999年提出、并在2004年得以完善的一种检测图像关键点（key points , 或者称为图像的interest points(兴趣点) )， 并对关键点提取其局部尺度不变特征的描绘子， 采用这个描绘子进行用于对两幅相关的图像进行匹配（matching）。

Matlab代码链接，见文后。或者点赞、收藏后，留言区留下email地址。

---

## 主要资料

SIFT学习资料，经过整理，主要包括：

* **2004 ijcv David G. Lowe Distinctive Image Features from Scale-Invariant Keypoints**

这是SIFT发明人Lowe在SIFT完善的时候，发表在计算机视觉最权威的期刊IJCV上的论文，讲述了SIFT的原理和构成。

![](https://pic4.zhimg.com/v2-a75d279b3d3a5fd80a123674dceb2563_b.jpg)

* 2015 Feature Extraction Of Real-Time Image Using Sift Algorithm
* 2017 Implementation of the Scale Invariant Feature Transform Algorithm in MATLAB 澳洲国防部

澳大利亚国防部的一个技术报告，将SIFT用于

* 2004 SIFT matlab tutorial

讲述了IFT在matlab中实现的案例。

## SIFT模型的来历

SIFT是由UBC（university of British Column）的教授**David Lowe** 于1999年提出、并在2004年得以完善的一种检测图像关键点（key points , 或者称为图像的interest points(兴趣点) )， 并对关键点提取其局部尺度不变特征的描绘子， 采用这个描绘子进行用于对两幅相关的图像进行匹配（matching）。

目前， SIFT可以说是所有图像局部特征描述特征子 中最广为接受的一种方法。关于SIFT的描述详见David Lowe与2004年发表的论文<`<Distinctive image features from scale-invariant key points>`>, International journal of Computer Vision 2004.。

SIFT发明者David Lowe 也一直是我比较佩服的计算机视觉领域的大牛之一。 下图是David Lowe的照片：

![](https://pic4.zhimg.com/v2-52f591ab7dc7a504f48ba885d6dca45f_b.jpg)

目前， SIFT已经申请了专利。 专利持有人是UCB大学。 这里简要介绍一下SIFT的原理和实现细节。

任何一个好的图像特征描绘子， 应该满足如下几点要求：

（1）Distinctive features(so that we can correctly matched against a large database of features from many images)。

（2）Invariance to image scale and rotation. 也就是说， 当图像发生了尺度变化（缩小， 放大等等）， 或者是图像发生旋转的时候， 我们能够保证我们提取到的特征是不变的， 没有任何的特征描述上的变化。这是SIFT的 **最主要特征** ，这也是为什么这个特征的名字叫做scale invariant features 吧。

（3） Robustness to Affine distortion, change in 3D viewpoint, Addition of noise, change in illumination。 也就是说， 我们的特征不会因为图像中场景中多了噪声， 发生了仿射失真， 或者亮度改变， 或者我们拍摄图片的时候发生了视角的变化等因素而变化。 我们的特征对这些影响保持鲁棒性。

![](https://pic3.zhimg.com/v2-78cfa60afa78bf7eb29b2473fa0349fe_b.jpg)

例如上面图片中的左右部分， 对同一个物体（车辆）， 从不同的角度拍摄得到的两幅图像， 虽然有occlusion， other objects， clutter， rotation等等， 但是当我们计算图像对应位置的SIFT features的时候， 二者对应的特征basically the same。 所以， 只要是相同的object， 我们就可以用SIFT匹配。

满足上面的几个特点的这样一个特征很难找到。 然而David Lowe 找到了， 这就是SIFT， 这也是SIFT产生的Motivation吧。 SIFT特征的优点由如下几种：

(1) Locality：特征是局部的， 所以对于遮挡（occusion）和复杂的背景（clutter）的鲁棒性非常好， very rebust.

(2) Distinctiveness: 特征具有独一无二性。 也就是说 individual features can be matched to a large database of objects.

(3)Quantity: 即使是一个很小的物体（objects）, 也能够产生大量的SIFT features.。

(4) Efficiency： SIFT接近于实时的性能， 效率很高。

在这里， 可能会遇到的问题：

## Q1： 什么是图像的局部特征描绘子。

前面说了SIFT是图像的局部特征描绘子。 英文就是Local features。 与之相对应的是全局特征（global features）。 给定一幅图像， 我们计算出整幅图像的直方图（histogram）， 那么这个直方图就是这幅图像的全局特征描绘子（global feature）， 如果采用滑动窗口的办法， 例如窗口的大小为4x4等， 然后计算窗口的所有像素的均值， 将这个均值作为中心像素的邻域描述， 那么这个就是局部特征。 类似的， SIFT计算关键点的SIFT特征的时候， 就用到了关键点周围的邻域（neighborhood）的信息统计， 所以SIFT就是局部特征描绘子。

## Q2： detector（检测子）和descriptor （描绘子）的区别是什么

首先， detector（检测子）告诉我们我们感兴趣的位置在哪里， descriptor 的作用是我们如何去描述这个感兴趣的位置（特征）。 一个是关于where， 一个是关于how 的问题。 举个例子， 光检测边缘的检测子（detector）就有几十种了， 例如Canny detector， Laplacian 啊 等等。 这些检测子只是告诉你边缘发生在图像的那些位置。 描述子（descriptor）是关于我们如何去描述的， 层次比detector高。 如果我们能够正确的描述一个问题， 那么我们基本上就成功一半了。 对于SIFT， 第一步是关于detector， 也就是检测出关键点（感兴趣点）， 我们是找到这些感兴趣点在哪里？ 这是我们的第一步， 第二是关于descriptor， 也就是描述我们所找到的感兴趣点。 这就是SIFT features.。 这样我们就可以利用这些SIFT features 去做计算机视觉中比较advanced tasks 了。

下面介绍一下SIFT找到（提取）图像中的Key points, 并且使用key Point descriptor 去描述这些关键点的步骤。

---

## SIFT的四个步骤

（1）尺度空间峰值选择（Scale space peak selection）, 这一步的目的是在尺度空间中选择选择潜在的满足尺度不变性和旋转不变性的关键点（或者称为兴趣点）。

（2）关键点定位（key point localization）: 这一步的目的就是精确定位出特征关键点的位置, 涉及到剔除伪关键点。

（3）方向分配（orientation assignment）： 这一步的目的就是基于关键点的局部梯度方向，给每个相应的关键点分配方向（Assign orientation to key points）。

(4) 关键点描述（Key Point descriptor）： 此步骤的目的就是对于每个关键点， 用一个高维度（high dimensional vector， 在下面的例子中是128 维的向量）的向量去描述每个关键点。

下面详细介绍上面的四个步骤。

## （1）尺度空间峰值选择

首先， 给定一幅图像， 要想使用SIFT提取出潜在的感兴趣点， 我们的第一步就是构造图像的尺度空间（Scale space）。我们采用的办法就是采用Laplacian 0f Gaussian(LoG) 去构建图像的尺度空间。 构造完成之后， 我们通过选择LoG 尺度空间上所有的的local maxima， 作为我们的interest point(兴趣点)，这也是stable features， 这样我们就完成了第一步。

具体解释如下：

给定一组图像：

![](https://pic4.zhimg.com/v2-9fb09a641ef24e2d47737a534e7306d3_b.png)

我们要提取出这幅图像的兴趣点， 第一步是不同variance（方差）的Gaussian filter （离散域是Gaussian kernel: 高斯核）进行滤波。

![](https://pic1.zhimg.com/v2-f420ae37a586d244f25e1c7bb7b35528_b.png)

其中， I（x, y ）表示原图像， G是高斯核。 由于在尺度空间中， G是三维的函数：

![](https://pic2.zhimg.com/v2-090b271ba4a867f01d9a526c04321f91_b.jpg)

我们通过选择不同的标准差（注意

![](https://pic2.zhimg.com/v2-c1a7b49628bccad461d9d4d9eaa1a845_b.png)

是标准差， 不是方差）， 从而得到与原图像对应的许多幅不同的尺度（这里尺度代表标准差）高斯滤波器平滑滤波（blur）后的图像。

对于下面的图像， 不同的方差的高斯核函数作用于这幅图像得到如下对应的几幅图像（参见下图）：

![](https://pic3.zhimg.com/v2-6ca1358173b7c27755468a3aac7748f2_b.jpg)

![](https://pic4.zhimg.com/v2-a1a3c44637328bb9f025daf749a018df_b.jpg)

不难发现， 随着方差t的变大， 滤波后的图像平滑的效果越来越明显， 相应的， 图像也就损失了越来越多的细节。 为了从直观上给出解释， 下面我们画出了一维Gaussian 图像的曲线图：

![](https://pic3.zhimg.com/v2-bb2f3acb08e35bff5c958b4d8ca60a9e_b.jpg)

不难看出， 随着方差的减小， 越来越尖， 平滑效果不明显， 随着方差增大， 高斯函数越来越flat， 平滑效果越来越明显。

言归正传， 对于图像：

![](https://pic4.zhimg.com/v2-9fb09a641ef24e2d47737a534e7306d3_b.png)

我们使用一系列的不同方差的高斯滤波函数对其滤波平滑后的图像， 然后使用对所有得到的图像采用Laplacian operator, 得到相对应的对应的图像。

![](https://pic4.zhimg.com/v2-be8d9dbffcfe61e8b2695a1ccfccb24f_b.png)

然而， 上述的做法分了两步， 也就是先平滑处理， 后采用Laplacian operator， 事实上， 我们可以一步到位， 对于原图像， 直接采用LoG (Laplacian of Gaussian)算子即可完成上面的操作。

这样， 我们对于图像， 使用一系列的方差的LoG算子得到如下：

![](https://pic2.zhimg.com/v2-3a3f5c0c3d37c8a98427959bdf6a89d1_b.jpg)

我们于是得到右边三维的空间（其中x表示行， y 表示列，

![](https://pic3.zhimg.com/v2-945a37f0da37589a7d86cf949d007cda_b.png)

表示不同的高斯标准方差）等等得到的图像。 上面的几幅图像都是在同一尺度的。 也就是说没有下采样。NOTE： 只有统一尺度（这里的尺度scale的意思是size）的图像， 才有相互比较的意义。

例如， 上面同一尺度（size）得到了三幅图像。 为了利用这个三维空间中找到图像的中的兴趣点， 我们需要找到这个尺度的3x3邻域的局部极值（由于这个尺度共有3幅图像， 所以共需要比较绿色区域中心点附近3x3 x 3 -1（即26个像素）， 看看这个这个中心点像素的的值是否大于周围的26个像素。 如果是， 我们就说这个中心点是局部极值。 这个点就是我们的兴趣点。 一次类推， 找到所有的 ***极值点（兴趣点）*** 。

题外话：

LoG可以用作“blob” detector， 如下， 对于不同大小的blob， 运用不同大小的LoG（即不同方差）（也即采用scale space 用于检测不同大小的blob）出现极值的情况， 不难看出， 对于一定大小的圆， 只有一定方差的LoG才能检测出极值的位置。

![](https://pic1.zhimg.com/v2-70eeed9b200526b689dfd51b66a0f534_b.jpg)

再比如， 对于下图：

![](https://pic3.zhimg.com/v2-e8ee6f5128f14c45fcd72724bfdfdabe_b.jpg)

不同的sigma的对应的尺度空间中进行blob 检测的结果如下：

![](https://pic1.zhimg.com/v2-9c07b158676b8b58b71b4daa921ed3cc_b.jpg)

如果我们利用上面的原始图像得到scale space， 对于每一个像素， 求出尺度空间中的局部极值， 我们最终会得到下图中检测到的blob的情况：

![](https://pic1.zhimg.com/v2-a26b270d5400a3d2e8c81bd37439a708_b.jpg)

上图中不同大小的红色圆圈表示不同的尺度空间（即不同sigma）中的极值（极值所位于的sigma的位置）。 相同大小的圆圈表示在相同的极值位于相同的sigma对应的图像中找到的。

言归正传， 回到SIFT中， 下面构建尺度空间。

为了得到尺度不变特征（scale invariant）， 我们必须在所有的尺度空间中检测出相应的兴趣点。 如果使用LoG去构建(build)尺度空间， 求二阶导数会导致计算量很大。

为了解决这个问题， 我们采用DoG去近似逼近LoG， 具体推导如下：

通常情况下， 我们选择：

关于为啥通常会选择者两个值， 后面将会给出解释。

![](https://pic3.zhimg.com/v2-b729ca1cdd37db4566ac0335127a4962_b.jpg)

构造尺度空间具体如下图：

![](https://pic4.zhimg.com/v2-4c16fc14fd7db4bcb45c1a4b99885207_b.jpg)

![](https://pic4.zhimg.com/v2-e03e57f93815888370f4d668b1ca167b_b.png)

对应的Gaussian的方差具体取值情况如下图所示：

![](https://pic2.zhimg.com/v2-db5ffe7594d87a50a8f42fe0d88b4f19_b.jpg)

对于上面的一幅图像， 我们需要注意如下几点：

（1）对于每一个octave， 共有5幅经过不同的标准差的Gaussian 滤波器过滤后的图像。 对于每一个octave的从下向上的图像， 作用于上一幅图像滤波器的标准差是与其相邻下一幅的滤波器标准差的k 倍。 不同的ocatve的最底层， 使用的滤波器上一octave的标准差是下一octave最底层的标准差的k^2倍。你可能会问问什么？ 原因是上一octave的图像是下一层图像行列个下采样2 every other row and column， 所以分辨率变小了， 所以滤波器的标准差也变成了原来的2倍（也就是k^2）。

（2）同一octave对应着相同的尺度（size 的图像）， 求出相邻图像的差， 也就是求出DoG， 得到了对应的尺度空间（在这里， 每一个octave对应着4幅DoG图像）。

当我们取值：

不难计算出不同octave， 不同的Gaussian滤波器对应的标准差如下表：

![](https://pic2.zhimg.com/v2-8b8600297acfed4a58265c00d8803251_b.jpg)

## Q： 每一个octave究竟需要多少个scale呢（也就是几幅不同标准差的Gaussian滤波后的图像）呢？

要回答这个问题， 需要一幅实验得到的图， David Lowe 经过试验测试， 给出了如下图：

![](https://pic3.zhimg.com/v2-c4da086887ae08642022180604624542_b.jpg)

有上面左侧图像， 可以看到重复率最好的位置是每个octave的scale数目为3的时候， 然而在超过5以后，就保持flat， 所以选择5 是一个比较好的折中。

Q2 sigma的初试值应该选为多少合适呢？

之前我们已经给出了sigma的一般取值是1.6. 这并不是凭空想象出来的。 这个值也是David Lowe 经过测试得到的最好的结果。 David 经过测试得到，上图右侧图像，由上表不难看出， 重复率最好的位置是当sigma取值为1.6的时候。

这样， 我们就构建出了尺度空间， 接下来， 我们需要根据我们得到的尺度空间得到每一个octave所有的局部极值（峰值检测）， 这个检测到的局部峰值就是我们的兴趣点。

详见下图（前面有介绍过， 这里不再赘述）。

至此为止， 上面完成了SIFT算法的第一步。 下面介绍第二步。

## （2）关键点定位（key point localization）:

这一步的目的就是精确定位出特征关键点的位置。

在上一步中， 我们求出了图像中所有的局部极值点。 我们选取其位置。

![](https://pic2.zhimg.com/v2-8b8600297acfed4a58265c00d8803251_b.jpg)

如上图。 但是并不是所有的极值点都是关键点。 而且由于对于一幅图像， 我们可能找到很多的极值点， 所以我们需要剔除掉一些不是关键点的点（可能是边缘的点， 因为DoG对于边缘有很强的响应）。

剔去这是伪关键点可以使用两大步。

1.initial outlier rejection

这一步可以说是粗剔去。 主要剔去的是低对比度的候选关键点（candidates） 和poorly localized candidates along an edge。

为了找到剔除判别标准， 我们下面对尺度空间(DoG)进行taylor series展开：

![](https://pic1.zhimg.com/v2-8307124fc9b60000ab30896859689cf8_b.jpg)

注意上面的x 是一个三维向量。 对D求导得到的是一个矩阵。

对上面的Taylor再次求导， 令其等于0， 得到的就是极值的准确的位置。

当然会有很多的极值点， 求出来的都是准确的兴趣点的位置。

下面给出我们剔除非关键点的标准：

![](https://pic1.zhimg.com/v2-3f04e2ddbc85dd1a65dd1876f647b904_b.png)

th 是判别阈值。 凡是大于th的兴趣点， 我们都保留下来， 凡是小于th 的兴趣点我们都剔除。

例如下例， 我们未剔除之前， 共有832个关键点， 使用th = 0.03 的规则之后， 最后剔除剩下了729个关键点：

![](https://pic2.zhimg.com/v2-0825b7724752db3c1edfa31990ae30c5_b.jpg)

## Further outlier rejection

上面剔除的方法只是比较粗糙的剔除。 下面我们采用边缘响应的方式更进一步的剔除关键点：

这里剔除关键点所基于的原理是DoG算子对于边缘具有很强的响应。

现在我们将DoG多尺度图像想想成一个surface。 我们接下来计算DoG surfec 的主曲率（principle curvatures）。 由于沿着边缘的方向（along the edge） 其中的一个主曲率的值会很小， 垂直边缘的方向（across the edge）主曲率会很高。 这一点和PCA是有点相似的， PCA关注数据的变化的两个方向。

所以我们接下来计算DoG surface D的Hessian matrix（其中

![](https://pic2.zhimg.com/v2-2965d9fad4789b230923cf6521e0eea9_b.png)

是D的特征值） :

![](https://pic1.zhimg.com/v2-c8cab4302ef55015717bcef4ade02c94_b.jpg)

接下来， 我们按照如下方式剔除非关键点。

![](https://pic2.zhimg.com/v2-d9596a28d258480ae6631c1fb01e2185_b.jpg)

不难看出， 当r = 1 的时候， 上式取得最小值， 为4， 而且右边的式子会随着r的增大而增大。

所以我们规定， 凡是使得r > 10的点， 我们都将该点视为outlier(即不是关键点)， 我们将这些点剔除。

效果见下图， 不难看出经过这一步， 关键点的数目有729 降为536个：

![](https://pic2.zhimg.com/v2-442c7ba88add57b6ddeafd21475a9439_b.jpg)

至此， 我们算是完成了第二步， 接下来， 我们开始SIFT的第三步： Orientation Assignment

## （3）给关键点方向分配

这是SIFT的第三步。 目的是实现SIFT特征的旋转不变性（orientation invariance）。

我们所做的就是， 对于在某个尺度平滑后的图像L， 我们计算出L在每个关键点出的中心导数， 进而计算出每个关键点（x, y）处的magnitude 和direction 信息。此时每个关键带你所有的信息有： 位置， 方向， 所处的尺度。

计算公式如下：

![](https://pic1.zhimg.com/v2-4c15b39c13a3ee0822bb7de9f221250c_b.png)

为了确定关键点的方向信息， 我们采用如下方法：

对于每一个关键点， 我们选取其周围16 x 16 的窗口（neighbourhood）（这个窗口很大）， 我们将这个窗口内的所有像素（共256个）的梯度量化到36个bin 的直方图中。

然后统计。 不难看出每个bin的方向范围是10度（360/36 = 10°/bin）。

值得注意的是， 我们的灰度直方图并不是简单的对方向个数进行统计， 而是weighted direction histogram in a neighbourhood of a key point (36bins)。

对于每一个像素点， 这些方向的weights 是该像素点的gradient magnitudes.。 举个例子， 如果某个像素点的梯度方向是15°， 在该像素点的梯度是12， 那么我们就在10 °- 19°这个bin中统计的时候加上12., 而不是简单的加上1了。

由于16 x 16 是很大的窗口， 简便起见， 我们下面以一个4 x 4 的窗口说明最终得到的方向图像：

![](https://pic2.zhimg.com/v2-385b998837cd471690c1a3590f5b5f79_b.jpg)

不难知道， 上面的方向的长短表示这个方向的权重（也就是这个像素点梯度的magnitude）。 我们对上述图像进行统计， 得到如下的统计直方图：

接下来， 我们选择直方图中的峰值（peak）作为这个关键点的方向。

注意， 有的是偶， 我们也会引入additional key points， 因为在这里也出现了local peaks(是Max peaks 的0.8)， 这个额外的关键点与本关键点的位置相同， 但是关键点的方向却不同， 如上图。

至此， 第三步已经完成了， 下面介绍

## 第四步，关键点描述子。

（1）关键点描述

完成了上面的SIFT的三大步， 接下来我们就需要找到一个关键点的局部图像描绘子了（Local Image Descriptors at key points）。

其中， 一个可能的描绘子就是将关键点邻域附近的灰度样本存储起来， 以便描述这个关键点点。 然而如果这样做的话， 效果会很差， 因为这个描绘子对于灰度光变化非常敏感等等

所以我们选择的是使用梯度方向直方图来描述这个关键点（gradient orientation histograms）。 这是一个关键点的robust representation。

因为梯度比较稳定， 记录的是周围的变坏。

具体的， 我们的做法如下：

计算出 每个关键点周围16x 16的neighbourhood周围相对的orientation和magnitude。 然后将这16 x 16 的window 分成4 x 4 的块。 所以每个关键点的这个neighbourhood 就总共有16个blocks.。 对于每一个blocks, 我们统计一个8bins的 weighted histogram。 权重是magnitude 和这个关键点所处的spatial Gaussian。

然后， 我们将这16个histogram (因为有16个块) concatenate ， 从而组成一个long vector of 128 dimensions（即16 x 8 = 128）。

为了简便分析， 下面假如去关键点附近的8 x 8 的neighbourhood。 将其分成4 x 4 的block， 这样就有2 x 2 （即4个）blocks.

![](https://pic4.zhimg.com/v2-93820198c918828447e4480135f9d9c3_b.jpg)

此时描述这样的一个关键点的的向量的维数为4 x 8 = 32 维.

Q: 为什么选择将16 x 16 的窗口分成了size4 x 4的block共16个blocks 呢？？

首先这样划分也不是David的主观臆断的， 而是经过试验测试的， 这样划分blocks 的实验效果好， 具体的如下：

![](https://pic3.zhimg.com/v2-b6ca48fe6490d961dc5f0ddf51aa93f2_b.jpg)

从上图中， 可以看出4 x 4 的block的效果比较好。

至此我们已经完成了第三步。 我们使用向量描绘出了所有的关键点。 为了更近一步的提高SIFT的性能， 我们可以将这个向量归一化为单位向量， 从而实现SIFT特征的illumination invariance， 以及对于affine changes 的invariance。

对于非线性的亮度变化， 我们将我们的单位向向量中的每一个元素限制在最大为0.2, 一旦有超过0.2的， 我们就而移除这个较大的梯度， 重新对单位向量归一化。

至此， 我们完成了所有SIFT的工作， 我们得到了SIFT特征描绘子（也就是所有的关键点都用了128维的特征向量表示了）。 接下来， 我们所做的就是关键点匹配了（Key Point Matching ）。

根据SIFT进行match， 生成了A、B两幅图的描述子，（分别是k1*128维和k2*128维），就将两图中各个scale（所有scale）的描述子进行匹配，匹配上128维即可表示两个特征点match上了。

匹配的准则就是找最近邻。 也就是说给定一幅图像的一个关键点的128维的向量描述， 从另一幅图像中的关键点中找到欧氏距离最小的那个关键点， 表示这两个关键点完成了匹配。

## SIFT特征点提取Matlab代码

```matlab
%% Title: SIFT_FeatureExtraction
% Authors: ruogu7
% Email: 380545156@qq.com
% Start time: 8:30 am,Jan 15th,2020
% Latest update: Jan 20th,2020
tic
clear;
clc;
row=256;
colum=256;
img=imread('lenna.jpg');
img=imresize(img,[row,colum]);
img=rgb2gray(img);
% img=histeq(img);
img=im2double(img);
origin=img;
% img=medfilt2(img);
toc
%% Scale-Space Extrema Detection
tic
% original sigma and the number of actave can be modified. the larger
% sigma0, the more quickly-smooth images
sigma0=sqrt(2);
octave=3;%6*sigma*k^(octave*level)<=min(m,n)/(2^(octave-2))
level=3;
D=cell(1,octave);
for i=1:octave
D(i)=mat2cell(zeros(row*2^(2-i)+2,colum*2^(2-i)+2,level),row*2^(2-i)+2,colum*2^(2-i)+2,level);
end
% first image in first octave is created by interpolating the original one.
temp_img=kron(img,ones(2));
temp_img=padarray(temp_img,[1,1],'replicate');
figure(2)
subplot(1,2,1);
imshow(origin)
%create the DoG pyramid.
for i=1:octave
    temp_D=D{i};
    for j=1:level
        scale=sigma0*sqrt(2)^(1/level)^((i-1)*level+j);
        p=(level)*(i-1);
        figure(1);
        subplot(octave,level,p+j);
        f=fspecial('gaussian',[1,floor(6*scale)],scale);
        L1=temp_img;
        if(i==1&&j==1)
        L2=conv2(temp_img,f,'same');
        L2=conv2(L2,f','same');
        temp_D(:,:,j)=L2-L1;
        imshow(uint8(255 * mat2gray(temp_D(:,:,j))));
        L1=L2;
        else
        L2=conv2(temp_img,f,'same');
        L2=conv2(L2,f','same');
        temp_D(:,:,j)=L2-L1;
        L1=L2;
        if(j==level)
            temp_img=L1(2:end-1,2:end-1);
        end
        imshow(uint8(255 * mat2gray(temp_D(:,:,j))));
        end
    end
    D{i}=temp_D;
    temp_img=temp_img(1:2:end,1:2:end);
    temp_img=padarray(temp_img,[1,1],'both','replicate');
end
toc
%% Keypoint Localistaion
% search each pixel in the DoG map to find the extreme point
tic
interval=level-1;
number=0;
for i=2:octave+1
    number=number+(2^(i-octave)*colum)*(2*row)*interval;
end
extrema=zeros(1,4*number);
flag=1;
for i=1:octave
    [m,n,~]=size(D{i});
    m=m-2;
    n=n-2;
    volume=m*n/(4^(i-1));
    for k=2:interval  
        for j=1:volume
            % starter=D{i}(x+1,y+1,k);
            x=ceil(j/n);
            y=mod(j-1,m)+1;
            sub=D{i}(x:x+2,y:y+2,k-1:k+1);
            large=max(max(max(sub)));
            little=min(min(min(sub)));
            if(large==D{i}(x+1,y+1,k))
                temp=[i,k,j,1];
                extrema(flag:(flag+3))=temp;
                flag=flag+4;
            end
            if(little==D{i}(x+1,y+1,k))
                temp=[i,k,j,-1];
                extrema(flag:(flag+3))=temp;
                flag=flag+4;
            end
        end
    end
end
idx= extrema==0;
extrema(idx)=[];
toc
[m,n]=size(img);
x=floor((extrema(3:4:end)-1)./(n./(2.^(extrema(1:4:end)-2))))+1;
y=mod((extrema(3:4:end)-1),m./(2.^(extrema(1:4:end)-2)))+1;
ry=y./2.^(octave-1-extrema(1:4:end));
rx=x./2.^(octave-1-extrema(1:4:end));
figure(2)
subplot(1,2,2);
imshow(origin)
hold on
plot(ry,rx,'r+');
%% accurate keypoint localization 
%eliminate the point with low contrast or poorly localised on an edge
% x:|,y:-- x is for vertial and y is for horizontal
% value comes from the paper.
tic
threshold=0.1;
r=10;
extr_volume=length(extrema)/4;
[m,n]=size(img);
secondorder_x=conv2([-1,1;-1,1],[-1,1;-1,1]);
secondorder_y=conv2([-1,-1;1,1],[-1,-1;1,1]);
for i=1:octave
    for j=1:level
        test=D{i}(:,:,j);
        temp=-1./conv2(test,secondorder_y,'same').*conv2(test,[-1,-1;1,1],'same');
        D{i}(:,:,j)=temp.*conv2(test',[-1,-1;1,1],'same')*0.5+test;
    end
end
for i=1:extr_volume
    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
    rx=x+1;
    ry=y+1;
    rz=extrema(4*(i-1)+2);
    z=D{extrema(4*(i-1)+1)}(rx,ry,rz);
    if(abs(z)<threshold)
        extrema(4*(i-1)+4)=0;
    end
end
idx=find(extrema==0);
idx=[idx,idx-1,idx-2,idx-3];
extrema(idx)=[];
extr_volume=length(extrema)/4;
x=floor((extrema(3:4:end)-1)./(n./(2.^(extrema(1:4:end)-2))))+1;
y=mod((extrema(3:4:end)-1),m./(2.^(extrema(1:4:end)-2)))+1;

rx=x./2.^(octave-1-extrema(1:4:end));
figure(2)
subplot(2,2,3);
imshow(origin)
hold on
plot(ry,rx,'g+');
for i=1:extr_volume
    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
    rx=x+1;
    ry=y+1;
    rz=extrema(4*(i-1)+2);
        Dxx=D{extrema(4*(i-1)+1)}(rx-1,ry,rz)+D{extrema(4*(i-1)+1)}(rx+1,ry,rz)-2*D{extrema(4*(i-1)+1)}(rx,ry,rz);
        Dyy=D{extrema(4*(i-1)+1)}(rx,ry-1,rz)+D{extrema(4*(i-1)+1)}(rx,ry+1,rz)-2*D{extrema(4*(i-1)+1)}(rx,ry,rz);
        Dxy=D{extrema(4*(i-1)+1)}(rx-1,ry-1,rz)+D{extrema(4*(i-1)+1)}(rx+1,ry+1,rz)-D{extrema(4*(i-1)+1)}(rx-1,ry+1,rz)-D{extrema(4*(i-1)+1)}(rx+1,ry-1,rz);
        deter=Dxx*Dyy-Dxy*Dxy;
        R=(Dxx+Dyy)/deter;
        R_threshold=(r+1)^2/r;
        if(deter<0||R>R_threshold)
            extrema(4*(i-1)+4)=0;
        end
    
end
idx=find(extrema==0);
idx=[idx,idx-1,idx-2,idx-3];
extrema(idx)=[];
extr_volume=length(extrema)/4;
x=floor((extrema(3:4:end)-1)./(n./(2.^(extrema(1:4:end)-2))))+1;
y=mod((extrema(3:4:end)-1),m./(2.^(extrema(1:4:end)-2)))+1;
ry=y./2.^(octave-1-extrema(1:4:end));
rx=x./2.^(octave-1-extrema(1:4:end));
figure(2)
subplot(2,2,4);
imshow(origin)
hold on
plot(ry,rx,'b+');
toc
%% Orientation Assignment(Multiple orientations assignment)
tic
kpori=zeros(1,36*extr_volume);
minor=zeros(1,36*extr_volume);
f=1;
flag=1;
for i=1:extr_volume
    %search in the certain scale
    scale=sigma0*sqrt(2)^(1/level)^((extrema(4*(i-1)+1)-1)*level+(extrema(4*(i-1)+2)));
    width=2*round(3*1.5*scale);
    count=1;
    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
    %make sure the point in the searchable area
    if(x>(width/2)&&y>(width/2)&&x<(m/2^(extrema(4*(i-1)+1)-2)-width/2-2)&&y<(n/2^(extrema(4*(i-1)+1)-2)-width/2-2))
        rx=x+1;
        ry=y+1;
        rz=extrema(4*(i-1)+2);
        reg_volume=width*width;%3? thereom
        % make weight matrix
        weight=fspecial('gaussian',width,1.5*scale);
        %calculate region pixels' magnitude and region orientation
        reg_mag=zeros(1,count);
        reg_theta=zeros(1,count);
    for l=(rx-width/2):(rx+width/2-1)
        for k=(ry-width/2):(ry+width/2-1)
            reg_mag(count)=sqrt((D{extrema(4*(i-1)+1)}(l+1,k,rz)-D{extrema(4*(i-1)+1)}(l-1,k,rz))^2+(D{extrema(4*(i-1)+1)}(l,k+1,rz)-D{extrema(4*(i-1)+1)}(l,k-1,rz))^2);
            reg_theta(count)=atan2((D{extrema(4*(i-1)+1)}(l,k+1,rz)-D{extrema(4*(i-1)+1)}(l,k-1,rz)),(D{extrema(4*(i-1)+1)}(l+1,k,rz)-D{extrema(4*(i-1)+1)}(l-1,k,rz)))*(180/pi);
            count=count+1;
        end
    end
    %make histogram 
    mag_counts=zeros(1,36);
    for x=0:10:359
        mag_count=0;
       for j=1:reg_volume
           c1=-180+x;
           c2=-171+x;
           if(c1<0||c2<0)
           if(abs(reg_theta(j))<abs(c1)&&abs(reg_theta(j))>=abs(c2))
               mag_count=mag_count+reg_mag(j)*weight(ceil(j/width),mod(j-1,width)+1);
           end
           else
               if(abs(reg_theta(j)>abs(c1)&&abs(reg_theta(j)<=abs(c2))))
                   mag_count=mag_count+reg_mag(j)*weight(ceil(j/width),mod(j-1,width)+1);
               end
           end
       end
          mag_counts(x/10+1)=mag_count;
    end
    % find the max histogram bar and the ones higher than 80% max
    [maxvm,~]=max(mag_counts);
     kori=find(mag_counts>=(0.8*maxvm));
     kori=(kori*10+(kori-1)*10)./2-180;
     kpori(f:(f+length(kori)-1))=kori;
     f=f+length(kori);
     temp_extrema=[extrema(4*(i-1)+1),extrema(4*(i-1)+2),extrema(4*(i-1)+3),extrema(4*(i-1)+4)];
     temp_extrema=padarray(temp_extrema,[0,length(temp_extrema)*(length(kori)-1)],'post','circular');
     long=length(temp_extrema);
     minor(flag:flag+long-1)=temp_extrema;
     flag=flag+long;
    end
end
idx= minor==0;
minor(idx)=[];
extrema=minor;
% delete unsearchable points and add minor orientation points
idx= kpori==0;
kpori(idx)=[];
extr_volume=length(extrema)/4;
toc
%% keypoint descriptor
tic
d=4;% In David G. Lowe experiment,divide the area into 4*4.
pixel=4;
feature=zeros(d*d*8,extr_volume);
for i=1:extr_volume
    descriptor=zeros(1,d*d*8);% feature dimension is 128=4*4*8;
    width=d*pixel;
    %x,y centeral point and prepare for location rotation
    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
    z=extrema(4*(i-1)+2);
        if((m/2^(extrema(4*(i-1)+1)-2)-pixel*d*sqrt(2)/2)>x&&x>(pixel*d/2*sqrt(2))&&(n/2^(extrema(4*(i-1)+1)-2)-pixel*d/2*sqrt(2))>y&&y>(pixel*d/2*sqrt(2)))
        sub_x=(x-d*pixel/2+1):(x+d*pixel/2);
        sub_y=(y-d*pixel/2+1):(y+d*pixel/2);
        sub=zeros(2,length(sub_x)*length(sub_y));
        j=1;
        for p=1:length(sub_x)
            for q=1:length(sub_y)
                sub(:,j)=[sub_x(p)-x;sub_y(q)-y];
                j=j+1;
            end
        end
        distort=[cos(pi*kpori(i)/180),-sin(pi*kpori(i)/180);sin(pi*kpori(i)/180),cos(pi*kpori(i)/180)];
    %accordinate after distort
        sub_dis=distort*sub;
        fix_sub=ceil(sub_dis);
        fix_sub=[fix_sub(1,:)+x;fix_sub(2,:)+y];
        patch=zeros(1,width*width);
        for p=1:length(fix_sub)
        patch(p)=D{extrema(4*(i-1)+1)}(fix_sub(1,p),fix_sub(2,p),z);
        end
        temp_D=(reshape(patch,[width,width]))';
        %create weight matrix.
        mag_sub=temp_D;    
        temp_D=padarray(temp_D,[1,1],'replicate','both');
        weight=fspecial('gaussian',width,width/1.5);
        mag_sub=weight.*mag_sub;
        theta_sub=atan((temp_D(2:end-1,3:1:end)-temp_D(2:end-1,1:1:end-2))./(temp_D(3:1:end,2:1:end-1)-temp_D(1:1:end-2,2:1:end-1)))*(180/pi);
        % create orientation histogram
        for area=1:d*d
        cover=pixel*pixel;
        ori=zeros(1,cover);
        magcounts=zeros(1,8);
        for angle=0:45:359
          magcount=0;
          for p=1:cover;
              x=(floor((p-1)/pixel)+1)+pixel*floor((area-1)/d);
              y=mod(p-1,pixel)+1+pixel*(mod(area-1,d));
              c1=-180+angle;
              c2=-180+45+angle;
              if(c1<0||c2<0)
                  if (abs(theta_sub(x,y))<abs(c1)&&abs(theta_sub(x,y))>=abs(c2))
                  
                      ori(p)=(c1+c2)/2;
                      magcount=magcount+mag_sub(x,y);
                  end
              else
                  if(abs(theta_sub(x,y))>abs(c1)&&abs(theta_sub(x,y))<=abs(c2))
                      ori(p)=(c1+c2)/2;
                      magcount=magcount+mag_sub(x,y);
                  end
              end          
          end
          magcounts(angle/45+1)=magcount;
        end
        descriptor((area-1)*8+1:area*8)=magcounts;
        end
        descriptor=normr(descriptor);
        % cap 0.2
        for j=1:numel(descriptor)
            if(abs(descriptor(j))>0.2)
            descriptor(j)=0.2;    
            end
        end
        descriptor=normr(descriptor);
        else
            continue;
        end
        feature(:,i)=descriptor';
end
index=find(sum(feature));
feature=feature(:,index);
toc
```
