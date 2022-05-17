# Harris Corner Detection

**1.角点特征的优越性：（WHY）**

按照之前对优秀的局部特征的定义，角点特征具有可重复性和显著性特征，所以是优秀的局部特征。

**2.角点检测的基础思想：（WHAT）**

**（1）窗口：**

对局部特征检测来说，基础的方法是滑动窗口在图向上滑动，检测窗口滑动的内容（如所有像素值梯度）或者内容变化情况（如像素密度变化的整体情况），观察窗口捕捉到的信息是否符合局部特征的特点。这种方法很符合对“局部性”的阐释。

**（2）角点区域的特征：**

对角点来说，它是边的交点，边的特征是梯度在某一方向上产生突变，那么这个交点应该表现为某一区域内有两个或多个方向上的“边缘性变化”的梯度信息。 **需要强调的是，寻找角点的目标或许是找到一个“点”，但是分析的对象并不能只是一个“点”（在图像处理中，点指的是[像素点](https://www.zhihu.com/search?q=%E5%83%8F%E7%B4%A0%E7%82%B9&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A2286352823%7D)），而应该是一个区域所有像素点的整体所呈现出来的特征（先记住这个“点”与“点所在区域”的区别，之后会再提到）。** 所以我们需要在一定范围内观察这个局部的整体，之前提到的窗口就起到了作用。

如果用滑动窗口来观察角点区域的话，窗口向多个方向滑动，都能感知到像素密度的强变化（梯度）。反观平坦区域和边缘区域，前者在任意滑动方向上不会产生强变化，后者只会在一个滑动方向上产生强变化。如下图：

![img](https://pic3.zhimg.com/50/v2-553528759d89a374f5ccc3538184c97e_720w.jpg?source=1940ef5c)

> 啥是角点？

先上定义：

* 一阶导数(即灰度的梯度)的局部最大所对应的像素点；
* 两条及两条以上边缘的交点；
* 图像中梯度值和梯度方向的变化速率都很高的点；
* 角点处的一阶导数最大，二阶导数为零，指示物体边缘变化不连续的方向。

在懵逼之前，可以先看看这个图。

![](https://pic2.zhimg.com/v2-9c5a618c7435a110513601c83ee8f135_b.jpg)

在上面的倒五边形中，从上到下分别取1/2/3三部分，他们分别代表了：

1. 边缘
2. 团块
3. 角点

这么分的依据是什么呢？从肉眼上看，非常容易辨别，当然我们也有过Canny检测这种专门的算法来识别边缘特征。那么怎么从微观或者说让计算机识别角点呢？

> 如何识别角点？

基本思想是使用一个固定窗口在图像上进行任意方向上的滑动，比较滑动前与滑动后两种情况，窗口中的像素灰度变化程度，如果存在任意方向上的滑动，都有着较大灰度变化，那么我们可以认为该窗口中存在角点。

角点的特性下面的图更清晰：

![](https://pic3.zhimg.com/v2-019e6852a6bb019469d2c3d50b1ff362_b.jpg)

而且当然不用我等工程狗去根据这种特征来设计算法，跟边缘检测一样，大神们早就做过了，单识别角点的算法，比较有名的就有：

* Harri算法
* Shi-Tomasi算法
* 亚像素级角点检测

在图像处理中，检测角点特征的算法有很多，这里先介绍最常用的，也是最基础的 Harris 角点检测器(Harris Corner Detection)。

角点是两条边缘的交点，它表示两条边方向改变的地方，所以**角点在任意一个方向上做微小移动，都会引起该区域的梯度图的方向和幅值发生很大变化。**

也就是一阶导数(即灰度图的梯度)中的局部最大所对应的像素点就是角点。

于是我们可以利用这一点来检测角点。

使一个固定尺寸的窗口在图像上某个位置以任意方向做微小滑动，如果窗口内的灰度值（在梯度图上）都有较大的变化，那么这个窗口所在区域就存在角点。

这样就可以将 Harris 角点检测算法分为以下三步：

1. 当窗口（小的图像片段）同时向 x 和 y 两个方向移动时，计算窗口内部的像素值变化量 ![[公式]](https://www.zhihu.com/equation?tex=E%28u%2C+v%29) ；
2. 对于每个窗口，都计算其对应的一个角点响应函数 R；
3. 然后对该函数进行阈值处理，如果 R > threshold，表示该窗口对应一个角点特征。

接下来对每一步进行详细介绍。

### 2.1、第一步

如何确定哪些窗口会引起较大的灰度值变化？

让一个窗口的中心位于[灰度图像](https://www.zhihu.com/search?q=%E7%81%B0%E5%BA%A6%E5%9B%BE%E5%83%8F&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2283064609%22%7D)的一个位置 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2C+y%29) ，这个位置的像素灰度值为 ![[公式]](https://www.zhihu.com/equation?tex=I%28x%2C+y%29) ，如果这个窗口分别向 x 和 y 方向移动一个小的位移u和v，到一个新的位置 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Bu%2C+y%2Bv%29) ，这个位置的像素灰度值就是 ![[公式]](https://www.zhihu.com/equation?tex=I%28x%2Bu%2C+y%2Bv%29) 。

![[公式]](https://www.zhihu.com/equation?tex=%5BI%28x%2Bu%2C+y%2Bv%29+-+I%28x%2C+y%29%5D) 就是窗口移动引起的灰度值的变化值。

设 ![[公式]](https://www.zhihu.com/equation?tex=w%28x%2Cy%29) 为位置 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29) 处的窗口函数，表示窗口内各像素的权重，最简单的就是把窗口内所有像素的权重都设为1。

有时也会把 ![[公式]](https://www.zhihu.com/equation?tex=w%28x%2Cy%29) 设定为以窗口中心为原点的[高斯分布](https://www.zhihu.com/search?q=%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2283064609%22%7D)（二元正态分布）。如果窗口中心点像素是角点，那么窗口移动前后，中心点的灰度值变化非常强烈，所以该点权重系数应该设大一点，表示该点对灰度变化的贡献较大；而离窗口中心（角点）较远的点，这些点的灰度变化比较小，于是将权重系数设小一点，表示该点对灰度变化的贡献较小。

则窗口在各个方向上移动 ![[公式]](https://www.zhihu.com/equation?tex=%28u%2Cv%29) 所造成的像素灰度值的变化量公式如下：

![[公式]](https://www.zhihu.com/equation?tex=E%28u%2Cv%29+%3D+%5Csum%5Climits_%7B%28x%2Cy%29%7D+w%28x%2Cy%29+%5Ctimes+%5BI%28x%2Bu%2C+y%2Bv%29+-+I%28x%2Cy%29%5D%5E2+%5C%5C)

对于一个角点来说， ![[公式]](https://www.zhihu.com/equation?tex=E%28u%2Cv%29) 会非常大。因此，我们可以最大化上面这个函数来得到图像中的角点。

用上面的函数计算 ![[公式]](https://www.zhihu.com/equation?tex=E%28u%2Cv%29) 会非常慢。因此，我们使用泰勒展开式（只有一阶）来得到这个公式的近似形式。

对于二维的泰勒展开式公式为：

![[公式]](https://www.zhihu.com/equation?tex=T%28x%2Cy%29+%5Capprox+f%28u%2Cv%29+%2B+%28x-u%29f_x%28u%2Cv%29+%2B+%28y-v%29f_y%28u%2Cv%29+%2B+...+%5C%5C)

将 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdisplaystyle+I%28u%2Bx%2Cv%2By%29) 套用上面的公式，可以得到：

![[公式]](https://www.zhihu.com/equation?tex=I%28x%2Bu%2Cy%2Bv%29+%5Capprox+I%28x%2Cy%29+%2B+uI_x+%2B+vI_y+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=I_x) 和 ![[公式]](https://www.zhihu.com/equation?tex=I_y) 是 ![[公式]](https://www.zhihu.com/equation?tex=I) 的[偏微分](https://www.zhihu.com/search?q=%E5%81%8F%E5%BE%AE%E5%88%86&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2283064609%22%7D)，在图像中就是在 x 和 y 方向的 **梯度图** （可以通过 `cv2.Sobel()`来得到）：

![[公式]](https://www.zhihu.com/equation?tex=I_x+%3D+%5Cfrac+%7B%5Cpartial+I%28x%2Cy%29%7D%7B%5Cpartial+x%7D%2C%5C+%5C+I_y+%3D+%5Cfrac+%7B%5Cpartial+I%28x%2Cy%29%7D%7B%5Cpartial+y%7D+%5C%5C)

接下来继续推导：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D++E%28u%2Cv%29+%26%3D+%5Csum%5Climits_%7B%28x%2Cy%29%7D+w%28x%2Cy%29+%5Ctimes+%5BI%28x%2Cy%29+%2B+uI_x+%2B+vI_y+-+I%28x%2Cy%29%5D%5E2+%5C%5C+%26%3D+%5Csum%5Climits_%7B%28x%2Cy%29%7D+w%28x%2Cy%29+%5Ctimes+%28uI_x+%2B+vI_y%29%5E2+%5C%5C++%26%3D+%5Csum%5Climits_%7B%28x%2Cy%29%7D+w%28x%2Cy%29+%5Ctimes+%28u%5E2I_x%5E2+%2B+v%5E2I_y%5E2+%2B+2uvI_xI_y%29+%5Cend%7Baligned%7D+%5C%5C)

把 u 和 v 拿出来，得到最终的形式：

![[公式]](https://www.zhihu.com/equation?tex=E%28u%2Cv%29+%5Capprox+%5Cbegin%7Bbmatrix%7D+u%2C+v+%5Cend%7Bbmatrix%7D+M+%5Cbegin%7Bbmatrix%7D+u+%5C%5C+v+%5Cend%7Bbmatrix%7D+%5C%5C)

其中矩阵M为：

![[公式]](https://www.zhihu.com/equation?tex=M+%3D+%5Csum%5Climits_%7B%28x%2Cy%29%7D+w%28x%2Cy%29++%5Cbegin%7Bbmatrix%7D+I_x%5E2+%26+I_xI_y+%5C%5C+I_xI_y+%26+I_y%5E2+%5Cend%7Bbmatrix%7D++%5Crightarrow+R%5E%7B-1%7D+%5Cbegin%7Bbmatrix%7D+%5Clambda_1+%26+0+%5C%5C+0+%26+%5Clambda_2+%5Cend%7Bbmatrix%7D+R++%5C%5C)

最后是把实对称矩阵对角化处理后的结果，可以把R看成旋转因子，其不影响两个正交方向的变化分量。

经对角化处理后，将两个正交方向的变化分量提取出来，就是 λ1 和 λ2（[特征值](https://www.zhihu.com/search?q=%E7%89%B9%E5%BE%81%E5%80%BC&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2283064609%22%7D)）。

公式推导完了，现在回顾一下：

对于图像的每一个[像素点](https://www.zhihu.com/search?q=%E5%83%8F%E7%B4%A0%E7%82%B9&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2283064609%22%7D) ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29) ，对应一个以该像素为中心的窗口 ![[公式]](https://www.zhihu.com/equation?tex=w%28x%2Cy%29) ，然后该像素平移 ![[公式]](https://www.zhihu.com/equation?tex=%28u%2Cy%29) 得到新的像素点 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Bu%2C+y%2Bv%29) ，而 ![[公式]](https://www.zhihu.com/equation?tex=E%28u%2Cv%29) 就是 **窗口中所有像素的加权和乘以不同位置像素的灰度差值** 。

### 2.2、第二步

现在我们已经得到 ![[公式]](https://www.zhihu.com/equation?tex=E%28u%2Cv%29) 的最终形式，别忘了我们的目的是要找到会引起较大的灰度值变化的那些窗口。

灰度值变化的大小则取决于矩阵M，那么如何找到这些窗口，我们可以使用矩阵的特征值来实现。

计算每个窗口对应的得分（角点响应函数R）：

![[公式]](https://www.zhihu.com/equation?tex=R+%3D+%5Ctext+%7Bdet%7D%28M%29+-+k+%28%5Ctext+%7Btrace%28M%29%7D%29%5E2+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext+%7Bdet%7D%28M%29+%3D+%5Clambda_1+%5Clambda_2) 是矩阵的行列式， ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext+%7Btrace%28M%29%7D+%3D+%5Clambda_1+%2B+%5Clambda_2) 是矩阵的迹。

λ1 和 λ2 是矩阵M的特征值， ![[公式]](https://www.zhihu.com/equation?tex=k) 是一个经验常数，在范围 (0.04, 0.06) 之间。

### 2.3、第三步

根据 R 的值，将这个窗口所在的区域划分为平面、边缘或角点。为了得到最优的角点，我们还可以使用 **非极大值抑制** 。

注意：Harris 检测器具有旋转不变性，但不具有尺度不变性，也就是说尺度变化可能会导致角点变为边缘，如下图所示：

![](https://pic1.zhimg.com/v2-eeb16e97576beaa149b2db469cd82b64_b.jpg)

（插播一句，想要尺度不变特性的话，可以关注SIFT特征）

因为特征值 λ1 和 λ2 决定了 R 的值，所以我们可以用特征值来决定一个窗口是平面、边缘还是角点：

* **平面:** 该窗口在平坦区域上滑动，窗口内的灰度值基本不会发生变化，所以 ![[公式]](https://www.zhihu.com/equation?tex=%7CR%7C) 值非常小，在水平和竖直方向的变化量均较小，即 ![[公式]](https://www.zhihu.com/equation?tex=I_x) 和 ![[公式]](https://www.zhihu.com/equation?tex=I_y) 都较小，那么 λ1 和 λ2 都较小；
* **边缘:** ![[公式]](https://www.zhihu.com/equation?tex=R) 值为负数，仅在水平或竖直方向有较大的变化量，即 ![[公式]](https://www.zhihu.com/equation?tex=I_x) 和 ![[公式]](https://www.zhihu.com/equation?tex=I_y) 只有一个较大，也就是 λ1>>λ2 或 λ2>>λ1；
* **角点:** ![[公式]](https://www.zhihu.com/equation?tex=R) 值很大，在水平、竖直两个方向上变化均较大的点，即 ![[公式]](https://www.zhihu.com/equation?tex=I_x) 和 ![[公式]](https://www.zhihu.com/equation?tex=I_y) 都较大，也就是 λ1 和 λ2 都很大

用图片表示如下：

![](https://pic3.zhimg.com/v2-95cc2824eac2378cc04bf9282dbec912_b.jpg)

Harris 角点检测的结果是带有这些分数 R 的灰度图像，设定一个阈值，分数大于这个阈值的像素就对应角点


## 三、Shi-Tomasi 角点检测器

知道了什么是 Harris 角点检测，后来有大佬在论文《[Good_Features_to_Track](https://www.zhihu.com/search?q=Good_Features_to_Track&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2283064609%22%7D)》中提出了它的改进版——Shi-Tomasi 角点检测，Shi-Tomasi 方法在很多情况下可以得到比 Harris 算法更好的结果。

Harris 角点检测中每个窗口的分数公式是将矩阵 M 的行列式与 M 的迹相减：

![[公式]](https://www.zhihu.com/equation?tex=R+%3D+%CE%BB_1%CE%BB_2+%E2%88%92+k+%28%CE%BB_1+%2B+%CE%BB_2%29%5E2+%5C%5C)

由于 Harris 角点检测算法的稳定性和 k 值有关，而 k 是个经验值，不好设定最佳值。

Shi-Tomasi 发现，角点的稳定性其实和矩阵 M 的较小特征值有关，于是直接用较小的那个特征值作为分数。这样就不用调整k值了。

所以 Shi-Tomasi 将分数公式改为如下形式：

![[公式]](https://www.zhihu.com/equation?tex=R+%3D+min+%28%CE%BB_1%2C+%CE%BB_2%29+%5C%5C)

和 Harris 一样，如果该分数大于设定的阈值，我们就认为它是一个角点。

我们可以把它绘制到 λ1 ～ λ2 空间中，就会得到下图：

![img](https://pic1.zhimg.com/v2-56a5bd60f1e79bc8b6f0b652dee589bc_b.jpg)


## 四、OpenCV 代码实现

### 4.1 Harris 角点检测

在opencv中有提供实现 Harris 角点检测的函数 [cv2.cornerHarris](https://link.zhihu.com/?target=https%3A//docs.opencv.org/master/dd/d1a/group__imgproc__feature.html%23gac1fc3598018010880e370e2f709b4345)，我们直接调用的就可以，非常方便。

函数原型：`cv2.cornerHarris(src, blockSize, ksize, k[, dst[, borderType]])`

对于每一个像素 (x,y)，在 (blockSize x blockSize) 邻域内，计算梯度图的[协方差矩阵](https://www.zhihu.com/search?q=%E5%8D%8F%E6%96%B9%E5%B7%AE%E7%9F%A9%E9%98%B5&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2283064609%22%7D) ![[公式]](https://www.zhihu.com/equation?tex=M%28x%2Cy%29) ，然后通过上面第二步中的角点响应函数得到结果图。图像中的角点可以为该结果图的局部最大值。

即可以得到输出图中的局部最大值，这些值就对应图像中的角点。

涉及到几个参数：

* **src** - 输入灰度图像，float32类型
* **blockSize** - 用于角点检测的邻域大小，就是上面提到的窗口的尺寸
* **ksize** - 用于计算梯度图的Sobel算子的尺寸
* **k** - 用于计算角点响应函数的参数k，取值范围常在0.04~0.06之间

```python
import cv2 as cv
import numpy as np

# detector parameters
block_size = 5
sobel_size = 3
k = 0.04

image = cv2.imread('bird.jpg')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# modify the data type setting to 32-bit floating point 
gray_img = np.float32(gray_img)

# detect the corners with appropriate values as input parameters
corners_img = cv2.cornerHarris(gray_img, block_size, sobel_size, k)

# result is dilated for marking the corners, not necessary
dst = cv.dilate(corners_img, None)

# Threshold for an optimal value, marking the corners in Green
image[corners_img>0.01*corners_img.max()] = [0,0,255]

cv2.imwrite('new_bird.jpg', image)
```

### 4.2、Shi-Tomasi 角点检测

OpenCV 提供了 Shi-Tomasi 的函数：  **cv2.goodFeaturesToTrack()** ，来获取图像中前 N 个最好的角点。

函数原型如下：

```text
goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
```

其中的参数如下：

* image：输入灰度图像，float32类型
* maxCorners：返回角点的最大数目，值为0表表示没有设置最大值限制，返回所有检测到的角点。
* qualityLevel：质量系数（小于1.0的正数，一般在0.01-0.1之间），表示可接受角点的最低质量水平。该系数乘以最好的角点分数（也就是上面较小的那个特征值），作为可接受的最小分数；例如，如果最好的角点分数值为1500且质量系数为0.01，那么所有质量分数小于15的角都将被忽略。
* minDistance：角之间最小欧式距离，忽略小于此距离的点。
* corners：输出角点坐标
* mask：可选的感兴趣区域，指定想要检测角点的区域。
* blockSize：默认为3，角点检测的邻域大小（窗口尺寸）
* useHarrisDetector：用于指定角点检测的方法，如果是true则使用Harris角点检测，false则使用Shi Tomasi算法。默认为False。
* k：默认为0.04，Harris角点检测时使用。

设定好这些参数，函数就能在图像上找到角点。所有低于质量水平的角点都会被忽略，然后再把合格角点按角点质量进行降序排列。

然后保留质量最高的一个角点，将它附近（最小距离之内）的角点都删掉（类似于非极大值抑制），按这样的方式最后得到 N 个最佳角点。

```python3
import numpy as np
import cv2

maxCorners = 100
qualityLevel = 0.01
minDistance = 10

img = cv2.imread('bird.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)

corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),2,(0,0,255),-1)
  
cv2.imwrite('new_bird.jpg', img)
```

最后得到的结果如下：

![](https://pic2.zhimg.com/v2-5af148b5fbf60ddde518852892f11451_b.jpg)
三张图分别为原图，harris图和Shi-Tomasi图

Harris 和 Shi-Tomasi 都是基于梯度计算的角点检测方法，Shi-Tomasi 的效果要好一些。基于梯度的检测方法有一些缺点: 计算复杂度高，图像中的噪声可以阻碍梯度计算。

想要提高检测速度的话，可以考虑基于模板的方法：FAST角点检测算法。该算法原理比较简单，但实时性很强。

有兴趣的同学可以去了解一下。
