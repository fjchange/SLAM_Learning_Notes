# 0. SIFT

SIFT(Scale-Invariant Feature Transform)自1999年由David Lowe提出以后被广泛的应用于CV的各种领域：图像识别，图像检索，3D重建等等，可谓无人不知无人不晓。其应用太广泛了，而且Lowe还申请了专利，以至于想商业用的人都很担心。它的步骤可以主要分两步：1）特征点检出 keypoint localisation，2）特征点描述 feature description。

**特征点检出**主要是用了DoG，就是把图像做不同程度的高斯模糊blur，平滑的区域或点肯定变化不大，而纹理复杂的比如边缘，点，角之类区域肯定变化很大，这样变化很大的点就是特征点。当然为了找到足够的点，还需要把图像放大缩小几倍(Image Pyramids)来重复这个步骤找特征点。其实DoG并不是Lowe提出的，很久以前就有了，读过SIFT专利的人都知道，SIFT的专利里面也不包括这部分。可代替特征点检出还有很多其他方法如MSER等。

**特征点描述**就是一个简单版的HOG，即以检出的特征点为中心选16x16的区域作为local patch，这个区域又可以均分为4x4个子区域，每个子区域中各个像素的梯度都可以分到8个bin里面，这样就得到了4x4x8=128维的特征向量。特征点检出以后还需要一个很重要的步骤就是[归一化](https://www.zhihu.com/search?q=%E5%BD%92%E4%B8%80%E5%8C%96&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2222476595%22%7D)，计算这个patch的主方向，然后根据这个主方向把patch旋转到特定方向，这样计算的特征就有了 **方向不变性** ，也需要根据patch各像素梯度大小把patch缩放到一定的尺度，这样特征就有了 **尺度不变性** 。

很多人都直接把这两步统称为SIFT，其实确切来说SIFT是指第二步的特征，而SIFT features或SIFT descriptor的说法应该比较准确。
