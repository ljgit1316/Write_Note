# OpenCv

## 计算机中的图像

### 1.像素

日常生活中常见的图像是RGB三原色图。RGB图上的每个点都是由红（R）、绿（G）、蓝（B）三个颜色按照一定比例混合而成的，几乎所有颜色都可以通过这三种颜色按照不同比例调配而成。在计算机中，RGB三种颜色被称为RGB三通道，根据这三个通道存储的像素值，来对应不同的颜色。例如，在使用“画图”软件进行自定义调色时，其数值单位就是像素。如下图所示：

![red](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/red.png)

![blue](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/blue.png)

![green](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/green.png)

### 2.图像

计算机采用0/1编码的系统，数字图像也是利用0/1来记录信息，我们平常接触的图像都是8位数图像，包含0～255灰度，其中0，代表最黑，1，表示最白。

#### 2.1二值图

一幅二值图像的二维矩阵仅由0、1两个值构成，“0”代表黑色，“1”代白色。由于每一像素（矩阵中每一元素）取值仅有0、1两种可能，所以计算机中二值图像的数据类型通常为1个二进制位。二值图像通常用于文字、线条图的扫描识别（OCR）和掩膜图像的存储。

![cat2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat2.png)

#### 2.2灰度图

每个像素只有一个采样颜色的图像，这类图像通常显示为从最暗黑色到最亮的白色的灰度，尽管理论上这个采样可以任何颜色的不同深浅，甚至可以是不同亮度上的不同颜色。灰度图像与黑白图像不同，在计算机图像领域中黑白图像只有黑色与白色两种颜色；但是，灰度图像在黑色与白色之间还有许多级的颜色深度。灰度图像经常是在单个电磁波频谱如可见光内测量每个像素的亮度得到的，用于显示的灰度图像通常用每个采样像素8位的非线性尺度来保存，这样可以有256级灰度（如果用16位，则有65536级）。

![cat3](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat3.png)

#### 2.3彩色图

每个像素通常是由红（R）、绿（G）、蓝（B）三个分量来表示的，分量介于（0，255）。RGB图像与索引图像一样都可以用来表示彩色图像。与索引图像一样，它分别用红（R）、绿（G）、蓝（B）三原色的组合来表示每个像素的颜色。但与索引图像不同的是，RGB图像每一个像素的颜色值（由RGB三原色表示）直接存放在图像矩阵中，由于每一像素的颜色需由R、G、B三个分量来表示，M、N分别表示图像的行列数，三个M x N的二维矩阵分别表示各个像素的R、G、B三个颜色分量。RGB图像的数据类型一般为8位无符号整形，通常用于表示和存放真彩色图像。

![cat1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat1.png)

实例：生成一个512*512大小的彩色图片  每一个像素点随机颜色

```python
import cv2
import numpy as np
# 设置图像尺寸
height, width = 512, 512

# 创建一个空白的彩色图像，其深度为3（代表BGR三个通道）
img = np.zeros((height, width, 3), dtype=np.uint8)

# 为每个像素生成随机的BGR值
# OpenCV中颜色范围是0-255
img[:] = np.random.randint(0, 256, img.shape)

# 显示图像
cv2.imshow('pic', img)
cv2.waitKey(0)

```

![random](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/random.png)

## OpenCv基础的图像操作

### 1.读取图像

- 语法

  ```python
  cv2.imread(path,[读取方式])
  ```

- 参数

  - path:要读取的图像路径
  - 读取方式的标志(彩色-默认,灰色等等)

- 实例

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  ```

### 2.显示图像

- 语法

  ```python
  cv2.imshow(arg1,arg2)
  ```

- 参数

  - arg1:显示图像的窗口名称，以字符串类型表示
  - arg2:要加载的图像

- 实例

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  cv2.imshow('pic',pic)
  cv2.waitKey(0)
  ```

![QQ图片20240812170336](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/imshow_pic.png)

### 3.保存图像

- 语法

  ```python
  cv2.imwrite(arg1,arg2)
  ```

- 参数

  - arg1:文件名，要保存在哪里
  - arg2:要保存的图像

- 实例

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  cv2.imshow('pic',pic)
  cv2.imwrite('pic01.png',pic)
  cv2.waitKey(0)
  ```

## OpecnCv绘制几何图形

### 1.绘制直线

- 语法

  ```python
  cv2.line(img,start,end,color,thickness)
  ```

- 参数

  - img:要绘制直线的图像
  - Start,end: 直线的起点和终点
  - color: 线条的颜色
  - Thickness: 线条宽度

- 实例

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  cv2.line(pic,(50,50),(100,100),(255,255,0),10)
  cv2.imshow('pic',pic)
  cv2.waitKey(0)
  ```

![line_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/line_pic.png)

### 2.绘制圆形

- 语法

  ```python
  cv.circle(img,centerpoint, r, color, thickness)
  ```

- 参数

  - img:要绘制圆形的图像
  - Centerpoint圆心
  - r: 半径
  - color: 线条的颜色
  - Thickness: 线条宽度，为-1时生成闭合图案并填充颜色

- 实例

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  cv2.circle(pic,(500,500),100,(255,0,255),5)
  cv2.imshow('pic',pic)
  cv2.waitKey(0)
  ```

![circle_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/circle_pic.png)

### 3.绘制矩形

- 语法

  ```python
  cv.rectangle(img,leftupper,rightdown,color,thickness)
  ```

- 参数

  - img:要绘制矩形的图像
  - Leftupper, rightdown: 矩形的左上角和右下角坐标
  - color: 线条的颜色
  - Thickness: 线条宽度

- 实例

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  cv2.rectangle(pic,(200,200),(300,300),(100,100,0),5)
  cv2.imshow('pic',pic)
  cv2.waitKey(0)
  ```

![tangle_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/tangle_pic.png)

### 4.图像中添加文字

- 语法

  ```python
  cv.putText(img,text,station, font, Fontscale ,color,thickness,cv2.LINE_AA)
  ```

- 参数

  - img: 图像

  - text：要写入的文本数据

  - station：文本的放置位置

  - font：字体样式

  - Fontscale :字体大小

  - thickness字体线条宽度

  - cv2.LINE_AA

    最后一个参数 `cv2.LINE_AA` 表示使用反走样（Anti-Aliasing）技术来绘制文本边框。

    - 反走样是一种提高图形质量的技术，它通过混合颜色和像素边缘以减少锯齿状效果，使文本看起来更加平滑、清晰。
    - 在 OpenCV 中，`cv2.LINE_AA` 是一种高级线条类型，用于实现文本边界的高质量渲染。相比于其他线型如 `cv2.LINE_8`（默认值），它能提供更好的视觉效果，特别是在文本较小或者需要高精度显示的情况下

- 实例

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  
  cv2.putText(pic,'Picture',(50,200),cv2.FONT_ITALIC,4,(100,100,100),5,cv2.LINE_AA)
  cv2.imshow('pic',pic)
  cv2.waitKey(0)
  ```

![txt_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/txt_pic.png)

### 5.获取并修改图像中的像素点

- 实例

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  print(pic.shape)
  px=pic[100,345]
  pic[100,345]=[0,0,255]
  print(px)
  cv2.imshow('pic',pic)
  cv2.waitKey(0)
  ```

![point_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/point_pic.png)

### 6.捕获摄像头的实时视频流

- 语法

  ```python
  cap = cv2.VideoCapture(path)
  #path视频流资源路径设置为0代表从默认摄像头捕获视频流
  
  ret, frame = cap.read()
  #返回值cap 调用read()方法可以得到一个布尔值和一帧图像。布尔值表示是否成功读取到帧，如果为False，可能是因为视频结束或读取失败；如果为True，第二项则是当前帧的图像数据。
  ```

- 实例

  ```python
  import cv2
  import numpy as np
  cap=cv2.VideoCapture(0)
  while True:
      flag,frame=cap.read()
      if flag==False or cv2.waitKey(100)==ord('q'):
          break
      else:
          cv2.imshow('crame',frame)
  cap.release()
  cv2.destroyAllWindows()
  ```

## 图像灰度化处理

​		灰度图与彩色图最大的不同就是：彩色图是由R、G、B三个通道组成，而灰度图只有一个通道，也称为单通道图像，所以彩色图转成灰度图的过程本质上就是将R、G、B三通道合并成一个通道的过程。

### 1.最大值法

- 定义

  对于彩色图像的每个像素，它会从R、G、B三个通道的值中选出最大的一个，并将其作为灰度图像中对应位置的像素值。

- 图示

  ![max](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/max.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  pic_gray=np.zeros((pic.shape[0],pic.shape[1]),dtype=np.int16)
  for i in range(pic.shape[0]):
      for j in range(pic.shape[1]):
          pic_gray[i][j]=max(pic[i][j][0],pic[i][j][1],pic[i][j][2])
  cv2.imshow('gray',pic_gray)
  cv2.waitKey(0)
  ```

  ![max_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/max_pic.png)

### 2.平均值法

- 定义

  对于彩色图像的每个像素，它会将R、G、B三个通道的像素值全部加起来，然后再除以三，得到的平均值就是灰度图像中对应位置的像素值。

- 图示

  ![overage](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/overage.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  pic_gray=np.zeros((pic.shape[0],pic.shape[1]),dtype=np.uint8)
  
  for i in range(pic.shape[0]):
      for j in range(pic.shape[1]):
          pic_gray[i][j]=(pic[i][j][0]+pic[i][j][1]+pic[i][j][2])//3
  
  cv2.imshow('gray',pic_gray)
  cv2.waitKey(0)
  ```

  ![overage_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/overage_pic.png)

### 3.加权均值法

- 定义

  对于彩色图像的每个像素，它会按照一定的权重去乘以每个通道的像素值，并将其相加，得到最后的值就是灰度图像中对应位置的像素值。

- 图示

  ![weight](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/weight.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  pic_gray=np.zeros((pic.shape[0],pic.shape[1]),dtype=np.uint8)
  
  wr,wg,wb=0.2,0.45,0.35
  
  for i in range(pic.shape[0]):
      for j in range(pic.shape[1]):
          pic_gray[i][j]=(pic[i][j][0]*wb+pic[i][j][1]*wg+pic[i][j][2]*wr)
  
  cv2.imshow('gray',pic_gray)
  cv2.waitKey(0)
  ```

  ![weight_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/weight_pic.png)

### 4.极端的灰度值

- 图示

  | ![black](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/black.png) | ![white](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/white.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 灰度值为0（纯黑）                                            | 灰度值为255（纯白）                                          |

## 图像二值化处理

​		二值化，顾名思义，就是将某张图像的所有像素改成只有两种值之一，其操作的图像也必须是灰度图。也就是说，二值化的过程，就是将一张灰度图上的像素根据某种规则修改为0和maxval（maxval表示最大值，一般为255，显示白色）两种像素值，使图像呈现黑白的效果，能够帮助我们更好地分析图像中的形状、边缘和轮廓等特征。

### 1.阈值法（THRESH_BINARY）

- 定义

  阈值法就是通过设置一个阈值，将灰度图中的每一个像素值与该阈值进行比较，小于等于阈值的像素就被设置为0（黑），大于阈值的像素就被设置为maxval。

- 图示

  ![yu](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/yu.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg',cv2.IMREAD_GRAYSCALE)#读入图片时，已转换为灰度图
  ret,pic_two=cv2.threshold(pic,200,255,cv2.THRESH_BINARY)
  cv2.imshow('two',pic_two)
  cv2.waitKey(0)
  ```

  ![yu_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/yu_pic.png)

### 2.反阈值法（THRESH_BINARY_INV）

- 定义

  反阈值法是当灰度图的像素值大于阈值时，该像素值将会变成0（黑），当灰度图的像素值小于等于阈值时，该像素值将会变成maxval。

- 图示

  ![fan_yu](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/fan_yu.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg',cv2.IMREAD_GRAYSCALE)#读入图片时，已转换为灰度图
  ret,pic_two=cv2.threshold(pic,127,255,cv2.THRESH_BINARY_INV)
  cv2.imshow('two',pic_two)
  cv2.waitKey(0)
  ```

  ![fan_yu_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/fan_yu_pic.png)

### 3.截断阈值法（THRESH_TRUNC）

- 定义

  截断阈值法，指将灰度图中的所有像素与阈值进行比较，像素值大于阈值的部分将会被修改为阈值，小于等于阈值的部分不变。换句话说，经过截断阈值法处理过的二值化图中的最大像素值就是阈值。(与传统二值图有所不同，值不为二选一)

- 图示

  ![jie_yu](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/jie_yu.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg',cv2.IMREAD_GRAYSCALE)#读入图片时，已转换为灰度图
  ret,pic_two=cv2.threshold(pic,200,255,cv2.THRESH_TRUNC)
  cv2.imshow('two',pic_two)
  cv2.waitKey(0)
  ```

  ![jie_yu_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/jie_yu_pic.png)

### 4.低阈值零处理（THRESH_TOZERO）

- 定义

  低阈值零处理，字面意思，就是像素值小于等于阈值的部分被置为0（也就是黑色），大于阈值的部分不变。

- 图示

  ![di_yu](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/di_yu.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg',cv2.IMREAD_GRAYSCALE)#读入图片时，已转换为灰度图
  ret,pic_two=cv2.threshold(pic,127,255,cv2.THRESH_TOZERO)
  cv2.imshow('two',pic_two)
  cv2.waitKey(0)
  ```

  ![di_yu_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/di_yu_pic.png)

### 5.超阈值零处理（THRESH_TOZERO_INV）

- 定义

  超阈值零处理就是将灰度图中的每个像素与阈值进行比较，像素值大于阈值的部分置为0（也就是黑色），像素值小于等于阈值的部分不变。

- 图示

  ![chao_yu](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/chao_yu.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg',cv2.IMREAD_GRAYSCALE)#读入图片时，已转换为灰度图
  ret,pic_two=cv2.threshold(pic,127,255,cv2.THRESH_TOZERO_INV)
  cv2.imshow('two',pic_two)
  cv2.waitKey(0)
  ```

  ![chao_yu_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/chao_yu_pic.png)

### 6.OTSU阈值法

- 定义

  OTSU算法是通过一个值将这张图分前景色和背景色（也就是灰度图中小于这个值的是一类，大于这个值的是一类），通过统计学方法（最大类间方差）来验证该值的合理性，当根据该值进行分割时，使用最大类间方差计算得到的值最大时，该值就是二值化算法中所需要的阈值。通常该值是从灰度图中的最小值加1开始进行迭代计算，直到灰度图中的最大像素值减1，然后把得到的最大类间方差值进行比较，来得到二值化的阈值。
  
 ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/OTSU_pic.png)
  
- 图示

  ![double](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/double.png)

  下面举个例子，有一张大小为4×4的图片，假设阈值T为1:

  ![double_test](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/double_test.png)

  也就是这张图片根据阈值1分为了前景（像素为2的部分）和背景（像素为0）的部分，并且计算出了OTSU算法所需要的各个数据，根据上面的数据，我们给出计算方差的公式：

  $$
  g=\omega_{0}(\mu_{0}-\mu)^{2}+\omega_{1}(\mu_{1}-\mu)^{2}
  $$

  g就是前景与背景两类之间的方差，这个值越大，说明前景和背景的差别就越大，效果就越好。OTSU算法就是在灰度图的像素值范围内遍历阈值T，使得g最大，基本上双峰图片的阈值T在两峰之间的谷底。

  通过OTSU算法得到阈值之后，就可以结合上面的方法根据该阈值进行二值化，在本实验中有THRESH_OTSU和THRESH_INV_OTSU两种方法，就是在计算出阈值后结合了阈值法和反阈值法。

  注意：使用OTSU算法计算阈值时，组件中的thresh参数将不再有任何作用。

  | ![double_table1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/double_table1.png) | ![double_table2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/double_table2.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![double_3](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/double_3.png) | ![double_table4](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/double_table4.png) |
  | 方法                                                         | 效果                                                         |

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg',cv2.IMREAD_GRAYSCALE)#读入图片时，已转换为灰度图
  ret,pic_two=cv2.threshold(pic,127,255,cv2.THRESH_OTSU)
  cv2.imshow('two',pic_two)
  cv2.waitKey(0)
  ```

  ![double_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/double_pic.png)

## 图像自适应二值化

- 定义

  与二值化算法相比，自适应二值化更加适合用在明暗分布不均的图片，因为图片的明暗不均，**导致图片上的每一小部分都要使用不同的阈值进行二值化处理**，这时候传统的二值化算法就无法满足我们的需求了，于是就出现了自适应二值化。

  自适应二值化方法会对图像中的所有像素点计算其各自的阈值，这样能够更好的保留图片里的一些信息。

- 图示

  ![adapt_overage](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/adapt_overage.png)

- 语法

  ```python
  cv2.adaptiveThreshold(image_np_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 10)
  ```

- 参数说明

  ```python
  1. image_np_gray: 输入图像，这里必须是灰度图像（单通道）。
  
  2. 255: 输出图像的最大值。在二值化后，超过自适应阈值的像素会被设置为该最大值，通常为255表示白色；未超过阈值的像素将被设置为0，表示黑色。
  
  3. cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 自适应阈值类型。在这个例子中，使用的是高斯加权的累计分布函数（CDF），并添加一个常数 C 来计算阈值。另一种可选类型是 cv2.ADAPTIVE_THRESH_MEAN_C，它使用邻域内的平均值加上常数 C 计算阈值。
  
  4. cv2.THRESH_BINARY: 输出图像的类型。这意味着输出图像将会是一个二值图像（binary image），其中每个像素要么是0要么是最大值（在这里是255）。另外还有其他选项如 cv2.THRESH_BINARY_INV 会得到相反的二值图像。
  
  5. 7: blockSize 参数，表示计算每个像素阈值时所考虑的7x7邻域大小（正方形区域的宽度和高度），其值必须是奇数。
  
  6. 10: C 参数，即上面提到的常数值，在计算自适应阈值时与平均值或高斯加权值相加。正值增加阈值，负值降低阈值，具体效果取决于应用场景。
  ```

### 1.取均值

![adapt_overage_example](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/adapt_overage_example.png)

​		如我们使用的小区域是3\*3的，那么就会从图片的左上角开始（也就是像素值为162的地方）计算其邻域内的平均值，如果处于边缘地区就会对边界进行填充，填充值就是边界的像素点，如下图所示：

![adapt_overage](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/adapt_overage.png)

​		那么对于左上角像素值为162的这个点，161（也就是上图中括号内的计算结果，结果会进行取整）就是根据平均值计算出来的阈值，接着减去一个固定值C，得到的结果就是左上角这个点的二值化阈值了，接着根据选取的是阈值法还是反阈值法进行二值化操作。紧接着，向右滑动计算每个点的邻域内的平均值，直到计算出右下角的点的阈值为止。我们所用到的不断滑动的小区域被称之为核，比如3\*3的小区域叫做3\*3的核，并且核的大小都是奇数个，也就是3\*3、5\*5、7\*7等

```python
import cv2
import numpy as np
pic=cv2.imread('pic.jpg')
pic_gray=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)#转换为灰度图
pic_adaptive=cv2.adaptiveThreshold(pic_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,10)
cv2.imshow('gray',pic_gray)
cv2.imshow('adaptive',pic_adaptive)
cv2.waitKey(0)
```

![adaptive_overage_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/adaptive_overage_pic.png)

### 2.加权求和

- 定义

  对小区域内的像素进行加权求和得到新的阈值，其权重值来自于高斯分布。

  高斯分布，通过概率密度函数来定义高斯分布，一维高斯概率分布函数为：
  
![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/gauss_spacx.png)


- 图示

  通过改变函数中和的值，我们可以得到如下图像，其中均值为，标准差为。

  ![tai](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/tai.png)

  此时我们拓展到二维图像，一般情况下我们使x轴和y轴的相等并且，此时我们可以得到二维高斯函数的表达式为：

  ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/gauss_spacx_two.png)

  高斯概率函数是相对于二维坐标产生的，其中（x,y）为点坐标，要得到一个高斯滤波器模板，应先对高斯函数进行离散化，将得到的值作为模板的系数。例如：要产生一个3\*3的高斯权重核，以核的中心位置为坐标原点进行取样，其周围的坐标如下图所示（x轴水平向右，y轴竖直向上）

  ![spax](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/spax.png)

  将坐标带入上面的公式中，即可得到一个高斯权重核。

  而在opencv里，当kernel(小区域)的尺寸为1、3、5、7并且用户没有设置sigma的时候(sigma \<= 0),核值就会取固定的系数，这是一种默认的值是高斯函数的近似。


  | kernel尺寸 | 核值                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 1          | [1]                                                          |
  | 3          | [0.25, 0.5, 0.25]                                            |
  | 5          | [0.0625, 0.25, 0.375, 0.25, 0.0625]                          |
  | 7          | [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125] |

  比如kernel的尺寸为3\*3时，使用

![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/kernel_one.png)

  进行矩阵的乘法，就会得到如下的权重值，其他的类似。

![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/kernel_two.png)

  通过这个高斯核，即可对图片中的每个像素去计算其阈值，并将该阈值减去固定值得到最终阈值，然后根据二值化规则进行二值化。

  而当kernels尺寸超过7的时候,如果sigma设置合法(用户设置了sigma),则按照高斯公式计算.当sigma不合法(用户没有设置sigma),则按照如下公式计算sigma的值：

$$
  \sigma=0.3*\big((k s i z e-1)*0.5-1\big)+0.8
$$

  某像素点的阈值计算过程如下图所示：

  ![gauss_count](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/gauss_count.png)

  首先还是对边界进行填充，然后计算原图中的左上角（也就是162像素值的位置）的二值化阈值，其计算过程如上图所示，再然后根据选择的二值化方法对左上角的像素点进行二值化，之后核向右继续计算第二个像素点的阈值，第三个像素点的阈值…直到右下角（也就是155像素值的位置）为止。

  当核的大小不同时，仅仅是核的参数会发生变化，计算过程与此是一样的。

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg')
  pic_gray=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)#转换为灰度图
  pic_adaptive=cv2.adaptiveThreshold(pic_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,10)
  cv2.imshow('gray',pic_gray)
  cv2.imshow('adaptive',pic_adaptive)
  cv2.waitKey(0)
  ```

  ![adaptive_gauss_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/adaptive_gauss_pic.png)


## 图形形态学变换

​		形态学变换是一种基于形状的简单变换，它的处理对象通常是二值化图像。形态学变换有两个输入，一个输出：输入为原图像、核（结构化元素），输出为形态学变换后的图像。其基本操作有腐蚀和膨胀，这两种操作是相反的，即较亮的像素会被腐蚀和膨胀。

### 1.核

- 定义

  核（kernel）是一个小区域，通常为3\*3、5\*5、7\*7大小，有着其自己的结构，比如矩形结构、椭圆结构、十字形结构，通过不同的结构可以对不同特征的图像进行形态学操作的处理。

- 图示

  ![kernel](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/kernel.png)

### 2.腐蚀

- 定义

  （找核区域中最小值，替换目标元素值）

  ​		使用核在原图（二值化图）上进行从左到右、从上到下的滑动（也就是从图像的左上角开始，滑动到图像的右下角）。**在滑动过程中，令核值为1的区域与被核覆盖的对应区域进行相乘，得到其最小值，该最小值就是卷积核覆盖区域的中心像素点的新像素值，接着继续滑动。**

  ​		由于操作图像为二值图，所以不是黑就是白，这就意味着，**在被核值为1覆盖的区域内，只要有黑色（像素值为0），那么该区域的中心像素点必定为黑色（0）。**这样做的结果就是会将二值化图像中的白色部分尽可能的压缩，如下图所示，该图经过腐蚀之后，“变瘦”了。

- 图示

  | ![erode_1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/erode_1.png) |
  | ------------------------------------------------------------ |
  | ![erode_2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/erode_2.png) |

- 流程

  1. **初始化**：

     - 设置一个起始位置（通常从图像的左上角开始）。
     - 准备好结构元素（structuring element），它是一个小的矩阵，大小通常是奇数，并且有一个明确的中心点。

  2. **逐像素处理**： 对于输入图像中的每一个像素，执行以下步骤：

     a. **定位**： 将结构元素移动到当前待处理像素的位置，使得结构元素的中心与该像素对齐。

     b. **区域覆盖**： 结构元素会覆盖图像上的一个局部邻域，这个邻域由结构元素的尺寸决定。

     c. **条件检查**： 检查结构元素覆盖区域内所有图像像素的颜色。对于二值图像来说，就是看这些像素是否都是白色（前景像素）。如果所有被结构元素覆盖的像素均为白色，则继续下一个步骤；否则，跳过此步骤，将中心像素视为背景像素。

     d. **侵蚀决策**： 如果结构元素覆盖的所有像素都是白色，则原图像中的中心像素保持不变（在输出图像中仍为白色）；否则，将中心像素变为黑色（在输出图像中变为背景色）。

  3. **迭代移动**： 结构元素沿着图像从左到右、从上到下逐行逐列地移动，重复上述过程，直到整个图像都被结构元素遍历过。

  4. **循环处理**： 如果指定了多个迭代次数，那么在整个图像完成一次遍历后，再次从头开始进行同样的遍历和侵蚀决策，直到达到指定的迭代次数。

- 代码

  ```python
  pic=cv2.imread('black_word.png',cv2.IMREAD_GRAYSCALE)
  kernel=np.ones((3,3),dtype=np.uint8)
  pic_erode=cv2.erode(pic,kernel,iterations=3)
  #iterations：迭代次数，默认值为1。如果设置大于1的值，那么腐蚀操作将连续执行指定的次数。每次迭代都会使得图像中的白色（高亮或前景）区域根据结构元素的形状进一步收缩。
  cv2.imshow('word',pic)
  cv2.imshow('erode_word',pic_erode)
  cv2.waitKey(0)
  ```

  | ![word_yuan](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_yuan.png) | ![word_erode](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_erode.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 灰度图                                                       | 腐蚀图                                                       |

### 3.膨胀

- 定义

  （找核区域中最大值，替换目标元素值）

  ​		**膨胀与腐蚀刚好相反**，膨胀操作就是使用核在原图（二值化图）上进行从左到右、从上到下的滑动（也就是从图像的左上角开始，滑动到图像的右下角），**在滑动过程中，令核值为1的区域与被核覆盖的对应区域进行相乘，得到其最大值，该最大值就是核覆盖区域的中心像素点的新像素值，接着继续滑动。**

  ​		由于操作图像为二值图，所以不是黑就是白，**这就意味着，在卷积核覆盖的区域内，只要有白色（像素值为255），那么该区域的中心像素点必定为白色（255）。这样做的结果就是会将二值化图像中的白色部分尽可能的扩张**，如下图所示，该图经过膨胀之后，“变胖”了。

- 图示

  ![word_dilate](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_dilate.png)

- 流程

  1. **初始化**：

     - 设置一个起始位置（通常从图像的左上角开始）。
     - 准备好结构元素（structuring element），它是一个小的矩阵，大小通常是奇数，并且有一个明确的中心点。

  2. **逐像素处理**： 对于输入图像中的每一个像素，执行以下步骤：

     a. **定位**： 将结构元素移动到当前待处理像素的位置，使得结构元素的中心与该像素对齐。

     b. **区域覆盖**： 结构元素会覆盖图像上的一个局部邻域，这个邻域由结构元素的尺寸决定。

     c. **条件检查**： 检查结构元素覆盖区域内是否存在白色（前景）像素。对于二值图像来说，如果有任何一个被结构元素覆盖的像素是白色的，则继续下一步；否则，将中心像素保持原样（黑色或非目标物体像素不变）。

     d. **膨胀决策**： 如果在结构元素覆盖的范围内找到了至少一个白色像素，则无论原中心像素是什么颜色，都将输出图像中的该中心像素设置为白色（前景色）。这表示即使原中心像素可能是背景像素，但只要其周围有白色像素存在，就认为该位置也应属于前景区域。

     e. **更新输出**： 根据上述判断结果更新输出图像对应位置的像素值。

  3. **迭代移动**： 结构元素沿着图像从左到右、从上到下逐行逐列地移动，重复上述过程，直到整个图像都被结构元素遍历过。

  4. **循环处理**： 如果指定了多个迭代次数，那么在整个图像完成一次遍历后，再次从头开始进行同样的遍历和膨胀决策，直到达到指定的迭代次数。

- 代码

  ```python
  pic=cv2.imread('black_word.png',cv2.IMREAD_GRAYSCALE)
  kernel=np.ones((3,3),dtype=np.uint8)
  pic_dilate=cv2.dilate(pic,kernel,iterations=3)
  cv2.imshow('word',pic)
  cv2.imshow('dilate_word',pic_dilate)
  cv2.waitKey(0)
  ```

| ![word_yuan](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_yuan.png) | ![word_dia](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_dia.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 灰度图                                                       | 膨胀图                                                       |

### 4.开运算

开运算是先腐蚀后膨胀，其**作用**是：分离物体，消除小区域。**特点**：消除噪点，去除小的干扰块，而不影响原来的图像

```python
pic=cv2.imread('black_word.png',cv2.IMREAD_GRAYSCALE)
kernel=np.ones((3,3),dtype=np.uint8)
pic_open=cv2.morphologyEx(pic,cv2.MORPH_OPEN,kernel)
cv2.imshow('word',pic)
cv2.imshow('open_word',pic_open)
cv2.waitKey(0)
```

| ![word_yuan](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_yuan.png) | ![word_open](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_open.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 灰度图                                                       | 开运算效果图                                                 |

### 5.闭运算

闭运算与开运算相反，是先膨胀后腐蚀，**作用**是消除/“闭合”物体里面的孔洞，**特点**：可以填充闭合区域

```python
pic=cv2.imread('black_word.png',cv2.IMREAD_GRAYSCALE)
kernel=np.ones((3,3),dtype=np.uint8)
pic_close=cv2.morphologyEx(pic,cv2.MORPH_CLOSE,kernel)
cv2.imshow('word',pic)
cv2.imshow('erode_word',pic_close)
cv2.waitKey(0)
```

| ![word_yuan](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_yuan.png) | ![word_close](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_close.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 灰度图                                                       | 闭运算效果图                                                 |

### 6.礼帽运算

- 定义

  ​		**原图像与“开运算“的结果图之差,**因为开运算带来的结果是放大了裂缝或者局部低亮度的区域，因此，从原图中减去开运算后的图，得到的效果图突出了比原图轮廓周围的区域更明亮的区域，且这一操作和选择的核的大小相关。

  ​		礼帽运算用来分离比邻近点亮一些的斑块。当一幅图像具有大幅的背景的时候，而微小物品比较有规律的情况下，可以使用礼帽运算进行背景提取

- 代码

  ```python
  pic=cv2.imread('black_word.png',cv2.IMREAD_GRAYSCALE)
  kernel=np.ones((5,5),dtype=np.uint8)
  pic_tophat=cv2.morphologyEx(pic,cv2.MORPH_BLACKHAT,kernel)
  cv2.imshow('pic_tophat',pic_tophat)
  cv2.waitKey(0)
  ```

  | ![word_yuan](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_yuan.png) | ![word_tophat](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_tophat.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 灰度图                                                       | 礼帽运算                                                     |

### 7.黑帽运算

- 定义

  黑帽运算为”闭运算“的结果图与原图像之差,

  黑帽运算后的效果图突出了比原图轮廓周围的区域更暗的区域，且这一操作和选择的核的大小相关。

  黑帽运算用来分离比邻近点暗一些的斑块

- 代码

  ```python
  pic=cv2.imread('black_word.png',cv2.IMREAD_GRAYSCALE)
  kernel=np.ones((5,5),dtype=np.uint8)
  blackhat=cv2.morphologyEx(pic,cv2.MORPH_BLACKHAT,kernel)
  cv2.imshow('blackhat',blackhat)
  cv2.waitKey(0)
  ```

  | ![word_yuan](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_yuan.png) | ![word_blackhat](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_blackhat.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 灰度图                                                       | 黑帽运算                                                     |

### 8.形态学梯度

- 定义

  形态学梯度是一个基于结构元素的图像处理方法，它通过比较原图像与膨胀图和腐蚀图之间的差异来突出图像边缘特征。

  具体来说，对于图像中的每个像素点，其形态学梯度值是**该像素点在膨胀后的图像值与其在腐蚀后的图像值之差**。这样得到的结果通常能够强化图像的边缘信息，并且对噪声有一定的抑制作用

- 代码

  ```python
  pic=cv2.imread('black_word.png',cv2.IMREAD_GRAYSCALE)
  kernel=np.ones((5,5),dtype=np.uint8)
  gradient=cv2.morphologyEx(pic,cv2.MORPH_GRADIENT,kernel)
  cv2.imshow('blackhat',gradient)
  cv2.waitKey(0)
  ```

  | ![word_yuan](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_yuan.png) | ![word_gradient](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/word_gradient.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 灰度图                                                       | 梯度图                                                       |

## 图像颜色识别

### 1.RGB颜色空间

​		在图像处理中，最常见的就是RGB颜色空间。RGB颜色空间是我们接触最多的颜色空间，是一种用于表示和显示彩色图像的一种颜色模型。RGB代表红色(Red)、绿色(Green)和蓝色(Blue)，这三种颜色通过不同强度的光的组合来创建其他颜色。

​		RGB颜色模型基于笛卡尔坐标系，如下图所示，RGB原色值位于3个角上，二次色青色、红色和黄色位于另外三个角上，黑色位于原点处，白色位于离原点最远的角上。因为黑色在RGB三通道中表现为（0，0，0），所以映射到这里就是原点；而白色是（255，255，255），所以映射到这里就是三个坐标为最大值的点。

![RGB_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/RGB_pic.jpeg)



​		在OpenCV中，颜色是以BGR的方式进行存储的，而不是RGB，这也是上面红色的像素值是（0，0，255）而不是（255，0，0）的原因。

### 2.颜色加法

- 定义

  使用OpenCV的cv.add()函数把两幅图像相加，或者可以简单地通过numpy操作添加两个图像，如res = img1 + img2。两个图像应该具有相同的大小和类型。

- cv加法和numpy加法

  ```python
  #OpenCV的加法是饱和操作
  #Numpy添加是模运算
  x = np.uint8([250])
  y = np.uint8([10])
  print( cv.add(x,y) ) # 250+10 = 260 => 255 => [[255]]
  print( x+y )         # 250+10 = 260 % 256 = 4 =>[4]
  ```

- 实例

  ```python
  ##OpenCV的加法是饱和操作
  
  pic1=cv2.imread('hua512.png')
  pic2=cv2.imread('pic512.png')
  pic3=cv2.add(pic1,pic2)
  cv2.imshow('pic1',pic1)
  cv2.imshow('pic2',pic2)
  cv2.imshow('pic3',pic3)
  cv2.waitKey(0)
  ```

  | ![add_pic1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/add_pic1.png) | ![add_pic2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/add_pic2.png) | ![add_pic3](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/add_pic3.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 图1                                                          | 图2                                                          | 颜色加法图                                                   |

  ```python
  #Numpy加法
  
  pic1=cv2.imread('hua512.png')
  pic2=cv2.imread('pic512.png')
  pic3=pic1+pic2
  cv2.imshow('pic1',pic1)
  cv2.imshow('pic2',pic2)
  cv2.imshow('pic3',pic3)
  cv2.waitKey(0)
  ```

  | ![add_pic1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/add_pic1.png) | ![add_pic2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/add_pic2.png) | ![add_num_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/add_num_pic.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 图1                                                          | 图2                                                          | numpy加法图                                                  |

### 3.颜色加权法

- 定义

  这其实也是加法，但是不同的是两幅图像的权重不同，图像混合的计算公式如下：

  `g(x) = (1−α)f0(x) + αf1(x)`

  把两幅图混合在一起，第一幅图的权重是0.7，第二幅图的权重是0.3。函数cv2.addWeighted()可以按下面的公式对图片进行混合操作，γ可取为零。

  `dst = α⋅img1 + β⋅img2 + γ`

- 实例

  ```python
  pic1=cv2.imread('hua512.png')
  pic2=cv2.imread('pic512.png')
  pic3=cv2.addWeighted(pic1,0.5,pic2,0.5,0)
  #pic1=0.5权重，pic2=0.5权重，γ=0
  cv2.imshow('pic1',pic1)
  cv2.imshow('pic2',pic2)
  cv2.imshow('pic3',pic3)
  cv2.waitKey(0)
  ```

  | ![add_pic1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/add_pic1.png) | ![add_pic2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/add_pic2.png) | ![add_weight_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/add_weight_pic.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 图1                                                          | 图2                                                          | 颜色加法权重                                                 |

### 4.HSV颜色空间

- 定义

  ​		HSV颜色空间指的是HSV颜色模型，这是一种与RGB颜色模型并列的颜色空间表示法。RGB颜色模型使用红、绿、蓝三原色的强度来表示颜色，是一种加色法模型，即颜色的混合是添加三原色的强度。

  ​		HSV颜色空间使用色调（Hue）、饱和度（Saturation）和亮度（Value）三个参数来表示颜色，色调H表示颜色的种类，如红色、绿色、蓝色等；饱和度表示颜色的纯度或强度，如红色越纯，饱和度就越高；亮度表示颜色的明暗程度，如黑色比白色亮度低。

- 模型

  ![HSV](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/HSV.png)

- 说明

  1.色调H：

  使用角度度量，取值范围为0°\~360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°，蓝色为240°。它们的补色是：黄色为60°，青色为180°，紫色为300°。通过改变H的值，可以选择不同的颜色

  2.饱和度S：

  饱和度S表示颜色接近光谱色的程度。一种颜色可以看成是某种光谱色与白色混合的结果。其中光谱色所占的比例越大，颜色接近光谱色的程度就越高，颜色的饱和度就越高。饱和度越高，颜色就越深而艳，光谱色的白光成分为0，饱和度达到最高。通常取值范围为0%\~100%，其中0%表示灰色或无色，100%表示纯色，通过调整饱和度的值，可以使颜色变得更加鲜艳或者更加灰暗。

  3.明度V：

  明度表示颜色明亮的程度，对于光源色，明度值与发光体的光亮度有关；对于物体色，此值和物体的透射比或反射比有关。通常取值范围为0%（黑）到100%（白），通过调整明度的值，可以使颜色变得更亮或者更暗。

- 颜色区分范围

  一般对颜色空间的图像进行有效处理都是在HSV空间进行的，然后对于基本色中对应的HSV分量需要给定一个严格的范围，下面是通过实验计算的模糊范围：

  H: 0— 180

  S: 0— 255

  V: 0— 255

  此处把部分红色归为紫色范围：

  ![HSV_table](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/HSV_table.png)

- 作用与意义

  - 符合人类对颜色的感知方式：人类对颜色的感知是基于色调、饱和度和亮度三个维度的，而HSV颜色空间恰好就是通过这三个维度来描述颜色的。因此，使用HSV空间处理图像可以更直观地调整颜色和进行色彩平衡等操作，更符合人类的感知习惯。
  - 颜色调整更加直观：在HSV颜色空间中，色调、饱和度和亮度的调整都是直观的，而在RGB颜色空间中调整颜色不那么直观。例如，在RGB空间中要调整红色系的颜色，需要同时调整R、G、B三个通道的数值，而在HSV空间中只需要调整色调和饱和度即可。
  - 降维处理有利于计算：在图像处理中，降维处理可以减少计算的复杂性和计算量。HSV颜色空间相对于RGB颜色空间，减少了两个维度（红、绿、蓝），这有利于进行一些计算和处理任务，比如色彩分割、匹配等。

### 5图像制作掩膜

- 定义

  掩膜（Mask）是一种在图像处理中常见的操作，它用于选择性地遮挡图像的某些部分，以实现特定任务的目标。**掩膜通常是一个二值化图像**，并且与原图像的大小相同，**其中目标区域被设置为1（或白色），而其他区域被设置为0（或黑色）**，并且目标区域可以根据HSV的颜色范围进行修改。

- 图示

  ![mask_show](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/mask_show.png)

- 实例

  ```python
  pic=cv2.imread('pic.jpg')
  pic_hsv=cv2.cvtColor(pic,cv2.COLOR_BGR2HSV)#BGR颜色空间转换为HSV颜色空间
  green_low=np.array([35,43,46])#绿色空间HSV最低值
  green_high=np.array([77,255,255])#绿色空间HSV最大值
  
  pic_hsv_mask=cv2.inRange(pic_hsv,green_low,green_high)
   #cv2.inRange用于进行多通道图像（尤其是彩色图像）的阈值操作。它将图像中的每个像素值与指定的颜色范围进行比较，并根据比较结果生成一个二值图像（通常称为掩模或标记图像），其中白色像素代表原图中对应位置的像素颜色在设定范围内，而黑色像素则表示不在该范围内。
      
  cv2.imshow('origin',pic)
  cv2.imshow('hsv_mask',pic_hsv_mask)
  cv2.imshow('hsv',pic_hsv)
  cv2.waitKey(0)
  ```

  | ![imshow_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/imshow_pic.png) | ![hsv_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/hsv_pic.png) | ![hsv_mask](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/hsv_mask.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 原图                                                         | HSV图                                                        | Mask图                                                       |

### 6.图像与运算

- 定义

  在图像处理中，“与”运算被用来对图像的像素值进行操作。具体来说，就**是将两个图像中所有的对应像素值一一进行“与”运算，**从而得到新的图像。从上面的图片我们可以看出，**掩膜中有很多地方是黑色的，其像素值为0，那么在与原图像进行“与”运算的时候，得到的新图像的对应位置也是黑色的。**

- 图示

  ![and_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/and_pic.png)

  通过掩膜与原图的与运算，我们就可以提取出图像中被掩膜覆盖的区域(扣图)。

- 实例

  ```python
  pic=cv2.imread('pic.jpg')
  pic_hsv=cv2.cvtColor(pic,cv2.COLOR_BGR2HSV)#BGR颜色空间转换为HSV颜色空间
  green_low=np.array([35,43,46])#绿色空间HSV最低值
  green_high=np.array([77,255,255])#绿色空间HSV最大值
  pic_hsv_mask=cv2.inRange(pic_hsv,green_low,green_high)#创建掩膜
  
  pic_and=cv2.bitwise_and(pic,pic,mask=pic_hsv_mask)
    # 输出图像中的每个像素值将是输入的两个图像对应像素值进行位与操作的结果。但当提供了第三个参数 mask存在时，该掩模图像会决定哪些位置上的像素进行实际的位与计算。掩模图像通常是一个单通道图像，其中非零（例如白色或高值）像素表示在位与操作中要保留的位置。
    # 注意 参数一pic和参数二pic的每一位像素进行按位与操作，由于是同一张图片，因此相同像素按位与后还是当前像素
    #相同的数值“与”运算后还是这个数值不变
  cv2.imshow('pic_and',pic_and)
  cv2.imshow('origin',pic)
  cv2.imshow('hsv_mask',pic_hsv_mask)
  cv2.imshow('hsv',pic_hsv)
  cv2.waitKey(0)
  ```

  | ![imshow_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/imshow_pic.png) | ![hsv_mask](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/hsv_mask.png) | ![mask_and](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/mask_and.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 原图                                                         | 掩膜                                                         | 与运算图                                                     |

## 图像颜色替换

### 1.图像制作掩膜

![mask_show](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/mask_show.png)

**通过这个掩膜，我们就可以对掩膜中的白色区域所对应的原图中的区域（也就是原图中的红色区域）进行像素值的修改，从而完成颜色替换的功能。**

### 2.颜色替换

- 定义

  由于掩膜与原图的大小相同，并且像素位置一一对应，那么我们就可以得到掩膜中白色（也就是像素值为255）区域的坐标，并将其带入到原图像中，即可得到原图中的红色区域的坐标，然后就可以修改像素值了，这样就完成了颜色的替换。

- 图示

  ![replace](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/replace.png)

- 实例

  ```python
  pic=cv2.imread('pic.jpg')
  pic_hsv=cv2.cvtColor(pic,cv2.COLOR_BGR2HSV)#BGR颜色空间转换为HSV颜色空间
  green_low=np.array([35,43,46])#绿色空间HSV最低值
  green_high=np.array([77,255,255])#绿色空间HSV最大值
  
  pic_hsv_mask=cv2.inRange(pic_hsv,green_low,green_high)
  
  pic[pic_hsv_mask>0]=(0,0,255)#布尔索引获取掩膜中大于0的像素点的坐标，并替换颜色
  cv2.imshow('replce_pic',pic)
  cv2.imshow('hsv_mask',pic_hsv_mask)
  cv2.imshow('hsv',pic_hsv)
  cv2.waitKey(0)
  ```

  | ![imshow_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/imshow_pic.png) | ![hsv_mask](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/hsv_mask.png) | ![replace_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/replace_pic.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 原图                                                         | 掩膜                                                         | 颜色替换                                                     |

## 图像ROI切割

- 定义

  ROI：Region of Interest，翻译过来就是感兴趣的区域。比如对于一个人的照片，假如我们要检测眼睛，因为眼睛肯定在脸上，所以我们感兴趣的只有脸这部分，其他都不care，所以可以单独把脸截取出来，这样就可以大大节省计算量，提高运行速度。

- 操作

  我们在使用OpenCV进行读取图像时，图像数据会被存储为Numpy数组，这也意味着我们可以使用Numpy数组的一些操作来对图像数据进行处理，比如切片。

- 说明

  在OpenCV中，坐标的x轴的正方向是水平向右，y轴的正方向是垂直向下，与数学上的二维坐标并不相同。

  在计算机视觉中，当我们使用OpenCV读取RGB三通道图像时，它会被转换成一个三维的Numpy数组。这个数组里的每个元素值都表示图像的一个像素值。这个三维数组的第一个维度（即轴0）通常代表图像的高度，第二个维度（即轴1）代表图像的宽度，而第三个维度（即轴2）代表图像的三个颜色通道（B、G、R，）OpenCV读取到的图像以BGR的方式存储所对应的像素值。

- 实例

  ```python
  #用矩形提取兴趣区域
  pic=cv2.imread('pic.jpg')
  
  x_min,x_max=100,200
  y_min,y_max=300,500
  
  pic_rect=cv2.rectangle(pic,(x_min-1,y_min-1),(x_max+1,y_max+1),(255,0,0),1)
  
  pic_get=pic[y_min:y_max,x_min:x_max]
  
  cv2.imshow('origin',pic)
  cv2.imshow('pic_get',pic_get)
  cv2.waitKey(0)
  ```

  | ![ROI](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/ROI.png) | ![ROI_get](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/ROI_get.png) |
  | ------------------------------------------------------ | ------------------------------------------------------------ |
  | 原图                                                   | 截取图                                                       |

## 图像旋转

图像旋转是指图像以某一点为旋转中心，将图像中的所有像素点都围绕该点旋转一定的角度，并且旋转后的像素点组成的图像与原图像相同。

### 1.单点旋转

​	以最简单的一个点的旋转为例子，且以最简单的情况举例，令旋转中心为坐标系中心O(0，0)，假设有一点P0(x0,y0)，P0离旋转中心O的距离为r，OP与坐标轴x轴的夹角为α，P0绕O顺时针旋转Θ角后对应的点为P(x,y):

![one_point](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/one_point.png)

那么我们可以得到如下关系：


$$
y_{0}=r\times\sin\alpha
$$

$$
x_{0}=r\times\cos\alpha
$$


$$
x=r\times\cos(\alpha-\theta)=r\cos\alpha\cos\theta+r\sin\alpha\sin\theta=x_{0}\cos\theta+y_{0}\sin\theta
$$

$$
y=r\times\sin(\alpha-\theta)=r\sin\alpha\cos\theta-r\cos\alpha\sin\theta=-x_{0}\sin\theta+y_{0}\cos\theta
$$

用矩阵来表示就是

  ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M1.png)

然而，**在OpenCV中，旋转时是以图像的左上角为旋转中心，且以逆时针为正方向，因此上面的例子中其实是个负值，**那么该矩阵可写为：

  ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M2.png)
  
其中，

  ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M3.png)
  
也被称作旋转矩阵。然而我们所要的不仅仅是可以围绕图像左上角进行旋转，而是可以围绕任意点进行旋转。那么我们可以将其转化成绕原点的旋转，其过程为：

1. **首先将旋转点移到原点**
2. **按照上面的旋转矩阵进行旋转得到新的坐标点**
3. **再将得到的旋转点移回原来的位置**

也就是说，在以任意点为旋转中心时，除了要进行旋转之外，还要进行平移操作。那么当点经过平移后得到P点时，如下图所示：

![random_point](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/random_point.png)

那么我们就可以得到：

$$
y=y_{0}+t_{y}
$$

$$
x=x_{0}+t_{x}
$$

写成矩阵的形式为：

![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M4.png)

于是

![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M5.png)

也被叫做平移矩阵，相反的，从P移到点时，其平移矩阵为：

![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M6.png)

我们将原始的旋转矩阵也扩展到3\*3的形式：

![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M7.png)

从平移和旋转的矩阵可以看出，3x3矩阵的前2x2部分是和旋转相关的，第三列与平移相关。有了上面的表达式之后，我们就可以得到二维空间中绕任意点旋转的旋转矩阵了，只需要将旋转矩阵先左乘

![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M8.png)

，再右乘

![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M9.png)

即可得到最终的矩阵，其结果为：

![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M10.png)

于是我们就可以根据这个矩阵计算出图像中任意一点绕某点旋转后的坐标了，这个矩阵学名叫做**仿射变换矩阵**，而仿射变换是一种二维坐标到二维坐标之间的线性变换，也就是只涉及一个平面内二维图形的线性变换，图像旋转就是仿射变换的一种。它保持了二维图形的两种性质：

1. **平直性：直线经过变换后依然是直线。**
2. **平行性：平行线经过变换后依然是平行线。**

### 2.图片旋转

- 定义

  将图像里的每个像素点都带入仿射变换矩阵里，从而得到旋转后的新坐标

  在OpenCV中，要得到仿射变换矩阵可以使用cv2.getRotationMatrix2D()，通过这个函数即可直接获取到上面的旋转矩阵。

- 语法

  ```python
  cv2.getRotationMatrix2D(Center,Angle,Scale)
  ```

- 参数

  - Center：表示旋转的中心点，是一个二维的坐标点(x,y)
  - Angle：表示旋转的角度
  - Scale：表示缩放比例，可以通过该参数调整图像相对于原始图像的大小变化

- 操作说明

  ​		由于三角函数的值是小数，那么其乘积也会是小数，虽然OpenCV中会对其进行取整操作，但是像素点旋转之后的取整结果也有可能重合，这样就会导致可能会在旋转的过程中丢失一部分原始的像素信息。并且如果使用了scale参数进行图像的缩放的话，**当图像放大时，比如一个10\*10的图像放大成20\*20，图像由100个像素点变成400个像素点，多余的300个像素点是怎么来的？当图像缩小时，比如一个20\*20的图像缩小为10\*10的图像，需要丢掉300个像素点，如何保持图像正确？**为了保证图像的完整性，这种方法就叫做**插值法**。

- 实例

  ```python
  pic=cv2.imread('hua512.png')
  
  height,width,_=pic.shape
  M=cv2.getRotationMatrix2D((height//2,width//2),angle=45,scale=1)
  #取得对应点的旋转矩阵(旋转中心点,旋转角度,缩放比例)
  
  rotation_pic=cv2.warpAffine(pic,M,dsize=(width,height))
  #根据旋转矩阵，进行旋转图片
  """
  cv2.warpAffine(pic,M,
  (img_shape[0],img_shape[1]),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT_101)
  
  """
  cv2.imshow('hua',pic)
  cv2.imshow('rotation_pic',rotation_pic)
  
  cv2.waitKey(0)
  ```

| ![add_pic1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/add_pic1.png) | ![rotation_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/rotation_pic.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 原图                                                         | 旋转图                                                       |

### 3.插值方法

​		图像处理中常用于处理图像的放大、缩小、旋转、变形等操作，以及处理图像中的像素值。

​		图像插值算法是为了解决图像缩放或者旋转等操作时，由于像素之间的间隔不一致而导致的信息丢失和图像质量下降的问题。当我们对图像进行缩放或旋转等操作时，需要在新的像素位置上计算出对应的像素值，而插值算法的作用就是根据已知的像素值来推测未知位置的像素值。

#### 3.1最近邻插值

- 语法

  ```python
  CV2.INTER_NEAREST
  ```

- 公式


  ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M11.png)
  

  **dstX**表示目标图像中某点的x坐标，**srcWidth**表示原图的宽度，dstWidth表示目标图像的宽度；**dstY**表示目标图像中某点的y坐标，**srcHeight**表示原图的高度，**dstHeight**表示目标图像的高度。而**srcX**和**srcY**则表示目标图像中的某点对应的原图中的点的x和y的坐标。

- 图示

  比如一个2\*2的图像放大到4\*4，如下图所示，其中红色的为每个像素点的坐标，黑色的则表示该像素点的像素值。

  ![near_show](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/near_show.png)

  根据公式我们就可以计算出放大后的图像（0，0）点对应的原图像中的坐标为：
  
 ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M12.png)
 

  也就是原图中的（0，0）点，而最近邻插值的原则是：目标像素点的像素值与经过该公式计算出来的对应的像素点的像素值相同，**如出现小数部分需要进行取整(向下取整)**。那么放大后图像的（0，0）坐标处的像素值就是原图像中（0，0）坐标处的像素值，也就是10。

  接下来就是计算放大后图像（1，0）点对应的原图像的坐标，还是带入公式：
  
![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M13.png)


  也就是原图中的（0.5，0）点，因此需要对计算出来的坐标值进行取整，取整后的结果为（0，0），也就是说放大后的图像中的（1，0）坐标处对应的像素值就是原图中（0，0）坐标处的像素值，其他像素点计算规则与此相同。

#### 3.2双线性插值

- 语法

  ```python
  CV2.INTER_LINEAR
  ```

- 原理

  对图像进行变换时，特别是尺寸变化时，原始图像的某些像素坐标可能不再是新图像中的**整数位置**，这时就需要使用插值算法来确定这些非整数坐标的像素值。

  1. 假设要查找目标图像上坐标为 `(x', y')` 的像素值，在原图像上对应的浮点坐标为 `(x, y)`。

  2. 在原图像上找到四个最接近`(x, y)`的像素点，通常记作 `P00(x0, y0)`, `P01(x0, y1)`, `P10(x1, y0)`, `P11(x1, y1)`，它们构成一个2x2的邻域矩阵。

  3. 分别在**水平方向和垂直方向上做线性插值**：

     - 水平方向：根据 `x` 与 `x0` 和 `x1` 的关系计算出 `P00` 和 `P10` 之间的插值结果。
     - 垂直方向：将第一步的结果与 `y` 与 `y0` 和 `y1` 的关系结合，再在 `P00-P01` 和 `P10-P11` 对之间做一次线性插值。

  4. 综合上述两次线性插值的结果，得到最终位于 `(x', y')` 处的新像素的估计值。

- 公式

  假如已知两个点(X0,Y0)和(x1,y1)，我们要计算[x0,y0]区间内某一位置x在直线上的y值，那么计算过程为：
  
![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M14.png)


  仔细看公式，其实就是计算距离，并将距离作为一个权重用于y0和y1的加权求和。这就是线性插值，而双线性插值本质上就是在两个方向上做线性插值。

  还是给出目标点与原图像中点的计算公式：
  
![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M15.png)

- 图示

  根据上述公式计算出了新图像中的某点所对应的原图像的点P，其周围的点分别为Q12、Q22、Q11、Q21， 要插值的P点不在其周围点的连线上，这时候就需要用到双线性插值了。首先延申P点得到P和Q11、Q21的交点R1与P和Q12、Q22的交点R2，如下图所示：

  ![liner_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/liner_pic.png)

  然后根据Q11、Q21得到R1的插值，根据Q12、Q22得到R2的插值，然后根据R1、R2得到P的插值即可，这就是双线性插值。

  首先计算R1和R2的插值：
  
 ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M16.png)
 

  然后根据R1和R2计算P的插值：
  
 ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M17.png)
 
  这样就得到了P点的插值。注意此处如果先在y方向插值、再在x方向插值，其结果与按照上述顺序双线性插值的结果是一样的。

- 出现的问题

  1.根据坐标系的不同，产生的结果不同，这张图是左上角为坐标系原点的情况，我们可以发现最左边x=0的点都会有概率直接复制到目标图像中（至少原点肯定是这样），而且就算不和原图像中的点重合，也相当于进行了1次单线性插值.

  而且无论我们采用什么坐标系，最左侧和最右侧（最上侧和最下侧）的点是“不公平的”。

  2.整体的图像相对位置会发生变化。左侧是原图像(3，3)，右侧是目标图像(5，5)，原图像的几何中心点是(1,1)，目标图像的几何中心点是(2,2)，根据对应关系，目标图像的几何中心点对应的原图像的位置是(1.2,1.2)，那么问题来了，目标图像的原点(0,0)和原始图像的原点是重合的，但是目标图像的几何中心点相对于原始图像的几何中心点偏右下，那么整体图像的位置会发生偏移。

  ![problem](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/problem.png)

- 改进公式



![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M18.png)


#### 3.3像素区域插值

- 语法

  ```python
  cv2.INTER_AREA
  ```

- 原理

  像素区域插值主要分两种情况:缩小图像和放大图像

  缩小图像时：像素区域插值方法，它就会变成一个**均值滤波器**，对一个区域内的像素值取平均值。

  放大图像时：如果图像放大的比例**是整数倍**，那么其工作原理与**最近邻插值类似**；如果放大的比例**不是整数倍**，那么就会调用双线性插值进行放大。

- 公式



![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M19.png)


#### 3.4双三次插值

- 语法

  ```python
  cv2.INTER_CUBIC
  ```

- 原理

  与双线性插值法相同，该方法也是通过映射，在映射点的邻域内通过加权来得到放大图像中的像素值。不同的是，双三次插值法需要原图像中**近邻的16个点来加权**。

- 公式

  目标像素点与原图像的像素点的对应公式如下所示：
  
![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M20.png)


- 说明

  假设原图像A大小为m\*n，缩放后的目标图像B的大小为M\*N。其中A的每一个像素点是已知的，B是未知的，我们想要求出目标图像B中每一个像素点（X,Y）的值，必须先找出像素（X,Y）在原图像A中对应的像素（x,y），再根据原图像A距离像素（x,y）最近的16个像素点作为计算目标图像B（X,Y）处像素值的参数，利用**BiCubic基函数**求出16个像素点的权重，图B像素（x,y）的值就等于16个像素点的加权叠加。

  下图中的P点就是目标图像B在（X,Y）处根据上述公式计算出的对应于原图像A中的位置，P的坐标位置会出现小数部分，所以我们假设P点的坐标为（x+u,y+v），其中x、y表示整数部分，u、v表示小数部分，那么我们就可以得到其周围的最近的16个像素的位置，我们用a（i，j）（i，j=0,1,2,3）来表示。

  ![cubi_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cubi_pic.png)

  BiCubic函数：
  ![aaaa](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/aaaa.png)

  其中，**a一般取-0.5或-0.75。**

  我们要做的就是将上面的16个点的坐标带入函数中，获取16像素所对应的权重W(x)。然而BiCubic函数是一维的，所以我们需要将像素点的行与列分开计算，比如a00这个点，我们需要将x=0带入BiCubic函数中，计算a00点对于P点的x方向的权重，然后将y=0带入BiCubic函数中，计算a00点对于P点的y方向的权重，其他像素点也是这样的计算过程，最终我们就可以得到P所对应的目标图像B在（X,Y）处的像素值为：
  
![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M21.png)

#### 3.5Lanczos插值

- 语法

  ```
  cv2.INTER_LANCZOS4
  ```

- 原理

  Lanczos插值方法与双三次插值的思想是一样的，不同的就是其需要的原图像周围的像素点的范围变成了8\*8，并且不再使用BiCubic函数来计算权重，而是换了一个公式计算权重。

- 公式

  目标像素点与原图像的像素点的对应公式如下所示：
  
![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M22.png)


- 说明

  假设原图像A大小为m\*n，缩放后的目标图像B的大小为M\*N。其中A的每一个像素点是已知的，B是未知的，我们想要求出目标图像B中每一个像素点（X,Y）的值，必须先找出像素（X,Y）在原图像A中对应的像素（x,y），再根据原图像A距离像素（x,y）最近的64个像素点作为计算目标图像B（X,Y）处像素值的参数，利用权重函数求出64个像素点的权重，图B像素（x,y）的值就等于64个像素点的加权叠加。

  假如下图中的P点就是目标图像B在（X,Y）处根据上述公式计算出的对应于原图像A中的位置，P的坐标位置会出现小数部分，所以我们假设P点的坐标为（x+u,y+v），其中x、y表示整数部分，u、v表示小数部分，那么我们就可以得到其周围的最近的64个像素的位置，我们用a（i，j）（i，j=0,1,2,3,4,5,6,7）来表示。

  ![Lan_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/Lan_pic.png)

  权重公式：

![bbb](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/bbb.png)

**其中a通常取2或者3，当a=2时，该算法适用于图像缩小。a=3时，该算法适用于图像放大。**

与双三次插值一样，这里也需要将像素点分行和列分别带入计算权重值，其他像素点也是这样的计算过程，最终我们就可以得到P所对应的目标图像B在（X,Y）处的像素值为：

![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M23.png)


#### 3.7插值方法小结

最近邻插值的计算**速度最快**，但是可能会导致图像出现锯齿状边缘和失真，效果较差。双线性插值的计算速度慢一点，但效果有了大幅度的提高，适用于大多数场景。双三次插值、Lanczos插值的计算速度都很慢，但是效果都很好。

**在OpenCV中，关于插值方法默认选择的都是双线性插值**

### 4.边缘填充

左图在逆时针旋转45度之后原图的四个顶点在右图中已经看不到了，右图的四个顶点区域其实是什么都没有的，因此我们需要对空出来的区域进行一个填充。右图就是对空出来的区域进行了像素值为（0，0，0）的填充，也就是黑色像素值的填充。

| ![add_pic1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/add_pic1.png) | ![rotation_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/rotation_pic.png) |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| 原图                                                        | 旋转图                                                       |

#### 4.1边界复制（BORDER_REPLICATE）

- 定义

  边界复制会将边界处的像素值进行复制，然后作为边界填充的像素值

- 图示

  ![replicate](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/replicate.png)

- 实例

  ```python
  pic=cv2.imread('cat1.jpg')
  height,width,_=pic.shape
  
  M=cv2.getRotationMatrix2D((height//2,width//2),angle=45,scale=1)#取得对应旋转点的旋转矩阵
  
  rotation_pic=cv2.warpAffine(pic,M,dsize=(width,height),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)#根据旋转矩阵，进行旋转图片，插值为最近邻方法,填充为边界复制
  cv2.imshow('cat',pic)
  cv2.imshow('rotation_pic',rotation_pic)
  cv2.waitKey(0)
  ```

  | ![cat1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat1.png) | ![cat_rote](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_rote.png) | ![cat_replacte](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_replacte.png) |
  | --------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
  | 原图                                                | 旋转图                                                      | 填充图                                                       |

#### 4.2边界反射（BORDER_REFLECT）

- 图示

  ![reflect](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/reflect.png)

- 实例

  ```python
  pic=cv2.imread('cat1.jpg')
  height,width,_=pic.shape
  
  M=cv2.getRotationMatrix2D((height//2,width//2),angle=45,scale=0.5)#取得对应旋转点的旋转矩阵
  
  rotation_pic=cv2.warpAffine(pic,M,dsize=(width,height),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT)#根据旋转矩阵，进行旋转图片,插值为最近邻方法，填充为边界反射
  cv2.imshow('cat',pic)
  cv2.imshow('rotation_pic',rotation_pic)
  cv2.waitKey(0)
  ```

  | ![cat1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat1.png) | ![cat_rote](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_rote.png) | ![cat_reflect](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_reflect.png) |
  | --------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
  | 原图                                                | 旋转图                                                      | 边界反射                                                     |

#### 4.3边界反射_101（BORDER_REFLECT_101）

与边界反射不同的是，不再反射边缘的像素点

- 图示

  ![101](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/101.png)

- 实例

  ```python
  pic=cv2.imread('cat1.jpg')
  height,width,_=pic.shape
  
  M=cv2.getRotationMatrix2D((height//2,width//2),angle=45,scale=0.5)#取得对应旋转点的旋转矩阵
  
  rotation_pic=cv2.warpAffine(pic,M,dsize=(width,height),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT101)#根据旋转矩阵，进行旋转图片,插值为最近邻方法,填充为边界反射101
  cv2.imshow('cat',pic)
  cv2.imshow('rotation_pic',rotation_pic)
  cv2.waitKey(0)
  ```

  | ![cat1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat1.png) | ![cat_rote](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_rote.png) | ![cat_101](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_101.png) |
  | --------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------- |
  | 原图                                                | 旋转图                                                      | 边界反射101                                               |

#### 4.4边界常数（BORDER_CONSTANT）

当选择边界常数时，还要指定常数值是多少，默认的填充常数值为0

- 图示

  ![constant](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/constant.png)

- 实例

  ```python
  pic=cv2.imread('cat1.jpg')
  height,width,_=pic.shape
  
  M=cv2.getRotationMatrix2D((height//2,width//2),angle=45,scale=0.5)#取得对应旋转点的旋转矩阵
  
  rotation_pic=cv2.warpAffine(pic,M,dsize=(width,height),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,0,0))#根据旋转矩阵，进行旋转图片插值,为最近邻方法,填充为边界常数，数值为蓝色
  cv2.imshow('cat',pic)
  cv2.imshow('rotation_pic',rotation_pic)
  cv2.waitKey(0)
  ```

  | ![cat1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat1.png) | ![cat_rote](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_rote.png) | ![cat_constant](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_constant.png) |
  | --------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
  | 原图                                                | 旋转图                                                      | 边界常熟                                                     |

#### 4.5边界包裹（BORDER_WRAP）

- 图示

  ![wrap](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/wrap.png)

- 实例

  ```python
  pic=cv2.imread('cat1.jpg')
  height,width,_=pic.shape
  
  M=cv2.getRotationMatrix2D((height//2,width//2),angle=45,scale=0.5)#取得对应旋转点的旋转矩阵
  
  rotation_pic=cv2.warpAffine(pic,M,dsize=(width,height),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_WRAP)#根据旋转矩阵，进行旋转图片,插值为最近邻方法,填充为边界包裹
  cv2.imshow('cat',pic)
  cv2.imshow('rotation_pic',rotation_pic)
  cv2.waitKey(0)
  ```

  | ![cat1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat1.png) | ![cat_rote](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_rote.png) | ![cat_wrap](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_wrap.png) |
  | --------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
  | 原图                                                | 旋转图                                                      | 边界包裹                                                    |

## 图片镜像旋转

- 定义

  图像的旋转是围绕一个特定点进行的，而图像的镜像旋转则是围绕坐标轴进行的。图像的镜像旋转分为水平翻转、垂直翻转、水平垂直翻转三种。

  水平翻转就是将图片的像素点沿y轴翻转，具体到像素点来说就是令其坐标从（x,y）翻转为（-x,y）。

  垂直翻转就是将图片的像素点沿x轴翻转，具体到像素点来说就是其坐标从（x,y）翻转为（x,-y）

  水平垂直翻转就是水平翻转和垂直翻转的结合，具体到像素点来说就是其坐标从（x,y）翻转为（-x,-y）。

- 语法

  ```
  cv2.flip(image, flipCode)
  ```

- 公式

  ![aaa](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/aaa.png)

- 说明

  图像在旋转的时候需要有旋转中心，而图像的镜像旋转虽然都是围绕x轴和y轴进行旋转，但是我们**也需要确定x轴和y轴的坐标**。在OpenCV中，**图片的镜像旋转是以图像的中心为原点进行镜像翻转的**。也就是说，**水平翻转时，图像的左侧和右侧会关于中心点进行交换，垂直翻转时，图像的上侧和下侧会关于中心点进行交换。**

  - 0：垂直翻转
  - 大于0：水平翻转
  - 小于0：水平垂直翻转

- 实例

  ```python
  pic=cv2.imread('cat1.jpg')
  
  mirror_pic_vis=cv2.flip(pic,0)
  mirror_pic_flat=cv2.flip(pic,1)
  mirror_pic_vis_flat=cv2.flip(pic,-1)
  
  cv2.imshow('mirror_pic_vis',mirror_pic_vis)
  cv2.imshow('mirror_pic_flat',mirror_pic_flat)
  cv2.imshow('mirror_pic_vis_flat',mirror_pic_vis_flat)
  
  cv2.waitKey(0)
  ```

  | ![cat1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat1.png) | ![cat_vis](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_vis.png) | ![cat_flat](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_flat.png) | ![cat_flat_via](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_flat_via.png) |
  | --------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
  | 原图                                                | 垂直翻转                                                  | 水平翻转                                                    | 垂直水平翻转                                                 |

## 图像缩放

- 语法

  ```python
  cv2.resize(image_np, dsize=None, fx=0.6, fy=0.6, interpolation=cv2.INTER_LINEAR)
  ```

- 参数

  1. **image_np**:
     - 这是待缩放的原始图像，它是一个numpy数组表示的图像。
  2. **dsize**:
     - 这个参数代表目标图像的尺寸（宽度和高度），在这里设置为 `None` 意味着我们不直接指定新图像的具体尺寸，而是通过接下来的 `fx` 和 `fy` 参数来按比例缩放原图像。**若设置了dsize，则fx，fy失效**。
  3. **fx**:
     - 这是一个缩放因子，用来控制图像水平方向（宽度）的缩放比例。如果 `fx=0.6`，那么原图像的宽度将缩小到原来宽度的60%。
  4. **fy**:
     - 同样是一个缩放因子，但它控制的是图像垂直方向（高度）的缩放比例。当 `fy=0.6` 时，原图像的高度将缩小到原来高度的60%。
  5. **interpolation**:
     - 插值方法，用于决定如何计算新尺寸图像中的像素值。`cv2.INTER_LINEAR` 表示双线性插值，这是一种常用的、平滑且质量相对较高的插值方式，能够较好地保留图像细节和连续性。

- 实例

  ```python
  pic=cv2.imread('cat1.jpg')
  pic_rsize=cv2.resize(pic,dsize=None,fx=0.7,fy=0.3,interpolation=cv2.INTER_LANCZOS4)
  cv2.imshow('rsize',pic_rsize)
  cv2.waitKey(0)
  ```

  | ![cat1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat1.png) | ![cat_rsize](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/cat_rsize.png) |
  | --------------------------------------------------- | ------------------------------------------------------------ |
  | 原图                                                | 缩放图                                                       |

## 图片矫正

- 矫正原理

  仿射变换：

  ​		是把一个二维坐标系转换到另一个二维坐标系的过程，转换过程坐标点的相对位置和属性不发生变换，是一个线性变换，该过程只发生旋转和平移过程。因此，一个平行四边形经过仿射变换后还是一个平行四边形。

  ![ping](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/ping.png)

  透视变换：

  ​		是**把一个图像投影到一个新的视平面的过程**，在现实世界中，我们观察到的物体在视觉上会受到透视效果的影响，即远处的物体看起来会比近处的物体小。透视投影是指将三维空间中的物体投影到二维平面上的过程，这个过程会导致物体在图像中出现形变和透视畸变。透视变换可以通过数学模型来校正这种透视畸变，使得图像中的物体看起来更符合我们的直观感受。

  ​		图1在经过透视变换后得到了图2的结果，带入上面的话就是图像中的车道线（目标物体）的被观察视角从平视视角变成了俯视视角，这就是透视变换的作用。

  ![per_1 (1)](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/per_1.png)

  ![per_1 (2)](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/per_2.png)

- 数学关系

  透视变换矩阵：


  
![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M24.png)

  即
  
 ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M5.png)
 

  由此可得新的坐标的表达式为：
  
  ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/M26.png)
  

  其中x、y是原始图像点的坐标，x‘、y’是变换后的坐标，a11，a12，…,a33则是一些旋转量和平移量，由于透视变换矩阵的推导涉及三维的转换，所以这里不具体研究该矩阵，只要会使用就行。

- 语法函数

  ```python
  getPerspectiveTransform(src,dst)
  ```

  src：原图像上需要进行透视变化的四个点的坐标，这四个点用于定义一个原图中的四边形区域。

  dst：透视变换后，src的四个点在新目标图像的四个新坐标。

  该函数会返回一个透视变换矩阵。

  ```python
  cv2.warpPerspective(src, M, dsize, flags, borderMode)
  ```

  src：输入图像。

  M：透视变换矩阵。这个矩阵可以通过getPerspectiveTransform函数计算得到。

  dsize：输出图像的大小。它可以是一个Size对象，也可以是一个二元组。

  flags：插值方法的标记。

  borderMode：边界填充的模式。

- 实例

  ```python
  pic=cv2.imread('card.png')
  print(pic.shape)
  height,width,_=pic.shape
  print(height,width)
  pic1_loc=np.float32([[223,149],[440,269],[102,242],[319,380]])
  pic2_loc=np.float32([[0,0],[width,0],[0,height],[width,height]])
  print(pic1_loc.shape,pic2_loc.shape)
  M=cv2.getPerspectiveTransform(pic1_loc,pic2_loc)
  
  pic_per=cv2.warpPerspective(pic,M,dsize=(width,height),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT)
  
  cv2.imshow('card',pic)
  cv2.imshow('card_per',pic_per)
  cv2.waitKey(0)
  ```

  | ![card](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/card.png) | ![per_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/per_pic.png) |
  | --------------------------------------------------- | --------------------------------------------------------- |
  | 原图                                                | 矫正图                                                    |

## 图像添加水印

- 定义

  将一张图片中的某个物体或者图案提取出来，然后叠加到另一张图片上。

- 原理

  通过将原始图片转换成灰度图，并进行二值化处理，去除背景部分，得到一个类似掩膜的图像。然后将这个二值化图像与另一张图片中要添加水印的区域进行“与”运算，使得目标物体的形状出现在要添加水印的区域。最后，将得到的目标物体图像与要添加水印的区域进行相加，就完成了添加水印的操作。

- 实例

  ```python
  #自定义方法
  pic=cv2.imread('pic.jpg')
  logo=cv2.imread('logo.png')
  
  
  logo_height,logo_width,_=logo.shape#获取logo长宽值
  
  hsv_logo=cv2.cvtColor(logo,cv2.COLOR_BGR2HSV)#水印图转HSV颜色空间，方便提取文字和字母
  
  logo_word_low=np.array([0,43,46])#文字颜色空间最低值
  logo_word_high=np.array([10,255,255])#文字颜色空间最高值
  
  log_alpha_low=np.array([0,0,46])#字母颜色空间最低值
  log_alpha_high=np.array([180,43,220])#字母颜色空间最高值
  
  hsv_logo_word_mask=cv2.inRange(hsv_logo,logo_word_low,logo_word_high)#获得文字掩膜
  hsv_logo_alpha_mask=cv2.inRange(hsv_logo,log_alpha_low,log_alpha_high)#获得字母掩膜
  
  pic_logo=pic[200:200+logo_height,200:200+logo_width]#在目标图片中指定位置切出logo大小区域
  
  pic_logo[hsv_logo_word_mask>0]=logo[hsv_logo_word_mask>0]#替换文字
  pic_logo[hsv_logo_alpha_mask>0]=logo[hsv_logo_alpha_mask>0]#替换字母
  
  print(pic_logo.shape)
  
  cv2.imshow('hsv_logo',hsv_logo_word_mask)
  cv2.imshow('logo',pic_logo)
  cv2.imshow('pic',pic)
  cv2.waitKey(0)
  ```

  ```python
  #标准方法
  import cv2
  import numpy as np
  
  #引入两个图片，第二个是logo
  img1 = cv2.imread("./bg.png")
  img2 = cv2.imread('logohq.png')
  r1,c1,ch1 = img1.shape
  r2,c2,ch2 = img2.shape
  roi = img1[:r2,:c2]#取出img1中跟img2同样大小的区域
  # cv2.imshow("roi",roi)
  gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)#得到logo的灰度图
  ret, ma1 = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)#得到logo的掩膜:黑化的logo
  # cv2.imshow("ma1",ma1)
  
  fg1 = cv2.bitwise_and(roi,roi,mask=ma1)#roi中跟ma1中黑色重叠的部分也变成黑色 其他地方颜色不变
  cv2.imshow("fg1",fg1)
  
  
  ret, ma2 = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)#得到logo的掩膜:白色化的logo
  # cv2.imshow("ma2",ma2)
  fg2 = cv2.bitwise_and(img2,img2,mask = ma2)#img2原logo中跟ma2中白色重叠的部分保留 其他地方颜色变黑
  cv2.imshow("fg2",fg2)
  
  roi[:] = cv2.add(fg1, fg2)#合到一起
  
  cv2.imshow('img1',img1)#修改roi的数据相当于修改原图:看下面案例
  
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

  | ![point_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/point_pic.png) | ![logo](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/logo.png) | ![water_logo](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/water_logo.png) |
  | ------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------ |
  | 图1                                                          | 图2                                                 | 水印                                                         |

  
## 图像噪点消除

​		噪声：指图像中的一些干扰因素，通常是由图像采集设备、传输信道等因素造成的，表现为图像中随机的亮度，也可以理解为有那么一些点的像素值与周围的像素值格格不入。

​		常见的噪声类型包括高斯噪声和椒盐噪声。高斯噪声是一种分布符合正态分布的噪声，会使图像变得模糊或有噪点。椒盐噪声则是一些黑白色的像素值分布在原图像中。

```python
#生成椒盐噪点
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noise = np.zeros(image.shape, dtype=np.uint8)
    salt_locations = np.random.rand(*image.shape) < salt_prob
    pepper_locations = np.random.rand(*image.shape) < pepper_prob
    noise[salt_locations] = 255
    noise[pepper_locations] = 0
    noisy_image = cv2.add(image, noise)
    return noisy_image

# 读取图像
image = cv2.imread('pic.jpg')
# 添加椒盐噪声
noisy_image = add_salt_and_pepper_noise(image, 0.05, 0.05)
# 显示原始图像和添加噪声后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.waitKey(0)

```

```python
#生成高斯噪点
pic=cv2.imread('pic.jpg')
noise=np.random.normal(0,25,pic.shape).astype(np.uint8)
gauss_image=cv2.add(pic,noise)

cv2.imshow('show',gauss_image)
cv2.waitKey(0)

```

| ![pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/pic.jpg) | ![gauss_noise](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/gauss_noise.png) | ![salt_noise](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/salt_noise.png) |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 原图                                              | 高斯噪点                                                     | 椒盐噪点                                                     |

​		滤波器：也可以叫做卷积核，与自适应二值化中的核一样，本身是一个小的区域，有着特定的核值，并且工作原理也是在原图上进行滑动并计算中心像素点的像素值。

​		滤波器可分为线性滤波和非线性滤波，线性滤波对邻域中的像素进行线性运算，如在核的范围内进行加权求和，常见的线性滤波器有均值滤波、高斯滤波等。非线性滤波则是利用原始图像与模板之间的一种逻辑关系得到结果，常见的非线性滤波器中有中值滤波器、双边滤波器等。

滤波与模糊联系与区别：

- 它们都属于卷积，不同滤波方法之间只是卷积核不同（对线性滤波而言）
- 低通滤波器是模糊，高通滤波器是锐化
- 低通滤波器就是允许低频信号通过，在图像中边缘和噪点都相当于高频部分，所以低通滤波器用于去除噪点、平滑和模糊图像。高通滤波器则反之，用来增强图像边缘，进行锐化处理。

### 1.均值滤波器

- 定义

  均值滤波是一种最简单的滤波处理，它取的是卷积核区域内元素的均值。

  如3×3的卷积核：


 ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM1.png)

- 图示

  如有一张4\*4的图片，现在使用一个3\*3的卷积核进行均值滤波时，其过程如下所示：

  边界的像素点，则会进行边界填充，以确保卷积核的中心能够对准边界的像素点进行滤波操作。

  在OpenCV中，**默认边缘填充的是使用BORDER_REFLECT_101的方式进行填充**，下面的滤波方法中**除了中值滤波使用的是BORDER_REPLICATE进行填充之外**，其他默认也是使用这个方式进行填充。

  ![average_bo](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/average_bo.png)

  ![average_bo2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/average_bo2.png)

- 实例

  ```python
  gauss_pic=cv2.imread('gauss_noise.png')
  salit_pic=cv2.imread('salt_noise.png')
  
  deal_gauss=cv2.blur(gauss_pic,(3,3))#（3，3）核大小
  deal_salt=cv2.blur(salit_pic,(3,3))
  
  cv2.imshow('deal_gauss',deal_gauss)
  cv2.imshow('deal_salt',deal_salt)
  cv2.waitKey(0)
  ```

  | ![average_deal_gauss](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/average_deal_gauss.png) | ![average_deal_salt](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/average_deal_salt.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 均值处理高斯噪点                                             | 均值处理椒盐噪点                                             |

### 2.方框滤波器

- 定义

  方框滤波跟均值滤波很像，如3×3的滤波核如下：


   ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM2.png)

- 图示

  ![box_bo2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/box_bo2.png)

  ksize：代表卷积核的大小，eg：ksize=3，则代表使用3×3的卷积核。

  ddepth：输出图像的深度，-1代表使用原图像的深度。

  normalize：**当normalize为True的时候，方框滤波就是均值滤波，公式中的a就等于1/9；normalize为False的时候，a=1，相当于求区域内的像素和。**

  其滤波的过程与均值滤波一模一样，都采用卷积核从图像左上角开始，逐个计算对应位置的像素值，并从左至右、从上至下滑动卷积核，直至到达图像右下角。

  | ![box_bo3](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/box_bo3.png) | ![box_bo](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/box_bo.png) |
  | --------------------------------------------------------- | ------------------------------------------------------- |
  | normalize=True                                            | normalize=False                                         |

- 实例

  ```python
  gauss_pic=cv2.imread('gauss_noise.png')
  salit_pic=cv2.imread('salt_noise.png')
  
  deal_gauss=cv2.boxFilter(gauss_pic,-1,(3,3),normalize=True)
  deal_salt=cv2.boxFilter(salit_pic,-1,(3,3),normalize=True)
  """
  ksize：代表卷积核的大小，eg：ksize=3，则代表使用3×3的卷积核。
  
  ddepth：输出图像的深度，-1代表使用原图像的深度。
  
  normalize：**当normalize为True的时候，方框滤波就是均值滤波，公式中的a就等于1/9；normalize为False的时候，a=1，相当于求区域内的像素和。**
  
  """
  cv2.imshow('deal_gauss',deal_gauss)
  cv2.imshow('deal_salt',deal_salt)
  cv2.waitKey(0)
  ```

  | ![average_deal_gauss](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/average_deal_gauss.png) | ![average_deal_salt](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/average_deal_salt.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 方框处理高斯噪点                                             | 方框处理椒盐噪点                                             |

### 3.高斯滤波器

- 定义

  高斯滤波的卷积核权重并不相同：中间像素点权重最高，越远离中心的像素权重越小。这里跟自适应二值化里生成高斯核的步骤是一样的，都是以核的中心位置为坐标原点，然后计算周围点的坐标。

  


   ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM3.png)
  
  其中的值也是与自适应二值化里的一样，当时会取固定的系数，当kernel大于7并且没有设置时，会使用固定的公式进行计算σ的值：


   ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM4.png)


  以3\*3的卷积核为例，其核值如下所示：


 ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM5.png)

- 图示

  ![gauss_bo](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/gauss_bo.png)

  ksize：代表卷积核的大小，eg：ksize=3，则代表使用3×3的卷积核。

  sigmaX：就是高斯函数里的值，σ值越大，模糊效果越明显。高斯滤波相比均值滤波效率要慢，但可以有效消除高斯噪声，能保留更多的图像细节，所以经常被称为最有用的滤波器。

- 实例

  ```python
  gauss_pic=cv2.imread('gauss_noise.png')
  
  
  deal_gauss=cv2.GaussianBlur(gauss_pic,ksize=(3,3),sigmaX=1)
  """
  ksize：代表卷积核的大小，eg：ksize=3，则代表使用3×3的卷积核。
  
  sigmaX：就是高斯函数里的值，σ值越大，模糊效果越明显。
  
  """
  cv2.imshow('deal_gauss',deal_gauss)
  
  cv2.waitKey(0)
  ```

| ![gauss_noise](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/gauss_noise.png) | ![gauss_deal](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/gauss_deal.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 原图                                                         | 高斯滤波                                                     |

### 4.中值滤波器

- 定义

  中值又叫中位数，是所有数排序后取中间的值。中值滤波没有核值，而是在原图中从左上角开始，将卷积核区域内的像素值进行排序，并选取中值作为卷积核的中点的像素值。

- 图示

  ![mid_bo](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/mid_bo.png)

  中值滤波就是用区域内的中值来代替本像素值，所以那种孤立的斑点，如0或255很容易消除掉，**适用于去除椒盐噪声和斑点噪声。**中值是一种非线性操作，效率相比前面几种线性滤波要慢。

- 实例

  ```python
  gauss_pic=cv2.imread('gauss_noise.png')
  salit_pic=cv2.imread('salt_noise.png')
  
  deal_gauss=cv2.medianBlur(gauss_pic,ksize=3)
  deal_salt=cv2.medianBlur(salit_pic,ksize=3)
  
  cv2.imshow('deal_gauss',deal_gauss)
  cv2.imshow('deal_salt',deal_salt)
  cv2.waitKey(0)
  ```

  | ![mid_deal_gauss](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/mid_deal_gauss.png) | ![mid_deal_salt](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/mid_deal_salt.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 中值处理高斯噪点图                                           | 中值处理椒盐噪点图                                           |

### 5.双边滤波器

- 定义

  模糊操作基本都会损失掉图像细节信息，尤其前面介绍的线性滤波器，图像的边缘信息很难保留下来。然而**，边缘（edge）**信息是图像中很重要的一个特征，所以这才有了双边滤波。

  双边滤波的基本思路是同时考虑将要被滤波的像素点的**空域信息（周围像素点的位置的权重）和值域信息（周围像素点的像素值的权重）。**

  双边滤波采用了两个高斯滤波的结合，**一个负责计算空间邻近度的权值（也就是空域信息）**，也就是上面的高斯滤波器，**另一个负责计算像素值相似度的权值（也就是值域信息）**，也是一个高斯滤波器。

- 公式

  


   ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM6.png)

  可以看到，对于ωs来说，这就是普通的高斯滤波函数，其带入的坐标是坐标值，Σs是程序输入值，该函数是在空间临近度上计算的。而ωr是计算像素值相似度，也是高斯函数带入坐标值，然后得到对应点的像素值，进行两个点像素值插值的绝对值的平方。也就是说，双边滤波的核值不再是一个固定值，而是随着滤波的过程在不断发生变化的。

- 参数说明

  - ksize：卷积核的大小
  - d：过滤时周围每个像素领域的直径
  - sigmaColor：在color space中过滤sigma。参数越大，临近像素将会在越远的地方mix。
  - sigmaSpace：在coordinate space中过滤sigma。参数越大，那些颜色足够相近的的颜色的影响越大。

  关于2个sigma参数：

  简单起见，可以令2个sigma的值相等；

  如果他们很小（小于10），那么滤波器几乎没有什么效果；

  如果他们很大（大于150），那么滤波器的效果会很强，使图像显得非常卡通化。

  关于参数d：

  过大的滤波器（d\>5）执行效率低。

  对于实时应用，建议取d=5；

  对于需要过滤严重噪声的离线应用，可取d=9；

- 实例

  ```python
  gauss_pic=cv2.imread('gauss_noise.png')
  salit_pic=cv2.imread('salt_noise.png')
  
  deal_gauss=cv2.bilateralFilter(gauss_pic,d=9,sigmaColor=100,sigmaSpace=100)
  deal_salt=cv2.bilateralFilter(salit_pic,d=9,sigmaColor=100,sigmaSpace=100)
  """
  - ksize：卷积核的大小
  - d：过滤时周围每个像素领域的直径
  - sigmaColor：在color space中过滤sigma。参数越大，临近像素将会在越远的地方mix。
  - sigmaSpace：在coordinate space中过滤sigma。参数越大，那些颜色足够相近的的颜色的影响越大。
  
  """
  
  cv2.imshow('deal_gauss',deal_gauss)
  cv2.imshow('deal_salt',deal_salt)
  cv2.waitKey(0)
  ```

  | ![double_deal_salt](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/double_deal_salt.png) | ![double_deal_gauss](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/double_deal_gauss.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 双边处理椒盐噪点                                             | 双边处理高斯噪点                                             |

### 6.小结

- 在不知道用什么滤波器好的时候，优先高斯滤波，然后均值滤波。

- 斑点和椒盐噪声优先使用中值滤波。

- 要去除噪点的同时尽可能保留更多的边缘信息，使用双边滤波。

- 线性滤波方式：均值滤波、方框滤波、高斯滤波（速度相对快）。

- 非线性滤波方式：中值滤波、双边滤波（速度相对慢）。

## 图像梯度处理

### 1.图像的梯度

​		把图片想象成连续函数，因为边缘部分的像素值是与旁边像素明显有区别的，所以对图片局部求极值，就可以得到整幅图片的边缘信息了。不过图片是二维的离散函数，**导数就变成了差分，这个差分就称为图像的梯度。**

### 2.垂直边缘提取

#### 1.提取垂直边缘（得水平梯度）

- 卷积核

  


   ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM7.png)

  

- 图示

  ![vertical_work](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/vertical_work.png)

  当前列左右两侧的元素进行差分，由于边缘的值明显小于（或大于）周边像素，所以边缘的差分结果会明显不同，这样就提取出了垂直边缘。**(小于0值设为0，大于255设为255)**

  ![vertical_work2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/vertical_work2.png)



#### 2.提取水平边缘（得垂直梯度）

- 卷积核

  把上面那个矩阵转置一下，就是提取水平边缘。


   ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM8.png)

- 图示

  ![vertical_horzion](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/vertical_horzion.png)

#### 3.梯度处理函数

- 语法

  filter2D函数是用于对图像进行二维卷积（滤波）操作。它允许自定义卷积核（kernel）来实现各种图像处理效果，如平滑、锐化、边缘检测等

  ```
  cv2.filter2D(src, ddepth, kernel)
  ```

- 参数

  - `src`: 输入图像，一般为`numpy`数组。
  - `ddepth`: 输出图像的深度，可以是负值（表示与原图相同）、正值或其他特定值（常用-1 表示输出与输入具有相同的深度）。
  - `kernel`: 卷积核，一个二维数组（通常为奇数大小的方形矩阵），用于计算每个像素周围邻域的加权和。

- 实例

  ```python
  num_pic=cv2.imread('num_du.png')
  kernel_vertical=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  kernel_horizen=kernel_vertical.T
  
  num_pic_vertical=cv2.filter2D(num_pic,-1,kernel_vertical)
  num_pic_horizen=cv2.filter2D(num_pic,-1,kernel_horizen)
  """
  - `src`: 输入图像，一般为`numpy`数组。
  - `ddepth`: 输出图像的深度，可以是负值（表示与原图相同）、正值或其他特定值（常用-1 表示输出与输入具有相同的深度）。
  - `kernel`: 卷积核，一个二维数组（通常为奇数大小的方形矩阵），用于计算每个像素周围邻域的加权和。
  """
  cv2.imshow('vertical',num_pic_vertical)
  cv2.imshow('horizen',num_pic_horizen)
  cv2.waitKey(0)
  ```

  | ![num_du](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/num_du.png) | ![vertical](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/vertical.png) | ![horizen](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/horizen.png) |
  | ------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------- |
  | 原图                                                    | 垂直边缘                                                    | 水平边缘                                                  |

### 3.sobel算子

以上两个卷积核都叫做Sobel算子，只是方向不同，它先在垂直方向计算梯度:


$$
G_{x}=k_{1}\times s r c
$$


再在水平方向计算梯度:


$$
G_{y}=k_{2}\times s r c
$$


最后求出总梯度：


$$
G={\sqrt{G x^{2}+G y^{2}}}
$$

- 语法

  ```
  **sobel_image = cv2.Sobel(src, ddepth, dx, dy, ksize)**
  ```

- 图示

  ![sobel](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/sobel11.png)

- 参数说明

  **src**：这是输入图像，通常应该是一个灰度图像（单通道图像），因为 Sobel 算子是基于像素亮度梯度计算的。在彩色图像的情况下，通常需要先将其转换为灰度图像。

  **ddepth**：这个参数代表输出图像的深度，即输出图像的数据类型。在 OpenCV 中，-1 表示输出图像的深度与输入图像相同。

  **dx,dy**：当组合为dx=1,dy=0时求x方向的一阶导数，在这里，设置为1意味着我们想要计算图像在水平方向（x轴）的梯度。当组合为    dx=0,dy=1时求y方向的一阶导数（如果同时为1，通常得不到想要的结果,想两个方向都处理的比较好 学习使用后面的算子）

  **ksize**：Sobel算子的大小，可选择3、5、7，默认为3。

- 实例

  ```python
  num_pic=cv2.imread('num_du.png')
  kernel_vertical=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  kernel_horizen=kernel_vertical.T
  
  num_pic_vertical=cv2.Sobel(num_pic,-1,dx=1,dy=0,ksize=3)
  num_pic_horizen=cv2.Sobel(num_pic,-1,dx=0,dy=1,ksize=3)
  
  cv2.imshow('vertical',num_pic_vertical)
  cv2.imshow('horizen',num_pic_horizen)
  cv2.waitKey(0)
  ```

  | ![num_du](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/num_du.png) | ![vertical](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/vertical.png) | ![horizen](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/horizen.png) |
  | ------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------- |
  | 原图                                                    | 垂直边缘                                                    | 水平边缘                                                  |

### 4.拉普拉斯算子（Laplacian）

- 卷积核公式推导

  二阶导计算梯度：


   ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM9.png)
  
  同样提取前面的系数，那么二维的Laplacian滤波核就是：


   ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM10.png)
  
  这就是Laplacian算子的图像卷积模板，有些资料中在此基础上考虑斜对角情况，将卷积核拓展为：


  ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM11.png)

- 语法

  ```
  cv2.Laplacian(src, ddepth)
  ```

- 图示

  ![la](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/la.png)

- 参数说明

  **src**：这是输入图像

  **ddepth**：这个参数代表输出图像的深度，即输出图像的数据类型。在 OpenCV 中，-1 表示输出图像的深度与输入图像相同。

- 实例

  ```python
  num_pic=cv2.imread('num_du.png')
  kernel_vertical=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
  kernel_horizen=kernel_vertical.T
  
  
  num_pic_vertical=cv2.Laplacian(num_pic,-1)
  num_pic_horizen=cv2.Laplacian(num_pic,-1)
  
  cv2.imshow('vertical',num_pic_vertical)
  cv2.imshow('horizen',num_pic_horizen)
  cv2.waitKey(0)
  ```

  | ![num_du](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/num_du.png) | ![lapulasi_deal](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/lapulasi_deal.png) |
  | ------------------------------------------------------- | ------------------------------------------------------------ |
  | 原图                                                    | 拉普拉斯算子处理                                             |

## 图像边缘检测

检测步骤：

1. 读取图像
2. 二值化图像。
3. 高斯滤波。
4. 计算图像的梯度和方向。
5. 非极大值抑制。
6. 双阈值筛选。

### 1.高斯滤波

边缘检测对噪点比较敏感，为了是边缘检测更加准确，需要对图像进行降噪处理，这里采用高斯降噪处理，卷积核为5\*5的高斯核：




 ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM12.png)


### 2.计算图像的梯度和方向

使用sobel算子卷积核来计算图像的梯度值：


 ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM13.png)
 

获得两个方向的梯度值后，这不是图像真正的梯度值，可以使用勾股定理处理两个方向梯度值，在opencv中常用G=|Gx+Gy|来计算实际梯度值

在等到实际梯度值后，要需要确定梯度方向，通过如下公式获得梯度方向：


 ![](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/NUM14.png)
 
**角度值其实是当前边缘的梯度的方向**，然后根据梯度方向获取边缘的方向。

获取边缘的两种选择方式：

1.如果梯度方向不是0°、45°、90°、135°这种特定角度，那么就要用到插值算法来计算当前像素点在其方向上进行插值的结果了，然后进行比较并判断是否保留该像素点。这里使用的是**单线性插值**，通过A1和A2两个像素点获得dTmp1与dTmp2处的插值，然后与**中心点C进行比较(非极大值抑制)。**

2.得到θ的值之后，就可以对边缘方向进行分类，一般将其归为四个方向：水平方向、垂直方向、45°方向、135°方向。并且：

当θ值为-22.5°\~22.5°，或-157.5°\~157.5°，则认为边缘为水平边缘；

当法线方向为22.5°\~67.5°，或-112.5°\~-157.5°，则认为边缘为45°边缘；

当法线方向为67.5°\~112.5°，或-67.5°\~-112.5°，则认为边缘为垂直边缘；

当法线方向为112.5°\~157.5°，或-22.5°\~-67.5°，则认为边缘为135°边缘；

| ![first](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/first.png) | ![second](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/second.png) |
| ----------------------------------------------------- | ------------------------------------------------------- |
| 第一种选择方式                                        | 第二种选择方式                                          |

### 3.非极大值抑制

​		边缘不经过处理是没办法使用的，因为高斯滤波的原因，边缘会变得模糊，导致经过第二步后得到的边缘像素点非常多，因此我们需要对其进行一些过滤操作，而非极大值抑制就是一个很好的方法，它会对得到的边缘像素进行一个排除，使边缘尽可能细一点。

​		检查每个像素点的梯度方向上的相邻像素，并保留梯度值最大的像素，将其他像素抑制为零。假设当前像素点为（x，y），其梯度方向是0°（水平方向梯度，x轴），梯度值为G（x，y），那么我们就需要比较G（x，y）与两个相邻像素的梯度值：G（x-1，y）和G（x+1，y）。如果G（x，y）是三个值里面最大的，就保留该像素值，否则将其抑制为零。

​		**非极大值抑制的目的**是在已经计算出图像梯度幅度图像的基础上，进一步细化边缘位置，减少假响应并确保边缘轮廓的一致性和单像素宽度。

- 原理
  - 在Canny算法中，首先通过高斯滤波和平滑图像，然后计算每个像素点的梯度幅值和方向。
  - 对于每一个像素点，假设已知其梯度幅值（通常通过Sobel或其他导数算子计算得到）以及梯度的方向（通常是精确到某个离散的角度集合）。
  - 非极大值抑制会沿着梯度方向检查像素点的梯度幅值是否是其邻域内（包括梯度方向指向的临近像素点）的最大值。
  - 如果像素点的梯度幅值不是其梯度方向上局部极大值，则认为这个点不是边缘点，并将其梯度幅值置零或者忽略它。
  - 这样做可以去除那些位于边缘两侧但由于噪声或者其他原因导致的不准确的梯度响应，从而保证最终得到的边缘只出现在梯度方向的极大值处，形成连续的、单像素宽的边缘。

### 4.双阈值筛选

- 定义

  经过非极大值抑制之后，我们还需要设置阈值来进行筛选，当阈值设的太低，就会出现假边缘，而阈值设的太高，一些较弱的边缘就会被丢掉，因此使用了双阈值来进行筛选。

- 图示

  ![threshold](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/threshold.png)

  当某一像素位置的幅值**超过最高阈值时，该像素必是边缘像素**；当幅值低**于最低像素时，该像素必不是边缘像素**；幅值处于最高像素与最低像素之间时，**如果它能连接到一个高于阈值的边缘时，则被认为是边缘像素，否则就不会被认为是边缘。**

  上图中的A和C是边缘，B不是边缘。因为C虽然不超过最高阈值，但其与A相连，所以C就是边缘。

- 原理

  - 设定两个阈值，一般称为高阈值（`highThreshold`）和低阈值（`lowThreshold`）。
  - 如果一个像素点的梯度幅值大于等于高阈值，则标记为强边缘像素；
  - 若梯度幅值介于高阈值和低阈值之间，则标记为潜在边缘像素；
  - 若梯度幅值低于低阈值，则认为是非边缘像素，不予考虑。
  - 接下来，采用某种形式的连通性分析，例如Hysteresis（滞后效应），只有当一个弱边缘像素与一个强边缘像素相邻时，才保留这个弱边缘像素作为最终的边缘点。否则，弱边缘像素会被丢弃。

### 5.API

- 语法

  ```
  edges = cv2.Canny(image, threshold1, threshold2)
  ```

- 参数

  - `image`：输入的灰度/二值化图像数据。（彩色图也可以）
  - `threshold1`：低阈值，用于决定可能的边缘点。
  - `threshold2`：高阈值，用于决定强边缘点。

- 实例

  ```python
  #彩图采集边缘
  pic=cv2.imread('pic.jpg')
  edge=cv2.Canny(pic,60,200)
  cv2.imshow('candy',edge)
  cv2.waitKey(0)
  
  #标准化步骤
  pic=cv2.imread('pic.jpg')
  
  pic_gray=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)#灰度化
  
  ret,pic_binary=cv2.threshold(pic_gray,127,255,cv2.THRESH_BINARY)
  #二值化
  
  edge=cv2.Canny(pic_binary,60,200)
  #采集边缘
  
  cv2.imshow('candy',edge)
  cv2.waitKey(0)
  ```

  | ![pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/pic.jpg) | ![edge_ori](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/edge_ori.png) | ![edg_bina](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/edg_bina.png) |
  | ------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
  | 原图                                              | 原图采集边缘                                                | 灰度二值化采集边缘                                          |

## 绘制图像轮廓

轮廓是一系列相连的点组成的曲线，代表了物体的基本外形。相对于边缘，轮廓是连续的，边缘不一定连续，如下图所示。其实边缘主要是作为图像的特征使用，比如可以用边缘特征可以区分脸和手，而轮廓主要用来分析物体的形态，比如物体的周长和面积等，可以说边缘包括轮廓。

![countour](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/countour.png)

### 1.寻找轮廓

- 原理

  ​		寻找轮廓需要将图像做一个**二值化处理**，并且根据图像的不同选择不同的二值化方法来将图像中要**绘制轮廓的部分置为白色，其余部分置为黑色。**也就是说，我们需要对原始的图像进行灰度化、二值化的处理，令目标区域显示为白色，其他区域显示为黑色。

  ​		然后对图像中的像素进行**遍历**，当一个**白色像素相邻（上下左右及两条对角线）位置有黑色像素存在或者一个黑色像素相邻（上下左右及两条对角线）位置有白色像素存在时**，那么该像素点就会被认定为边界像素点，轮廓就是有无数个这样的边界点组成的。

- 语法

  ```
  contours,hierarchy = cv2.findContours(image，mode，method)
  ```

- 参数

  - contours：表示获取到的轮廓点的列表。检测到有多少个轮廓，该列表就有多少子列表，每一个子列表都代表了一个轮廓中所有点的坐标。
  - hierarchy：表示轮廓之间的关系。对于第i条轮廓，hierarchy[i][0] , hierarchy[i][1] , hierarchy[i][2] , hierarchy[i][3]**分别表示其后一条轮廓、前一条轮廓、（同层次的第一个）子轮廓、父轮廓的索引（如果没有对应的索引，则为负数）**。该参数的使用情况会比较少。
  - image：表示输入的二值化图像。
  - mode：表示轮廓的检索模式。
  - method：轮廓的表示方法。

#### 1.1mode参数

mode参数共有四个选项分别为：RETR_LIST，RETR_EXTERNAL，RETR_CCOMP，RETR_TREE。

**RETR_LIST**

表示列出所有的轮廓。并且在hierarchy里的轮廓关系中，每一个轮廓只有前一条轮廓与后一条轮廓的索引，而没有父轮廓与子轮廓的索引。

**RETR_EXTERNAL**

表示只列出最外层的轮廓。并且在hierarchy里的轮廓关系中，每一个轮廓只有前一条轮廓与后一条轮廓的索引，而没有父轮廓与子轮廓的索引。

**RETR_CCOMP**

表示列出所有的轮廓。并且在hierarchy里的轮廓关系中，轮廓会按照成对的方式显示。

**RETR_TREE**

表示列出所有的轮廓。并且在hierarchy里的轮廓关系中，轮廓会按照树的方式显示，其中最外层的轮廓作为树根，其子轮廓是一个个的树枝。

#### 1.2method参数

method参数有三个选项：CHAIN_APPROX_NONE、CHAIN_APPROX_SIMPLE、CHAIN_APPROX_TC89_L1。

**CHAIN_APPROX_NONE**：表示将所有的轮廓点都进行存储；

**CHAIN_APPROX_SIMPLE**：表示只存储有用的点，比如直线只存储起点和终点，四边形只存储四个顶点，默认使用这个方法；CHAIN_APPROX_TC89_L1表示使用Teh-Chin链逼近算法进行轮廓逼近。这种方法使用的是Teh-Chin链码，它是一种边缘检测算法，可以对轮廓进行逼近，减少轮廓中的冗余点，从而更加准确地表示轮廓的形状。

**CHAIN_APPROX_TC89_L1**：是一种较为精确的轮廓逼近方法，适用于需要较高精度的轮廓表示的情况。

#### 1.3参数总结

**对于mode和method这两个参数来说，一般使用RETR_EXTERNAL和CHAIN_APPROX_SIMPLE这两个选项。**

### 2.绘制轮廓

- 语法

  ```python
  cv2.drawContours(image, contours, contourIdx, color, thickness)
  ```

- 参数

  - **image**：原始图像，一般为单通道或三通道的 numpy 数组。
  - **contours**：包含多个轮廓的列表，每个轮廓本身也是一个由点坐标构成的二维数组（numpy数组）。
  - **contourIdx**：要绘制的轮廓索引。如果设为 `-1`，则会绘制所有轮廓。
  - **color**：绘制轮廓的颜色，可以是 BGR 值或者是灰度值（对于灰度图像）。
  - **thickness**：轮廓线的宽度，如果是正数，则画实线；如果是负数，则填充轮廓内的区域。

- 实例

  ```python
  pic=cv2.imread('pic.jpg')
  
  pic_gray=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
  
  ret,pic_binary=cv2.threshold(pic_gray,127,255,cv2.THRESH_BINARY)
  
  contours,hierarchy = cv2.findContours(pic_binary,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
  
  countour_pic=cv2.drawContours(pic,contours,-1,color=(0,0,255),thickness=1)
  
  cv2.imshow('contour',pic)
  
  cv2.waitKey(0)
  ```

  | ![pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/pic.jpg) | ![counter](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/counter.png) |
  | ------------------------------------------------- | --------------------------------------------------------- |
  | 原图                                              | 轮廓图                                                    |
## 凸包特征检测

### 1.获取凸包点

- 定义

  ​		凸包其实就是将一张图片中物体的最外层的点连接起来构成的凸多边形，它能包含物体中所有的内容。

- 检测流程

  假设有如下点图：

  ![tu_1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/tu_1.png)

  经过凸包检测并绘制之后，其结果应该如下图所示：

  ![tu_5](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/tu_5.png)

  第一步，找出处于最左边和最右边的点：

  ![tu_2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/tu_2.png)

  第二步，将这两个点连接，并将点集分为上半区和下半区，我们以上半区为例：

  ![tu_3](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/tu_3.png)

  第三步，找到离直线最远的点，由于直线有两个点的坐标，直线方程是已知的，处于直线上方的点的坐标也是已知的，根据点到直线距离公式来计算哪个点到直线的距离最远。设直线方程Ax+By+C=0,点(x0,y0),点到直线的距离公式为：

  ![N1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/N1.png)

  得到距离这条线最远的点，将其与左右两点连起来，并分别命名为y1和y2，如下图所示：

  ![tu_4](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/tu_4.png)

  分别根据点的坐标求出y1和y2的直线方程，之后将**上半区的每个点**的坐标带入下面公式中，得到所有点的d值：

  ![N1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/N1.png)

  **当d=0时**，表明该点在直线上；

  **当d\>0时，**表明点在直线的上方，在这里就代表该点在上图所围成的三角形的外面，也就意味着该三角形并没有完全包围住上半区的所有点，需要重新寻找凸包点；

  **当d\<0时**，表明点在直线的下方，在这里就代表该点在上图所围成的三角形的里面，也就代表着这类点就不用管了。

​		当出现**d\>0**时，我们需要将出现这种结果的两个计算对象：**某点和y1或y2这条线标记**，重新计算出现这种现象的点集到y1或y2的距离来获取新的凸包点的坐标，在这个例子中也就是如下图所示的点和y2这条直线：

![tu_4](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/tu_4.png)

​		本例子中**只有这一个点**在这个三角形之外，所以毫无疑问的它就是一个凸包点，因此直接将它与y2直线的两个端点相连即可。

​		当有很多点在y2直线外时，就需要计算每个点到y2的距离，然后**找到离得最远**的点与y2构建三角形，并重新计算是否还有点在该三角形之外，如果没有，那么这个点就是新的凸包点，如果有，那就需要重复上面的步骤，直到所有的点都能被包围住，那么构建直线的点就是凸包点。

​		下半区寻找凸包点的思路与此一模一样，只不过是需要筛选d\<0（也就是点在直线的下方）的点，并重新构建三角形，寻找新的凸包点。

​		对于彩色图像，我们需要将其转换为二值图像，并使用轮廓检测技术来获取轮廓边界的点的坐标。然后，我们才能进行上述寻找凸包点的过程。因此，在处理图像时，我们需要将彩色图像转换为二值图像，并通过轮廓检测技术来获取轮廓边界的点的坐标，然后才能进行凸包点的寻找过程。

- 语法

  **cv2.convexHull(points,hull=None,clockwise=False,returnPoints=True)**

- 参数

  - `points`：输入参数，图像的轮廓
  - `hull`（可选）：输出参数，用于存储计算得到的凸包顶点序列，如果不指定，则会创建一个新的数组。
  - `clockwise`（可选）：布尔值，如果为True，则计算顺时针方向的凸包，否则默认计算逆时针方向的凸包。
  - `returnPoints`（可选）：布尔值，如果为True（默认），则函数返回的是原始点集中的点构成的凸包顶点序列；如果为False，则返回的是凸包顶点对应的边界框内的索引。

### 2.绘制凸包

- 语法

  **cv2.polylines(image, pts, isClosed, color, thickness=1)**

- 参数

  - `image`：要绘制线条的目标图像，它应该是一个OpenCV格式的二维图像数组（如numpy数组）。
  - `pts`：一个二维 numpy 数组，每个元素是一维数组，代表一个多边形的一系列顶点坐标。
  - `isClosed`：布尔值,表示是否闭合多边形,如果为 True，会在最后一个顶点和第一个顶点间自动添加一条线段，形成封闭的多边形。
  - `color`：线条颜色，可以是一个三元组或四元组，分别对应BGR或BGRA通道的颜色值，或者是灰度图像的一个整数值。
  - `thickness`（可选）：线条宽度，默认值为1。

- 实例

  ```python
  pic=cv2.imread('tu_pic.png')
  convex_pic=pic.copy()#深层拷贝，在这张图绘制凸包点
  pic_gray=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)#转化灰度图
  _,pic_binary=cv2.threshold(pic_gray,127,255,cv2.THRESH_BINARY)
  #转化二值图
  
  countour,hier=cv2.findContours(pic_binary,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
  #找到二值图中轮廓数组
  
  tu=cv2.convexHull(countour[0])
  #在二值图的轮廓数组中第一个数组，找到凸包点
  cv2.polylines(convex_pic,[tu],isClosed=True,color=(0,0,255),thickness=2)#凸包点 需要列表化[tu]
  #绘制凸包点
  cv2.imshow('show',convex_pic)
  cv2.waitKey(0)
  ```

  | ![tu_pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/tu_pic.png) | ![tu_pic2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/tu_pic2.png) |
  | ------------------------------------------------- | --------------------------------------------------- |
  | 原图                                              | 凸包绘制                                            |

## 图像轮廓特征查找

### 1.外接矩形

外接矩形可根据获得到的轮廓坐标中最上、最下、最左、最右的点的坐标来绘制外接矩形，也就是下图中的绿色矩形。

![wai_1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/wai_1.png)

```python
pic=cv2.imread('tu_pic.png')
pic_copy=pic.copy()

pic_copy_gray=cv2.cvtColor(pic_copy,cv2.COLOR_BGR2GRAY)
ret,image=pic_copy_gray_two=cv2.threshold(pic_copy_gray,127,255,cv2.THRESH_BINARY)

countour,heric=cv2.findContours(image,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(pic_copy,countour,-1,(0,255,0),2)#绘制轮廓


circle_pic=pic_copy.copy()#复制图像画外接矩阵
for i in countour:
    x,y,w,h=cv2.boundingRect(i)#索引左上角和右下角坐标
    # print(x,y,w,h)
    cv2.rectangle(circle_pic,(x,y),(x+w,y+h),(0,0,255),2)#绘制矩形

cv2.imshow('con',pic_copy)
cv2.imshow('circle',circle_pic)
cv2.waitKey(0)
```

| ![wai_rect](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/wai_rect.png) | ![wai_rect_2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/wai_rect_2.png) |
| ----------------------------------------------------- | --------------------------------------------------------- |
| 轮廓图                                                | 外接矩阵图                                                |

### 2.最小外接矩阵

- 定义

  上图所示的蓝色矩形，寻找最小外接矩形使用的算法叫做旋转卡壳法。

  旋转卡壳法思路：用到了**凸包概念**，凸包就是一个点集的凸多边形，它是这个点集所有点的凸壳，点集中所有的点都处于凸包里，构成凸包的点我们叫做凸包点。而旋转卡壳法就是基于凸包点进行的。

  旋转卡壳法有一个**很重要的前提条件**：对于多边形P的一个外接矩形存在一条边与原多边形的边共线。

- 原理

  根据前提条件，凸多边形的最小外接矩形与凸多边形的某条边是共线的。因此我们**只需要以其中的一条边为起始边**，然后**按照逆时针方向计算每个凸包点**与起始边的距离，并将距离最大的点记录下来。

  ![min_1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/min_1.png)

  我们首先以a、b两点为起始边，并计算出e点离起始边最远，那么e到起始边的距离就是一个矩形的高度，因此我们只需要再找出矩形的宽度即可。对于矩形的最右边，以向量ab为基准，然后分别计算**凸包点在向量ab上的投影的长度**，**投影最长的凸包点所在的垂直于起始边的直线**就是矩形最右边所在的直线。

  ![min_2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/min_2.png)

  d点就是在向量ab上投影最长的凸包点，那么通过d点垂直于直线ab的直线就是矩形的右边界所在的直线。矩形的左边界的也是这么计算的，不同的是使用的向量不是ab而是ba。 

  ![min_3](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/min_3.png)

  h点垂直于ab的直线就是以ab为起始边所计算出来的矩形所在的左边界所在的直线。其中矩形的高就是e点到直线ab的距离，矩形的宽是h点在向量上ba的投影加上d点在向量ab上的投影减去ab的长度，即：

  ![N2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/N2.png)

  

  ![min_4](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/min_4.png)

  综上，我们就有了以ab为起始边所构成的外接矩形的宽和高，这样就**可以得到该矩形的面积**。然后**再以bc为起始边，并计算其外接矩形的面积。也就是说凸多边形有几个边，就要构建几次外接矩形，**然后**找到其中面积最小**的矩形作为该凸多边形的最小外接矩形。

- 语法

  使用**cv2.minAreaRect()**来获取最小外接矩形，该函数只需要输入一个参数，就是凸包点的坐标，然后会返回最小外接矩形的中心点坐标、宽高以及旋转角度。通过返回的内容信息，即可绘制凸多边形的的最小外接矩形。

  ```python
  rect = cv2.minAreaRect(cnt)
  
  传入的cnt参数为contours中的轮廓,可以遍历contours中的所有轮廓,然后计算出每个轮廓的小面积外接矩形
  rect 是计算轮廓最小面积外接矩形
  rect 结构通常包含中心点坐标 `(x, y)`、宽度 `width`、高度 `height` 和旋转角度 `angle`
  ```

  ```python
  box = np.int0(cv2.boxPoints(rect))
  
  cv2.boxPoints(rect)返回 是一个形状为 4行2列的数组，每一行代表一个点的坐标（x, y），顺序按照逆时针或顺时针方向排列
  
  将最小外接矩形转换为边界框的四个角点，并转换为整数坐标
  ```

  ```python
  cv2.drawContours(image, contours, contourIdx, color, thickness)
  - **image**：原图像，一般为 numpy 数组，通常为灰度或彩色图像。
  - **contours**：一个包含多个轮廓的列表,可以用上一个api得到的 [box] 
  - **contourIdx**：要绘制的轮廓索引。如果设置为 `-1`，则绘制所有轮廓。
  - **color**：轮廓的颜色，可以是 BGR 颜色格式的三元组，例如 `(0, 0, 255)` 表示红色。
  - **thickness**：轮廓线的粗细，如果是正数，则绘制实线；如果是 0，则绘制轮廓点；如果是负数，则填充轮廓内部区域。
  ```

- 实例

  ```python
  pic=cv2.imread('tu_pic.png')
  pic_copy=pic.copy()#绘制轮廓图层
  pic_min_contor=pic.copy()#绘制最小矩形图层
  
  pic_gray=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)#灰度化
  ret,image=pic_gray_two=cv2.threshold(pic_gray,127,255,cv2.THRESH_BINARY)#二值化
  contour,heric=cv2.findContours(image,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)#找出轮廓点
  
  cv2.drawContours(pic_copy, contour, -1, (255, 0, 0), 2)#绘制轮廓
  
  for cnt in contour:
      rect=cv2.minAreaRect(cnt)
      box=np.int0(cv2.boxPoints(rect))
      cv2.drawContours(pic_min_contor,[box],-1,(0,0,255),2)
  """
  传入的cnt参数为contours中的轮廓,可以遍历contours中的所有轮廓,然后计算出每个轮廓的小面积外接矩形
  rect 结构通常包含中心点坐标 `(x, y)`、宽度 `width`、高度 `height` 和旋转角度 `angle`
  
  
  cv2.boxPoints(rect)返回 是一个形状为 4行2列的数组，每一行代表一个点的坐标（x, y），顺序按照逆时针或顺时针方向排列
  
  将最小外接矩形转换为边界框的四个角点，并转换为整数坐标
  """
  cv2.imshow('ccc',pic_min_contor)
  cv2.waitKey(0)
  ```

### 3.最小外接圆

- 定义

  最小外接圆使用的算法是Welzl算法。Welzl算法基于一个定理：**希尔伯特圆定理**表明，对于平面上的任意三个不在同一直线上的点，存在一个唯一的圆同时通过这三个点，且该圆是最小面积的圆（即包含这三个点的圆中半径最小的圆，也称为最小覆盖圆）。

  进一步推广到任意 n 个不在同一圆上的点，总存在一个唯一的最小覆盖圆包含这 n 个点。

  若已经存在平面上互不共线（或共圆）的 n 个点，并确定了它们的最小覆盖圆，那么添加第 n+1 个点，并且要求这个点不在原来的最小覆盖圆内（即在圆外），为了使新的包含 n+1 个点的最小覆盖圆的半径增大，**新加入的点必须位于由原 n 个点确定的最小覆盖圆的边界上（即圆周上）**。

  有了这个定理，就可以先取3个点建立一个圆（不共线的三个点即可确定一个圆，如果共线就取距离最远的两个点作为直径建立圆），然后遍历剩下的所有点，对于遍历到的点P来说：

  如果该点在圆内，那么最小覆盖圆不变。

  如果该点在圆外，根据上述定理，该点一定在想要求得的最小覆盖圆的圆周上，又因为三个点才能确定一个圆，所以需要枚举P点之前的点来找其余的两个点。当找到与P点组成的圆能够将所有点都包含在圆内或圆上，该圆就是这些点的最小外接圆。

  ![circle_1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/circle_1.png)

- 语法

  使用cv2.minEnclosingCircle()来获取最小外接圆，该函数只需要输入一个参数，就是要绘制最小外接圆的点集的坐标，然后会返回最小外接圆的圆心坐标与半径。通过该函数返回的内容信息即可绘制某点集的最小外接圆。

  ```python
  **cv2.minEnclosingCircle(points) -> (center, radius)**
  ```

  - `points`：输入参数图片轮廓数据

  返回值：

  - `center`：一个包含圆心坐标的二元组 `(x, y)`。
  - `radius`：浮点数类型，表示计算得到的最小覆盖圆的半径。

  ```
  **cv2.circle(img, center, radius, color, thickness)**
  ```

  - `img`：输入图像，通常是一个numpy数组，代表要绘制圆形的图像。
  - `center`：一个二元组 `(x, y)`，表示圆心的坐标位置。
  - `radius`：整型或浮点型数值，表示圆的半径长度。
  - `color`：颜色标识，可以是BGR格式的三元组 `(B, G, R)`，例如 `(255, 0, 0)` 表示红色。
  - `thickness`：整数，表示圆边框的宽度。如果设置为 `-1`，则会填充整个圆。

- 实例

  ```python
  pic=cv2.imread('tu_pic.png')
  pic_copy=pic.copy()#绘制轮廓
  pic_min_circle=pic.copy()#绘制最小外接圆
  
  pic_gray=cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
  ret,image=pic_gray_two=cv2.threshold(pic_gray,127,255,cv2.THRESH_BINARY)
  contour,heric=cv2.findContours(image,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(pic_copy, contour, -1, (255, 0, 0), 2)
  for cnt in contour:
      (x,y),r=cv2.minEnclosingCircle(cnt)
      (x,y,r)=np.int0((x,y,r))#取整
      cv2.circle(pic_min_circle,(x,y),r,(0,255,0),2)
  """
  - `points`：输入参数图片轮廓数据
  
  返回值：
  
  - `center`：一个包含圆心坐标的二元组 `(x, y)`。
  - `radius`：浮点数类型，表示计算得到的最小覆盖圆的半径。
  
  
  - `img`：输入图像，通常是一个numpy数组，代表要绘制圆形的图像。
  - `center`：一个二元组 `(x, y)`，表示圆心的坐标位置。
  - `radius`：整型或浮点型数值，表示圆的半径长度。
  - `color`：颜色标识，可以是BGR格式的三元组 `(B, G, R)`，例如 `(255, 0, 0)` 表示红色。
  - `thickness`：整数，表示圆边框的宽度。如果设置为 `-1`，则会填充整个圆。
  """
  cv2.imshow('ccc',pic_min_circle)
  cv2.waitKey(0)
  ```

  | ![wai_rect](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/wai_rect.png) | ![min_circle](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/min_circle.png) |
  | ----------------------------------------------------- | --------------------------------------------------------- |
  | 轮廓图                                                | 最小外接圆                                                |

## 直方图均衡化

### 1.直方图定义

直方图是对数据进行统计的一种方法，并且将统计值组织到一系列实现定义好的 bin 当中。其中， bin 为直方图中经常用到的一个概念，可以译为 “直条” 或 “组距”，其数值是从数据中计算出的特征统计量，这些数据可以是诸如梯度、方向、色彩或任何其他特征。

![d00925fd2a51c6513df3a9585d7bbcee](E:\AI-Study\python学习\opencv笔记\opencv笔记\20直方图均衡化\media\d00925fd2a51c6513df3a9585d7bbcee.png)

### 2.绘制直方图

- 语法

  以像素值为横坐标，像素值的个数为纵坐标绘制一个统计图。

  ```python
  hist=cv2.calcHist(images, channels, mask, histSize, ranges)
  ```

  - `images`：输入图像列表，可以是一幅或多幅图像（通常是灰度图像或者彩色图像的各个通道）。
  - `channels`：一个包含整数的列表，指示在每个图像上计算直方图的通道编号。如果输入图像是灰度图，它的值就是 [0]；如果是彩色图像的话，传入的参数可以是 [0]，[1]，[2] 它们分别对应着通道 B，G，R。 
  - `mask`（可选）：一个与输入图像尺寸相同的二值掩模图像，其中非零元素标记了参与直方图计算的区域,None为全部计算。
  - `histSize`：一个整数列表，也就是直方图的区间个数(BIN 的数目)。用中括号括起来，例如：[256]。 
  - `ranges`：每维数据的取值范围，它是一个二维列表，每一维对应一个通道的最小值和最大值，例如对灰度图像可能是 `[0, 256]`。

  返回值hist 是一个长度为255的数组，数组中的每个值表示图像中对应灰度等级的像素计数

  

  获取直方图的最小值、最大值及其对应最小值的位置索引、最大值的位置索引

  ```python
   minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
  ```

  

  ```
  cv2.line(img, pt1, pt2, color, thickness)
  ```

  - **img**：原始图像，即要在上面画线的numpy数组（一般为uint8类型）。
  - **pt1** 和 **pt2**：分别为线段的起点和终点坐标，它们都是元组类型，例如 `(x1, y1)` 和 `(x2, y2)` 分别代表线段两端的横纵坐标。
  - **color**：线段的颜色，通常是一个包含三个元素的元组 `(B, G, R)` 表示BGR色彩空间的像素值，也可以是灰度图像的一个整数值。
  - **thickness**：线段的宽度，默认值是1，如果设置为负数，则线宽会被填充。

- 实例

  ```python
  def calcAndDrawHist(image, color):
      hist=cv2.calcHist(image,[1],None,[255],[0,256])
      #统计彩色图像中二号通道所有像素中0-255出现的频率
      minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
      histImg=np.zeros([256,256,3],np.uint8)
      #初始化一个新图像，尺寸为256x256，用于绘制直方图，且每个像素点初始值为黑色（三通道均为0）
      hpt=int(0.9*256)
      for h in range(255):
          intensity=int(hist[h]*hpt/maxVal)
          cv2.line(histImg,(h,256),(h,256-intensity),color)
      return histImg
  
  pic=cv2.imread('pic.jpg')
  pic_jpg=calcAndDrawHist(pic,(255,0,0))
  cv2.imshow('pic_jpg',pic_jpg)
  cv2.waitKey(0)
  ```

  | ![pic](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/pic.jpg) | ![color_pus](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/color_pus.png) |
  | ------------------------------------------- | ------------------------------------------------------- |
  | 原图                                        | 通道2色域值分布                                         |

  

### 3.直方图均衡化

#### 3.1自适应直方图均衡化

- 定义

  自适应直方图均衡化（Adaptive Histogram Equalization, AHE），通过调整图像像素值的分布，使得图像的对比度和亮度得到改善。

- 过程

  ![straingt_2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/straingt_2.png)

  设有一个3\*3的图像，其灰度图的像素值如上图所示，现在我们要对其进行直方图均衡化，首先就是统计其每个像素值的个数、比例以及其累计比例。

  ![straingt_1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/straingt_1.png)

  接下来我们就要进行计算，就是将要缩放的范围（通常是缩放到0-255，所以就是255-0）乘以累计比例，得到新的像素值，并将新的像素值放到对应的位置上，**比如像素值为50的像素点，将其累计比例乘以255，也就是0.33乘以255得到84.15，取整后得到84，并将84放在原图像中像素值为50的地方，像素值为100、210、255的计算过程类似**，最终会得到如下图所示的结果，这样就完成了最基本的直方图均衡化的过程。

  ![straingt_2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/straingt_2.png)

  ![straingt_3](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/straingt_3.png)

- 语法

  ```python
  #imgGray为需要直方图均衡化的灰度图返回值为处理后的图像
  **dst = cv.equalizeHist(imgGray)**
  ```

- 实例

  ```python
  import  cv2
  import numpy as np
  img=cv2.imread("./zhifang.png",cv2.IMREAD_GRAYSCALE)
  cv2.imshow("img",img)
  cv2.imshow("img2",cv2.equalizeHist(img))
  cv2.waitKey(0)
  ```

  该方法适用于图像的灰度分布不均匀，且灰度分布集中在更窄的范围，图像的细节不够清晰且对比度较低的情况，然而，传统的直方图均衡化方法会引入噪声，并导致图像中出现过度增强的区域。这是因为直方图均衡化方法没有考虑到图像的局部特征和全局对比度的差异。

  ![straight_5](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/straight_5.png)

#### 3.对比度受限的自适应直方图均衡图

- 作用

  很明显，因为全局调整亮度和对比度的原因，脸部太亮，大部分细节都丢失了。自适应均衡化就是用来解决这一问题的：它在每一个小区域内（默认8×8）进行直方图均衡化。当然，如果有噪点的话，噪点会被放大，需要对小区域内的对比度进行了限制，所以这个算法全称叫：**对比度受限的自适应直方图均衡化**

- 步骤

  1. **图像分块（Tiling）**：
     - 图像首先被划分为多个不重叠的小块（tiles）。这样做的目的是因为在全局直方图均衡化中，单一的直方图无法反映图像各个局部区域的差异性。通过局部处理，AHE能够更好地适应图像内部的不同光照和对比度特性。（tiles 的 大小默认是 8x8）
  2. **计算子区域直方图**：
     - 对于每个小块，独立计算其内部像素的灰度直方图。直方图反映了该区域内像素值的分布情况。
  3. **子区域直方图均衡化**：
     - 对每个小块的直方图执行直方图均衡化操作。这涉及重新分配像素值，以便在整个小块内更均匀地分布。均衡化过程会增加低频像素的数量，减少高频像素的数量，从而提高整个小块的对比度。
  4. **对比度限制（Contrast Limiting）**：
     - 如果有噪声的话，噪声会被放大。为了防止过大的对比度增强导致噪声放大，出现了限制对比度自适应直方图均衡化（CLAHE）。CLAHE会在直方图均衡化过程中引入一个对比度限制参数。当某一小块的直方图在均衡化后出现极端值时，会对直方图进行平滑处理（使用线性或非线性的钳制函数），确保对比度增强在一个合理的范围内。
  5. **重采样和邻域像素融合**：
     - 由于小块之间是不重叠的，直接拼接经过均衡化处理的小块会产生明显的边界效应。因此，在CLAHE中通常采用重采样技术来消除这种效应，比如通过双线性插值将相邻小块的均衡化结果进行平滑过渡，使最终图像看起来更为自然和平滑。
  6. **合成输出图像**：
     - 将所有小块均衡化后的结果整合在一起，得到最终的自适应直方图均衡化后的图像。

- 语法

  ```python
  clahe =cv2.createCLAHE(clipLimit=None,tileGridSize=None)
  """
  - clipLimit（可选）：对比度限制参数，用于控制直方图均衡化过程中对比度增强的程度。如果设置一个大于1的值（如2.0或4.0），CLAHE会限制对比度增强的最大程度，避免过度放大噪声。如果不设置，OpenCV会使用一个默认值。
  - tileGridSize（可选）：图像分块的大小，通常是一个包含两个整数的元组，如`(8, 8)`，表示将图像划分成8x8的小块进行独立的直方图均衡化处理。分块大小的选择会影响到CLAHE的效果以及处理速度。
  
  """
  
  #创建CLAHE对象后，可以使用 `.apply()` 方法对图像进行CLAHE处理：
  
  img=clahe.apply(image)
  """
  - image:要均衡化的图像。
  - img均衡后的图像
  """
  ```

- 实例

  ```python
  import cv2
  import numpy as np
  def calcAndDrawHist(image, color):
      hist = cv2.calcHist([image], [0], None, [256], [0, 256])
      minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
      histImg = np.zeros([256, 256, 3], np.uint8)
      hpt = int(0.9 * 256)
      for h in range(256):
          intensity = int(hist[h] * hpt / maxVal)
          cv2.line(histImg, (h, 256), (h, 256 - intensity), color)
      return histImg
  if __name__ == "__main__":
      path = "./zhifang.png"
      image_np = cv2.imread(path)
      color = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0)}
      # 灰度
      image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
      # 绘制
      hist_image = calcAndDrawHist(image_np_gray, color["blue"])
      # 普通的直方图均衡化
      equ_hist_image_np = cv2.equalizeHist(image_np_gray)
      equ_hist_image = calcAndDrawHist(equ_hist_image_np, color["blue"])
      #自适应直方图均衡化
      clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
      equ_hist_image_np = clahe.apply(image_np_gray)
      # 返回处理正确后的内容
      cv2.imshow("image_np_gray", image_np_gray)
      cv2.imshow("hist_image", hist_image)
      cv2.imshow("equ_hist_image_np", equ_hist_image_np)
      cv2.imshow("equ_hist_image", equ_hist_image)
      cv2.waitKey(0)
  ```

  ![straight_4](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/straight_4.png)

## 模板匹配

### 1.模板匹配

模板匹配就是用模板图（通常是一个小图）在目标图像（通常是一个比模板图大的图片）中不断的滑动比较，通过某种比较方法来判断是否匹配成功。

![model](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/model.png)

## 2.匹配方法

- 语法

  ```python
  res=cv2.matchTemplate(image, templ, method)
  ```

- 参数

  - image：原图像，这是一个灰度图像或彩色图像（在这种情况下，匹配将在每个通道上独立进行）。

  - templ：模板图像，也是灰度图像或与原图像相同通道数的彩色图像。

  - method：匹配方法，可以是以下之一：

    - cv2.TM_CCOEFF
    - cv2.TM_CCOEFF_NORMED
    - cv2.TM_CCORR
    - cv2.TM_CCORR_NORMED
    - cv2.TM_SQDIFF
    - cv2.TM_SQDIFF_NORMED
    - 这些方法决定了如何度量模板图像与原图像子窗口之间的相似度。

  - 返回值res

    函数在完成图像模板匹配后返回一个结果矩阵，这个矩阵的大小与原图像相同。矩阵的每个元素表示原图像中相应位置与模板图像匹配的相似度。

    匹配方法不同，返回矩阵的值的含义也会有所区别。以下是几种常用的匹配方法及其返回值含义：

    1. `cv2.TM_SQDIFF` 或 `cv2.TM_SQDIFF_NORMED`：

       返回值越接近0，表示匹配程度越好。最小值对应的最佳匹配位置。

    2. `cv2.TM_CCORR` 或 `cv2.TM_CCORR_NORMED`：

       返回值越大，表示匹配程度越好。最大值对应的最佳匹配位置。

    3. `cv2.TM_CCOEFF` 或 `cv2.TM_CCOEFF_NORMED`：

       返回值越大，表示匹配程度越好。最大值对应的最佳匹配位置。

#### 2.1平方差匹配

```
cv2.TM_SQDIFF
```

以模板图与目标图所对应的像素值使用平方差公式来计算，其结果越小，代表匹配程度越高，计算过程举例如下。

注意：模板匹配过程皆不需要边缘填,直接从目标图像的左上角开始计算。

![ping_match](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/ping_match.png)

#### 2.2归一化平方差匹配

```
cv2.TM_SQDIFF_NORMED
```

与平方差匹配类似，只不过需要将值统一到0到1，计算结果越小，代表匹配程度越高，计算过程举例如下。

![gui_ping_match](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/gui_ping_match.png)

#### 2.3相关匹配

```
cv2.TM_CCORR
```

使用对应像素的乘积进行匹配，乘积的结果越大其匹配程度越高，计算过程举例如下。

![relative](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/relative.png)

#### 2.4归一化相关匹配

```
cv2.TM_CCORR_NORMED
```

与相关匹配类似，只不过是将其值统一到0到1之间，值越大，代表匹配程度越高，计算过程举例如下。

![gui_relative](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/gui_relative.png)

#### 2.5相关系数匹配

```
cv2.TM_CCOEFF
```

需要先计算模板与目标图像的均值，然后通过每个像素与均值之间的差的乘积再求和来表示其匹配程度，1表示完美的匹配，-1表示最差的匹配，计算过程举例如下。

![relative_num](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/relative_num.png)

#### 2.6归一化相关系数匹配

```
cv2.TM_CCOEFF_NORMED
```

也是将相关系数匹配的结果统一到0到1之间，值越接近1代表匹配程度越高，计算过程举例如下。

![gui_relative_num](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/gui_relative_num.png)

### 3.绘制轮廓

找的目标图像中匹配程度最高的点，我们可以设定一个匹配阈值来筛选出多个匹配程度高的区域。

```python
**loc=np.where(array > 0.8)** #loc包含array中所有大于0.8的**元素索引**的数组

**zip(*loc)**

x=list([[1,2,3,4,3],[23,4,2,4,2]])
print(list(zip(*x)))#[(1, 23), (2, 4), (3, 2), (4, 4), (3, 2)]
```

```python
import cv2
import numpy as np

if __name__ == "__main__":
    path_search = "./game.png"
    image_np = cv2.imread(path_search)
    image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)  # 转为灰度图(原图)
    path_target = "./temp.png"
    template = cv2.imread(path_target)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # 转为灰度图(匹配模板)
    h, w = template_gray.shape[:2]
    res = cv2.matchTemplate(image_np_gray, template_gray, cv2.TM_CCOEFF_NORMED)  # 模板匹配
    threshold = 0.8
    loc = np.where(res >= threshold)  # 匹配程度大于threshold的坐标y,x: 两个列表第一个是所有y,第二个是所有x
    # print(res,res[loc[0][1],loc[1][1]],res[loc[0][20],loc[1][20]])
    for pt in zip(*loc):
        right_bottom = (pt[1] + w, pt[0] + h)
        cv2.rectangle(image_np, pt[::-1], right_bottom, (0, 0, 255), 2)
    # 返回处理正确后的内容
    cv2.imshow("image_np", image_np)
    cv2.waitKey(0)
```

## 霍夫变换

### 1.霍夫变换图示

![huofu1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/huofu1.png)

### 2.霍夫直线变换

对于一条直线（不垂直于x轴的直线），都可以用$y=k x+b$来表示，此时，x和y是横纵坐标，k和b是一个固定参数。当我们换一种方式来看待这个式子，我们就可以得到：
$$
b=-kx+y
$$
此时，以k和b	为横纵坐标，x和y为固定参数，变换如下图所示：

![bianhuan](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/bianhuan.png)

从上图可以看出，在直角坐标系下的一个直线，在变换后的空间中仅仅表示为一点，对于变换后的空间，我们称其为霍夫空间。也就是说，直角坐标系下的一条直线对应了霍夫空间中的一个点。类似的，霍夫空间中的一条直线也对应了直角坐标系中的一个点，如下图所示：

![bianhuan_1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/bianhuan_1.png)

那么对于一个二值化后的图形来说，其中的每一个目标像素点（这里假设目标像素点为白色像素点）都对应了霍夫空间的一条直线，当霍夫空间中有两条直线相交时，就代表了直角坐标系中某两个点所构成的直线。而当霍夫空间中有很多条线相交于一点时，说明直角坐标系中有很多点能构成一条直线，也就意味着这些点共线，因此我们就可以通过检测霍夫空间中有最多直线相交的交点来找到直角坐标系中的直线。

然而对于x=1这种直线来说，y已经不存在了，那么就没办法使用上面的方法进行检测了，为了解决这个问题，我们就将直角坐标系转化为极坐标系，然后通过极坐标系与霍夫空间进行相互转化。

![huofu2](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/huofu2.png)

在极坐标系下是一样的，极坐标中的点对于霍夫空间中的线，霍夫空间中的点对应极坐标中的直线。并且此时的霍夫空间不再是以k为横轴、b为纵轴，而是以为θ横轴、ρ(上图中的r)为纵轴。上面的公式中，x、y是直线上某点的横纵坐标（直角坐标系下的横纵坐标），和是极坐标下的坐标，因此我们只要知道某点的x和y的坐标，就可以得到一个关于θ-ρ的表达式，如下图所示：

![hhhh](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/hhhh.png)

根据上图，霍夫空间在极坐标系下，一点可以产生一条三角函数曲线，而多条这样的曲线可能会相交于同一点。因此，我们可以通过设定一个阈值，来检测霍夫空间中的三角函数曲线相交的次数。如果一个交点的三角函数曲线相交次数超过阈值，那么这个交点所代表的直线就可能是我们寻找的目标直线。

- 语法

  ```
  **lines=cv2.HoughLines(image, rho, theta, threshold)**
  ```

- 参数

  - `image`：输入图像，通常为二值图像，其中白点表示边缘点，黑点为背景。
  - `rho`：r的精度，以像素为单位，表示霍夫空间中每一步的距离增量,  值越大，考虑越多的线。
  - `theta`：角度θ的精度，通常以弧度为单位，表示霍夫空间中每一步的角度增量。值越小，考虑越多的线。
  - `threshold`：累加数阈值，只有累积投票数超过这个阈值的候选直线才会被返回。

  返回值：`cv2.HoughLines` 函数返回一个二维数组，每一行代表一条直线在霍夫空间中的参数 `(rho, theta)`。

- 代码

  ```python
  import cv2
  import numpy as np
  
  if __name__ == "__main__":
      path = "./huofu.png"
      image_np = cv2.imread(path)
      image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)  # 转为灰度图
      image_edges = cv2.Canny(image_np_gray, 30, 70)  # 进行canny边缘检测
      # 使用霍夫变换检测直线
      lines = cv2.HoughLines(image_edges, 0.8, 0.01745, 90)
      # 遍历并绘制检测到的直线
      for line in lines:
          rho, theta = line[0]
          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a * rho
          y0 = b * rho
          x1 = int(x0 + 1000 * (-b))#1000是让绘制的直线变长
          y1 = int(y0 + 1000 * (a))
          x2 = int(x0 - 1000 * (-b))
          y2 = int(y0 - 1000 * (a))
          cv2.line(image_np, (x1, y1), (x2, y2), (0, 0, 255))
      # 返回处理正确后的内容
      cv2.imshow("image_np", image_np)
      cv2.waitKey(0)
  ```

### 3.统计概率霍夫直线变换

- 定义

  前面的方法又称为**标准霍夫变换**，它会计算图像中的每一个点，计算量比较大，另外它得到的是整一条线（r和θ），并不知道原图中直线的端点。所以提出了**统计概率霍夫直线变换**(Probabilistic Hough Transform)，是一种改进的霍夫变换，它在获取到直线之后，会检测原图中在该直线上的点，并获取到两侧的端点坐标，然后通过两个点的坐标来计算该直线的长度，通过直线长度与最短长度阈值的比较来决定该直线要不要被保留。

- 语法

  ```
  **lines=cv2.HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=0, maxLineGap=0)**
  ```

- 参数

  - `image`：输入图像，通常为二值图像，其中白点表示边缘点，黑点为背景。
  - `rho`：极径分辨率，以像素为单位，表示极坐标系中的距离分辨率。
  - `theta`：极角分辨率，以弧度为单位，表示极坐标系中角度的分辨率。
  - `threshold`：阈值，用于过滤掉弱检测结果，只有累计投票数超过这个阈值的直线才会被返回。
  - `lines`（可选）：一个可初始化的输出数组，用于存储检测到的直线参数。
  - `minLineLength`（可选）：最短长度阈值，比这个长度短的线会被排除。
  - `maxLineGap`（可选）：同一直线两点之间的最大距离。当霍夫变换检测到一系列接近直角的线段时，这些线段可能是同一直线的不同部分。`maxLineGap`参数指定了在考虑这些线段属于同一直线时，它们之间最大可接受的像素间隔。
    - 如果`maxLineGap`设置得较小，那么只有相邻且间距很小的线段才会被连接起来，这可能导致检测到的直线数量较多，但更准确地反映了图像中的局部直线结构。
    - 如果`maxLineGap`设置得较大，则线段间的间距可以更大，这样可能会合并更多的局部线段成为更长的直线，但有可能会将原本不属于同一直线的线段误连接起来。

  返回值lines：`cv2.HoughLinesP` 函数返回一个二维数组，每个元素是一个包含4个元素的数组，分别表示每条直线的起始点和结束点在图像中的坐标（x1, y1, x2, y2）。

- 代码

  ```python
  import cv2
  import numpy as np
  
  if __name__ == "__main__":
      path = "./huofu.png"
      image_np = cv2.imread(path)
      image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)  # 转为灰度图
      image_edges = cv2.Canny(image_np_gray, 30, 70)  # 进行canny边缘检测
      # 统计概率霍夫直线变换
      lines = cv2.HoughLinesP(image_edges, 0.8, 0.01745, 90, minLineLength=50, maxLineGap=10)
      # 遍历并绘制检测到的直线
      for line in lines:
          x1, y1, x2, y2 = line[0]
          cv2.line(image_np, (x1, y1), (x2, y2), (0, 0, 255), 1, lineType=cv2.LINE_AA)
      # 返回处理正确后的内容
      cv2.imshow("image_np", image_np)
      cv2.waitKey(0)
  ```

### 4.霍夫圆变换

- 定义

  霍夫圆变换跟直线变换类似，它可以从图像中找出潜在的圆形结构，并返回它们的中心坐标和半径。只不过线是用(r,θ)表示，圆是用(x_center,y_center,r)来表示，从二维变成了三维，数据量变大了很多；所以一般使用霍夫梯度法减少计算量。

- 语法

  ```
  circles=cv2.HoughCircles(image, method, dp, minDist, param1, param2)
  ```

- 参数

  - `image`：输入图像，通常是灰度图像。

  - `method`：使用的霍夫变换方法:霍夫梯度法，可以是 `cv2.HOUGH_GRADIENT`，这是唯一在OpenCV中用于圆检测的方法。

  - `dp`：累加器分辨率与输入图像分辨率之间的降采样比率，用于加速运算但不影响准确性。设置为1表示霍夫梯度法中累加器图像的分辨率与原图一致

  - `minDist`：检测到的圆心之间的最小允许距离，以像素为单位。在霍夫变换检测圆的过程中，可能会检测到许多潜在的圆心。`minDist` 参数就是为了过滤掉过于接近的圆检测结果，避免检测结果过于密集。当你设置一个较小的 `minDist` 值时，算法会尝试找出尽可能多的圆，即使是彼此靠得很近的圆也可能都被检测出来。相反，当你设置一个较大的 `minDist` 值时，算法会倾向于只检测那些彼此间存在一定距离的独立的圆。

    例如，如果你设置 `minDist` 很小，可能在真实图像中存在的一个大圆旁边的一些噪声点会被误判为多个小圆；而如果设置 `minDist` 较大，则可以排除这种情况，只保留明显分离的圆的检测结果。

    

  - `param1` 和 `param2`：这两个参数是在使用 `cv2.HOUGH_GRADIENT` 方法时的特定参数，分别为：

    - `param1`(可选)：阈值1，决定边缘强度的阈值。

    - `param2`：阈值2，控制圆心识别的精确度。较大的该值会使得检测更严格的圆。`param2` 通常被称为圆心累积概率的阈值。在使用霍夫梯度方法时，`param2` 设置的是累加器阈值，它决定了哪些候选圆点集合被认为是有效的圆。较高的 `param2` 值意味着对圆的检测更严格，只有在累加器中积累了足够高的响应值才认为是真实的圆；较低的 `param2` 值则会降低检测的门槛，可能会检测到更多潜在的圆，但也可能包含更多的误检结果。

      举个例子，如果你将 `param2` 设置得较高，那么算法只会返回那些边缘强烈符合圆形特征且周围有足够的支持像素的圆；而如果设置得较低，即使边缘特征不是很强烈，只要有一些证据支持就可能将其视为一个圆。

  返回值：`cv2.HoughCircles` 返回一个二维numpy数组，包含了所有满足条件的圆的参数。

- 代码

  ```python
  import cv2
  import numpy as np
  
  if __name__ == "__main__":
      path = "./huofu.png"
      image_np = cv2.imread(path)
      image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)  # 转为灰度图
      image_edges = cv2.Canny(image_np_gray, 30, 70)  # 进行canny边缘检测
      # 霍夫圆变换
      circles = cv2.HoughCircles(image_edges, cv2.HOUGH_GRADIENT, 1, 20, param2=30)
      circles = np.int_(np.around(circles))
      # 将检测的圆画出来
      for i in circles[0, :]:
          cv2.circle(image_np, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 画出外圆
          cv2.circle(image_np, (i[0], i[1]), 2, (0, 0, 255), 3)  # 画出圆心
      # 返回处理正确后的内容
      cv2.imshow("image_np", image_np)
      cv2.waitKey(0)
  ```

## 图像亮度变换

### 1.亮度变换

对比度调整：图像暗处像素强度变低，图像亮处像素强度变高，从而拉大中间某个区域范围的显示精度。

亮度调整：图像像素强度整体变高或者变低。

![light_1](https://github.com/ljgit1316/Picture_resource/blob/main/OpenCv_Pic/light_1.png)

上图中，(a)把亮度调高，就是图片中的所有像素值加上了一个固定值；(b)把亮度调低，就是图片中的所有像素值减去了一个固定值；(c)增大像素对比度（白的地方更白，黑的地方更黑）；(d)减小像素对比度（整幅图都趋于一个颜色）；

OpenCV调整图像对比度和亮度时，公式为：$g(i,j)=\alpha f(i,j)+\beta$。但是不能浅显的讲$\alpha$是控制对比度，$\beta$是控制亮度的。

对比度：需要通过$\alpha、\beta$一起控制（仅调整$\alpha$只能控制像素强度0附近的对比度，而这种做法只会导致像素强度大于0的部分更亮而已，根本看不到对比度提高的效果）。

亮度：通过$\beta$控制。

### 2.线性变换

- 定义

  使用 `cv2.addWeighted()` 函数，可以对图像的像素值进行加权平均，进而改变图像的整体亮度。亮度增益可以通过向每个像素值添加一个正值来实现。

- 语法

  ```
  **cv2.addWeighted(src1, alpha, src2, beta, gamma)**
  ```

- 参数

  - `src1`：第一张输入图像，它将被赋予权重 `alpha`。

  - `alpha`：第一个输入图像的权重。

  - `src2`：第二张输入图像，它将被赋予权重 `beta`。

  - `beta`：第二个输入图像的权重。

  - `gamma`：一个标量，将被添加到权重求和的结果上，可用于调整总体亮度。

    计算公式为: dst = src1 * alpha + src2 * beta + gamma

- 代码

  ```python
  import cv2
  import numpy as np
  # 加载图像
  img = cv2.imread('./1.jpg')
  # 设定亮度增益，例如设置为1.5倍亮度
  alpha = 1.5
  # 提升图像亮度
  brightened_img = cv2.addWeighted(img, alpha, np.zeros_like(img), 0, 0)
  # 显示和/或保存处理后的图像
  cv2.imshow('Brightened Image', brightened_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  ```

### 3.直接像素值修改

- 定义

  如果只需要增加或减少固定的亮度值，可以直接遍历图像像素并对每个像素值进行加减操作。

- 语法

  ```
  numpy.clip(a, a_min, a_max)
  ```

   用于对数组中的元素进行限定，将超出指定范围的元素值截断至指定的最小值和最大值之间

- 参数

  - `a`：输入数组。

  - `a_min`：指定的最小值，数组中所有小于 `a_min` 的元素将被替换为 `a_min`。

  - `a_max`：指定的最大值，数组中所有大于 `a_max` 的元素将被替换为 `a_max`。

- 代码

  ```python
  import cv2
  import numpy as np
  
  
  window_name = 'Trackbar Demo'
  cv2.namedWindow(window_name)
  def on_trackbar_change(x):
      x=x/255*(255--255)-255
      # cv2.destroyWindow("brightness_conversion_image")
      # 读取图片路径
      path = "./1.jpg"
      # 读取图片
      image_np = cv2.imread(path)
      # 亮度变换是对图像的每个通道的每个像素进行统一的加某个值
      # np.clip是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。
      # np.uint8是将值转换为0-255的整数
      brightness_conversion_img = np.uint8(np.clip(image_np + x, 0, 255))
      cv2.imshow("brightness_conversion_image", brightness_conversion_img)
      cv2.imshow("image_np", image_np)
      print(x)
  # 创建滑动条并设置参数
  trackbar_name = 'Threshold'
  max_value = 255
  initial_value = 100
  on_trackbar_change(initial_value)
  cv2.createTrackbar(trackbar_name, window_name, initial_value, max_value, on_trackbar_change)
  cv2.waitKey(0)
  ```

  


  




