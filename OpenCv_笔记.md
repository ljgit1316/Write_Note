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

  



