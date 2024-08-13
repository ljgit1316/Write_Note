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
