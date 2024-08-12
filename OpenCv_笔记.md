# OpenCv

## 计算机中的图像

### 1.像素

日常生活中常见的图像是RGB三原色图。RGB图上的每个点都是由红（R）、绿（G）、蓝（B）三个颜色按照一定比例混合而成的，几乎所有颜色都可以通过这三种颜色按照不同比例调配而成。在计算机中，RGB三种颜色被称为RGB三通道，根据这三个通道存储的像素值，来对应不同的颜色。例如，在使用“画图”软件进行自定义调色时，其数值单位就是像素。如下图所示：

![red](C:\Users\13167\Desktop\OpenCv_Pic_test\red.png)

![blue](C:\Users\13167\Desktop\OpenCv_Pic_test\blue.png)

![green](C:\Users\13167\Desktop\OpenCv_Pic_test\green.png)

### 2.图像

计算机采用0/1编码的系统，数字图像也是利用0/1来记录信息，我们平常接触的图像都是8位数图像，包含0～255灰度，其中0，代表最黑，1，表示最白。

#### 2.1二值图

一幅二值图像的二维矩阵仅由0、1两个值构成，“0”代表黑色，“1”代白色。由于每一像素（矩阵中每一元素）取值仅有0、1两种可能，所以计算机中二值图像的数据类型通常为1个二进制位。二值图像通常用于文字、线条图的扫描识别（OCR）和掩膜图像的存储。

![cat2](C:\Users\13167\Desktop\OpenCv_Pic_test\cat2.png)

#### 2.2灰度图

每个像素只有一个采样颜色的图像，这类图像通常显示为从最暗黑色到最亮的白色的灰度，尽管理论上这个采样可以任何颜色的不同深浅，甚至可以是不同亮度上的不同颜色。灰度图像与黑白图像不同，在计算机图像领域中黑白图像只有黑色与白色两种颜色；但是，灰度图像在黑色与白色之间还有许多级的颜色深度。灰度图像经常是在单个电磁波频谱如可见光内测量每个像素的亮度得到的，用于显示的灰度图像通常用每个采样像素8位的非线性尺度来保存，这样可以有256级灰度（如果用16位，则有65536级）。

![cat3](C:\Users\13167\Desktop\OpenCv_Pic_test\cat3.png)

#### 2.3彩色图

每个像素通常是由红（R）、绿（G）、蓝（B）三个分量来表示的，分量介于（0，255）。RGB图像与索引图像一样都可以用来表示彩色图像。与索引图像一样，它分别用红（R）、绿（G）、蓝（B）三原色的组合来表示每个像素的颜色。但与索引图像不同的是，RGB图像每一个像素的颜色值（由RGB三原色表示）直接存放在图像矩阵中，由于每一像素的颜色需由R、G、B三个分量来表示，M、N分别表示图像的行列数，三个M x N的二维矩阵分别表示各个像素的R、G、B三个颜色分量。RGB图像的数据类型一般为8位无符号整形，通常用于表示和存放真彩色图像。

![cat1](C:\Users\13167\Desktop\OpenCv_Pic_test\cat1.png)

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

![random](C:\Users\13167\Desktop\OpenCv_Pic_test\random.png)

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

![QQ图片20240812170336](C:\Users\13167\Desktop\OpenCv_Pic_test\imshow_pic.png)

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

![line_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\line_pic.png)

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

![circle_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\circle_pic.png)

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

![tangle_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\tangle_pic.png)

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

![txt_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\txt_pic.png)

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

![point_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\point_pic.png)

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

  ![max](C:\Users\13167\Desktop\OpenCv_Pic_test\max.png)

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

  ![max_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\max_pic.png)

### 2.平均值法

- 定义

  对于彩色图像的每个像素，它会将R、G、B三个通道的像素值全部加起来，然后再除以三，得到的平均值就是灰度图像中对应位置的像素值。

- 图示

  ![overage](C:\Users\13167\Desktop\OpenCv_Pic_test\overage.png)

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

  ![overage_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\overage_pic.png)

### 3.加权均值法

- 定义

  对于彩色图像的每个像素，它会按照一定的权重去乘以每个通道的像素值，并将其相加，得到最后的值就是灰度图像中对应位置的像素值。

- 图示

  ![weight](C:\Users\13167\Desktop\OpenCv_Pic_test\weight.png)

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

  ![weight_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\weight_pic.png)

### 4.极端的灰度值

- 图示

  | ![black](E:\AI-Study\python学习\opencv笔记\opencv笔记\02灰度实验\media\black.png) | ![white](E:\AI-Study\python学习\opencv笔记\opencv笔记\02灰度实验\media\white.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 灰度值为0（纯黑）                                            | 灰度值为255（纯白）                                          |

## 图像二值化处理

​		二值化，顾名思义，就是将某张图像的所有像素改成只有两种值之一，其操作的图像也必须是灰度图。也就是说，二值化的过程，就是将一张灰度图上的像素根据某种规则修改为0和maxval（maxval表示最大值，一般为255，显示白色）两种像素值，使图像呈现黑白的效果，能够帮助我们更好地分析图像中的形状、边缘和轮廓等特征。

### 1.阈值法（THRESH_BINARY）

- 定义

  阈值法就是通过设置一个阈值，将灰度图中的每一个像素值与该阈值进行比较，小于等于阈值的像素就被设置为0（黑），大于阈值的像素就被设置为maxval。

- 图示

  ![yu](C:\Users\13167\Desktop\OpenCv_Pic_test\yu.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg',cv2.IMREAD_GRAYSCALE)#读入图片时，已转换为灰度图
  ret,pic_two=cv2.threshold(pic,200,255,cv2.THRESH_BINARY)
  cv2.imshow('two',pic_two)
  cv2.waitKey(0)
  ```

  ![yu_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\yu_pic.png)

### 2.反阈值法（THRESH_BINARY_INV）

- 定义

  反阈值法是当灰度图的像素值大于阈值时，该像素值将会变成0（黑），当灰度图的像素值小于等于阈值时，该像素值将会变成maxval。

- 图示

  ![fan_yu](C:\Users\13167\Desktop\OpenCv_Pic_test\fan_yu.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg',cv2.IMREAD_GRAYSCALE)#读入图片时，已转换为灰度图
  ret,pic_two=cv2.threshold(pic,127,255,cv2.THRESH_BINARY_INV)
  cv2.imshow('two',pic_two)
  cv2.waitKey(0)
  ```

  ![fan_yu_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\fan_yu_pic.png)

### 3.截断阈值法（THRESH_TRUNC）

- 定义

  截断阈值法，指将灰度图中的所有像素与阈值进行比较，像素值大于阈值的部分将会被修改为阈值，小于等于阈值的部分不变。换句话说，经过截断阈值法处理过的二值化图中的最大像素值就是阈值。(与传统二值图有所不同，值不为二选一)

- 图示

  ![jie_yu](C:\Users\13167\Desktop\OpenCv_Pic_test\jie_yu.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg',cv2.IMREAD_GRAYSCALE)#读入图片时，已转换为灰度图
  ret,pic_two=cv2.threshold(pic,200,255,cv2.THRESH_TRUNC)
  cv2.imshow('two',pic_two)
  cv2.waitKey(0)
  ```

  ![jie_yu_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\jie_yu_pic.png)

### 4.低阈值零处理（THRESH_TOZERO）

- 定义

  低阈值零处理，字面意思，就是像素值小于等于阈值的部分被置为0（也就是黑色），大于阈值的部分不变。

- 图示

  ![di_yu](C:\Users\13167\Desktop\OpenCv_Pic_test\di_yu.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg',cv2.IMREAD_GRAYSCALE)#读入图片时，已转换为灰度图
  ret,pic_two=cv2.threshold(pic,127,255,cv2.THRESH_TOZERO)
  cv2.imshow('two',pic_two)
  cv2.waitKey(0)
  ```

  ![di_yu_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\di_yu_pic.png)

### 5.超阈值零处理（THRESH_TOZERO_INV）

- 定义

  超阈值零处理就是将灰度图中的每个像素与阈值进行比较，像素值大于阈值的部分置为0（也就是黑色），像素值小于等于阈值的部分不变。

- 图示

  ![chao_yu](C:\Users\13167\Desktop\OpenCv_Pic_test\chao_yu.png)

- 代码

  ```python
  import cv2
  import numpy as np
  pic=cv2.imread('pic.jpg',cv2.IMREAD_GRAYSCALE)#读入图片时，已转换为灰度图
  ret,pic_two=cv2.threshold(pic,127,255,cv2.THRESH_TOZERO_INV)
  cv2.imshow('two',pic_two)
  cv2.waitKey(0)
  ```

  ![chao_yu_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\chao_yu_pic.png)

### 6.OTSU阈值法

- 定义

  OTSU算法是通过一个值将这张图分前景色和背景色（也就是灰度图中小于这个值的是一类，大于这个值的是一类），通过统计学方法（最大类间方差）来验证该值的合理性，当根据该值进行分割时，使用最大类间方差计算得到的值最大时，该值就是二值化算法中所需要的阈值。通常该值是从灰度图中的最小值加1开始进行迭代计算，直到灰度图中的最大像素值减1，然后把得到的最大类间方差值进行比较，来得到二值化的阈值。
  $$
  \begin{aligned}
  &T：阈值\\
  &N_{0}：前景像素点数\\
  &N_{1}：背景像素点数\\
  &\omega_{0}：前景的像素点数占整幅图像的比例\\
  &\omega_{1}：背景的像素点数占整幅图像的比例\\
  &\mathcal{U_{0}}：前景的平均像素值\\
  &\mathcal{U_{1}}：背景的平均像素值\\
  &\mathcal{U}：整幅图的平均像素值\\
  &rows×cols：图像的行数和列数
  \end{aligned}
  $$
  
- 图示

  ![double](C:\Users\13167\Desktop\OpenCv_Pic_test\double.png)

  下面举个例子，有一张大小为4×4的图片，假设阈值T为1:

  ![double_test](C:\Users\13167\Desktop\OpenCv_Pic_test\double_test.png)

  也就是这张图片根据阈值1分为了前景（像素为2的部分）和背景（像素为0）的部分，并且计算出了OTSU算法所需要的各个数据，根据上面的数据，我们给出计算方差的公式：

  $$
  g=\omega_{0}(\mu_{0}-\mu)^{2}+\omega_{1}(\mu_{1}-\mu)^{2}
  $$

  g就是前景与背景两类之间的方差，这个值越大，说明前景和背景的差别就越大，效果就越好。OTSU算法就是在灰度图的像素值范围内遍历阈值T，使得g最大，基本上双峰图片的阈值T在两峰之间的谷底。

  通过OTSU算法得到阈值之后，就可以结合上面的方法根据该阈值进行二值化，在本实验中有THRESH_OTSU和THRESH_INV_OTSU两种方法，就是在计算出阈值后结合了阈值法和反阈值法。

  注意：使用OTSU算法计算阈值时，组件中的thresh参数将不再有任何作用。

  | ![double_table1](C:\Users\13167\Desktop\OpenCv_Pic_test\double_table1.png) | ![double_table2](C:\Users\13167\Desktop\OpenCv_Pic_test\double_table2.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![double_3](C:\Users\13167\Desktop\OpenCv_Pic_test\double_3.png) | ![double_table4](C:\Users\13167\Desktop\OpenCv_Pic_test\double_table4.png) |
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

  ![double_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\double_pic.png)

## 图像自适应二值化

- 定义

  与二值化算法相比，自适应二值化更加适合用在明暗分布不均的图片，因为图片的明暗不均，**导致图片上的每一小部分都要使用不同的阈值进行二值化处理**，这时候传统的二值化算法就无法满足我们的需求了，于是就出现了自适应二值化。

  自适应二值化方法会对图像中的所有像素点计算其各自的阈值，这样能够更好的保留图片里的一些信息。

- 图示

  ![adapt_overage](C:\Users\13167\Desktop\OpenCv_Pic_test\adapt_overage.png)

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

![adapt_overage_example](C:\Users\13167\Desktop\OpenCv_Pic_test\adapt_overage_example.png)

​		如我们使用的小区域是3\*3的，那么就会从图片的左上角开始（也就是像素值为162的地方）计算其邻域内的平均值，如果处于边缘地区就会对边界进行填充，填充值就是边界的像素点，如下图所示：

![adapt_overage](C:\Users\13167\Desktop\OpenCv_Pic_test\adapt_overage.png)

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

![adaptive_overage_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\adaptive_overage_pic.png)

### 2.加权求和

- 定义

  对小区域内的像素进行加权求和得到新的阈值，其权重值来自于高斯分布。

  高斯分布，通过概率密度函数来定义高斯分布，一维高斯概率分布函数为：

  $$
  p(y)={\frac{1}{\sigma{\sqrt{2\pi}}}}e^{{\frac{-(y-\mu)^{2}}{2\sigma^{2}}}}
  $$

- 图示

  通过改变函数中和的值，我们可以得到如下图像，其中均值为，标准差为。

  ![tai](C:\Users\13167\Desktop\OpenCv_Pic_test\tai.png)

  此时我们拓展到二维图像，一般情况下我们使x轴和y轴的相等并且，此时我们可以得到二维高斯函数的表达式为：

  $$
  g(x,y)=\frac{1}{2\pi\sigma ^{2}}e^{-\frac{(x^{2}+y^{2})}{2\sigma^{2}}}
  $$

  高斯概率函数是相对于二维坐标产生的，其中（x,y）为点坐标，要得到一个高斯滤波器模板，应先对高斯函数进行离散化，将得到的值作为模板的系数。例如：要产生一个3\*3的高斯权重核，以核的中心位置为坐标原点进行取样，其周围的坐标如下图所示（x轴水平向右，y轴竖直向上）

  ![spax](C:\Users\13167\Desktop\OpenCv_Pic_test\spax.png)

  将坐标带入上面的公式中，即可得到一个高斯权重核。

  而在opencv里，当kernel(小区域)的尺寸为1、3、5、7并且用户没有设置sigma的时候(sigma \<= 0),核值就会取固定的系数，这是一种默认的值是高斯函数的近似。


  | kernel尺寸 | 核值                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 1          | [1]                                                          |
  | 3          | [0.25, 0.5, 0.25]                                            |
  | 5          | [0.0625, 0.25, 0.375, 0.25, 0.0625]                          |
  | 7          | [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125] |

  比如kernel的尺寸为3\*3时，使用

$$
  \left[\begin{array}{c}{{0.25}}\\ {{0.5}}\\ {{0.25}}\end{array}\right]\times\left[0.25~~~~0.5~~~~0.25\right]
$$

  进行矩阵的乘法，就会得到如下的权重值，其他的类似。

$$
  kernel=\left[\begin{array}{c}{{0.0625~~~0.125~~~0.0625}}\\{{0.125~~~~0.25~~~~0.125}}\\
  {{0.0625~~~0.125~~~0.0625}}
  \end{array}\right]
$$

  通过这个高斯核，即可对图片中的每个像素去计算其阈值，并将该阈值减去固定值得到最终阈值，然后根据二值化规则进行二值化。

  而当kernels尺寸超过7的时候,如果sigma设置合法(用户设置了sigma),则按照高斯公式计算.当sigma不合法(用户没有设置sigma),则按照如下公式计算sigma的值：

$$
  \sigma=0.3*\big((k s i z e-1)*0.5-1\big)+0.8
$$

  某像素点的阈值计算过程如下图所示：

  ![gauss_count](C:\Users\13167\Desktop\OpenCv_Pic_test\gauss_count.png)

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

  ![adaptive_gauss_pic](C:\Users\13167\Desktop\OpenCv_Pic_test\adaptive_gauss_pic.png)