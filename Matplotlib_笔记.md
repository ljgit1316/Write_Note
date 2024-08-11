

# Matplotlib

## plot绘制图案

- 语法

  ```python
  # 画单条线
  #API怎么看: [arg] 代表可以省略这个参数  arg=None代表默认为None
  plot([x], y, [fmt], *, data=None, **kwargs)
  
  # 画多条线
  plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)
  ```

- 参数说明

  - **x和 y：**

    - **含义：** 分别表示 x 轴和 y 轴上的数据点。
    - **类型：** 可以是列表、数组或其他可迭代对象。

  - **format：**

    - **含义：** 可选参数，用于指定线条的样式、颜色和标记。

    - **示例：** `'b-'` 表示蓝色实线，`'ro'` 表示红色圆点。`"r-."`表示红色点线

    - ### 1. 颜色字符：

      - `'b'`：蓝色
      - `'g'`：绿色
      - `'r'`：红色
      - `'c'`：青色
      - `'m'`：洋红色
      - `'y'`：黄色
      - `'k'`：黑色
      - `'w'`：白色

      ### 2. 线型字符：

      - `'-'`：实线
      - `'--'`：虚线
      - `':'`：点线
      - `'-.'`：点划线

      ### 3. 标记字符：

      - `'.'`：点标记
      - `'o'`：圆圈标记
      - `'s'`：正方形标记
      - `'D'`：菱形标记
      - `'^'`：上三角标记
      - `'v'`：下三角标记
      - `'<'`：左三角标记
      - `'>'`：右三角标记
      - `'p'`：五边形标记
      - `'h'`：六边形标记

  - **其他关键字参数** (`**kwargs`)：

    - 其他参数用于调整线条、标记、颜色等属性，例如 `linewidth`（线宽）、`markersize`（标记大小）、`color`（颜色）等。

    - **`color`：**

      - **说明：** 指定线条的颜色。
      - **示例：** `color='red'` 或 `color='#FF0000'`。

    - **`linestyle`：**

      - **说明：** 指定线型。
      - **示例：** `linestyle='--'`。

    - **`linewidth`：**

      - **说明：** 指定线条的宽度。
      - **示例：** `linewidth=2`。

    - **`markersize`：**

      - **说明：** 指定标记的大小。
      - **示例：** `markersize=8`。

    - **`marker`：**

      - **marker** 可以定义的符号如下：

        | 标记               | 符号                           | 描述                                         |
        | :----------------- | :----------------------------- | :------------------------------------------- |
        | "."                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem00.png) | 点                                           |
        | ","                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem01.png)| 像素点                                       |
        | "o"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem02.png) | 实心圆                                       |
        | "v"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem03.png) | 下三角                                       |
        | "^"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem04.png)| 上三角                                       |
        | "<"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem05.png)| 左三角                                       |
        | ">"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem06.png)| 右三角                                       |
        | "1"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem07.png)| 下三叉                                       |
        | "2"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem08.png)| 上三叉                                       |
        | "3"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem09.png) | 左三叉                                       |
        | "4"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem10.png)| 右三叉                                       |
        | "8"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem11.png)| 八角形                                       |
        | "s"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem12.png) | 正方形                                       |
        | "p"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem13.png) | 五边形                                       |
        | "P"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem23.png)| 加号（填充）                                 |
        | "*"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem14.png) | 星号                                         |
        | "h"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem15.png) | 六边形 1                                     |
        | "H"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem16.png) | 六边形 2                                     |
        | "+"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem17.png)| 加号                                         |
        | "x"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem18.png) | 乘号 x                                       |
        | "X"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem24.png)| 乘号 x (填充)                                |
        | "D"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem19.png)| 菱形                                         |
        | "d"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem20.png)| 瘦菱形                                       |
        | "\|"               | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem21.png)| 竖线                                         |
        | "_"                | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem22.png)| 横线                                         |
        | 0 (TICKLEFT)       | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem25.png) | 左横线                                       |
        | 1 (TICKRIGHT)      | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem26.png) | 右横线                                       |
        | 2 (TICKUP)         | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem26.png) | 上竖线                                       |
        | 3 (TICKDOWN)       | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem27.png) | 下竖线                                       |
        | 4 (CARETLEFT)      | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem29.png)| 左箭头                                       |
        | 5 (CARETRIGHT)     | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem30.png)| 右箭头                                       |
        | 6 (CARETUP)        | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem31.png)| 上箭头                                       |
        | 7 (CARETDOWN)      | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem32.png)| 下箭头                                       |
        | 8 (CARETLEFTBASE)  | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem33.png) | 左箭头 (中间点为基准)                        |
        | 9 (CARETRIGHTBASE) | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem34.png)| 右箭头 (中间点为基准)                        |
        | 10 (CARETUPBASE)   | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem35.png)| 上箭头 (中间点为基准)                        |
        | 11 (CARETDOWNBASE) | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem36.png) | 下箭头 (中间点为基准)                        |
        | "None", " " or ""  |                                | 没有任何标记                                 |
        | '$...$'            | ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypicturem37.png)| 渲染指定的字符。例如 "$f$" 以字母 f 为标记。 |

### 1.绘制直线图

```python
import matplotlib.pyplot as plt
import numpy as np

xp = np.array([1, 10])
yp = np.array([2, 6])
plt.plot(xp, yp)
plt.show()
```

### 2.绘制折线图

```python
import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([10, 6, 7, 3])
plt.plot(x, y)  # 绘制图案
plt.show()  # 将图案显示出来
```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231125202226743.png)


### 3.省略x轴的值

```python
import matplotlib.pyplot as plt
import numpy as np

y = np.array([3.6, 9.0, 1.2, 11.2])
# 这里缺省x轴数值，默认数值是x轴的单位刻度长度  比如0,1,2,3
plt.plot(y)
plt.show()
```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231125204420220.png)


### 4.绘制sin和cos图案

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)
y = np.sin(x)
z = np.cos(x)

plt.plot(x, y, "r-",  x, z, "b--", linewidth=4)  #这里的linewidth必须放在最后，但也可以写多条plot函数调用
plt.show()
```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231125204841299.png)


### 5.绘制圆形图案

```python
import numpy as np
import matplotlib.pyplot as plt

# 圆是2π,R是1，相当于将圆上的点等差分为1000份
t = np.linspace(0, 2 * np.pi, 1000)
# 三角函数cos(sita) = x/r
# 三角函数sin(sita) = y/r
x = np.cos(t)
y = np.sin(t)

plt.plot(x, y, "r")
plt.show()

```

![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231128132338919.png)


## Matplotlib轴标签和标题

**函数：** `xlabel()` 和 `ylabel()`

- 参数：

  - `xlabel()`：x轴标签的文本字符串。
  - `ylabel()`：y轴标签的文本字符串。

- **返回值：** 无，这两个函数主要用于设置图表的标签，不会返回特定的对象。

**函数：** `title()`

- 参数：
  - `title()`：图表标题的文本字符串。
- **返回值：** 无，该函数用于设置图表的标题，不会返回特定的对象。

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)
y = np.sin(x)

plt.xlabel("x轴")
plt.ylabel("y轴")
plt.title("正旋函数")
plt.plot(x, y, linewidth=2, marker="o")
plt.show()
```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231127131014775.png)


## Matplotlib显示中文问题

**使用操作系统中的字体库**

```python
# 设置系统已安装字体的名称
plt.rcParams['font.family'] = ['SimHei']  #  或者'Microsoft YaHei' , 'SimSun'等
```

```python
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['SimHei']

x = np.arange(0, 10, 0.1)
y = np.sin(x)
plt.xlabel("x轴")
plt.ylabel("y轴")
plt.title("正旋函数")
plt.plot(x, y, linewidth=2, marker="o")
plt.show()
```

## Matplotlib绘制网格线

- 语法

  ```python
  matplotlib.pyplot.grid(b=None, which='major', axis='both', )
  ```

- 参数说明

  - **b**：可选，默认为 None，可以设置布尔值，true 为显示网格线，false 为不显示，如果设置 **kwargs 参数，则值为 true。
  - **which**：可选，可选值有 'major'、'minor' 和 'both'，默认为 'major'，表示应用更改的网格线。
  - **axis**：可选，设置显示哪个方向的网格线，可以是取 'both'（默认），'x' 或 'y'，分别表示两个方向，x 轴方向或 y 轴方向。
  - ***\*kwargs**：可选，设置网格样式，可以是 color='r', linestyle='-' 和 linewidth=2，分别表示网格线的颜色，样式和宽度。

- 实例

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  
  x = np.array([1, 1, 3, 3, 1])
  y = np.array([0, 3, 3, 0, 0])
  # x = np.linspace(1, 50, 50)
  # print(x)
  # y = 2 * x + 1
  plt.plot(x, y, linewidth=3)
  plt.minorticks_on()# 启用次要刻度线
  plt.grid(True, which="both", axis="both", color="r", linestyle="--")
  # 参数1：True前不能写b=True,直接写成True即可
  # 参数2：which显示的是major主或minor次刻度线，both标签全部开启，次刻度没效果需要使用plt.minorticks_on()开启次刻度
  # 参数3：axis显示那个轴的刻度，x，y，或者both
  # 参数4：color设置刻度颜色
  plt.show()
  ```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231128222924086.png)

  

## Matplotlib绘制多图

我们可以使用 pyplot 中的 **subplot()** 和 **subplots()** 方法来绘制多个子图。

**subplot()** 方法在绘图时需要指定位置，**subplots()** 方法可以一次生成多个，在调用时只需要调用生成对象的 ax 即可。

### 1.subplot

```python
subplot(nrows, ncols, index, **kwargs)
subplot(pos, **kwargs)
subplot(**kwargs)
subplot(ax)

"""
以上函数将整个绘图区域分成 nrows 行和 ncols 列，然后从左到右，从上到下的顺序对每个子区域进行编号 **1...N** ，左上的子区域的编号为 1、右下的区域编号为 N，编号可以通过参数 **index** 来设置。

"""
"""
设置 numRows ＝ 1，numCols ＝ 2，就是将图表绘制成 1x2 的图片区域, 对应的坐标为：(1, 1), (1, 2)
"""
"""
**plotNum ＝ 1**, 表示的坐标为(1, 1), 即第一行第一列的子图。

**plotNum ＝ 2**, 表示的坐标为(1, 2), 即第一行第二列的子图。
"""
```

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 30)
y = x ** 2 + 1
plt.subplot(2, 2, 1)
plt.plot(x, y)

x = np.linspace(1, 10, 10)
y = 2 * x + 1
plt.subplot(2, 2, 2)
plt.plot(x, y)

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.subplot(2, 2, 3)
plt.plot(x, y)

x = np.linspace(0, 10, 100)
y = np.cos(x)
#plt.subplot(2, 2, 4)  #subplot(nrows, ncols, index, **kwargs)这种写法
#plt.subplot(224)      #subplot(pos, **kwargs)这种写法这种写法
#plt.subplot(2, 2, 4)  #subplot(**kwargs)这种写法
plt.plot(x, y)
plt.show()
```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231128224940046.png)



### 2.subplots

- 语法

  ```python
  matplotlib.pyplot.subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw)
  ```

- 参数说明

  - **nrows**：默认为 1，设置图表的行数。
  - **ncols**：默认为 1，设置图表的列数。
  - **sharex、sharey**：设置 x、y 轴是否共享属性，默认为 false，可设置为 'none'、'all'、'row' 或 'col'。 False 或 none 每个子图的 x 轴或 y 轴都是独立的，True 或 'all'：所有子图共享 x 轴或 y 轴，'row' 设置每个子图行共享一个 x 轴或 y 轴，'col'：设置每个子图列共享一个 x 轴或 y 轴。
  - **squeeze**：布尔值，默认为 True，表示额外的维度从返回的 Axes(轴)对象中挤出，对于 N*1 或 1*N 个子图，返回一个 1 维数组，对于 N*M，N>1 和 M>1 返回一个 2 维数组。如果设置为 False，则不进行挤压操作，返回一个元素为 Axes 实例的2维数组，即使它最终是1x1。
  - **subplot_kw**：可选，字典类型。把字典的关键字传递给 add_subplot() 来创建每个子图。
  - **gridspec_kw**：可选，字典类型。把字典的关键字传递给 GridSpec 构造函数创建子图放在网格里(grid)。
  - ***\*fig_kw**：把详细的关键字参数传给 figure() 函数。

- 实例

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  
  # fig是整个图像窗口，ax是图像轴的子图
  fig, ax = plt.subplots(2, 2)
  x = np.linspace(-10, 10, 30)
  y = x ** 2 + 1
  ax[0, 0].plot(x, y)
  
  x = np.linspace(1, 10, 10)
  y = 2 * x + 1
  ax[0, 1].plot(x, y)
  
  x = np.linspace(0, 10, 100)
  y = np.sin(x)
  ax[1, 0].plot(x, y)
  
  x = np.linspace(0, 10, 100)
  y = np.cos(x)
  ax[1, 1].plot(x, y)
  plt.show()
  ```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231128231058822.png)


  

## Matplotlib绘制散点图

- scatter()语法

  ```python
  matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None, **kwargs)
  ```

- 参数说明

  **x，y**：长度相同的数组，也就是我们即将绘制散点图的数据点，输入数据。

  **s**：点的大小，默认 20，也可以是个数组，数组每个参数为对应点的大小。

  **c**：点的颜色，默认蓝色 'b'，也可以是个 RGB 或 RGBA 二维行数组。

  **marker**：点的样式，默认小圆圈 'o'。

  **cmap**：Colormap，默认 None，标量或者是一个 colormap 的名字，只有 c 是一个浮点数数组的时才使用。如果没有声明就是 image.cmap。

  **norm**：Normalize，默认 None，数据亮度在 0-1 之间，只有 c 是一个浮点数的数组的时才使用。

  **vmin，vmax：**：亮度设置，在 norm 参数存在时会忽略。

  **alpha：**：透明度设置，0-1 之间，默认 None，即不透明。

  **linewidths：**：标记点的长度。

  **edgecolors：**：颜色或颜色序列，默认为 'face'，可选值有 'face', 'none', None。

  **plotnonfinite：**：布尔值，设置是否使用非限定的 c ( inf, -inf 或 nan) 绘制点。

  ***\*kwargs：**：其他参数。

- 实例

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  import random
  
  x = np.random.rand(20)
  y = np.random.rand(20)
  color_src = ["red", "green", "orange", "yellow", "blue", "purple"]
  color = list(range(20))
  for i in range(20):
      color[i] = color_src[random.randint(0, 5)]
  
  plt.scatter(x, y, s=40, c=color)
  plt.show()
  ```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231128233840102.png)




## Matplotlib柱形图

- bar() 方法

  ```python
  matplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
  ```

- 参数说明

  **x**：浮点型数组，柱形图的 x 轴数据。

  **height**：浮点型数组，柱形图的高度。

  **width**：浮点型数组，柱形图的宽度。

  **bottom**：浮点型数组，底座的 y 坐标，默认 0。

  **align**：柱形图与 x 坐标的对齐方式，'center' 以 x 位置为中心，这是默认值。 'edge'：将柱形图的左边缘与 x 位置对齐。要对齐右边缘的条形，可以传递负数的宽度值及 align='edge'。

  ***\*kwargs：**：其他参数。

- 实例

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  plt.rcParams["font.family"]=["SimHei"]
  x = np.array(["张三", "李四", "王五", "老七"])
  y = np.array([88, 66, 99, 100])
  color = np.array(["red", "yellow", "purple", "blue"])
  
  fig, ax = plt.subplots(2, 2)
  # 如果是水平两个子图ax[0],ax[1]当成一维数组访问即可
  ax[0, 0].bar(x, y, width=0.5, color=color)
  ax[0, 1].barh(x, y, height=0.5, color=color)
  ax[1, 0].bar(x, y, width=0.5, color=color)
  ax[1, 1].barh(x, y, height=0.5, color=color)
  plt.show()
  ```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231129000427860.png)


  

## Matplotlib饼图

- pie() 语法

  ```python
  matplotlib.pyplot.pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=0, radius=1, counterclock=True, wedgeprops=None, textprops=None, center=0, 0, frame=False, rotatelabels=False, *, normalize=None, data=None)[source]
  ```

- 参数说明

  - **x**：浮点型数组或列表，用于绘制饼图的数据，表示每个扇形的面积。
  - **explode**：数组，表示各个扇形之间的间隔，默认值为0。
  - **labels**：列表，各个扇形的标签，默认值为 None。
  - **colors**：数组，表示各个扇形的颜色，默认值为 None。
  - **autopct**：设置饼图内各个扇形百分比显示格式，**%d%%** 整数百分比，**%0.1f** 一位小数， **%0.1f%%** 一位小数百分比， **%0.2f%%** 两位小数百分比。
  - **labeldistance**：标签标记的绘制位置，相对于半径的比例，默认值为 1.1，如 **<1**则绘制在饼图内侧。
  - **pctdistance：**类似于 labeldistance，指定 autopct 的位置刻度，默认值为 0.6。
  - **shadow：**布尔值 True 或 False，设置饼图的阴影，默认为 False，不设置阴影。
  - **radius：**设置饼图的半径，默认为 1。
  - **startangle：**用于指定饼图的起始角度，默认为从 x 轴正方向逆时针画起，如设定 =90 则从 y 轴正方向画起。
  - **counterclock**：布尔值，用于指定是否逆时针绘制扇形，默认为 True，即逆时针绘制，False 为顺时针。
  - **wedgeprops：**字典类型，默认值 None。用于指定扇形的属性，比如边框线颜色、边框线宽度等。例如：wedgeprops={'linewidth':5} 设置 wedge 线宽为5。
  - **textprops：**字典类型，用于指定文本标签的属性，比如字体大小、字体颜色等，默认值为 None。
  - **center：**浮点类型的列表，用于指定饼图的中心位置，默认值：(0,0)。
  - **frame：**布尔类型，用于指定是否绘制饼图的边框，默认值：False。如果是 True，绘制带有表的轴框架。
  - **rotatelabels：**布尔类型，用于指定是否旋转文本标签，默认为 False。如果为 True，旋转每个 label 到指定的角度。
  - **data**：用于指定数据。如果设置了 data 参数，则可以直接使用数据框中的列作为 x、labels 等参数的值，无需再次传递。

  除此之外，pie() 函数还可以返回三个参数：

  - `wedges`：一个包含扇形对象的列表。
  - `texts`：一个包含文本标签对象的列表。
  - `autotexts`：一个包含自动生成的文本标签对象的列表。

- 实例

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  
  x = np.array([50, 80, 66, 13, 21])
  label = np.array(["python", "c", "c++", "go", "R"])
  
  fig, ax = plt.subplots(1, 2)
  ax[0].pie(x, labels=label, autopct="%.1f%%")
  
  explode = [0, 0.1, 0, 0, 0]
  ax[1].pie(x, explode=explode, labels=label, autopct="%.1f%%")
  plt.show()
  ```

  ![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231129001813176.png)



## Matplotlib直方图

- hist() 语法

  ```python
  matplotlib.pyplot.hist(x, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, **kwargs)
  ```

- 参数说明

  - `x`：表示要绘制直方图的数据，可以是一个一维数组或列表。
  - `bins`：可选参数，表示直方图柱子个数。默认为10。
  - `range`：可选参数，表示直方图的值域范围，可以是一个二元组或列表。默认为None，即使用数据中的最小值和最大值。
  - `density`：可选参数，表示是否将直方图归一化。默认为False，即直方图的高度为每个箱子内的样本数，而不是频率或概率密度。
  - `weights`：可选参数，表示每个数据点的权重。默认为None。
  - `cumulative`：可选参数，表示是否绘制累积分布图。默认为False。
  - `bottom`：可选参数，表示直方图的起始高度。默认为None。
  - `histtype`：可选参数，表示直方图的类型，可以是'bar'、'barstacked'、'step'、'stepfilled'等。默认为'bar'。
  - `align`：可选参数，表示直方图箱子的对齐方式，可以是'left'、'mid'、'right'。默认为'mid'。
  - `orientation`：可选参数，表示直方图的方向，可以是'vertical'、'horizontal'。默认为'vertical'。
  - `rwidth`：可选参数，表示每个箱子的宽度。默认为None。
  - `log`：可选参数，表示是否在y轴上使用对数刻度。默认为False。
  - `color`：可选参数，表示直方图的颜色。
  - `label`：可选参数，表示直方图的标签。
  - `stacked`：可选参数，表示是否堆叠不同的直方图。默认为False。
  - `**kwargs`：可选参数，表示其他绘图参数。

- 实例

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  
  # 生成正态分布随机数
  x = np.random.randn(1000)
  # bins是柱的个数，alpha是透明度
  plt.hist(x, bins=40, alpha=1)
  plt.show()
  ```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231129141932090.png)
  

  

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  
  """
  numpy.random.normal(loc=0.0, scale=1.0, size=None)
  loc: 均值（正态分布轴线位置），默认为 0。
  scale: 标准差(数据的离散程度)，默认为 1。
  size: 数组的形状，可以是整数，元组或者 None。表示生成的随机数的数量和形状。
  """
  x = np.random.normal(loc=-5, scale=1, size=1000)
  plt.hist(x, bins=40, alpha=0.5, label="Data1")
  
  x = np.random.normal(loc=0, scale=1, size=1000)
  plt.hist(x, bins=40, alpha=0.5, label="Data2")
  
  x = np.random.normal(loc=5, scale=1, size=1000)
  plt.hist(x, bins=40, alpha=0.5, label="Data3")
  # plt.legend() 函数是 Matplotlib 库中用于添加图例的函数
  plt.legend()
  plt.show()
  ```

![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231129143746900.png)





## Matplotlib imshow() 方法

```python
"""
imshow() 函数是 Matplotlib 库中的一个函数，用于显示图像。

imshow() 函数常用于绘制二维的灰度图像或彩色图像。

imshow() 函数可用于绘制矩阵、热力图、地图等。

imshow() 方法语法格式如下：
"""

imshow(X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, shape=None, filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None, *, data=None, **kwargs)
```

- 参数说明
  - `X`：输入数据。可以是二维数组、三维数组、PIL图像对象、matplotlib路径对象等。
  - `cmap`：颜色映射。用于控制图像中不同数值所对应的颜色。可以选择内置的颜色映射，如`gray`、`hot`、`jet`等，也可以自定义颜色映射。
  - `norm`：用于控制数值的归一化方式。可以选择`Normalize`、`LogNorm`等归一化方法。
  - `aspect`：控制图像纵横比（aspect ratio）。可以设置为`auto`或一个数字。
  - `interpolation`：插值方法。用于控制图像的平滑程度和细节程度。可以选择`nearest`、`bilinear`、`bicubic`等插值方法。
  - `alpha`：图像透明度。取值范围为0~1。
  - `origin`：坐标轴原点的位置。可以设置为`upper`或`lower`。
  - `extent`：控制显示的数据范围。可以设置为`[xmin, xmax, ymin, ymax]`。
  - `vmin`、`vmax`：控制颜色映射的值域范围。
  - `filternorm 和 filterrad`：用于图像滤波的对象。可以设置为`None`、`antigrain`、`freetype`等。
  - `imlim`： 用于指定图像显示范围。
  - `resample`：用于指定图像重采样方式。
  - `url`：用于指定图像链接。
- 实例

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成10*10的二维数组
img = np.random.rand(10, 10)
# cmap指定颜色 "gray"灰色
"""
supported values are 'Accent', 'Accent_r', 
'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 
'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 
'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 
'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 
'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 
'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 
'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 
'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 
'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu',
'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 
'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 
'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 
'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 
'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 
'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 
'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r',
'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r',
'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',
'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 
'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 
'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 
'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 
'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 
'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 
'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b',
'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 
'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r',
'viridis', 'viridis_r', 'winter', 'winter_r'
      
"""
plt.imshow(img, cmap="summer_r")
plt.colorbar()
plt.show()
```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231129151728426.png)




```python
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as im

image = im.open("./map.jpeg")
data = np.array(image)
plt.imshow(data)
plt.axis("off")  # 关闭轴的显示
plt.show()
```

![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231129152508552.png)



## Matplotlib imsave()方法		

```python
"""
imsave() 方法是 Matplotlib 库中用于将图像数据保存到磁盘上的函数。

通过 imsave() 方法我们可以轻松将生成的图像保存到我们指定的目录中。

imsave() 方法保存图片支持多种图像格式，例如 PNG、JPEG、BMP 等。
"""

matplotlib.pyplot.imsave(fname, arr, **kwargs)
```

- 参数说明

  - `fname`：保存图像的文件名，可以是相对路径或绝对路径。
  - `arr`：表示图像的NumPy数组。
  - `kwargs`：可选参数，用于指定保存的图像格式以及图像质量等参数。

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  
  img = np.random.rand(20, 20, 3)
  plt.imshow(img)
  
  plt.imsave("img.png", img)
  plt.show()
  
  ```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231129163728602.png)




## Matplotlib imread() 方法

```python
"""
imread() 方法是 Matplotlib 库中的一个函数，用于从图像文件中读取图像数据。

imread() 方法返回一个 numpy.ndarray 对象，其形状是 **(nrows, ncols, nchannels)**，表示读取的图像的行数、列数和通道数：

- 如果图像是灰度图像，则 nchannels 为 1。
- 如果是彩色图像，则 nchannels 为 3 或 4，分别表示红、绿、蓝三个颜色通道和一个 alpha 通道。
"""

matplotlib.pyplot.imread(fname, format=None)
```

- 参数说明

  - `fname`：指定了要读取的图像文件的文件名或文件路径，可以是相对路径或绝对路径。
  - `format `：参数指定了图像文件的格式，如果不指定，则默认根据文件后缀名来自动识别格式。

  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  
  img = plt.imread("./map.jpeg")
  plt.imshow(img)
  plt.show()
  ```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Matplotlib_Pic/mypictureimage-20231129170157309.png)
  
