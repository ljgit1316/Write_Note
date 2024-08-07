# NumPy

## NumPy对象（ndarray）

### 1.ndarray定义

NumPy 最重要的一个特点是其 N 维数组对象 ndarray，它是一系列同类型数据的集合，以 0 下标为开始进行集合中元素的索引。

ndarray 对象是用于存放同类型元素的多维数组。

ndarray 中的每个元素在内存中都有相同存储大小的区域。

ndarray 内部由以下内容组成：

- 一个指向数据（内存或内存映射文件中的一块数据）的指针。
- 数据类型或 dtype，描述在数组中的固定大小值的格子。
- 一个表示数组形状（shape）的元组，表示各维度大小的元组。
- 一个跨度元组（stride），其中的整数指的是为了前进到当前维度下一个元素需要"跨过"的字节数。

### 2.语法创建

```python
numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
```

### 3.参数说明

| 名称   | 描述                                                         |
| ------ | ------------------------------------------------------------ |
| object | 数组或嵌套的数列                                             |
| dtype  | 数组元素的数据类型，可选                                     |
| copy   | 对象是否需要复制，可选                                       |
| order  | 创建数组的内存存储样式，C为行方向，F为列方向，A为任意方向（默认） |
| subok  | 默认返回一个与基类类型一致的数组                             |
| ndmin  | 指定生成数组的最小维度                                       |

- 创建一维数组

  ```python
  arr=np.array([100,200,300])
  print(arr)
  """
  [100 200 300]
  """
  ```

- 创建二维数组

  ```python
  arr=np.array([[100,200],[300,400]])
  print(arr)
  """
  [[100 200]
   [300 400]]
  """
  ```

- 指定最小维度

  ```python
  arr=np.array([100,200,300],ndmin=2)# 指定最小维度，可以升，不可降
  print(arr)
  """
  [[100 200 300]]
  """
  ```

- 设定数组元素类型

  ```python
  arr=np.array([1.0,2.3,2.8],dtype=np.int16)
  print(arr)
  """
  [1 2 2]
  """
  arr=np.array([1,2.3,2.8],dtype=np.float16)
  print(arr)
  """
  [1.  2.3 2.8]
  """
  ```

## NumPy数据类型

| 名称       | 描述                                                         |
| ---------- | ------------------------------------------------------------ |
| bool_      | 布尔型数据类型（True 或者 False）                            |
| int_       | 默认的整数类型（类似于 C 语言中的 long，int32 或 int64）     |
| intc       | 与 C 的 int 类型一样，一般是 int32 或 int 64                 |
| intp       | 用于索引的整数类型（类似于 C 的 ssize_t，一般情况下仍然是 int32 或 int64） |
| int8       | 字节（-128 to 127）                                          |
| int16      | 整数（-32768 to 32767）                                      |
| int32      | 整数（-2147483648 to 2147483647）                            |
| int64      | 整数（-9223372036854775808 to 9223372036854775807）          |
| uint8      | 无符号整数（0 to 255）                                       |
| uint16     | 无符号整数（0 to 65535）                                     |
| uint32     | 无符号整数（0 to 4294967295）                                |
| uint64     | 无符号整数（0 to 18446744073709551615）                      |
| float_     | float64 类型的简写                                           |
| float16    | 半精度浮点数，包括：1 个符号位，5 个指数位，10 个尾数位      |
| float32    | 单精度浮点数，包括：1 个符号位，8 个指数位，23 个尾数位      |
| float64    | 双精度浮点数，包括：1 个符号位，11 个指数位，52 个尾数位     |
| complex_   | complex128 类型的简写，即 128 位复数                         |
| complex64  | 复数，表示双 32 位浮点数（实数部分和虚数部分）               |
| complex128 | 复数，表示双 64 位浮点数（实数部分和虚数部分）               |

## NumPy数组属性

| 属性             | 说明                                        |
| ---------------- | ------------------------------------------- |
| ndarray.ndim     | 秩，即轴的数量或维度的数量                  |
| ndarray.shape    | 数组的维度，对于矩阵，n 行 m 列             |
| ndarray.size     | 数组元素的总个数，相当于 .shape 中 n*m 的值 |
| ndarray.dtype    | ndarray 对象的元素类型                      |
| ndarray.itemsize | ndarray 对象中每个元素的大小，以字节为单位  |
| ndarray.flags    | ndarray 对象的内存信息                      |

### 1.ndarray.ndim

```python
#返回数组维度数

arr=np.array([100,200,300])
print(arr.ndim)#1

arr=np.array([[100,200],[300,400]])
print(arr.ndim)#2

arr=np.array([[[100,200],[300,400]]])
print(arr.ndim)#3
```

### 2.ndarray.shape

```python
#用元组的形式，返回数组维度，如同返回矩阵（行列数）

arr=np.array([100,200,300])
print(arr.shape)#(3,)

arr=np.array([[100,200],[300,400]])
print(arr.shape)#(2, 2)

arr=np.array([[[100,200],[300,400]]])
print(arr.shape)#(1, 2, 2)
```

### 3.ndarray.size

```python
#返回数组元素总个数

arr=np.array([100,200,300])
print(arr.size)#3

arr=np.array([[100,200],[300,400]])
print(arr.size)#4

arr=np.array([[[100,200],[300,400],[500,600]]])
print(arr.size)#6
```

### 4.ndarray.dtype

```python
#返回ndarray对象中元素类型

arr=np.array([100.0,200.0,300.0])
print(arr.dtype)#float64

arr=np.array([[100,200],[300,400]])
print(arr.dtype)#int32

arr=np.array([[[100.0,200.0],[300,400],[500,600]]])
print(arr.dtype)#float64
#当对象中的元素既有int又有float型时，会以精度高的作为数据类型作为返回
```

### 5.ndarray.itemsize

```python
#返回每个元素的大小，以字节为单位

arr=np.array([100.0,200.0,300.0])
print(arr.itemsize)#8

arr=np.array([[100,200],[300,400]])
print(arr.itemsize)#4

arr=np.array([[[100.0,200.0],[300,400],[500,600]]])
print(arr.itemsize)#8
```

## NumPy创建数组方法

### 1.自定义创建数组

#### 1.1、numpy.empty

- 语法

  ```python
  numpy.empty(shape, dtype = float, order = 'C')
  ```

- 参数说明

  | 参数  | 描述                                                         |
  | ----- | ------------------------------------------------------------ |
  | shape | 数组形状                                                     |
  | dtype | 数据类型，可选                                               |
  | order | 有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序。 |

- 代码实现

  ```python
  arr=np.empty((5,5),dtype=np.int16)
  print(arr)
  """
  [[ 25613 -31730  23040  25610  25615]
   [-31728  23040  25611  25617 -31726]
   [ 23040  25612  21251  10496  -9708]
   [ 18436  16717  31299  21118  17222]
   [ 12832  12337   8244  19784  17217]]
  """
  #实现创建未初始化的指定维度数组，及其指定数据类型
  ```

#### 1.2、numpy.zeros

- 语法

  ```python
  numpy.zeros(shape, dtype = float, order = 'C')
  ```

- 参数说明

  | 参数  | 描述                                                |
  | ----- | --------------------------------------------------- |
  | shape | 数组形状                                            |
  | dtype | 数据类型，可选                                      |
  | order | 'C' 用于 C 的行数组，或者 'F' 用于 FORTRAN 的列数组 |

- 代码实现

  ```python
  #创建指定大小的数组，数组元素以 0 来填充
  
  arr=np.zeros((5,5),dtype=np.int16)
  print(arr)
  
  """
  [[0 0 0 0 0]
   [0 0 0 0 0]
   [0 0 0 0 0]
   [0 0 0 0 0]
   [0 0 0 0 0]]
  """
  ```

#### 1.3、numpy.ones

- 语法

  ```python
  numpy.ones(shape, dtype = None, order = 'C')
  ```

- 参数说明

  | 参数  | 描述                                                |
  | ----- | --------------------------------------------------- |
  | shape | 数组形状                                            |
  | dtype | 数据类型，可选                                      |
  | order | 'C' 用于 C 的行数组，或者 'F' 用于 FORTRAN 的列数组 |

- 代码实现

  ```python
  #创建指定形状的数组，数组元素以 1 来填充
  
  arr=np.ones((5,5),dtype=np.float16)
  print(arr)
  
  """
  [[1. 1. 1. 1. 1.]
   [1. 1. 1. 1. 1.]
   [1. 1. 1. 1. 1.]
   [1. 1. 1. 1. 1.]
   [1. 1. 1. 1. 1.]]
  """
  ```

#### 1.4、numpy.zeros_like

- 语法

  numpy.zeros_like 用于创建一个与给定数组具有相同形状的数组，数组元素以 0 来填充。

  ```python
  numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
  ```

- 参数说明

  | 参数  | 描述                                                         |
  | ----- | ------------------------------------------------------------ |
  | a     | 给定要创建相同形状的数组                                     |
  | dtype | 创建的数组的数据类型                                         |
  | order | 数组在内存中的存储顺序，可选值为 'C'（按行优先）或 'F'（按列优先），默认为 'K'（保留输入数组的存储顺序） |
  | subok | 是否允许返回子类，如果为 True，则返回一个子类对象，否则返回一个与 a 数组具有相同数据类型和存储顺序的数组 |
  | shape | 创建的数组的形状，如果不指定，则默认为 a 数组的形状。        |

- 代码实现

  ```python
  arr=np.array([[1,4,7],[2,5,8],[3,6,9]])
  print(arr)
  print(arr.shape)
  """
  [[1 4 7]
   [2 5 8]
   [3 6 9]]
   
  (3, 3)
  
  """
  array_=np.zeros_like(arr,dtype=np.int16)
  print(array_)
  print(array_.shape)
  """
  [[0 0 0]
   [0 0 0]
   [0 0 0]]
   
  (3, 3)
  
  """
  ```

#### 1.5、numpy.ones_like

- 语法

  numpy.ones_like 用于创建一个与给定数组具有相同形状的数组，数组元素以 1 来填充.

  ```
  numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)
  ```

- 参数说明

  | 参数  | 描述                                                         |
  | ----- | ------------------------------------------------------------ |
  | a     | 给定要创建相同形状的数组                                     |
  | dtype | 创建的数组的数据类型                                         |
  | order | 数组在内存中的存储顺序，可选值为 'C'（按行优先）或 'F'（按列优先），默认为 'K'（保留输入数组的存储顺序） |
  | subok | 是否允许返回子类，如果为 True，则返回一个子类对象，否则返回一个与 a 数组具有相同数据类型和存储顺序的数组 |
  | shape | 创建的数组的形状，如果不指定，则默认为 a 数组的形状。        |

- 代码实现

  ```python
  arr=np.array([[1,4,7],[2,5,8],[3,6,9]])
  print(arr)
  print(arr.shape)
  """
  [[1 4 7]
   [2 5 8]
   [3 6 9]]
   
  (3, 3)
  
  """
  array_=np.ones_like(arr,dtype=np.int16)
  print(array_)
  print(array_.shape)
  """
  [[1 1 1]
   [1 1 1]
   [1 1 1]]
   
  (3, 3)
  
  """
  ```

### 2.从已有数组创建数组

#### 2.1、numpy.asarray

- 语法

  ```python
  numpy.asarray(a, dtype = None, order = None)
  ```

- 参数说明

  | 参数  | 描述                                                         |
  | ----- | ------------------------------------------------------------ |
  | a     | 任意形式的输入参数，可以是，列表, 列表的元组, 元组, 元组的元组, 元组的列表，多维数组 |
  | dtype | 数据类型，可选                                               |
  | order | 可选，有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序。 |

- 代码实现

  ```python
  #列表转换ndarray，数据类型为int
  
  x=[1,7,5,9,1,2]
  arr=np.asarray(x,dtype=np.int16)
  print(arr)
  ```

  ```python
  #元组转换ndarray，数据类型为float
  
  x=(1,7,5,9,1,2)
  arr=np.asarray(x,dtype=np.float16)
  print(arr)
  ```

  ```python
  #元组列表转换ndarray，数据类型为float
  
  x=[(1,7,5),(9,1,2)]
  arr=np.asarray(x,dtype=np.float16)
  print(arr)
  """
  [[1. 7. 5.]
   [9. 1. 2.]]
  """
  ```

#### 2.2、numpy.fromiter

- 语法

  numpy.fromiter 方法从可迭代对象中建立 ndarray 对象，返回一维数组

  ```python
  numpy.fromiter(iterable, dtype, count=-1)
  ```

- 参数说明

  | 参数     | 描述                                   |
  | -------- | -------------------------------------- |
  | iterable | 可迭代对象                             |
  | dtype    | 返回数组的数据类型                     |
  | count    | 读取的数据数量，默认为-1，读取所有数据 |

- 代码实现

  ```python
  x=range(10)
  arr=np.fromiter(x,dtype=np.float16)
  print(arr)
  """
  [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
  """
  ```

### 3.从数值范围创建数组

#### 3.1、numpy.arange

- 语法

  numpy 包中的使用 arange 函数创建数值范围并返回 ndarray 对象

  ```python
  numpy.arange(start, stop, step, dtype)
  ```

- 参数说明

  | 参数    | 描述                                                         |
  | ------- | ------------------------------------------------------------ |
  | `start` | 起始值，默认为`0`                                            |
  | `stop`  | 终止值（不包含）                                             |
  | `step`  | 步长，默认为`1`                                              |
  | `dtype` | 返回`ndarray`的数据类型，如果没有提供，则会使用输入数据的类型。 |

- 代码实现

  ```python
  arr=np.arange(5,dtype=np.float16)
  print(arr)
  """
  [0. 1. 2. 3. 4.]
  """
  arr=np.arange(3,10,dtype=np.int16)
  print(arr)
  """
  [3 4 5 6 7 8 9]
  """
  arr=np.arange(3,10,2,dtype=np.float16)
  print(arr)
  """
  [3. 5. 7. 9.]
  """
  
  ```

#### 3.2、numpy.linspace

- 语法

  numpy.linspace 函数用于创建一个一维数组，数组是一个等差数列构成的

  ```python
  np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
  ```

- 参数说明

  | 参数       | 描述                                                         |
  | ---------- | ------------------------------------------------------------ |
  | `start`    | 序列的起始值                                                 |
  | `stop`     | 序列的终止值，如果`endpoint`为`true`，该值包含于数列中       |
  | `num`      | 要生成的等步长的样本数量，默认为`50`                         |
  | `endpoint` | 该值为 `true` 时，数列中包含`stop`值，反之不包含，默认是True。 |
  | `retstep`  | 如果为 True 时，生成的数组中会显示间距，反之不显示。         |
  | `dtype`    | `ndarray` 的数据类型                                         |

- 代码实现

  ```python
  arr=np.linspace(0,10,num=5,dtype=np.int16)
  print(arr)
  """
  [ 0  2  5  7 10]
  """
  
  arr=np.linspace(0,20,num=10,retstep=True,dtype=np.int16)
  print(arr)
  """
  (array([ 0,  2,  4,  6,  8, 11, 13, 15, 17, 20], dtype=int16), 2.2222222222222223)
  """
  ```

## NumPy切片和索引

- **ndarray对象的内容可以通过索引或切片来访问和修改**，**与 Python 中 list 的切片操作一样。**

- **ndarray 数组可以基于 0 - n 的下标进行索引，切片对象可以通过内置的 slice 函数，并设置 start, stop 及 step 参数进行，从原数组中切割出一个新数组。**

  ```python
  #单维数组切片及索引
  arr=np.arange(10)#[0 1 2 3 4 5 6 7 8 9]
  
  s=slice(1,9,2)
  print(arr[s])#[1 3 5 7]
  print(arr[1:9:2])#[1 3 5 7]
  print(arr[1:6])#[1 2 3 4 5]
  print(arr[7])#7
  ```

  ```python
  #多维数组切片及索引
  arr=np.array([[1,4,7],[2,5,8],[3,6,9]])
  """
  [[1 4 7]
   [2 5 8]
   [3 6 9]]
  """
  print(arr[1:])
  """
  [[2 5 8]
   [3 6 9]]
  """
  print(arr[0:2])
  """
  [[1 4 7]
   [2 5 8]]
  
  """
  print(arr[...,1])
  """
  [4 5 6]
  """
  print(arr[1,...])
  """
  [2 5 8]
  """
  print(arr[1:,...])
  """
  [[2 5 8]
   [3 6 9]]
  """
  ```

## NumPy高级索引

​		NumPy 中的高级索引指的是使用整数数组、布尔数组或者其他序列来访问数组的元素。相比于基本索引，高级索引可以访问到数组中的任意元素，并且可以用来对数组进行复杂的操作和修改。

### 1.整数数组索引

​		整数数组索引是指使用一个数组来访问另一个数组的元素。这个数组中的每个元素都是目标数组中某个维度上的索引值。

```python
arr=np.array([[1,4,7],[2,5,8],[3,6,9]])
"""
[[1 4 7]
 [2 5 8]
 [3 6 9]]
"""
print(arr[[0,1],[2,2]])#在数组中获取（0,2）,(1,2)位置处的元素
"""
[7 8]
"""

#获取四个角元素
row=[0,0,2,2]
col=[0,2,0,2]
print(arr[row,col])#(0,0)(0,2)(2,0)(2,2)
"""
[1 7 3 9]
"""
#切片与索引数组结合
a=arr[0:2,1:3]
"""
[[4 7]
 [5 8]]
"""
b=arr[0:2,[0,1]]
"""
[[1 4]
 [2 5]]
"""
c=arr[...,[0,1]]
"""
[[1 4]
 [2 5]
 [3 6]]
"""
```

### 2.布尔索引

布尔索引通过布尔运算（如：比较运算符）来获取符合指定条件的元素的数组

```python
arr=np.array([[1,4,7],[2,5,8],[3,6,9]])
"""
[[1 4 7]
 [2 5 8]
 [3 6 9]]
"""
#获取大于5的元素
print(arr[arr>5])
"""
[7 8 6 9]
"""
arr=np.array([[1,4,7],[2,np.nan,8],[3,np.nan,9]])
"""
[[ 1.  4.  7.]
 [ 2. nan  8.]
 [ 3. nan  9.]]
"""
#过滤掉np.nan
print(arr[~np.isnan(arr)])
"""
[1. 4. 7. 2. 8. 3. 9.]
"""
```

### 3.花式索引

- 花式索引指的是利用**整数数组**进行索引。

- **花式索引根据索引数组的值作为目标数组的某个轴的下标来取值。**
- 对于使用一维整型数组作为索引，如果目标是一维数组，那么索引的结果就是对应位置的元素，如果目标是二维数组，那么就是对应下标的行。

- 花式索引跟切片不一样，它总是将数据复制到新数组中。

### 4.花式索引一维数组

```python
#花式索引一维数组，索引的结果就是对应位置的元素
arr=np.arange(10)#[0 1 2 3 4 5 6 7 8 9]
print(arr[[1,6]])#[1 6]
print(arr[[5]])#[5]
```

### 5.花式索引二维数组

```python
#花式索引二维数组，索引的结果就是对应下标的行
arr=np.array([[1,4,7],[2,5,8],[3,6,9]])
"""
[[1 4 7]
 [2 5 8]
 [3 6 9]]
"""
print(arr[[0,1]])
"""
[[1 4 7]
 [2 5 8]]
"""
print(arr[[-1,-2]])
"""
[[3 6 9]
 [2 5 8]]
"""
```

```python
#传入多个索引数组
"""

np.ix_() 函数就是输入两个数组，产生笛卡尔积的映射关系。

笛卡尔乘积是指在数学中，两个集合 X 和 Y 的笛卡尔积（Cartesian product），又称直积，表示为 **X×Y**，第一个对象是X的成员而第二个对象是 Y 的所有可能有序对的其中一个成员。
例如 **A={a,b}, B={0,1,2}**，则：

A×B={(a, 0), (a, 1), (a, 2), (b, 0), (b, 1), (b, 2)}
B×A={(0, a), (0, b), (1, a), (1, b), (2, a), (2, b)}
"""
arr=np.arange(64).reshape(8,8)
"""
[[ 0  1  2  3  4  5  6  7]
 [ 8  9 10 11 12 13 14 15]
 [16 17 18 19 20 21 22 23]
 [24 25 26 27 28 29 30 31]
 [32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47]
 [48 49 50 51 52 53 54 55]
 [56 57 58 59 60 61 62 63]]
"""
print(arr[np.ix_([1,3,5,7],[4,5,6,7])])
"""
[[12 13 14 15]
 [28 29 30 31]
 [44 45 46 47]
 [60 61 62 63]]
"""
```
## NumPy广播（Broadcast）

### 1.定义

​		广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式， 对数组的算术运算通常在相应的元素上进行。

### 2.广播的规则

​	**广播的规则:**

- 让所有输入数组都向其中形状最长的数组看齐，形状中不足的部分都通过在前面加 1 补齐。
- 输出数组的形状是输入数组形状的各个维度上的最大值。
- 如果输入数组的某个维度和输出数组的对应维度的长度相同或者其长度为 1 时，这个数组能够用来计算，否则出错。
- 当输入数组的某个维度的长度为 1 时，沿着此维度运算时都用此维度上的第一组值。

**简单理解：**对两个数组，分别比较他们的每一个维度（若其中一个数组没有当前维度则忽略），满足：

- 数组拥有相同形状。
- 当前维度的值相等。
- 当前维度的值有一个是 1。

若条件不满足，抛出 **"ValueError: frames are not aligned"** 异常。

### 3.运算数组形状相同时

```python
#如果两个数组 a 和 b 形状相同，即满足 **a.shape == b.shape**，那么 a*b 的结果就是 a 与 b 数组对应位相乘。
arr1=np.array([[1,2],[3,4]])
"""
[[1 2]
 [3 4]]
"""
arr2=np.array([[5,6],[7,8]])
"""
[[5 6]
 [7 8]]
"""
result=arr2*arr1
"""
[[ 5 12]
 [21 32]]
"""
```

### 4.运算数组形状不相同时

```python
#当运算中的 2 个数组的形状不同时，numpy 将自动触发广播机制。
arr1=np.array([[1,2],[3,4]])
"""
[[1 2]
 [3 4]]
"""
arr2=np.array([5,6])
"""
[5 6]
"""
result=arr1*arr2
"""
[[ 5 12]
 [15 24]]
"""
"""
#arr2广播自动补齐为：
[[5 6]
 [5 6]]
"""
```

## NumPy副本和视图

- 副本是一个数据的完整的拷贝，对副本进行修改，它不会影响到原始数据

- 视图是数据的一个别称或引用，对视图进行修改，它会影响到原始数据

**视图(浅拷贝)一般发生在：**

- 1、numpy 的切片操作返回原数据的视图。
- 2、调用 ndarray 的 view() 函数产生一个视图。

**副本(深拷贝)一般发生在：**

- Python 序列的切片操作，调用deepCopy()函数。
- 调用 ndarray 的 copy() 函数产生一个副本。

### 1.无复制

```python
#简单的赋值不会创建数组对象的副本。 相反，它使用原始数组的相同id()来访问它。

arr=np.arange(10)
print(arr,id(arr))#[0 1 2 3 4 5 6 7 8 9] 2485302504112
arr_c=arr
print(arr_c,id(arr_c))#[0 1 2 3 4 5 6 7 8 9] 2485302504112
arr_c[0]=1316
print(arr)#[1316    1    2    3    4    5    6    7    8    9]
print(arr_c)#[1316    1    2    3    4    5    6    7    8    9]
```

### 2.视图或浅拷贝

```python
#使用ndarray.view() 实现数组的浅拷贝
arr=np.arange(10)
print(arr,id(arr))
#[0 1 2 3 4 5 6 7 8 9] 2485302504112

arr_c=arr.view()
print(arr_c,id(arr_c))
#[0 1 2 3 4 5 6 7 8 9] 2485302504112

arr_c[0]=1316
print(arr)
#[1316    1    2    3    4    5    6    7    8    9]
print(arr_c)
#[1316    1    2    3    4    5    6    7    8    9]
```

```python
#数组切片也是一种浅拷贝
arr=np.arange(6)
print(arr)
#[0 1 2 3 4 5]
a=arr[1:3]
#[1 2]
a[0]=666
print(a)
#[666   2]
print(arr)
#[  0 666   2   3   4   5]
```

### 3.副本或深拷贝

```python
#ndarray.copy() 函数创建一个副本。 对副本数据进行修改，不会影响到原始数据
arr=np.arange(6)
print(arr,id(arr))
#[0 1 2 3 4 5] 2756783745616
arr_c=arr.copy()
print(arr_c,id(arr_c))
#[0 1 2 3 4 5] 2756783745712
arr_c[0]=1316
print(arr)
#[0 1 2 3 4 5]
print(arr_c)
#[1316    1    2    3    4    5]
```

## NumPy数组操作

### 1.修改数组形状

#### 1.1、ndarray.reshape

- 语法

  ```python
  numpy.reshape(arr, newshape, order='C')
  ```

- 参数说明

  - `arr`：要修改形状的数组
  - `newshape`：整数或者整数数组，新的形状应当兼容原有形状
  - `order`：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'k' -- 元素在内存中的出现顺序。

- 代码实现

  ```python
  arr=np.arange(10)
  print(arr)
  #[0 1 2 3 4 5 6 7 8 9]
  arr=np.reshape(arr,(2,5))
  print(arr)
  """
  [[0 1 2 3 4]
   [5 6 7 8 9]]
  """
  ```

#### 1.2、ndarray.flat

- 语法

  ```python
  ndarray.flat是一个数组元素迭代器
  ```

- 代码实现

  ```py
  arr=np.arange(10).reshape(2,5)
  print(arr)
  """
  [[0 1 2 3 4]
   [5 6 7 8 9]]
  """
  for i in arr.flat:
      print(i)
  """
  0
  1
  2
  3
  4
  5
  6
  7
  8
  9
  """
  ```

#### 1.3、ndarray.flatten

- 语法

  ndarray.flatten 返回一份数组拷贝，对拷贝所做的修改不会影响原始数组

  ```python
  ndarray.flatten(order='C')
  ```

- 参数说明

  order(可选)：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'K' -- 元素在内存中的出现顺序。

- 代码实现

  ```python
  #ndarray.flatten，是个方法可直接用
  arr=np.arange(10).reshape(2,5)
  print(arr)
  """
  [[0 1 2 3 4]
   [5 6 7 8 9]]
  """
  arr=arr.flatten()
  print(arr)
  #[0 1 2 3 4 5 6 7 8 9]
  ```

#### 1.4、ndarray.ravel

- 语法

  numpy.ravel() 展平的数组元素(扁平化)，返回的是**数组视图****（修改会影响原始数组）**

  ```python
  ndarray.ravel(a, order='C')
  ```

- 参数说明

  order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'K' -- 元素在内存中的出现顺序。

- 代码实现

  ```python
  arr=np.arange(10).reshape(2,5)
  print(arr)
  """
  [[0 1 2 3 4]
   [5 6 7 8 9]]
  """
  cout=np.ravel(arr)
  print(cout)
  """
  [0 1 2 3 4 5 6 7 8 9]
  """
  cout[5]=100
  print(cout)
  print(arr)
  """
  [  0   1   2   3   4 100   6   7   8   9]
  
  [[  0   1   2   3   4]
   [100   6   7   8   9]]
  
  """
  ```

### 2.反转数组

| 函数        | 描述 |
| ----------- | ---- |
| `transpose` | 转置 |
| `ndarray.T` | 转置 |

#### 2.1、numpy.transpose

- 语法

  ```python
  numpy.transpose(arr, axes)
  ```

- 参数说明

  - `arr`：要操作的数组
  - `axes`：整数列表，对应维度，通常所有维度都会对换。

- 代码实现

  ```python
  arr=np.arange(10).reshape(2,5)
  print(arr)
  """
  [[0 1 2 3 4]
   [5 6 7 8 9]]
  """
  cout=np.transpose(arr)
  print(cout)
  """
  [[0 5]
   [1 6]
   [2 7]
   [3 8]
   [4 9]]
  """
  ```

#### 2.2、ndarray.T

- 语法

  ```python
  ndarray.T#类似于numpy.transpose
  ```

- 代码实现

  ```python
  arr=np.arange(10).reshape(2,5)
  print(arr)
  """
  [[0 1 2 3 4]
   [5 6 7 8 9]]
  """
  cout=arr.T
  print(cout)
  """
  [[0 5]
   [1 6]
   [2 7]
   [3 8]
   [4 9]]
  """
  ```

### 3.连接数组

#### 3.1、numpy.concatenate

- 语法

  numpy.concatenate 函数用于沿指定轴连接**相同形状**的两个或多个数组

  ```python
  numpy.concatenate((a1, a2, ...), axis)
  ```

- 参数说明

  - `a1, a2, ...`：相同类型的数组
  - `axis`：沿着它连接数组的轴，默认为 0

- 代码实现

  ```python
  arr=np.arange(10).reshape(2,5)
  print(arr)
  """
  [[0 1 2 3 4]
   [5 6 7 8 9]]
  """
  next=np.arange(10,20).reshape(2,5)
  print(next)
  """
  [[10 11 12 13 14]
   [15 16 17 18 19]]
  """
  total=np.concatenate((arr,next),axis=0)#沿行轴
  print(total)
  """
  [[ 0  1  2  3  4]
   [ 5  6  7  8  9]
   [10 11 12 13 14]
   [15 16 17 18 19]]
  """
  total=np.concatenate((arr,next),axis=1)#沿列轴
  print(total)
  """
  [[ 0  1  2  3  4 10 11 12 13 14]
   [ 5  6  7  8  9 15 16 17 18 19]]
  """
  ```

### 4.分割数组

| 函数     | 数组及操作                             |
| -------- | -------------------------------------- |
| `split`  | 将一个数组分割为多个子数组             |
| `hsplit` | 将一个数组水平分割为多个子数组（按列） |
| `vsplit` | 将一个数组垂直分割为多个子数组（按行） |

#### 4.1、numpy.split

- 语法

  numpy.split 函数沿特定的轴将数组分割为子数组

  ```python
  numpy.split(ary, indices_or_sections, axis)
  ```

- 参数说明

  - `ary`：被分割的数组
  - `indices_or_sections`：如果是一个整数，就用该数平均切分，如果是一个数组，为沿轴切分的位置（左开右闭）
  - `axis`：设置沿着哪个方向进行切分，默认为 0，横向切分，即水平方向。为 1 时，纵向切分，即竖直方向。

- 代码实现

  ```python
  #一维数组分割
  arr=np.arange(10)
  print(arr)
  #[0 1 2 3 4 5 6 7 8 9]
  arr=np.split(arr,2)
  print(arr)
  #[array([0, 1, 2, 3, 4]), array([5, 6, 7, 8, 9])]
  
  arr_c=np.arange(10)
  #[0 1 2 3 4 5 6 7 8 9]
  arr=np.split(arr_c,[2,6])
  print(arr)
  #[array([0, 1]), array([2, 3, 4, 5]), array([6, 7, 8, 9])]
  ```

  ```python
  #多维数组，axis=1水平分割，axis=0纵向分割
  arr=np.arange(16).reshape(4,4)
  print(arr)
  """
  [[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]
   [12 13 14 15]]
  """
  arr=np.split(arr,2,axis=1)
  print(arr)
  """
  [array([[ 0,  1],
         [ 4,  5],
         [ 8,  9],
         [12, 13]]), 
         
   array([[ 2,  3],
         [ 6,  7],
         [10, 11],
         [14, 15]])]
  """
  
  arr=np.split(arr,2,axis=0)
  print(arr)
  """
  [array([[0, 1, 2, 3],
         [4, 5, 6, 7]]), 
   array([[ 8,  9, 10, 11],
         [12, 13, 14, 15]])]
  
  """
  ```

#### 4.2、numpy.hsplit

- 语法

  numpy.hsplit 函数用于水平分割数组

  ```python
  numpy.hsplit
  ```

- 代码实现

  ```python
  arr=np.arange(16).reshape(4,4)
  print(arr)
  """
  [[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]
   [12 13 14 15]]
  """
  arr_h=np.hsplit(arr,2)
  print(arr_h)
  """
  [array([[ 0,  1],
         [ 4,  5],
         [ 8,  9],
         [12, 13]]), 
         
   array([[ 2,  3],
         [ 6,  7],
         [10, 11],
         [14, 15]])]
  """
  ```

#### 4.3、numpy.vsplit

- 语法

  numpy.vsplit 沿着垂直轴分割

  ```python
  numpy.vsplit
  ```

- 代码实现

  ```python
  arr=np.arange(16).reshape(4,4)
  print(arr)
  """
  [[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]
   [12 13 14 15]]
  """
  arr_v=np.vsplit(arr,2)
  print(arr_v)
  """
  [array([[0, 1, 2, 3],
         [4, 5, 6, 7]]), 
   array([[ 8,  9, 10, 11],
         [12, 13, 14, 15]])]
  """
  ```

  

## NumPy数学函数

#### 1、三角函数

```python
#NumPy 提供了标准的三角函数：sin()、cos()、tan()等
#np.pi/180=转化为弧度

arr=np.array([0,45,90,180])
sin=np.sin(arr*np.pi/180)
cos=np.cos(arr*np.pi/180)
tan=np.tan(arr*np.pi/180)
print(sin)
#[0.00000000e+00 7.07106781e-01 1.00000000e+00 1.22464680e-16]
print(cos)
#[1.00000000e+00 7.07106781e-01 6.12323400e-17 -1.00000000e+00]
print(tan)
#[0.00000000e+00 1.00000000e+00  1.63312394e+16 -1.22464680e-16]
```

```python
#numpy.degrees()可将反三角函数arcsin，arccos,arctan弧度转换为角度
arr=np.array([0,45,90,180])
sin=np.sin(arr*np.pi/180)
cos=np.cos(arr*np.pi/180)
tan=np.tan(arr*np.pi/180)

ars=np.arcsin(sin)
arc=np.arccos(cos)
art=np.arctan(tan)

print(np.degrees(ars))
#[0.0000000e+00 4.5000000e+01 9.0000000e+01 7.0167093e-15]
print(np.degrees(arc))
#[  0.  45.  90. 180.]
print(np.degrees(art))
#[ 0.0000000e+00  4.5000000e+01  9.0000000e+01 -7.0167093e-15]
```

#### 2、舍入函数

##### 2.1、numpy.around()

- 语法

  numpy.around() 函数返回指定数字的四舍五入值

  ```python
  numpy.around(a,decimals)
  ```

- 参数说明

  - a: 数组
  - decimals: 舍入的小数位数。 默认值为0。 如果为负，整数将四舍五入到小数点左侧的位置

- 代码实现

  ```python
  arr=np.array([1.52,5.6,2.21,7.1,6.7955])
  arr=np.around(arr,decimals=1)
  print(arr)
  #[1.5 5.6 2.2 7.1 6.8]
  ```

##### 2.2、numpy.floor() 

- 语法

  numpy.floor() 返回小于或者等于指定表达式的最大整数，即向下取整

  ```python
  numpy.floor(arr) 
  ```

- 代码实现

  ```python
  arr=np.array([1.52,5.6,2.21,7.1,6.7955])
  arr=np.floor(arr)
  print(arr)
  #[1. 5. 2. 7. 6.]
  ```

##### 2.3、numpy.ceil()

- 语法

- numpy.ceil() 返回大于或者等于指定表达式的最小整数，即向上取整

  ```python
  numpy.ceil(arr)
  ```

- 代码实现

  ```python
  arr=np.array([1.52,5.6,2.21,7.1,6.7955])
  arr=np.ceil(arr)
  print(arr)
  #[2. 6. 3. 8. 7.]
  ```

#### 3、随机数函数

##### 3.1、numpy.random.seed()

- 语法

  给定的种子值，使用相同的随机数种子可以得到相同的随机数

  ```python
  numpy.random.seed(10)
  ```

##### 3.2、numpy.random.random()

- 语法

  返回一个值在[0.0, 1.0)内的随机浮点数或N维浮点数组

  ```python
  numpy.random.random(size=None)
  ```

- 代码实现

  ```python
  arr=np.random.random((2, 2)) #生成2行2列从[0,1)中随机选取的浮点数
  print(arr)
  """
  [[0.24642123 0.46204823]
   [0.65305885 0.33316291]]
  """
  ```

##### 3.3、numpy.random.randint()

- 语法

  返回一个随机整数或N维整数数组，若high不为None时，取数范围[low,high)之间的随机整数，否则取[0,low)之间的随机整数。

  ```python
  numpy.random.randint(low, high=None, size=None, dtype=np.int16)
  ```

- 代码实现

  ```python
  arr=np.random.randint(3, size=3)
  tcc=np.random.randint(2, high=10, size=(2,2))
  print(arr)
  #[0 2 2]
  print(tcc)
  """
  [[5 3]
   [7 4]]
  """
  ```

##### 3.4、numpy.random.choice()

- 语法

  从a中以概率p随机抽取元素，返回一个随机数 或 指定大小的ndarray数组

  ```python
  numpy.random.choice(a, size=None, replace=True, p=None)
  ```

- 参数说明

  - a：一维的ndarray数组/列表/元组/字符串或一个整数值。若a为整数值，则随机抽取np.range(a)中的元素；若a为数组/列表/元组/字符串，则随机抽取a中的元素。
  - size：返回样本的大小，默认1。
  - replace：True表示可以取相同元素，False表示不允许有重复值。
  - p：表示取a中每个元素的概率，默认选取每个元素的概率相同，大小与a相同。

- 代码实现

  ```python
  arr=np.arange(10)
  print(arr)
  #[0 1 2 3 4 5 6 7 8 9]
  x=np.random.choice(arr)
  print(x)
  #5
  tcc=np.random.choice(arr,size=(2,2))
  print(tcc)
  """
  [[3 9]
   [9 2]]
  """
  ```

##### 3.5、numpy.random.shuffle()

- 语法

  对x进行重排序，改变x本身，无返回值（返回None）；
  对于多维数组，只对第一维进行重排序，而不改变其他维度元素的顺序。

  ```python
  numpy.random.shuffle(x)
  ```

- 代码实现

  ```python
  x = [1,2,3,4]
  a=np.random.shuffle(x) 
  print(a)  # None
  print(x)  # [4, 1, 3, 2]
  ```

##### 3.6、numpy.random.uniform()

- 语法

  从一个均匀分布[low,high)中随机采样

  ```python
  numpy.random.uniform(low=0.0, high=1.0, size=None)
  ```

- 参数说明

  size: 输出样本数目，为int或元组(tuple)类型，缺省时输出1个值

- 代码实现

  ```python
  arr=np.random.uniform(0,10)
  print(arr)
  #8.09838607762593
  tcc=np.random.uniform(0.0, 1.0, size=(2,2))
  print(tcc)
  """
  [[0.80768654 0.07585118]
   [0.22854273 0.54718549]]
  """
  ```

##### 3.7、numpy.random.rand()

- 语法

- 0-1均匀分布：返回一个在[0,1)上均匀分布的随机浮点数或随机N维浮点数组

  ```python
  numpy.random.rand(d0, d1, ..., dn)
  ```

- 参数说明

  当函数括号内没有参数时，则返回一个浮点数；

  当函数括号内有一个参数时，则返回秩为1的数组，不能表示向量和矩阵；
  当函数括号内有两个及以上参数时，则返回对应维度的数组，能表示向量或矩阵

- 代码实现

  ```python
  np.random.rand()  # 0.4316369830815895
  np.random.rand(1)  # array([0.35502719])
  np.random.rand(2)  # array([0.4837686, 0.0589509])
  np.random.rand(2,2) 
  # array([[0.36035899, 0.15746599],[0.92904081, 0.78280208]])
  ```

##### 3.8、numpy.random.randn()

- 语法

  标准正态分布：返回一个服从标准正态分布的随机浮点数或随机N维浮点数组

  ```python
  numpy.random.randn(d0, d1, ..., dn)
  ```

- 参数说明

  输入通常为整数，但是如果为浮点数，则会自动直接截断转换为整数；
  标准正态分布是以0为均值、以1为标准差的正态分布，记为N(0,1)

- 代码实现

  ```python
  arr=np.random.randn(3)
  print(arr)
  #[ 0.92294172 -0.85579141  1.88739405]
  ```

##### 3.9、numpy.random.normal()

- 语法

  正态分布：从正态分布（高斯分布）中抽取随机样本，返回一个size大小的数组。

  ```python
  numpy.random.normal(loc=0.0, scale=1.0, size=None)
  ```

- 参数说明

  > loc：均值（数学期望） μ
  > scale：标准差 σ ，方差 σ2

- 代码实现

  ```python
  arr=np.random.normal(loc=0.0, scale=1.0)
  tcc=np.random.normal(5,9)
  bdd=np.random.normal(size=(2, 4))
  print(arr)
  #-0.28641454622082635
  print(tcc)
  #20.67116257829486
  print(bdd)
  """
  [[ 1.06432688  0.32725135  2.04405066  0.63729957]
   [-1.50038329  0.24918225 -1.79831511 -1.23835005]]
  """
  ```

##### 3.10、numpy.random.standard_normal()

- 语法

  标准正态分布：与np.random.randn()一样的功能，返回一个服从标准正态分布的随机浮点数或N维浮点数组

  ```python
  numpy.random.standard_normal(size=None)
  ```

- 参数说明

  不同的是它的shape由size参数指定，对于多维数组，size必须是元组形式。

- 代码实现

  ```python
  arr=np.random.standard_normal()
  tcc=np.random.standard_normal(4)
  bdd=np.random.standard_normal(size=(2, 4))
  print(arr)
  #0.6063328986898614
  print(tcc)
  #[ 1.15663018  1.05697384 -0.25251242 -0.92866212]
  print(bdd)
  """
  [[ 0.58690998  0.72101993  0.71599569 -0.27951657]
   [ 0.21214347  0.6327988  -0.07155567 -1.76459184]]
  """
  ```

## NumPy算术函数

- NumPy 算术函数包含简单的加减乘除: **add()**，**subtract()**，**multiply()** 和 **divide()**。
- **需要注意的是数组必须具有相同的形状或符合数组广播规则。**

#### 1、numpy.reciprocal()

- 语法

  numpy.reciprocal() 函数返回参数逐元素的倒数。如 **1/4** 倒数为 **4/1**。

  ```python
  numpy.reciprocal(arr)
  ```

- 代码实现

  ```python
  arr=np.array([0.25,0.5,2,10])
  arr=np.reciprocal(arr)
  print(arr)
  #[4.  2.  0.5 0.1]
  ```

#### 2、numpy.power()

- 语法

  numpy.power() 函数将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂。

  ```python
  numpy.power(a,b)
  ```

- 代码实现

  ```python
  arr=np.arange(5)
  arr=np.power(arr,2)
  print(arr)
  #[ 0  1  4  9 16]
  ```

#### 3、numpy.mod()

- 语法

  numpy.mod() 计算输入数组中相应元素的相除后的余数

  ```python
  numpy.mod(a,b)#a（mode）b
  ```

- 代码实现

  ```python
  a=np.arange(1,6)
  print(a)
  #[1 2 3 4 5]
  b=np.arange(2,7)
  print(b)
  #[2 3 4 5 6]
  print(np.mod(b,a))
  #[0 1 1 1 1]
  ```

## NumPy矩阵库

#### 1、转置矩阵

- 语法

  m 行 n 列的矩阵，使用 array.T就能转换为 n 行 m 列的矩阵

  ```
  array.T
  ```

- 代码实现

  ```python
  arr=np.arange(20).reshape(4,5)
  print(arr)
  """
  [[ 0  1  2  3  4]
   [ 5  6  7  8  9]
   [10 11 12 13 14]
   [15 16 17 18 19]]
  """
  tcc=arr.T
  print(tcc)
  """
  [[ 0  5 10 15]
   [ 1  6 11 16]
   [ 2  7 12 17]
   [ 3  8 13 18]
   [ 4  9 14 19]]
  """
  ```

#### 2、matlib.empty()

- 语法

  matlib.empty() 函数返回一个新的矩阵

  ```
  numpy.empty(shape, dtype, order)
  ```

- 参数说明

  - **shape**: 定义新矩阵形状的整数或整数元组
  - **Dtype**: 可选，数据类型
  - **order**: C（行序优先） 或者 F（列序优先）

- 代码实现

  ```python
  arr=np.matlib.empty((5,5),dtype=np.int16)
  print(arr)
  """
  [[ 25613 -31730  23040  25610  25615]
   [-31728  23040  25611  25617 -31726]
   [ 23040  25612  21251  10496  -9708]
   [ 18436  16717  31299  21118  17222]
   [ 12832  12337   8244  19784  17217]]
  """
  ```

#### 3、numpy.zeros()

```python
#numpy.zeros() 函数创建一个以 0 填充的矩阵

arr=np.zeros((2,2))
print(arr)
"""
[[0. 0.]
 [0. 0.]]
"""
```

#### 4、numpy.ones()

```python
#numpy.zeros() 函数创建一个以 0 填充的矩阵

arr=np.ones((2,2))
print(arr)
"""
[[1. 1.]
 [1. 1.]]
"""
```

#### 5、numpy.eye()

- 语法

  numpy.eye() 函数返回一个矩阵，对角线元素为 1，其他位置为零

  ```
  numpy.eye(n, M,k, dtype)
  ```

- 参数说明

  - **n**: 返回矩阵的行数
  - **M**: 返回矩阵的列数，默认为 n
  - **k**: 对角线的索引
  - **dtype**: 数据类型

- 代码实现

  ```python
  arr=np.eye(3,3,0,dtype=np.int16)
  print(arr)
  """
  [[1 0 0]
   [0 1 0]
   [0 0 1]]
  """
  ```

#### 6、numpy.identity()

- numpy.identity() 函数返回给定大小的单位矩阵

- 单位矩阵是个方阵，从左上角到右下角的对角线（称为主对角线）上的元素均为 1，除此以外全都为 0。

- 代码实现

  ```python
  arr=np.identity(5,dtype=np.int16)
  print(arr)
  """
  [[1 0 0 0 0]
   [0 1 0 0 0]
   [0 0 1 0 0]
   [0 0 0 1 0]
   [0 0 0 0 1]]
  """
  ```


## NumPy线性代数

| 函数          | 描述                 |
| ------------- | -------------------- |
| `dot`         | 两个数组的点积。     |
| `vdot`        | 两个向量的点积       |
| `inner`       | 两个数组的内积       |
| `matmul`      | 两个数组的矩阵积     |
| `determinant` | 数组的行列式         |
| `solve`       | 求解线性矩阵方程     |
| `inv`         | 计算矩阵的乘法逆矩阵 |

#### 1、numpy.dot()

- 语法

  ​		numpy.dot() 对于两个一维的数组，计算的是这两个数组对应下标元素的乘积和(数学上称之为向量点积)；

  ​		对于二维数组，计算的是两个数组的矩阵乘积；

  ​		对于多维数组，它的通用计算公式如下，即结果数组中的每个元素都是：数组a的最后一维上的所有元素与数组b的倒数第二位上的所有元素的乘积和

  ```python
  numpy.dot(a, b, out=None) 
  ```

- 参数说明

  - **a** : ndarray 数组
  - **b** : ndarray 数组
  - **out** : ndarray, 可选，用来保存dot()的计算结果

- 代码实现

  ```python
  arr=np.array([1,2])
  tcc=np.array([2,3])
  print(np.dot(arr,tcc))#8
  #1*2+2*3=8
  
  arr=np.arange(4).reshape(2,2)
  print(arr)
  """
  [[0 1]
   [2 3]]
  """
  tcc=np.arange(5,9).reshape(2,2)
  print(tcc)
  """
  [[5 6]
   [7 8]]
  """
  total=np.dot(arr,tcc)
  print(total)
  """
  [[ 7  8]
   [31 36]]
  """
  #[[0*5+1*7, 0*6+1*8],[2*5+3*7, 2*6+3*8]]
  
  ```

#### 2、numpy.vdot()

​	numpy.vdot() 函数是两个向量的点积

- 代码实现

  ```python
  arr=np.arange(4).reshape(2,2)
  print(arr)
  """
  [[0 1]
   [2 3]]
  """
  tcc=np.arange(5,9).reshape(2,2)
  print(tcc)
  """
  [[5 6]
   [7 8]]
  """
  total=np.vdot(arr,tcc)
  print(total)#44
  #0*5+1*6+2*7+3*8=44
  ```


#### 3、numpy.inner()

​		numpy.inner() 函数返回一维数组的向量内积。对于更高的维度，它返回最后一个轴上的和的乘积

- 代码实现

  ```python
  #一维数组
  arr=np.array([1,2])
  tcc=np.array([2,3])
  print(np.dot(arr,tcc))#8
  #1*2+2*3=8
  
  #多维数组
  arr=np.arange(4).reshape(2,2)
  print(arr)
  """
  [[0 1]
   [2 3]]
  """
  tcc=np.arange(5,9).reshape(2,2)
  print(tcc)
  """
  [[5 6]
   [7 8]]
  """
  total=np.inner(arr,tcc)
  print(total)
  """
  [[ 6  8]
   [28 38]]
  """
  #[[0*5+1*6,0*7+1*8],[2*5+3*6,2*7+3*8]]
  ```


#### 4、numpy.matmul

​	numpy.matmul 函数返回两个数组的矩阵乘积。 虽然它返回二维数组的正常乘积对于二维数组，它就是矩阵乘法

- 代码实现

  ```python
  arr=np.array([1,2])
  tcc=np.array([2,3])
  print(np.dot(arr,tcc))#8
  #1*2+2*3=8
  
  arr=np.arange(4).reshape(2,2)
  print(arr)
  """
  [[0 1]
   [2 3]]
  """
  tcc=np.arange(5,9).reshape(2,2)
  print(tcc)
  """
  [[5 6]
   [7 8]]
  """
  total=np.dot(arr,tcc)
  print(total)
  """
  [[ 7  8]
   [31 36]]
  """
  ```

  ```python
  a = [[1,2],[3,4]]
  b = [1,2]
  print (np.matmul(a,b))
  #[ 5 11]
  print (np.matmul(b,a))
  #[ 7 10]
  ```

#### 5、numpy.linalg.det()

​	numpy.linalg.det() 函数计算输入矩阵的行列式

- 代码实现

  ```python
  a = np.array([[1,2],[3,4]])
  print(a)
  """
  [[1 2]
   [3 4]]
  """
  b=np.array([[1,2,3],[4,5,6],[7,8,9]])
  print(b)
  """
  [[1 2 3]
   [4 5 6]
   [7 8 9]]
  """
  print (np.linalg.det(a))#-2.0000000000000004
  print (np.linalg.det(b))#0.0
  ```


#### 6、numpy.linalg.solve()

​		numpy.linalg.solve() 函数给出了矩阵形式的线性方程的解。

- 解以下方程

  ```
  x + 3y + z = 3
  
  3x + 4y + 2z = 9
  
  -1x - 5y + 4z = 10
  
  ```

- 代码实现

  ```python
  a=np.array([[1,3,1],[3,4,2],[-1,-5,4]])
  b=np.array([3,9,10])
  print(np.linalg.solve(a,b))
  #[ 2.03703704 -0.48148148  2.40740741]
  ```

  

#### 7、numpy.linalg.inv()

- numpy.linalg.inv() 函数计算矩阵的乘法逆矩阵

- 代码实现

  ```python
  import numpy as np  
  x = np.array([[1,2],[3,4]]) 
  y = np.linalg.inv(x) 
  print (x)
  print (y)
  print (np.dot(x,y))
  """
  [[1.0000000e+00 0.0000000e+00]
   [8.8817842e-16 1.0000000e+00]]
  """
  ```

  
