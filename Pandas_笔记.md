# Pandas

## Pandas简介

- Pandas 是一个开源的数据分析和数据处理库，它是基于 Python 编程语言的。

- Pandas 提供了易于使用的数据结构和数据分析工具，特别适用于处理结构化数据，如表格型数据（类似于Excel表格）。

- Pandas 是数据科学和分析领域中常用的工具之一，它使得用户能够轻松地从各种数据源中导入数据，并对数据进行高效的操作和分析。
- Pandas 主要引入了两种新的数据结构：**DataFrame** 和 **Series**。

## Pandas 应用

- **数据清洗和预处理：** Pandas被广泛用于清理和预处理数据，包括处理缺失值、异常值、重复值等。它提供了各种方法来使数据更适合进行进一步的分析。
- **数据分析和统计：** Pandas使数据分析变得更加简单，通过DataFrame和Series的灵活操作，用户可以轻松地进行统计分析、汇总、聚合等操作。从均值、中位数到标准差和相关性分析，Pandas都提供了丰富的功能。
- **数据可视化：** 将Pandas与Matplotlib、Seaborn等数据可视化库结合使用，可以创建各种图表和图形，从而更直观地理解数据分布和趋势。这对于数据科学家、分析师和决策者来说都是关键的。
- **时间序列分析：** Pandas在处理时间序列数据方面表现出色，支持对日期和时间进行高效操作。这对于金融领域、生产领域以及其他需要处理时间序列的行业尤为重要。
- **机器学习和数据建模：** 在机器学习中，数据预处理是非常关键的一步，而Pandas提供了强大的功能来处理和准备数据。它可以帮助用户将数据整理成适用于机器学习算法的格式。
- **数据库操作：** Pandas可以轻松地与数据库进行交互，从数据库中导入数据到DataFrame中，进行分析和处理，然后将结果导回数据库。这在数据库管理和分析中非常有用。
- **实时数据分析：** 对于需要实时监控和分析数据的应用，Pandas的高效性能使其成为一个强大的工具。结合其他实时数据处理工具，可以构建实时分析系统。

## Pandas 数据结构 - Series

### 1.定义

```python
Pandas Series 类似表格中的一个列（column），类似于一维数组，可以保存任何数据类型。
```

- **索引：** 每个 `Series` 都有一个索引，它可以是整数、字符串、日期等类型。如果没有显式指定索引，Pandas 会自动创建一个默认的整数索引。

- **数据类型：** `Series` 可以容纳不同数据类型的元素，包括整数、浮点数、字符串等。

- 语法

  ```python
  pandas.Series( data, index, dtype, name, copy)
  ```
![](https://github.com/ljgit1316/Picture_resource/blob/main/Pandas_Pic/1.png)
### 2.参数说明

- **data**：一组数据(ndarray 类型)。
- **index**：数据索引标签，如果不指定，默认从 0 开始。
- **dtype**：数据类型，默认会自己判断。
- **name**：设置名称。
- **copy**：拷贝数据，默认为 False。

### 3.代码实现

#### 1.无索引读取数据

```python
import pandas as pd
import numpy as np
#如果没有指定索引，索引值就从 0 开始
arr=np.arange(1,10)
print(arr)
#[1 2 3 4 5 6 7 8 9]
my_panda=pd.Series(arr)
print(my_panda)
"""
0    1
1    2
2    3
3    4
4    5
5    6
6    7
7    8
8    9
dtype: int32
"""
```

#### 2.根据索引读取数据

```python
import pandas as pd
import numpy as np

arr=np.arange(1,5)
print(arr)
#[1 2 3 4]
my_panda=pd.Series(arr,index=['a','b','c','d'])
print(my_panda)
"""
a    1
b    2
c    3
d    4
dtype: int32
"""
```

#### 3.字典读取创建Series

- 用类似字典创建Series，字典的 key 变成了索引值

  ```python
  arr={1:'字典',2:'元组',3:'列表',4:'集合'}
  my_panda=pd.Series(arr)
  print(arr)
  """
  1    字典
  2    元组
  3    列表
  4    集合
  dtype: object
  
  """
  ```

- 读取字典部分数据

  ```python
  arr={1:'字典',2:'元组',3:'列表',4:'集合'}
  my_panda=pd.Series(arr,index=[2,4])
  print(my_panda)
  """
  2    元组
  4    集合
  dtype: object
  """
  ```

- 设置Series 名称参数

  ```python
  arr={1:'字典',2:'元组',3:'列表',4:'集合'}
  my_panda=pd.Series(arr,index=[2,4],name='AI')
  print(my_panda)
  """
  2    元组
  4    集合
  Name: AI, dtype: object
  
  """
  ```

## Pandas 数据结构 - DataFrame

### 1.定义

​		DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。

- DataFrame 特点

  - **列和行：** `DataFrame` 由多个列组成，每一列都有一个名称，可以看作是一个 `Series`。同时，`DataFrame` 有一个行索引，用于标识每一行。
  - **二维结构：** `DataFrame` 是一个二维表格，具有行和列。可以将其视为多个 `Series` 对象组成的字典。
  - **列的数据类型：** 不同的列可以包含不同的数据类型，例如整数、浮点数、字符串等。
![](https://github.com/ljgit1316/Picture_resource/blob/main/Pandas_Pic/2.png)

- DataFrame 可视为由多个 Series 组成的数据结构：
![](https://github.com/ljgit1316/Picture_resource/blob/main/Pandas_Pic/3.png)

- 语法

  ```
  pandas.DataFrame( data, index, columns, dtype, copy)
  ```

### 2.参数说明

- **data**：一组数据(ndarray、series, map, lists, dict 等类型)。
- **index**：索引值，或者可以称为行标签。
- **columns**：列标签，默认为 RangeIndex (0, 1, 2, …, n) 。
- **dtype**：数据类型。
- **copy**：拷贝数据，默认为 False。
- Pandas DataFrame 是一个二维的数组结构。

### 3.代码实现

#### 1.创建DataFrame

- 列表创建DataFrame

  ```python
  data=[['张无忌',40],['赵敏',40],['张三丰',120],['张翠山',35]]
  my_panda=pd.DataFrame(data,columns=['name','age'])
  print(my_panda)
  """
    name  age
  0  张无忌   40
  1   赵敏   40
  2  张三丰  120
  3  张翠山   35
  """
  ```

- 字典/列表创建DataFrame

  ```python
  data={'name':['张无忌','赵敏','张三丰','张翠山'],'age':[40,40,120,35]}
  my_panda=pd.DataFrame(data)
  print(my_panda)
  """
  name  age
  0  张无忌   40
  1   赵敏   40
  2  张三丰  120
  3  张翠山   35
  """
  ```

- 使用列表/字典创建，其中字典的 key 为列名没有对应的部分数据为NaN

  ```python
  data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
  my_panda = pd.DataFrame(data)
  print (my_panda)
  """
     a   b     c
  0  1   2   NaN
  1  5  10  20.0
  """
  ```

#### 2.读取数据

使用 loc 属性返回指定行的数据，如果没有设置索引，第一行索引为 **0**，第二行索引为 **1**

```python
data={'name':['张无忌','赵敏','张三丰','张翠山'],'age':[40,40,120,35]}
my_panda=pd.DataFrame(data)
print(my_panda.loc[0])
"""
   name  age
0  张无忌   40
1   赵敏   40
2  张三丰  120
3  张翠山   35

name    张无忌
age      40

"""
```

返回多行数据，使用 [[ 1,2,3... ]] 格式，1,2,3... 为各行的索引，以逗号隔开

```python
data={'name':['张无忌','赵敏','张三丰','张翠山'],'age':[40,40,120,35]}
my_panda=pd.DataFrame(data)
print(my_panda.loc[[0,3]])
"""
   name  age
0  张无忌   40
1   赵敏   40
2  张三丰  120
3  张翠山   35

   name  age
0  张无忌   40
3  张翠山   35

"""
```

指定行索引值

```python
data={'name':['张无忌','赵敏','张三丰','张翠山'],'age':[40,40,120,35]}
my_panda=pd.DataFrame(data,index=['person1','person2','person3','person4'])
print(my_panda)
print(my_panda.loc[['person1']])
"""
         name  age
person1  张无忌   40
person2   赵敏   40
person3  张三丰  120
person4  张翠山   35


         name  age
person1  张无忌   40

"""

```

