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
## Pandas CSV 文件

- 定义

  ```
  CSV（Comma-Separated Values，逗号分隔值，有时也称为字符分隔值，因为分隔字符也可以不是逗号），其文件以纯文本形式存储表格数据（数字和文本）。
  CSV 是一种通用的、相对简单的文件格式，被用户、商业和科学广泛应用。
  ```

- 读取csv文件

  ```python
  import pandas as pd
  df = pd.read_csv('nba.csv')
  #print(df)
  print(df.to_string())#to_string() 用于返回 DataFrame 类型的数据，如果不使用该函数，则输出结果为数据的前面 5 行和末尾 5 行，中间部分以 ... 代替。
  ```

- 存储csv文件

  ```python
  import pandas as pd
  
  name = ["karen", "jack", "marry", "zhangsan"]
  likes = ["python", "numpy", "pandas", "matplotlib"]
  age = [20, 30, 40, 18]
  # 字典
  dict1 = {'name': name, 'likes': likes, 'age': age}
  df = pd.DataFrame(dict1)
  # 保存 dataframe
  df.to_csv('site.csv')
  ```

## 数据处理

- 1.head函数

  ```python
  #head( *n* ) 方法用于读取前面的 n 行，如果不填参数 n ，默认返回 5 行。
  
  import pandas as pd
  df = pd.read_csv('nba.csv')
  print(df.head())
  ```

- 2.tail函数

  ```python
  #tail( *n* ) 方法用于读取尾部的 n 行，如果不填参数 n ，默认返回 5 行，空行各个字段的值返回 **NaN**。
  
  import pandas as pd
  df = pd.read_csv('nba.csv')
  print(df.tail())
  ```

- 3.info函数

  ```python
  #info() 方法返回表格的一些基本信息
  
  import pandas as pd
  df = pd.read_csv('nba.csv')
  print(df.info())
  ```

  ```python
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 458 entries, 0 to 457          # 行数，458 行，第一行编号为 0
  Data columns (total 9 columns):            # 列数，9列
   #   Column    Non-Null Count  Dtype       # 各列的数据类型
  ---  ------    --------------  -----  
   0   Name      457 non-null    object 
   1   Team      457 non-null    object 
   2   Number    457 non-null    float64
   3   Position  457 non-null    object 
   4   Age       457 non-null    float64
   5   Height    457 non-null    object 
   6   Weight    457 non-null    float64
   7   College   373 non-null    object         # non-null，意思为非空的数据    
   8   Salary    446 non-null    float64
  dtypes: float64(4), object(5)                 # 类型
  ```

## Pandas Json

### 1.处理JSON字符串

```python
#JSON 对象与 Python 字典具有相同的格式，所以我们可以直接将 Python 字典转化为 DataFrame 数据

import pandas as pd
# 字典格式的 JSON                                                                                             
s = {
    "col1":{"row1":1,"row2":2,"row3":3},
    "col2":{"row1":"x","row2":"y","row3":"z"}
}
# 读取 JSON 转为 DataFrame  
df = pd.DataFrame(s)
print(df)

"""
      col1 col2
row1     1    x
row2     2    y
row3     3    z
"""
```

```python
import pandas as pd
data =[
    {
      "id": "A001",
      "name": "华清远见",
      "url": "www.hqyj.com",
      "likes": 61
    },
    {
      "id": "A002",
      "name": "百度",
      "url": "www.baidu.com",
      "likes": 124
    },
    {
      "id": "A003",
      "name": "淘宝",
      "url": "www.taobao.com",
      "likes": 45
    }
]
df = pd.DataFrame(data)
print(df)

"""
     id  name             url  likes
0  A001  华清远见    www.hqyj.com     61
1  A002    百度   www.baidu.com    124
2  A003    淘宝  www.taobao.com     45
"""
```

### 2.读取JSON文件

```python
import pandas as pd
df = pd.read_json('sites.json')
print(df.to_string())#to_string() 用于返回 DataFrame 类型的数据
```

### 3.从URL中读取JSON数据

```python
import pandas as pd
URL = 'http://localhost:7001/test.json'
df = pd.read_json(URL)
print(df)
```

### 4.不同JSON类型数据解析

- 嵌套JSON数据

  ```python
  import pandas as pd
  data={
      "school_name": "ABC primary school",
      "class": "Year 1",
      "students": [
      {
          "id": "A001",
          "name": "Tom",
          "math": 60,
          "physics": 66,
          "chemistry": 61
      },
      {
          "id": "A002",
          "name": "James",
          "math": 89,
          "physics": 76,
          "chemistry": 51
      },
      {
          "id": "A003",
          "name": "Jenny",
          "math": 79,
          "physics": 90,
          "chemistry": 78
      }]
  }
  df = pd.DataFrame(data)
  print(df)
  
  
  """
            school_name  ...                                           students
  0  ABC primary school  ...  {'id': 'A001', 'name': 'Tom', 'math': 60, 'phy...
  1  ABC primary school  ...  {'id': 'A002', 'name': 'James', 'math': 89, 'p...
  2  ABC primary school  ...  {'id': 'A003', 'name': 'Jenny', 'math': 79, 'p...
  
  [3 rows x 3 columns]
  
  """
  ```

- JSON展平数据

  ```python
  **json_normalize(data,record_path,meta)**展平数据
  
  data要展平的数据,record_path需要展平的字段,meta需要显示的元数据
  
  将students内嵌的数据完整的解析出来：
  
  import pandas as pd
  data={
      "school_name": "ABC primary school",
      "class": "Year 1",
      "students": [
      {
          "id": "A001",
          "name": "Tom",
          "math": 60,
          "physics": 66,
          "chemistry": 61
      },
      {
          "id": "A002",
          "name": "James",
          "math": 89,
          "physics": 76,
          "chemistry": 51
      },
      {
          "id": "A003",
          "name": "Jenny",
          "math": 79,
          "physics": 90,
          "chemistry": 78
      }]
  }
  
  
  df = pd.DataFrame(data)
  print(df)
  
  df_nested_list = pd.json_normalize(data, record_path =['students'])
  print(df_nested_list)
  """
            school_name  ...                                           students
  0  ABC primary school  ...  {'id': 'A001', 'name': 'Tom', 'math': 60, 'phy...
  1  ABC primary school  ...  {'id': 'A002', 'name': 'James', 'math': 89, 'p...
  2  ABC primary school  ...  {'id': 'A003', 'name': 'Jenny', 'math': 79, 'p...
  
  [3 rows x 3 columns]
  
       id   name  math  physics  chemistry
  0  A001    Tom    60       66         61
  1  A002  James    89       76         51
  2  A003  Jenny    79       90         78
  
  """
  ```

  ```python
  json_normalize() 使用了参数 **record_path** 并设置为 **['students']** 用于展开内嵌的 JSON 数据 **students**。
  
  显示结果还没有包含 school_name 和 class 元素，如果需要展示出来可以使用 meta 参数来显示这些元数据：
  import pandas as pd
  data={
      "school_name": "ABC primary school",
      "class": "Year 1",
      "students": [
      {
          "id": "A001",
          "name": "Tom",
          "math": 60,
          "physics": 66,
          "chemistry": 61
      },
      {
          "id": "A002",
          "name": "James",
          "math": 89,
          "physics": 76,
          "chemistry": 51
      },
      {
          "id": "A003",
          "name": "Jenny",
          "math": 79,
          "physics": 90,
          "chemistry": 78
      }]
  }
  
  # 展平数据
  df_nested_list = pd.json_normalize(
      data,
      record_path =['students'],
      meta=['school_name', 'class']
  )
  print(df_nested_list)
  
  
  """
  
       id   name  math  physics  chemistry         school_name   class
  0  A001    Tom    60       66         61  ABC primary school  Year 1
  1  A002  James    89       76         51  ABC primary school  Year 1
  2  A003  Jenny    79       90         78  ABC primary school  Year 1
  
  
  
  """
  ```

  ```python
  #更复杂的 JSON 数据，该数据嵌套了列表和字典
  
  import pandas as pd
  import json
  data={
     "school_name": "local primary school",
     "class": "Year 1",
     "info": {
       "president": "John Kasich",
       "address": "ABC road, London, UK",
       "contacts": {
          "email": "admin@e.com",
          "tel": "123456789"
       }
      },
      "students": [
      {
         "id": "A001",
          "name": "Tom",
         "math": 60,
          "physics": 66,
         "chemistry": 61
      },
      {
          "id": "A002",
          "name": "James",
         "math": 89,
          "physics": 76,
         "chemistry": 51
      },
      {
         "id": "A003",
         "name": "Jenny",
         "math": 79,
         "physics": 90,
         "chemistry": 78
      }]
  }
  
  df = pd.json_normalize(
      data,
      record_path =['students'],
      meta=[
          'class',
          ['info', 'president'],
          ['info', 'contacts', 'tel']
      ]
  )
  print(df)
  
  """
       id   name  math  physics  chemistry   class info.president info.contacts.tel
  0  A001    Tom    60       66         61  Year 1    John Kasich         123456789
  1  A002  James    89       76         51  Year 1    John Kasich         123456789
  2  A003  Jenny    79       90         78  Year 1    John Kasich         123456789
  
  """
  ```

### 5.JSON模块加载

```python
#如果是json文件中有嵌套数据,需要读取文件数据然后使用 Python JSON 模块载入数据

import pandas as pd
import json
# 使用 Python JSON 模块载入数据
with open('test.json','r') as f:
    data = json.loads(f.read())
    print(data)
    df = pd.json_normalize(
        data,
        record_path=['students'],
        meta=[
            'class',
            ['info', 'president'],
            ['info', 'contacts', 'tel']
        ]
    )
    print(df)
```

## Pandas数据清洗

```python
数据清洗是对一些没有用的数据进行处理的过程。

很多数据集存在数据缺失、数据格式错误、错误数据或重复数据的情况，如果要使数据分析更加准确，就需要对这些没有用的数据进行处理
```

![image-20240811175316799](C:\Users\13167\AppData\Roaming\Typora\typora-user-images\image-20240811175316799.png)

上表包含了四种空数据：

- n/a
- NA
- —
- na

### 1.Pandas清洗空值

```python
#通过 isnull() 判断各个单元格是否为空。

import pandas as pd
df = pd.read_csv('property-data.csv')
print (df['NUM_BEDROOMS'])
print (df['NUM_BEDROOMS'].isnull())
```

效果：

![image-20240811175637585](C:\Users\13167\AppData\Roaming\Typora\typora-user-images\image-20240811175637585.png)

```python
#Pandas 把 n/a 和 NA 当作空数据，na 不是空数据，可以指定空数据类型

import pandas as pd
missing_values = ["n/a", "na", "--"]
df = pd.read_csv('property-data.csv', na_values = missing_values)
print (df['NUM_BEDROOMS'])
print (df['NUM_BEDROOMS'].isnull())
```

#### 1.1删除包含空字段的行

- 函数

  ```python
  DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
  ```

- 参数说明

  - axis：默认为 **0**，表示逢空值剔除整行，如果设置参数 **axis＝1** 表示逢空值去掉整列。

  - how：默认为 **'any'** 如果一行（或一列）里任何一个数据有出现 NA 就去掉整行，如果设置 **how='all'** 一行（或列）都是 NA 才去掉这整行。

  - thresh：设置需要多少非空值的数据才可以保留下来的。

  - subset：设置想要检查的列。如果是多个列，可以使用列名的 list 作为参数。

    **注意: 行计算数量时  会把行的标记也算上**

  - inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。默认情况下，dropna() 方法返回一个新的 DataFrame，不会修改源数据。

- 代码

  ```python
  import pandas as pd
  df = pd.read_csv('property-data.csv')
  new_df = df.dropna()
  print(new_df.to_string())
  
  #要修改源数据 DataFrame, 可以使用 **inplace = True** 参数
  import pandas as pd
  df = pd.read_csv('property-data.csv')
  df.dropna(inplace = True)
  print(df.to_string())
  
  ```

#### 1.2删除指定列有空值的行

```python
import pandas as pd
df = pd.read_csv('property-data.csv')
df.dropna(subset=['ST_NUM'], inplace = True)
print(df.to_string())
```

#### 1.3替换空字段

```python
#函数fillna()
```

```python
import pandas as pd
df = pd.read_csv('property-data.csv')
df.fillna(12345, inplace = True)#使用 12345 替换空字段：
print(df.to_string())
```

```python
#指定某一个列来替换数据

import pandas as pd
df = pd.read_csv('property-data.csv')
df['PID'].fillna(12345, inplace = True)#使用 12345 替换 PID 为空数据：
print(df.to_string())
```

```python
#使用 mean() 方法计算列的均值并替换空单元格

import pandas as pd
df = pd.read_csv('property-data.csv')
x = df["ST_NUM"].mean()
df["ST_NUM"].fillna(x, inplace = True)
print(df.to_string())

```

```python
#使用 median() 方法计算列的中位数并替换空单元格

import pandas as pd
df = pd.read_csv('property-data.csv')
x = df["ST_NUM"].median()
df["ST_NUM"].fillna(x, inplace = True)
print(df.to_string())
```

```python
#使用 mode() 方法计算列的众数并替换空单元格

import pandas as pd
df = pd.read_csv('property-data.csv')
x = df["ST_NUM"].mode()
df["ST_NUM"].fillna(x, inplace = True)
print(df.to_string())
```

### 2.Pandas 清洗格式错误日期

```python
#**pd.to_datetime()**格式化日期

import pandas as pd
# 第三个日期格式错误
data = {
  "Date": ['2019/12/01', '2024/3/01' , '20201226'],
  "duration": [50, 40, 45]
}
df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
df['Date'] = pd.to_datetime(df['Date'],format='mixed')
print(df.to_string())
```

### 3.Pandas清洗错误的数据

```python
#替换错误年龄的数据

import pandas as pd
person = {
  "name": ['karen', 'jack' , 'marry'],
  "age": [50, 40, 12345]    # 12345 年龄数据是错误的
}
df = pd.DataFrame(person) 
df.loc[2, 'age'] = 30 # 修改数据
print(df.to_string())
```

```python
#设置条件语句,将 age 大于 120 的设置为 120

import pandas as pd
person = {
  "name": ['karen', 'jack' , 'marry'],
  "age": [50, 200, 12345]    
}
df = pd.DataFrame(person)
for x in df.index:
  if df.loc[x, "age"] > 120:
    df.loc[x, "age"] = 120
print(df.to_string())
```

```python
#将错误数据的行删除：将 age 大于 120 的删除

import pandas as pd
person = {
  "name":['karen', 'jack' , 'marry'],
  "age": [50, 40, 12345]    # 12345 年龄数据是错误的
}
df = pd.DataFrame(person)
for x in df.index:
  if df.loc[x, "age"] > 120:
    df.drop(x, inplace = True)
print(df.to_string())
```

### 4.Pandas清洗重复数据

```python
#如果我们要清洗重复数据，可以使用 **duplicated()** 和 **drop_duplicates()** 方法。
#如果对应的数据是重复的，**duplicated()** 会返回 True，否则返回 False。

import pandas as pd
person = {
  "name": ['python', 'numpy' , 'numpy' , 'pandas'],
  "age": [50, 40, 40, 23]  
}
df = pd.DataFrame(person)
print(df.duplicated())

#删除重复数据，可以直接使用**drop_duplicates()** 方法。
import pandas as pd
persons = {
  "name": ['Google', 'Runoob', 'Runoob', 'Taobao'],
  "age": [50, 40, 40, 23]  
}
df = pd.DataFrame(persons)
df.drop_duplicates(inplace = True)
print(df)
```


