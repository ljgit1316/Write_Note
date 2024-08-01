

# Python基础与语法

## day_1

### 1.PyCharm常用快捷键

```
ctrl + s 保存

ctrl + / 注释和取消注释

ctrl + d 可以复制当前行到下一行

ctrl + alt + L  可以将程序自动格式化
```

### 2.Python输入和输出

​	print() 函数

​		向终端输出文字信息，在控制台输出打印数据

```python
#格式
print(数据1, 数据2, 数据3, ..., sep=' ', end='\n')
#例子
print("hello world!")
print("我们班有多少人": 12)
print(1, 2, 3, 4)
print(1, 2, 3, 4, sep=" ", end="\n")
print(1, 2, 3, 4, sep="###", end="\n\n\n")
print(1, 2, 3, 4)
```

- sep 关键字参数用来定义多个数据之间的分割符，默认用空格分隔
- end 关键字参数用来定义数据输出完后用什么字符结束，默认是换行符('\n')

​	input() 函数

​		是个阻塞函数，让程序停下来，等待用户输入文字信息，返回用户输入文字的 字符串

```python
name = input('请输入您的姓名:')
```

### 3.Python注释

1. 井号(#)

   ```python
   # 这是一个注释print("Hello, World!")
   ```

2. 单引号(''')

   ```python
   '''这是多行注释，用三个单引号这是多行注释，用三个单引号 这是多行注释，用三个单引号'''
   ```

3. 双引号(""")

   ```python
   """这是多行注释，用三个双引号这是多行注释，用三个双引号 这是多行注释，用三个双引号"""
   ```

### 4.Python变量

#### 1.变量命名规则

1. 变量名必须是一个标识符

2. 标识符的命名规则

   2.1第一个字母必须是英文字母**[A-Z a-z]**或下划线 **[ _ ]**，数字不能开头

   2.2从第二个字母起(如果有)，**必须是英文字母、下划线、数字**

   示例：

   ```python
   a        a1           abc       ABC     a1b2c3d4
   one_hundred          count      _a      __ABC__
   
   getNameAge         get_name_age        GetNameAge
   # 小驼峰              匈牙利命名法         大驼峰
   ```

  	 2.3关键字(keywords),**关键字不能当成变量名使用**

​			python中的部分关键字:

```python
False      await      else       import     pass
None       break      except     in         raise
True       class      finally    is         return
and        continue   for        lambda     try
as         def        from       nonlocal   while
assert     del        global     not        with
async      elif       if         or         yield
```

#### 2.赋值语句

- 语法

  ```python
  变量名 = 数字类型
  变量名 = 表达式
  变量名1 = 变量名2 = 变量名3  = 数字类型
  变量名1, 变量名2, 变量名3  = 数字类型1, 数字类型2, 数字类型3
  ```

- 实例

  ```python
  one_hundred = 99 + 1
  a = b = c = 200
  a, b = 100, 200
  
  counter = 100          # 整型变量
  miles   = 1000.0       # 浮点型变量
  name    = "hqyj"     # 字符串
  print (counter)
  print (miles)
  print (name)
  ```

#### 3.变量类型

##### 字符串

###### 字符串使用和多重引号表达

1.字符串使用

```python
## 用 英文的 ' 或 " 或 ''' 或 """  开始或结束
## 此示例示意字符串的写法
print('同学们好')
print("同学们好")
print('''同学们好''')
print("""同学们好""")
```

2.多重引号表达

- 双引号的字符串的内部的单引号不算是结束符

- 单引号的字符串的内部的双引号不算是结束符

  ```python
  # I'm a teacher!
  print("I'm a teacher!")
  # 我有一只"够"
  print('我是"够"')
  ```

- 三引号字符串的内部可以包含单引号和双引号

  ```python
  # I'm a teacher! 我是"徐婧"
  print('''I'm a teacher! 我是"徐婧"''')
  ```

- 三引号字符串又称为所见即所得字符串, 每一次换行会转换成换行符 '\n'

  ```python
  # print("咏鹅\n鹅鹅鹅,\n曲项向天歌;\n白毛浮绿水，\n红掌拨清波。")
  # 等同于如下写法
  print('''咏鹅
  鹅鹅鹅,
  曲项向天歌;
  白毛浮绿水，
  红掌拨清波。''')
  ```

###### 字符串的转义使用

```python
print('我是单引号\',我是双引号\", 我是三单引号\'\'\', 我是三双引"""')
```

| 转义序列   | 含义                       | 注释  |
| ---------- | -------------------------- | ----- |
| `\newline` | 反斜杠加换行全被忽略       |       |
| `\\`       | 反斜杠 (`\`)               |       |
| `\'`       | 单引号 (`'`)               |       |
| `\"`       | 双引号 (`"`)               |       |
| `\a`       | ASCII 响铃 (BEL)           |       |
| `\b`       | ASCII 退格 (BS)            |       |
| `\f`       | ASCII 进纸 (FF)            |       |
| `\n`       | ASCII 换行 (LF)            |       |
| `\r`       | ASCII 回车 (CR)            |       |
| `\t`       | ASCII 水平制表 (TAB)       |       |
| `\v`       | ASCII 垂直制表 (VT)        |       |
| `\ooo`     | 八进制数 *ooo* 码位的字符  | (1,3) |
| `\xhh`     | 十六进制数 *hh* 码位的字符 | (2,3) |

###### 字符串的运算

- 字符串连接：+

  ```python
  s1 = '123'
  s2 = "456"
  s3 = s1 + s2
  print(s1)   # 123
  print(s2)   # 456
  print(s3)   # 123456
  ```

- 字符多次重复

  ```python
  x=3*('a' + 'b')
  print(x)#ababab
  ```

- in/not in 运算

  ```python
  >>> s = 'welcome to beijing!'
  >>> 'to' in s
  True
  >>> 'weimingze' in s
  False
  >>> 'beijing' not in s
  False
  >>> 'weimz' not in s
  True
  ```

- 字符串索引

  ```python
  >>> s = 'ABCDE'
  >>> s[1]
  B
  >>> s[1+2]
  D
  >>> s[-1]
  E
  ```

- 字符串切片

  ```python
       #   01234
  >>> s = 'ABCDE'
  >>> s[1:]    # 'BCDE'
  >>> s[-2:]   # 'DE'
  >>> s[1:4]   # 'BCD'
  >>> s[:]     # 'ABCDE'
  >>> s[:2]    # 'AB'
  >>> s[1:1]   # ''
  >>> s[4:2]   # ''
  >>> s[::]    # 等同于 s[::1] 'ABCDE'
  >>> s[::2]   # 'ACE'
  >>> s[2::2]  # 'CE'
  >>> s[-1::-2] # 'ECA'
  >>> s[::-1]   # 'EDCBA'
  
  >>> x='huaqyj'[0:2:2]
  >>> print(x)
  ```

###### 字符串的格式化表达式

- 作用

  生成具有一定格式的字符串

- 语法规则

  ```python
  格式化字符串 % 参数1
  # 或者
  格式化字符串 % (参数1, 参数2, 参数3)
  ```

- 其他类型占位符和类型码

  | 占位符和类型码 | 说明                                   |
  | :------------: | -------------------------------------- |
  |       %s       | 转成字符串, 使用 str(x) 函数转换(常用) |
  |       %d       | 转成 十进制的整数(常用)                |
  |       %o       | 转成 八进制的整数                      |
  |     %x,%X      | 转成 十六进制的整数                    |
  |     %e,%E      | 转成 指数格式的浮点数                  |
  |     %f,%F      | 转成小数格式的浮点数(常用)             |
  |     %g,%G      | 转成指数格式或小数格式的浮点数         |
  |       %%       | 转成一个%                              |

###### 字符串常用函数的API

- 语法规则

  对象.方法名(参数)

- 例子

  ```python
  >>> "abc".isalpha() 
  x = "abc"
  x.isalpha() 
  True
  >>> "123".isalpha()
  False
  >>> "123".isdigit()
  True
  ```

- 其他函数

  | 序号 | 方法及描述                                                   |
  | ---- | ------------------------------------------------------------ |
  | 1    | **capitalize**()将字符串的第一个字符转换为大写               |
  | 2    | **center**(width, fillchar)返回一个指定的宽度 width 居中的字符串，fillchar 为填充的字符，默认为空格。 |
  | 3    | **count**(str, beg= 0,end=len(string))返回 str 在 string 里面出现的次数，如果 beg 或者 end 指定则返回指定范围内 str 出现的次数 |
  | 4    | **endswith**(suffix, beg=0, end=len(string))检查字符串是否以 suffix 结束，如果 beg 或者 end 指定则检查指定的范围内是否以 suffix 结束，如果是，返回 True,否则返回 False。 |
  | 5    | **expandtabs**(tabsize=8)把字符串 string 中的 \t 符号转为空格，tab 符号默认的空格数是 8 。 |
  | 6    | **find**(str, beg=0, end=len(string))检测 str 是否包含在字符串中，如果指定范围 beg 和 end ，则检查是否包含在指定范围内，如果包含返回开始的索引值，否则返回-1 |
  | 7    | **index**(str, beg=0, end=len(string))跟find()方法一样，只不过如果str不在字符串中会报一个异常。 |
  | 8    | **isalnum**()非空字符串 中没有符号 就返回True                |
  | 9    | **isalpha**() 判断字符串是否是英文字符 是就返回True          |
  | 10   | **isdigit**()如果字符串只包含数字则返回 True 否则返回 False  |
  | 11   | **islower**() 用于检测字符串中的所有字符是否都是小写字母,字符都是小写，则返回 True，否则返回 False |
  | 12   | **isnumeric**()如果字符串中只包含数字字符，则返回 True，否则返回 False |
  | 13   | **isspace**()如果字符串中只包含空白，则返回 True，否则返回 False. |
  | 14   | **istitle**()如果字符串是标题化的(见 title())则返回 True，否则返回 False |
  | 15   | **isupper**()用于检测字符串中的所有字符是否都是大写字母,并且都是大写，则返回 True，否则返回 False |
  | 16   | **join**(seq)以指定字符串作为分隔符，将 seq 中所有的元素(的字符串表示)合并为一个新的字符串 |
  | 17   | **len**(string)返回字符串长度                                |
  | 18   | **ljust**(width, fillchar])返回一个原字符串左对齐,并使用 fillchar 填充至长度 width 的新字符串，fillchar 默认为空格。 |
  | 19   | **lower**()转换字符串中所有大写字符为小写.                   |
  | 20   | **lstrip**()截掉字符串左边的空格,\t,\r,\n或指定字符。        |
  | 21   | **maketrans**()创建字符映射的转换表，对于接受两个参数的最简单的调用方式，第一个参数是字符串，表示需要转换的字符，第二个参数也是字符串表示转换的目标。 |
  | 22   | **max**(str)返回字符串 str 中最大的字母。                    |
  | 23   | **min**(str)返回字符串 str 中最小的字母。                    |
  | 24   | **replace**(old, new , max)把 将字符串中的 old 替换成 new,如果 max 指定，则替换不超过 max 次。 |
  | 25   | **rfind**(str, beg=0,end=len(string))类似于 find()函数，不过是从右边开始查找. |
  | 26   | **rindex**( str, beg=0, end=len(string))类似于 index()，不过是从右边开始. |
  | 27   | **rjust**(width, fillchar)返回一个原字符串右对齐,并使用fillchar(默认空格）填充至长度 width 的新字符串 |
  | 38   | **rstrip**()删除字符串末尾的空格\t,\r,\n或指定字符。         |
  | 29   | **split**(sep="", maxsplit=string.count(str))以 sep为分隔符截取字符串，如果 maxsplit有指定值，则仅截取 maxsplit+1 个子字符串 |
  | 30   | **splitlines**(keepends)按照行('\r', '\r\n', \n')分隔，返回一个包含各行作为元素的列表，如果参数 keepends 为 False，不包含换行符，如果为 True，则保留换行符。 |
  | 31   | **startswith**(substr, beg=0,end=len(string))检查字符串是否是以指定子字符串 substr 开头，是则返回 True，否则返回 False。如果beg 和 end 指定值，则在指定范围内检查。 |
  | 32   | **strip**(chars)在字符串上执行 lstrip()和 rstrip()           |
  | 33   | **swapcase**()将字符串中大写转换为小写，小写转换为大写       |
  | 34   | **title**()返回"标题化"的字符串,就是说所有单词都是以大写开始，其余字母均为小写 |
  | 35   | **upper**()转换字符串中的小写字母为大写                      |
  | 36   | **zfill** (width) 在字符串左侧填充指定数量的零，确保整个字符串达到指定长度 |

###### 其它字符串

- 字节串（bytes）：表示二进制数据，以字节为单位，例如b'hello'。
- 空值（NoneType）：表示一个特殊的空值，通常用于表示缺失或未定义的值。

##### 数字型

###### 整型（int）

```python
## 十进制的写法
100        0         -5

## 二进制的写法 0b 开头 后跟 0~1
0b1101

## 八进制的写法 0o开头 后跟 0~7
0o777   等于  0b111111111   等于 511

## 十六进制的写法  0x 开头 后跟 0~9, a-f, A-F
0xA1B2C3D4
```

###### 浮点型（floa）

```python
## 小数写法
3.14         0.14       .14         3.0       3.      0.0
## 科学计数法
6.18E-1   # 等同于 0.618   
2.9979E8   # 等同于 299790000.0
```

###### 布尔型（bool）

```python
True    真(表示行，好，成立) ,值为1
False   假(表示不行，不好，不成立) ,值为0
```

##### 复合型

- 列表（list）：可变序列，用于存储一组值，可以包含不同类型的元素。
- 元组（tuple）：不可变序列，用于存储一组值，元素不能被修改。
- 字典（dict）：键值对映射，用于存储关联性数据，由键和对应的值组成。
- 集合（set）：无序集合，用于存储唯一的元素，不允许重复。
- 枚举类型（Enum）：本质上是一个类，它是标准库中的`enum`模块提供的一个功能，用于创建有限的、命名的枚举类型
- 自定义类（class）：创建自定义类来表示复杂的数据结构，具有自定义属性和方法。

按照是否可以修改划分:

- 不可变数据：Number（数字）、String（字符串）、Tuple（元组）
- 可变数据：List（列表）、Dictionary（字典）、Set（集合）

#### 4.删除变量

```python
x=100
del x
print(x)#报错name 'x' is not defined
```

## day_2

### 5.Python的数字操作

#### 1.数字类型转换

- **int(x)** 将x转换为十进制**整数**

- **float(x)** 将x转换到一个浮点数。

- **bin(x)**将x转换为二进制

- **oct(x)**将x转换为八进制

- **hex(x)**将x转换为十六进制

- **complex(x)** 将x转换到一个复数，实数部分为 x，虚数部分为 0。

- **complex(x, y)** 将 x 和 y 转换到一个复数，实数部分为 x，虚数部分为 y。x 和 y 是数字.

- **bool(x)**将 x 转化为布尔值

  ```python
   print(int(20.5))#20
   print(float(20))#20.0
   print(bin(3))#0b11
   print(oct(20))#0o24
   print(hex(29))#0x1d
   print(complex(29))#(29+0j)
   print(complex(10,3))#(10+3j)
  ```

#### 2.运算符操作

##### 1.算术运算符

- `+`：加法

- `-`：减法

- `*`：乘法

- `/`：除法

- `%`：取模（取余数）

- `**`：幂运算

- `//`：整除（向下取整数部分）

  ```python
  print(10+20.3)#加法运算:30.3
  print(17 / 3)  # 整数除法返回浮点型:5.666666666666667
  print(17 // 3)  # 整数除法返回向下取整后的结果:5
  print(17.0 // 3)  # 整数除法返回向下取整后的结果:5.0
  print(17 // 3.0)  # 整数除法返回向下取整后的结果:5.0
  print(17 % 3)  # ％操作符返回除法的余数:2
  print(2**3) #幂运算2的3次方: 8
  ```

##### 2.比较运算符

**比较运算符的运算结果为布尔值**，**比较运算符通常返回布尔类型的数, True, False**

- `==`：等于

- `!=`：不等于

- `<`：小于

- `>`：大于

- `<=`：小于等于

- `>=`：大于等于

  ```python
  print(10==10.0)#只比较值是否相等:True
  print(3.14!=3.1415)#True
  print(255>170)#True
  print(255<170)#False
  print(255>=255)#True
  print(255<=255)#True
  x=15
  print(5<x<20)#注意:两个符号同时参与比较  相当于 (5 < x) and (x < 20)
  
  x=100
  y=200
  z=-10
  a=True
  print(a is not x is not y < z)#a和x判定 然后和y判定 然后和z判定
  # False
  ```

- `a is b` 返回 `True` 当且仅当 `a` 和 `b` 是同一个对象（即它们的身份相同）。

- `a is not b` 返回 `True` 当且仅当 `a` 和 `b` 不是同一个对象（即它们的身份不同）

  ```python
  a = [1, 2, 3]
  b = a
  c = [1, 2, 3]
  
  print(a is b)      # True，因为 a 和 b 是同一个对象
  print(a is c)      # False，因为 a 和 c 是不同的对象，即使它们的内容相同
  print(a is not c)  # True，因为 a 和 c 不是同一个对象
  ```

- python 中假值对象

  ```python
  None   # 空值对象
  False  # 布尔类型的假值
  0      # 整数的0
  0.0    # 浮点数的0
  ''     # 字符串的空字符串
  []     # 空列表
  {}     # 空字典
  .....
  ```

##### 3.逻辑运算符

- `and`：与（逻辑与）

  两者（两个元素同时为真，结果才为真）

- `or`：或（逻辑或）

  两者（两个元素只要有一个为真，结果就为真）

- `not`：非（逻辑非）

  将表达式的结果取 **非** 操作

  ```python
  >>> True and True    # True
  >>> True and False   # False
  >>> False and True   # False
  >>> False and False  # False
  >>> True or True    # True
  >>> True or False   # Ture
  >>> False or True   # Ture
  >>> False or False  # False
  >>> not False       # True
  >>> not True        # Flase
  >>> 3.14 and 5      # 5
  >>> 0.0 and 5       # 0.0
  >>> 3.14 or 5       # 3.14
  >>> 0.0 or 0        # 0
  >>> not 3.14        # False
  >>> not 0.0         # True
  ```

##### 4.位运算符

- `&`：按位与

  - 表达式：`a & b`

  - 功能：对于每一位，如果a和b的相应位都是1，则结果位为1，否则为0。

    ```python
    # 示例：计算两个二进制数的按位与
    a = 0b1011  # 二进制表示的11
    b = 0b1101  # 二进制表示的13
    result_and = a & b  # 计算两者之间的按位与
    print(bin(result_and))  # 输出：0b1001 （十进制为9）
    ```

- `|`：按位或

  - 表达式：`a | b`

  - 功能：对于每一位，只要a和b中至少有一位是1，则结果位为1，否则为0。

    ```python
    # 示例：计算两个二进制数的按位或
    a = 0b1011
    b = 0b1101
    result_or = a | b  # 计算两者之间的按位或
    print(bin(result_or))  # 输出：0b1111 （十进制为15）
    ```

- `^`：按位异或

  - 表达式：`a ^ b`

  - 功能：对于每一位，如果a和b的相应位不同（一个为1，另一个为0），则结果位为1，否则为0。

    ```python
    # 示例：计算两个二进制数的按位异或
    a = 0b1011
    b = 0b1101
    result_xor = a ^ b  # 计算两者之间的按位异或
    print(bin(result_xor))  # 输出：0b110 （十进制为6）
    ```

- `~`：按位取反

  - 表达式：`~a`

  - 功能：对操作数a的每一个二进制位进行取反，即将1变为0，0变为1。

    ```python
    # 示例：计算一个二进制数的按位取反
    a = 0b1011
    result_not = ~a  # 计算a的按位取反
    print(bin(result_not))  # 输出：-0b1100
    ```

- `<<`：左移位

  - 表达式：`a << b`

  - 功能：将a的二进制表示向左移动b位，左边移出的部分会被丢弃，右边空出的位置补零。**相当于乘以2^n次方**

  - ```python
    # 示例：将一个二进制数向左移动两位
    a = 0b1011
    result_left_shift = a << 2  # 将a向左移动两位
    print(bin(result_left_shift))  # 输出：0b101100 （十进制为44）
    ```

- `>>`：右移位

  - 表达式：`a >> b`

  - 功能：将a的二进制表示向右移动b位，对于无符号整数，右边移出的部分会被丢弃，左边空出的位置补零（通常补0）；对于有符号整数，右移时取决于具体实现，可能是算术右移（符号位扩展）或者逻辑右移（补0）。**同理，相当于除以2^n

    ```python
    # 示例：将一个有符号二进制数向右移动一位
    a = -0b1000  # 十进制为-8
    result_right_shift = a >> 1  # 将a向右移动一位
    print(bin(result_right_shift))  # 输出：-0b100 （十进制为-4）
    
    # 对于无符号数的例子
    unsigned_a = 0b1000
    unsigned_result_right_shift = unsigned_a >> 1
    print(bin(unsigned_result_right_shift))  # 输出：0b100 （十进制为4）
    ```

##### 5.赋值运算符

- `=`：赋值

- `+=`：加法赋值

- `-=`：减法赋值

- `*=`：乘法赋值

- `/=`：除法赋值

- `%=`：取余赋值

- `**=`：幂运算赋值

- `//=`：整除赋值

  **python中没有 a++、  a-- 这种自增自减运算符；**

##### 6.运算符的优先级

括号运算>>算术运算>>位运算>>比较运算>>逻辑运算>>赋值运算

| 运算符                                                       | 描述                                                 |
| :----------------------------------------------------------- | :--------------------------------------------------- |
| `(expressions...)`,`[expressions...]`, `{key: value...}`, `{expressions...}` | 绑定或加圆括号的表达式，列表显示，字典显示，集合显示 |
| `x[index]`, `x[index:index]`, `x(arguments...)`, `x.attribute` | 抽取，切片，调用，属性引用                           |
| [`await`](https://docs.python.org/zh-cn/3/reference/expressions.html#await) `x` | await 表达式                                         |
| `**`                                                         | 乘方                                                 |
| `+x`, `-x`, `~x`                                             | 正，负，按位非 NOT                                   |
| `*`, `@`, `/`, `//`, `%`                                     | 乘，矩阵乘，除，整除，取余                           |
| `+`, `-`                                                     | 加和减                                               |
| `<<`, `>>`                                                   | 移位                                                 |
| `&`                                                          | 按位与 AND                                           |
| `^`                                                          | 按位异或 XOR                                         |
| `|`                                                          | 按位或 OR                                            |
| [`in`](https://docs.python.org/zh-cn/3/reference/expressions.html#in), [`not in`](https://docs.python.org/zh-cn/3/reference/expressions.html#not-in), [`is`](https://docs.python.org/zh-cn/3/reference/expressions.html#is), [`is not`](https://docs.python.org/zh-cn/3/reference/expressions.html#is-not), `<`, `<=`, `>`, `>=`, `!=`, `==` | 比较运算，包括成员检测和标识号检测                   |
| [`not`](https://docs.python.org/zh-cn/3/reference/expressions.html#not) `x` | 布尔逻辑非 NOT                                       |
| [`and`](https://docs.python.org/zh-cn/3/reference/expressions.html#and) | 布尔逻辑与 AND                                       |
| [`or`](https://docs.python.org/zh-cn/3/reference/expressions.html#or) | 布尔逻辑或 OR                                        |
| [`if`](https://docs.python.org/zh-cn/3/reference/expressions.html#if-expr) -- `else` | 条件表达式                                           |
| [`lambda`](https://docs.python.org/zh-cn/3/reference/expressions.html#lambda) | lambda 表达式                                        |
| `:=`                                                         | 赋值表达式                                           |

#### 3.数学函数

- 引入math块

  **import math**

- 常用数学函数

  | 函数            | 返回值 ( 描述 )                                              |
  | --------------- | ------------------------------------------------------------ |
  | abs(x)          | 返回数字的绝对值，如abs(-10) 返回 10                         |
  | math.ceil(x)    | 返回数字的上入整数，如math.ceil(4.1) 返回 5                  |
  | cmp(x, y)       | 如果 x < y 返回 -1, 如果 x == y 返回 0, 如果 x > y 返回 1。 **Python 3 已废弃，使用 (x>y)-(x<y) 替换**。 |
  | math.exp(x)     | 返回e的x次幂(ex),如math.exp(1) 返回2.718281828459045         |
  | math.fabs(x)    | 以浮点数形式返回数字的绝对值，如math.fabs(-10) 返回10.0      |
  | math.floor(x)   | 返回数字的下舍整数，如math.floor(4.9)返回 4                  |
  | math.log(x)     | 如math.log(math.e)返回1.0,math.log(100,10)返回2.0            |
  | math.log10(x)   | 返回以10为基数的x的对数，如math.log10(100)返回 2.0           |
  | max(x1, x2,...) | 返回给定参数的最大值，参数可以为序列。                       |
  | min(x1, x2,...) | 返回给定参数的最小值，参数可以为序列。                       |
  | math.modf(x)    | 返回x的整数部分与小数部分，两部分的数值符号与x相同，整数部分以浮点型表示。 |
  | math.pow(x, y)  | x**y 运算后的值。                                            |
  | round(x ,n)     | 返回浮点数 x 的四舍五入值，如给出 n 值，则代表舍入到小数点后的位数。**其实准确的说是保留值将保留到离上一位更近的一端。**  1.保留整数只有一个小数时:4舍6入5看齐,奇进偶不进  2.保留整数或小数超过一个小数时:看保留位的下下位是否存在 |
  | math.sqrt(x)    | 返回数字x的平方根。                                          |

#### 4.随机数

- 引入random块

  ```python
  import random
  ```

- 常用函数

  | 函数                                | 描述                                                         |
  | ----------------------------------- | ------------------------------------------------------------ |
  | random.choice(seq)                  | 从序列的元素中随机挑选一个元素，比如random.choice(range(10))，从0到9中随机挑选一个整数。 |
  | random.randrange (start, stop,step) | 从指定范围内，按指定基数递增的集合中获取一个随机数，基数默认值为 1 |
  | random.random()                     | 随机生成下一个实数，它在[0,1)范围内。                        |
  | random.shuffle(list)                | 将序列的所有元素随机排序,修改原list                          |
  | uniform(x, y)                       | 随机生成实数，它在[x,y]范围内.                               |

#### 5.三角函数

- 引入math块

   **import math**

- 常用三角函数

  | 函数             | 描述                                              |
  | ---------------- | ------------------------------------------------- |
  | math.acos(x)     | 返回x的反余弦弧度值。                             |
  | math.asin(x)     | 返回x的反正弦弧度值。                             |
  | math.atan(x)     | 返回x的反正切弧度值。                             |
  | math.atan2(y, x) | 返回给定的 X 及 Y 坐标值的反正切值。              |
  | math.cos(x)      | 返回x的弧度的余弦值。                             |
  | math.sin(x)      | 返回的x弧度的正弦值。                             |
  | math.tan(x)      | 返回x弧度的正切值。                               |
  | math.degrees(x)  | 将弧度转换为角度,如degrees(math.pi/2) ， 返回90.0 |
  | math.radians(x)  | 将角度转换为弧度                                  |

#### 6.数学常量

- 引入math块

   **import math**

- 常用数学常量

  | 常量    | 描述                                  |
  | ------- | ------------------------------------- |
  | math.pi | 数学常量 pi（圆周率，一般以π来表示）  |
  | math.e  | 数学常量 e，e即自然常数（自然常数）。 |

### 6.条件语句

#### 1.if

```python
year=1993
if year%4==0:
    print("year能被4整除")
```

#### 2.if-else

```python
year=1993
if year%4==0:
    print("year能被4整除")
else:
    print("year不能被4,400整除")
```

#### 3.ie-elif-else

```python
year=1992
if year%4==0:
    print("year能被4整除")
elif year%400==0:
    print("year能被400整除")
else:
    print("year不能被4,400整除")
```

#### 4.整体

```python
if 条件表达式1:
    语句块1
elif 条件表达式2:
    语句块2
elif 条件表达式3:
    语句块3
...
elif 条件表达式n:
    语句块n
else:
    语句块(其他)
    
```

#### 5.if嵌套

```python
if xxxx:
     if yyyy > 0:
          print('.....')
     else:
          print("fjdsfdf")
else:
    print("hello")
```

### 7.循环语句

#### 1.while语句

- 语法

```python
while 真值表达式:
    语句块1  (*此部分可能会重复执行)
else:
    语句块2
```

- 说明

  - else 子句可以省略
  - else 子句 当且仅当 真值表达式为假Flase的时候 会执行 else 里的语句块2
  - 如果 此 while 是因为 调用 break 语句而终止循环。则 else 子句里的语句不会执行

- while嵌套

  ```python
  while 真值表达式:
      ...
      while 真值表达式2:
          ...
      else:
          ...
      ......
  else:
      ...
  ```

#### 2.break语句

- 语法

  ```python
  break
  ```

- 说明

  - break 语句只能用在 while 语句或for语句的内部。
  - break 语句通常和 if 语句组合使用。
  - 当break 语句执行后，此循环语句break 之后的所有语句都不会执行（else 子句里的语句也不执行）
  - break 语句只能终止包含他的当前循环，当有循环嵌套时，只能跳出离他最近的一个循环

- 实例

  ```python
  i = 1
  while i <= 5:
  	print('i=', i)
      if i == 2
          break
  	i += 1
  else:
      print('循环结束: i=', i)
  ```

- 死循环

  ```python
  while True:
  	语句块
  ```

  - 死循环是指循环条件一直成立的循环
  - 死循环通常使用 break 语句来终止循环
  - 死循环的 else 子句中的语句永远不会执行

#### 3.for语句

- 作用

  用来遍历可迭代对象的数据元素

  可迭代对象

  1. 字符串
  2. ---- 以下后面才讲----
  3. 列表 list
  4. 字典 dict 
  5. ...

- 语法

  ```python
  for 变量列表 in 可迭代对象:
      语句块1
  else:
      语句块2
  ```

- 说明

  - else 子句可以省略
  - else 子句的语句块2 只有在 可迭代对象不再能提供数据的时候才会执行
  - 因为 语句块1 部分调用break 而终止循环式，else 子句部分不会执行。

- 实例

  ```python
  s = 'ABCDE'
  for ch in s:
      print('ch=', ch)
  else:
      print("遍历结束")
  ```

#### 4.range 函数

- 作用

  用来生成一个能够得到一系列整数的可迭代对象（也叫整数序列生成器）

- 使用格式

  ```python
  range(stop)                 # stop 停止整数
  range(start, stop)          # start 开始整数
  range(start, stop, step)    # step 步长
  ```

- 实例

  ```python
  >>> for x in range(4):
  ...     print(x)
  ... 
  0
  1
  2
  3
  >>> for x in range(3, 6):
  ...     print(x)
  ... 
  3
  4
  5
  >>> for x in range(1, 10, 2):
  ...      print(x)
  ... 
  1
  3
  5
  7
  9
  >>> for x in range(5, 0, -2):
  ...     print(x)
  ... 
  5
  3
  1
  
  ```

#### 5.continue语句

- 作用

  用于循环语句(while 语句和for语句)中， 不再执行本次循环内 continue 之后的语句，开始一次新的循环

- 语法

  ```python
  continue
  ```

- 说明

  - 在for 语句中, 执行continue 语句，for语句将会从可迭代对象向中获取下一个元素绑定变量后再次进行循环
  - 在while 中,执行continue 语句, 将会直接跳转到while 语句的真值表达式处，重新判断循环条件。

- 实例

  ```python
  for x in range(5):
      if x % 2 == 0:
          continue
      print(x)    # 1 3 
  ```

#### 6.pass语句

- 作用

  pass是空语句，是为了保持程序结构的完整性。

  pass 不做任何事情，一般用做占位语句

- 实例

  ```python
  for x in  range(10):
      if x == 7:
          pass
      else:
          print(x)
  ```

### 8.容器

#### 1.列表（list）

- 列表是一种可以存储任意个各种类型的序列容器
- 列表内的数据有先后顺序关系
- 列表是可变的容器

##### 1.列表创建

- 创建列表的字面值

  ```python
  >>> L = []     # 创建一个空的列表
  >>> L = ['北京', '上海', '广州', '西安']  # 创建一个含有4个字符串的列表
  >>> L = [1, 'Two', 3.14, True, False, None]
  >>> L = [1, 2, [3.1, 3.2], 4]   #  含有四个元素的列表，第三个元素是列表
  >>> L2 = [
  	['姓名','语文成绩','数学成绩'],
  	['小王', 90, 100],
  	['牛犇', 59, 26]
  ]
  ```

- 创建列表的构造函数 list

  ```python
  L = list()          # L = []
  L = list("ABC")     # L = ['A', 'B', 'C']
  L = list(range(5))  # L = [0, 1, 2, 3, 4]
  ```

##### 2.列表运算

- \+ 用于拼接列表

  ```python
  >>> [1, 2, 3] + [4, 5, 6]   # [1, 2, 3, 4, 5, 6]
  ```

- \+= 追加

  ```python
  >>> L = [1, 2, 3]
  >>> L += [4, 5]         # L = [1, 2, 3, 4, 5]
  >>> L = [1, 2, 3]
  >>> L += "ABC"          # L = [1, 2, 3, 'A', 'B', 'C']
  >>> L += range(2)
  ```

- \* 用于生产重复的列表

  ```python
  >>> [1, 2] * 3    # [1, 2, 1, 2, 1, 2]
  >>> L = [5, 6]
  >>> L *= 3        # L = [5, 6, 5, 6, 5, 6]
  ```

- ==  != 用于比较

  ```python
  >>> [1, 2, 3] == [1, 2, 3]    # True
  >>> [1, 2, 3] != [3, 2, 1]    # True
  ```

- in /not in 用于判断一个数据元素是否在列表中

  ```python
  >>> "hello" in [1, "hello", 'world']
  True
  >>> '红楼梦'  in ['三国演义', '西游记']
  False
  ```

##### 3.列表访问元素

- 索引

  ```python
  L = [1, 2, 3, 4, 5, 6]
  print(L[0])    # 1
  print(L[-1])   # 6
  ```

- 反向索引

  ```python
  list = ['red', 'green', 'blue', 'yellow', 'white', 'black']
  print( list[-1] )
  print( list[-2] )
  print( list[-3] )
  ```

- 切片

  ```python
  x = [1, 2, 3, 4, 5, 6, 7, 8]
  y1 = x[:4]     # y1 = [1, 2, 3, 4]
  y2 = x[::2]    # y2 = [1, 3, 5, 7]
  y3 = x[::-1]   # y3 = [8, 7, 6, 5, 4, 3, 2, 1]
  ```

##### 4.列表数据操作（增删改查）

###### 4.1添加数据

- 方法

  | 方法名(L代表列表)    | 说明                                             |
  | -------------------- | ------------------------------------------------ |
  | L.append(x)          | 向列表的末尾追加单个数据                         |
  | L.insert(index, obj) | 将某个数据obj 插入到 index这个索引位置的数据之前 |
  | L.extend(可迭代对象) | 等同于: L += 可迭代对象                          |

- 实例

  ```python
  mylist1 = [1, 3, 4]            # 目标是变成 [1, 2, 3, 4, 5]
  mylist1.append(5)               # mylist1 = [1, 3, 4, 5]
  mylist1.insert(1, 2)            # mylist1 = [1, 2, 3, 4, 5]
  mylist1.extend(range(6, 10))    # mylist1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  ```

###### 4.2修改数据

- 方法

  ```
  列表[整数表达式] = 表达式
  ```

- 实例

  ```python
  mylist2 = [1, 1.99, 3]   # 把1.99 改为2
  mylist2[1] = 2    # mylist2 = [1, 2, 3]
  ```

###### 4.3删除数据

- 方法

  | 方法        | 说明                                                         |
  | ----------- | ------------------------------------------------------------ |
  | L.remove(x) | 从列表L中删除第一次出现在列表中的数据元素，如果x不存在则报错 |
  | L.clear()   | 清空列表                                                     |

- 实例

  ```python
  L = [1, 2, 3, 4, 2, 2, 3, 4]
  L.remove(3)    #  L = [1, 2, 4, 2, 2, 3, 4]
  L.remove(3)    #  L = [1, 2, 4, 2, 2, 4]
  L.remove(3)    #  报错了
  L.clear()      #  L = []
  ```

- del 语句删除指定位置的数据元素

  ```python
  L = ['张飞', '赵云', '鲁班7号', '孙悟空']
  del L[2]    # L = ['张飞', '赵云', '孙悟空']
  del L       # 删除 L 变量
  ```

###### 4.4查找数据

```python
print(L[0])  # 取值
```

##### 5.列表嵌套

```python
a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]
print(x)#[['a', 'b', 'c'], [1, 2, 3]]
print(x[0])#['a', 'b', 'c']
print(x[0][1])#b
```

##### 6.常用列表API

###### 6.1操作列表函数

| 序号 | 函数                        |
| ---- | --------------------------- |
| 1    | len(list)列表元素个数       |
| 2    | max(list)返回列表元素最大值 |
| 3    | min(list)返回列表元素最小值 |
| 4    | list(seq)将元组转换为列表   |

###### 6.2列表常用方法

| 运算                                 | 结果                                                         |
| :----------------------------------- | :----------------------------------------------------------- |
| `s.index(x[, i[, j]])`               | *x* 在 *s* 中首次出现项的索引号（索引号在 *i* 或其后且在 *j* 之前） |
| `s.count(x)`                         | *x* 在 *s* 中出现的总次数                                    |
| `s.append(x)`                        | 将 *x* 添加到序列的末尾 (等同于 `s[len(s):len(s)] = [x]`)    |
| `s.clear()`                          | 从 *s* 中移除所有项 (等同于 `del s[:]`)                      |
| `s.copy()`                           | 创建 *s* 的浅拷贝 (等同于 `s[:]`)                            |
| `s.extend(t)` 或 `s += t`            | 用 *t* 的内容扩展 *s* (基本上等同于 `s[len(s):len(s)] = t`)  |
| `s.insert(i, x)`                     | 在由 *i* 给出的索引位置将 *x* 插入 *s* (等同于 `s[i:i] = [x]`) |
| `s.pop([i])`                         | 提取在 *i* 位置上的项，并将其从 *s* 中移除                   |
| `s.remove(x)`                        | 删除 *s* 中第一个 `s[i]` 等于 *x* 的项目。                   |
| `s.reverse()`                        | 就地将列表中的元素逆序。                                     |
| `s.sort( key=None, *reverse=False*)` | 对列表内的数据进行排序, reverse=False 升序排序，否则是降序排序 |

- 实例

  ```python
  s = [1, "二", 3]
  value = s.pop(1)   # s= [1, 3]; value = '二'
  >>> s = [1, 2, 3, 5]
  >>> s.reverse()   # 反转
  >>> s
  [5, 3, 2, 1]
  >>> s = [1, 2, 3, 4, 2, 2, 3, 4]
  >>> s.index(3)  # 返回第一次出现的位置的索引
  2
  >>> s.index(3, 4)
  6
  >>> s.index(100)  # 触发异常，要用try 语句处理
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ValueError: 100 is not in list
  >>> s.count(3)  # 在列表 s 中 找出所有值为3 的元素的个数，并返回
  2
  >>> s.count(2)
  3
  >>> L1 = [2, 4, 6, 8, 9, 1]
  >>> L1.reverse()
  >>> L1
  [1, 9, 8, 6, 4, 2]
  >>> s.sort()   # 排序，默认是升序排序
  >>> s
  [1, 2, 2, 2, 3, 3, 4, 4]
  >>> s.sort(reverse=True)
  >>> s
  [4, 4, 3, 3, 2, 2, 2, 1]
  ```

##### 7.深浅拷贝区别

- 浅拷贝（Shallow Copy):

  只复制对象本身，而不复制嵌套对象。对于嵌套对象，浅拷贝只复制其引用。

  修改浅拷贝中的嵌套对象会影响原对象中的嵌套对象。

- 深考贝（Deep Copy):

  ```python
  import copy
  ```

  递归地复制对象及其所有嵌套对象。

  修改深拷贝中的嵌套对象不会影响原对象中的嵌套对象。

#### 2.元组（tuple）

- 元组是不可改变的列表
- 同列表list 一样，元组可以存放任意类型的数据
- 但是，一旦创建将不可修改
- 元组使用小括号 ( )，列表使用方括号 [ ]

##### 1.元组的创建

- 创建元组的字面值

  ```python
  t = ()         # 空元组
  t = (100,)     # 含有一个数据元素的元组
  t = 100,       # 含有一个数据元素的元组，元组中只包含一个元素时，需要在元素后面添加逗号 ，否则括号会被当作运算符使用
  t = (1, 2, 3)  # 含有三个数据元素的元组
  t = ( 'hqyj', 2004) # 存放不同类型的元组
  t = 1, 2, 3    # 含有三个数据元素的元组
  ```

- 创建元组的函数

  ```python
  t = tuple()          # t = ()
  t = tuple(range(5))  # t = (0, 1, 2, 3, 4)
  ```

##### 2.元组的数据操作（删查）

###### 2.1元组删除

```python
tup = ('openAI', 'hqyj', 100, 200) 
print (tup)
del tup
print (tup)#name 'tup' is not defined
```

###### 2.2元组查找

```python
tup1 = ('python', 'hqyj', 100, 200)
tup2 = (1, 2, 3, 4, 5, 6, 7 )
print (tup1[0])#python
print (tup2[1:5])#(2, 3, 4, 5)
print (tup2[:4])#(1, 2, 3, 4)
print (tup2[2:])#(3, 4, 5, 6, 7)
```

###### 2.3元组运算

```python
tup1 = (12, 34.56)
tup2 = ('abc', 'xyz') 
# 创建一个新的元组
tup3 = tup1 + tup2
print (tup3)

>>> t = (1, 2, 3) + (4, 5, 6)
>>> t += (7, 8, 9)  # 等同于 t = t + (7, 8, 9)
>>> t = t * 2
>>> t *= 2
>>> 5 in t
True
```

##### 3.元组不可变

```python
tup = (1, 2, 3, 4, 5, 6, 7)
tup[1] = 100
print(tup)#报错'tuple' object does not support item assignment
```

##### 4.元组常用API

| 序号 | 方法        | 描述                   |
| ---- | ----------- | ---------------------- |
| 1    | len(tuple)  | 返回元组中元素个数。   |
| 2    | max(tuple)  | 返回元组中元素最大值。 |
| 3    | min(tuple)  | 返回元组中元素最小值。 |
| 4    | tuple(list) | 将列表转换为元组。     |

##### 5.元组常用方法

| 运算                 | 结果                                                         |
| :------------------- | :----------------------------------------------------------- |
| s.index(x[, i[, j]]) | *x* 在 *s* 中首次出现项的索引号（索引号在 *i* 或其后且在 *j* 之前） |
| s.count(x)           | *x* 在 *s* 中出现的总次数                                    |

#### 3.字典（dict）

- 字典是一种可变容器模型，且可存储任意类型对象。
- 字典的数据都是以键(key)-值(value)对的形式进行映射存储.
- 字典的每个键值对( key:value )用冒号分割，每个对之间用逗号分割，整个字典包括在花括号 {} 中 
- **d = {key1 : value1, key2 : value2, key3 : value3 }**
- 字典的数据是无序的
- 字典的键不能重复，且只能用不可变类型作为字典的键
- 字典中的数据只能用"键"key 进行索引，不能用整数进行索引

##### 1.字典创建

- 创建字典的字面值

  ```python
  d = {}    # 创建空字典
  d = {'name': "weimingze", "age": 35}
  d = {'a': [1, 2, 3]}
  d = {'b': {"bb": 222}}
  d = {1:'壹', 2:'贰', 5:'伍'}
  d = {(1, 2, 3):'壹贰伍'}
  ```

  - 键必须是唯一的，但值则不必。

  - 值可以取任何数据类型，但键必须是不可变的，如字符串，数字。

    ```python
    d = {'a': 1, 'b': 2, 'a': 3}  # 字典的键不能重复 d = {'a': 3, 'b': 2}  
    d = {[1, 2, 3]: 'a'}          # 不能用可变类型作为字典的键  # 报错
    ```

- 字典的创建函数

  ```python
  d = dict()   # d = {}
  d = dict([("name", "小王"), ("age", 35)])  # {'name': '小王', 'age': 35}
  d = dict(a=1, b=2, c=3)    # {'a':1, 'b':2, 'c':3}
  d = dict([1, 2, 3, 4])  # 错
  ```

##### 2.字典数据操作（增删改查）

###### 2.1增加修改字典

- 语法

  ```python
  字典[键key] = 表达式
  ```

- 实例

  ```python
  d = {}
  d['name'] = 'tarena'  # 添加键值对  d = {'name': 'tarena'}
  d['age'] = 18         # d = {'name': 'tarena', 'age': 18}
  d['age'] = 19         # 改变 'age' 键对应的值 d = {'name': 'tarena', 'age': 19}
  ```

###### 2.2访问字典的值

- 键索引查找

  语法：

  ```
  字典[键key]
  ```

  实例

  ```python
  d = {'one': 1, 'two': 2}
  print(d['two'])
  ```

- in / not in  运算符

  ```python
  >>> d = dict(a=1, b=2)   # d = {'a': 1, 'b': 2}
  >>> 'a' in d
  True
  >>> 1 in d
  False
  >>> 'hello' not in d
  True
  ```

###### 2.3删除字典元素

```python
mydic = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
 
del mydic['Name'] # 删除键 'Name'
mydic.clear()     # 清空字典
 
print (mydic['Age'])
print (mydic['School'])

del mydic         # 删除字典
```

##### 3.字典键的特性

- 字典值可以是任何的 python 对象，既可以是标准的对象，也可以是用户定义的，但键不行
- 不允许同一个键出现两次。创建时如果同一个键被赋值两次，后一个值会被记住
- 键必须不可变，所以可以用数字，字符串或元组充当，而用列表等就不行
- 四种可变类型：列表、字典、集合、字节数组

##### 4.字典常用API

###### 4.1操作字典的函数

| 序号 | 函数           | 描述                                               |
| ---- | -------------- | -------------------------------------------------- |
| 1    | len(dict)      | 计算字典元素个数，即键的总数。                     |
| 2    | str(dict)      | 输出字典，可以打印的字符串表示。                   |
| 3    | type(variable) | 返回输入的变量类型，如果变量是字典就返回字典类型。 |

###### 4.2字典的方法

| 序号 | 函数及描述                                                   |
| ---- | ------------------------------------------------------------ |
| 1    | dict.clear()删除字典内所有元素                               |
| 2    | dict.copy()返回一个字典的浅复制                              |
| 3    | dict.fromkeys(seq)创建一个新字典，以序列seq中元素做字典的键，val为字典所有键对应的初始值 |
| 4    | dict.get(key, default=None)返回指定键的值，如果键不在字典中返回 default 设置的默认值 |
| 5    | key in dict如果键在字典dict里返回true，否则返回false         |
| 6    | dict.items()以列表返回一个视图对象                           |
| 7    | dict.keys()返回一个视图对象                                  |
| 8    | dict.setdefault(key, default=None)和get()类似, 但如果键不存在于字典中，将会添加键并将值设为default |
| 9    | dict.update(dict2)把字典dict2的键/值对更新到dict里           |
| 10   | dict.values()返回一个视图对象                                |
| 11   | pop(key,default)删除字典 key（键）所对应的值，返回被删除的值。 |
| 12   | popitem()返回并删除字典中的最后一对键和值。                  |

- 实例

  ```python
  >>> d = {'name': 'tarena', 'age': 19}
  >>> 
  >>> d['age']
  19
  >>> d['address']
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  KeyError: 'address'
  >>> d.get('address', '未填写住址')
  '未填写住址'
  >>> d.get('age', 0)
  19
  >>> n = d.pop('name')
  'tarena'
  ```

##### 5.列表、元组、字典小结

- 列表和元组是有序的，字典的存储是无序的

- 列表、字典是可变的,元组是不可变的

- 字典的键索引速度快，列表的索引速度也快

- 列表和元组是顺序存储的，字典是散列存储的

- 字典的 in / not in 运算符快于列表的 in / not in 运算符

#### 4.集合（set）/固定集合（frozenset）

-  集合是可变的容器，固定集合是不可变的集合 
-  集合相当于只有键没有值的字典
-  集合是无序的不重复元素的存储结构

-  集合内的数据都是唯一的，不可变的(去重特性)

##### 1.集合的创建

- 创建集合的字面值

  ```python
  s = set()            # 用函数空集合
  s = {1, 2, 3, 4}     # 创建非空集合的字面值
  s = set(range(5))    # 调用 set(可迭代对象) 来创建集合 s = {0, 1, 2, 3, 4}
  s = set("ABC")       # s = {'B', 'C', 'A'}
  s = set("ABCCCCCCC")  # s = {'B', 'C', 'A'}
  s = set(['ABC'])      # s = {'ABC'} 使用 set()函数从列表创建集合
  s = set((4, 5, 6, 7))# 使用 set()函数从元组创建集合
  ```

- 创建固定集合frozensets

  ```python
  fs = frozenset()              # 空固定集合 fs = frozenset()
  fs = frozenset([1, 2, 3])     # fs =  frozenset({1, 2, 3})
  ```

##### 2.集合的数据操作

###### 2.1添加元素

- **s.add( x ) 添加元素到集合** 

- **s.update( x ) 添加元素到集合，且参数可以是列表，元组，字典等** ,**x 可以有多个，用逗号分开**

  ```python
  s1 = set((4, 5, 6, 7))
  s1.add(100)
  print(s1)
  s1.update([200,300])
  print(s1)
  ```

###### 2.2删除元素

- **s.remove( x )：将元素 x 从集合 s 中移除，如果元素不存在，则会发生错误。**

- **s.discard( x ))：将元素 x 从集合 s 中移除，如果元素不存在，不会发生错误。**

- **s.pop())：对集合进行无序的排列，然后将这个无序排列集合的左面第一个元素进行删除。** 

- **x in s  判断元素 x 是否在集合 s 中，存在返回 True，不存在返回 False。**

- **集合（set）没有索引查看的功能。集合是一种无序、不可重复的数据结构，用于存储唯一的元素。由于集合是无序的，所以不能通过索引来访问其中的元素。所以也没有对应的修改功能。**

  ```python
  s1 = {10, 20, 30}
  print(20 in s1)
  ```

  ```python
  s1 = {10, 20, 30}
  s1.remove(20)
  print(s1)
  s1.remove(40)#报错
  ```

  ```python
  s1 = {10, 20, 30}
  s1.discard(20)
  print(s1)
  s1.discard(40)
  ```

  ```python
  s1 = {10, 20, 30}
  s1.pop()
  print(s1)
  
  del s1 # 也可以直接删除整个集合
  ```



##### 3.集合的常用API

| 方法                          | 描述                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| add()                         | 为集合添加元素                                               |
| clear()                       | 移除集合中的所有元素                                         |
| copy()                        | 拷贝一个集合                                                 |
| difference()                  | 返回多个集合的差集                                           |
| difference_update()           | 移除集合中的元素，该元素在指定的集合也存在。                 |
| discard()                     | 删除集合中指定的元素                                         |
| intersection()                | 返回集合的交集                                               |
| intersection_update()         | 返回集合的交集。                                             |
| isdisjoint()                  | 判断两个集合是否包含相同的元素，如果没有返回 True，否则返回 False。 |
| issubset()                    | 判断指定集合是否为该方法参数集合的子集。                     |
| issuperset()                  | 判断该方法的参数集合是否为指定集合的子集                     |
| pop()                         | 随机移除元素                                                 |
| remove()                      | 移除指定元素                                                 |
| symmetric_difference()        | 返回两个集合中不重复的元素集合。                             |
| symmetric_difference_update() | 移除当前集合中在另外一个指定集合相同的元素，并将另外一个指定集合中不同的元素插入到当前集合中。 |
| union()                       | 返回两个集合的并集                                           |
| update()                      | 给集合添加元素                                               |
| len()                         | 计算集合元素个数                                             |

#### 5.容器总结

- 容器总结

  - 类型

    - 列表 list
    - 元组 tuple
    - 字典 dict
    - 集合 set    /  固定集合 frozenset

  - 可变和不可变

    - 可变的容器

    ```
    list   dict    set       
    ```

    - 不可变得容器

    ```
    tuple, frozenset
    ```

  - 有序和乱序

    - 有序

      ```
      list   tuple   
      ```

    - 无序

      ```
      dict    set    frozenset
      ```

## day_3

### Python推导式

#### 1.列表推导式

- 语法

  ```python
  [ 表达式 for 自定义变量 in 可迭代对象 ]
  # 或
  [ 表达式 for 自定义变量 in 可迭代对象 if 真值表达式 ]
  ```

- 实例

  ```python
  # 生成一个列表， 里面有 100 个数是[1, 4, 9, 16, 25, ...]
  # 用 for 语句实现
  L = []
  for x in range(1, 101):
      L.append(x ** 2)
  print(L)
  
  # 用列表推导式
  L2 = [ x ** 2 for x in range(1, 101)]
  print(L2)
  
  L3 = []
  for x in range(1, 101):
      if x % 2 == 0:
          L3.append(x ** 2)
  
  L3 = [ x ** 2 for x in range(1, 101) if x % 2 == 0]  # 取出所有的偶数
  # L3 = [4, 16, 36, ...]
  ```

#### 2.字典推导式

- 语法

  ```python
  { 键表达式: 值表达式 for 元素 in 集合 }
  { 键表达式: 值表达式 for 元素 in 集合 if 条件 }
  ```

- 实例

  ```python
  # 将列表中各字符串值为键，各字符串的长度为值，组成键值对
  listdemo = ['karen','jack', 'marry']
  newdict = {key:len(key) for key in listdemo}
  print(newdict)#{'karen': 5, 'jack': 4, 'marry': 5}
  
  
  #提供三个数字，以三个数字为键，三个数字的平方为值来创建字典：
  dic = {x: x**2 for x in (2, 4, 6)}
  print(dic)#{2: 4, 4: 16, 6: 36}
  ```

#### 3.集合推导式

- 语法

  ```python
  { 表达式 for 元素 in 序列 }
  { 表达式 for 元素 in 序列 if 条件 }
  ```

- 实例

  ```python
  #计算数字 1,2,3 的平方数：
  setnew = {i**2 for i in (1,2,3)}
  print(setnew)#{1, 4, 9}
  
  
  #判断不是 abc 的字母并输出：
  a = {x for x in 'abracadabra' if x not in 'abc'}
  print(a)#{'d', 'r'}
  ```

#### 4.元组推导式

- 特性

  元组推导式可以利用 range 区间、元组、列表、字典和集合等数据类型，快速生成一个满足指定需求的元组，元组推导式返回的结果是一个生成器对象。

- 语法

  ```python
  （表达式 for 元素 in 序列 ）
  （表达式 for 元素 in 序列 if 条件 ）
  ```

- 实例

  ```python
  #生成一个包含数字 1~9 的元组
  a = (x for x in range(1,10))
  print(a)#返回的是生成器对象
  print(tuple(a))#使用 tuple() 函数，可以直接将生成器对象转换成元组
  ```

  

### Python函数

#### 1.函数定义

##### 1.1 def语句

- 语法

  ```python
  def 函数名(形式参数列表):
  	语句块
  ```

- 说明

  1. 函数代码块以 def 关键词开头，后接函数标识符名称和圆括号 () 2.
  2.  函数名是一个变量，不要轻易对其赋值
  3.  函数有自己的名字空间，在函数外部不可以访问函数内部的变量，在函数内部可以访问函数外 部的变量，但不能轻易对其改变 
  4. 函数的形参列表如果不需要传入参数，形式参数列表可以为空 
  5. 任何传入参数和自变量必须放在圆括号中间，圆括号之间可以用于定义参数。
  6. 函数内容以冒号 : 起始，并且缩进。
  7. return [表达式] 结束函数，选择性地返回一个值给调用方，不带表达式的 return 相当于返回  None。

- 实例

  ```python
  # 定义一个函数，用 say_hello 变量绑定
  def say_hello():
   print("hello world!")
   print("hello tarena!")
   print("hello everyone!")
   # 定义一个函数，传入两个参数，让这个函数把最大的值打印到终端
  def mymax(a, b):
   if a > b:
   print("最大值是", a)
   else:
   print("最大值是", b)
  ```

##### 1.2 函数调用

- 语法

  ```python
  函数名(实际调用传递参数)
  ```

- 说明

  函数调用是一个表达式 

  如果函数内没有return 语句，函数执行完毕后返回 None 对象

- 实例

  ```python
  # 调用
  say_hello()  # 调用一次
  say_hello()  # 调用第二次
  # 调用
  mymax(100, 200)
  mymax(999, 1)
  mymax('abc', 'cba')
  ```

##### 1.3 return语句

- 语法

  ```python
  return [表达式]
  ```

- 说明

  1. return 语句后面的表达式可以省略，省略后相当于 return None
  2. 如果函数内部没有 return 语句, 则函数执行完毕后返回None, 相当于在最后一条语句后有一条 return None

- 实例

  ```python
  def say_hello():
   print("hello aaa")
   print("hello bbb")
   return 1 + 2
   print("hello ccc")
   r = say_hello()
   print(r)   
  # 3
  ```

#### 2.函数参数

##### 2.1函数调用传参方式

- 位置传参

  实际参数传递时，实参和形参按**位置**来依次对应

- 关键字传参

  实际参数传递时，实参和形参 按**名称依次**对应

- 序列实参、字典形参

  使用*、**分别拆分

- **位置传参要先于关键字传参**

- 实例

  ```python
  def myfun1(a, b, c):
   print('a=', a)
   print('b=', b)
   print('c=', c)
   ## 位置传参
  myfun1(1, 2, 3)
   ## 关键字传参
  myfun1(c=33, a=11, b=22)
   ## 位置传参要先于关键字传参
  myfun1(111, c=333, b=222)   # 正确
  
  #序列实参:使用星号将序列拆分后，与形参进行对应
  listo1 = [7,8,9]
  myfun01(*list01)
  #字典实参:使用双星号将字典拆分后，依次与形参对应
  dicto1 = {i"c":3 , "b":2, "a" : 1}
  myfun01(**dict01)
  ```

##### 2.2函数形参的传递方式

###### 2.2.1函数缺省参数（默认参数）

- 语法

  ```python
  def 函数名(形参名1=默认实参1, 形参名2=默认实参2, ... ):
  	语句块
  ```

- 说明

  缺省参数即默认实参，必须自右向左依次存在(即,如果一个参数有缺省参数，则其右侧的所有参数都必须 有缺省参数

- 实例

  ```python
  def myadd4(a, b, c=0, d=0):    
  print(a)
   print(b)
   print(c)
   print(d)
   return a + b + c + d
   print(myadd4(1, 2))
   print(myadd4(1, 2, 3))
   print(myadd4(1, 2, 3, 4))
  ```

###### 2.2.2位置形参

- 语法

  ```python
   def 函数名(形参名1, 形参名2, ...):
  	 pass
  ```

###### 2.2.3星号元组形参

- 语法

  ```python
  def 函数名(*元组形参名):
  	 pass
  ```

- 说明

  收集多余的位置实参,元组形参名一般命名为args

- 实例

  ```python
  def myfunc2(*args):
   	print("len(args)=", len(args))
   	print('args=', args)
  myfunc2()           
  myfunc2(1, 2, 3)    
  # args=()
  # args=(1, 2, 3)
  def myfunc3(a, b, *args):
  	print(a, b, args)
  myfunc3(1, 2)        
  # 1-->a, 2-->b, ()--->args
  myfunc3(1, 2, 3, 4)  # # 1-->a, 2-->b, (3, 4)--->args
  ```

###### 2.2.4命名关键字形参

- 语法

  ```python
  def 函数名(*, 命名关键字形参1, 命名关键字形参2, ...):
   	pass
   # 或者
  def 函数名(*args, 命名关键字形参1, 命名关键字形参2, ...):
  	pass
  ```

- 说明

  强制，所有的参数都必须用关键字传参

- 实例

  ```python
  def myfunc4(a, b,*args, c, d):
   	print(a, b, c, d)
  myfunc4(1, 2, d=4, c=3)   # 正确,c,d 必须关键字传参
  myfunc4(1, 2, 3, 4)   # 错误
  ```

###### 2.2.5双星号字典形参

- 语法

  ```python
  def 函数名(**字典形参名):
   	pass
  ```

- 说明

  收集多余的关键字传参 

  字典形参名最多有一个， 

  字典形参名 一般命名为 kwargs

- 实例

  ```python
  def myfunc5(**kwargs):
   	print(kwargs)
  # {'name': 'tarena', 'age': 18}-->kwargs
  myfunc5(name='tarena', age=18) 
  ```

###### 2.2.6混合使用

- 位置形参，星号元组形参，命名关键字参数，双星号字典形参，缺省参数可以混合使用。 

- 函数的形参定义自左至右的顺序为：位置形参，星号元组形参，命名关键字参数，双星号字典形参

- 实例

  ```python
  def fn(a, b, *args, c, d, **kwargs):
       print(a)
       print(b)
       print(*args)
       print(c)
       print(d)
       print(kwargs)
  fn(100, 200, 300, 400, c='C',name='tarena',d='D')
  ```

##### 2.3可变与不可变

在 python 中，strings, tuples, 和 numbers 是不可更改的对象，而 list,dict 等则是可以修改的对象。

- **不可变类型**：变量赋值 a=5 后再赋值 a=10，这里实际是新生成一个 int 值对象 10，再让 a 指向 它，而 5 被丢弃，不是改变 a 的值，相当于新生成了 a。 
- **可变类型**：变量赋值 la=[1,2,3,4] 后再赋值 la[2]=5 则是将 list la 的第三个元素值更改，本身la没有 动，只是其内部的一部分值被修改了。

python 函数的参数传递：

- **不可变类型**：值传递:  如整数、字符串、元组。如 fun(a)，传递的只是 a 的值，没有影响 a 对象本 身。如果在 fun(a) 内部修改 a 的值，则是新生成一个 a 的对象。 

- **可变类型**：引用传递:  如 列表，字典。如 fun(la)，则是将 la 真正的传过去，修改后 fun 外部的 la  也会受影响

- 传可变对象实例

  ```python
  def changeme( mylist ):
   	mylist.append(40)
   	print (mylist)
  mylist = [10,20,30]
  changeme(mylist)#[10, 20, 30, 40]
  print (mylist)#[10, 20, 30, 40]
   #总结:传入函数的和在末尾添加新内容的对象用的是同一个引用
  ```

- 传不可变对象实例

  ```python
  def change(a):
   	print(id(a))
  a=10
  print(id(a))#1428793408
  a=1
  print(id(a))#1428793264
   #总结:可以看见在调用函数前后，形参和实参指向的是同一个对象（对象 id 相同），在函数内部修改形参后，形参指向的是不同的id。
  ```

#### 3.函数返回值

函数可以使用 return 语句来返回一个或多个值

如果没有明确的 return 语句，函数将默认返回None

```python
#没有return
 def fn1():
 	print("调用了函数")
 re=fn1()
 print(re)#None
 #返回一个值

def fn2(x):
 	return  x+100
re2=fn2(100)
print(re2)#200

 #返回多个值
def fn3(x):
 	return x*x,x%3
re3=fn3(100)
print(re3)#(10000, 1)
x,y=fn3(100)
print(x,y)#10000 1
```

#### 4.匿名函数

在Python中，匿名函数通常使用 lambda 关键字来创建。匿名函数也被称为lambda函数，它是一种简单的、一行的函数，常用于临时需要一个小函数的地方。

- 语法

  ```python
  lambda  [函数的参数列表]: 表达式
  ```

- 说明

  创建一个匿名函数对象 

  **lambda** 是关键字，表示你正在定义一个匿名函数。

  同 def 类似，但不提供函数名 

  **[函数的参数列表]**是函数的参数，可以有零个或多个参数，参数之间用逗号分隔。 

  **: 表达式** 是函数的返回值，通常是一个表达式，匿名函数会计算这个表达式并返回结果。

  lambda 表达式 的创建函数只能包含一个表达式

- 实例

  ```python
  def myadd(x, y):
   	return x + y
  print('1 + 2 =', myadd(1, 2))  # 3
  
  # myadd 函数可以改写成 
  myadd2 = lambda x, y: x + y
  print('3 + 4 =', myadd2(3, 4))  # 7
  
  square = lambda x: x * x
  print(square(5))  # 输出: 25
  
  # 1. 有参数有返回值
  # def func01(a,b):
   #     
  return a > b
   #
   # print(func01(10,20))
   func01 = lambda a,b:a > b
   print(func01(10,20))
   # 2. 无参数有返回值
  # def func02():
   #     
  return "ok"
   #
   # print(func02())
   func02 = lambda :"ok"
   print(func02())
   # 3. 无参数无返回值
  # def func03():
   #     
  print("ok")
   #
   # func03()
   func03 = lambda :print("ok")
   func03()
  # 4. 有参数无返回值
  # def func03(a):
   #     
  print(a)
   #
   # func03(100)
   func03 = lambda a:print(a)
   func03(100)
   # 5. lambda 不支持赋值语句
  # def func05(iterable):
   #     
  iterable[0] = 100
   #
   # list01 = [1]
   # func05(list01)
   # print(list01)
   # func05 = lambda iterable: iterable[0] = 100 报错
  # 6. lambda 不支持多条语句
  # def func06(a,b):
   #     
  print(a)
   #     
  print(b)
   #
   # func06(10,20)
   # func06 = lambda a,b: print(a);print(b)
  ```

#### 5.变量作用域

##### 5.1变量作用域的定义

​		一个变量声明以后,在哪里能够被访问使用,就是这个变量"起作用"的区域:也就是这个变量的作用域 一般来说,变量的作用域,是在函数内部和外部的区域 来体现,因此常常与函数有关

##### 5.2局部变量和全局变量

- 局部变量 

​		定义在函数内部的变量称为局部变量(函数的形参也是局部变量) 

​		函数内部的变量只能在函数内部或者函数内部的函数内部访问 ，函数外部不能访问 

​		局部变量在函数调用时才能够被创建，在函数调用之后会自动销毁

- 局部变量实例

  ```python
  >>> def fn(a, b):
   ...     c = 100
   ...     print(a, b, c)    # a, b, c三个都是局部变量
  >>> fn(1, 2)
  >>> print(a, b, c)   # 报错， 因为a,b,c 在调用后就销毁了
  ```

- 全局变量

  定义在函数外部，模块内部的变量称为全局变量 

  全局变量, 所有的函数都可以直接访问(取值,但函数内部不能直接将其赋值改变)

- 全局变量实例

  ```python
  a = 100  # 全局变量
  def fx(b, c):   # b, c 局部变量
  	d = 400     # d 局部变量
  	print(a, b, c, d)
  fx(200, 300)
  print(a)  # 100
  print(b)  # 报错, 因为此时 b 不存在了
  ```

  ```python
  a = 100  # 全局变量
  def fx(b):
   	a = 666  # 创建局部变量，不是改变全局变量
  	c = 300
   	print(a, b, c)  # 优先访问局部变量
  fx(200)  # 666 200 300
  print(a)  # 100
  ```

  

##### 5.3局部变量域修改全局变量

- 语法

  ```python
  global 变量名1, 变量名2, ....
  ```

- 说明

  **全局变量如果要在函数内部被赋值，则必须经过全局声明 global** 

  **默认全局变量在函数内部可以使用，但只能取值，不能赋值** 

  **不能先声明为局部变量，再用 global 声明为全局变量，此做法不符合语法规则** 

  **函数的形参已经是局部变量，不能用 global 声明为全局变量**

- 实例

  ```python
  # 如何用一个变量来记录一个函数调用的次数
  count = 0
  
  def hello(name):
   	global count  # 声明 global 是全局变量
  	print('hello', name)
   	count += 1  #  等同于 count = count + 1
  hello('小张')
  hello('小李')
  hello('小魏')
  
  print('您共调用hello函数', count, '次')  # 3 次
  ```

##### 5.4局部变量域修改外部变量

- 变量查找顺序

  **LEGB顺序**

  ​	**Local (L):** 

  ​		本地作用域，指当前函数内部的变量。 

  ​		当你在函数内部定义变量并尝试访问它时，Python 首先会在函数内部查找这个变量。

  ```python
  def outer_function():
  	 x = 10  # Local variable
   	 print(x)  # 查找顺序从 Local 开始
  ```

  ​	**Enclosing (E):** 

  ​		闭包函数外的函数作用域，指嵌套函数的外部函数中定义的变量。

   	   如果在当前函数内部找不到变量，Python 会查找外层（闭包）函数中的变量。

  ```python
  def outer_function():
   	x = 10  # Enclosing variable
   	def inner_function():
   		print(x)  # 查找顺序从 Enclosing 开始
  	inner_function()
  ```

  ​	 **Global (G):** 

  ​		全局作用域，指模块级别定义的变量。 

  ​		如果在本地和闭包函数中找不到变量，Python 会查找全局作用域的变量。

  ```python
  x = 20  # Global variable
  def outer_function():
   	print(x)  # 查找顺序从 Global 开始
  ```

  ​	**Built-in (B):**

  ​		 内建作用域，指 Python 预定义的变量、函数等，如  len 、 sum 等。

  ​		 如果在以上三个作用域中都找不到变量，Python 会查找内建作用域。

  ```python
  def outer_function():
  	print(len)  # 查找顺序从 Built-in 开始
  ```

- 语法

  ```python
  nonlocal 变量名
  ```

- 说明

  （1）变量的查找顺序还是遵从：LEGB顺序  在 Python 中，LEGB 代表四种作用域的查找顺序：Local、Enclosing、Global 和 Built-in。 

  （2）局部作用域中若要修改外部函数嵌套作用域中的变量需要使用：nonlocal  语句 

- 实例

  ```python
  """
  	外部嵌套作用域
  """
  
  def func01():
   	a = 10
   	def func02():
  		# 内部函数,可以访问外部嵌套变量
  		# print(a)
  		# 内部函数,如果修改外部嵌套变量,需要使用nonlocal语句声明
  		nonlocal a
   		a = 20
   	func02()
   	print(a)
  func01()
  ```

#### 6.函数内存分配

**1、将函数的代码存储到代码区，函数体中的代码不执行。** 

**2、调用函数时，在内存中开辟空间（栈帧），存储函数内部定义的变量。** 

**3、函数调用后，栈帧立即被释放。**

- 实例

  ```python
  def func(a, b):
  	a = 20
   	b[0] = 20
   a = 10
   b = [100]
   func(a, b)
   print(a) # 10
   print(b) # 20
  ```

​	**(1) 不可变类型参数有: 数值型(整数，浮点数)、 布尔值bool、None空值、字符串str、 元组tuple**

​	**(2) 可变类型参数有: 列表 list、 字典 dict、 集合 set**

​	**(3)不可变类型的数据传参时，函数内部不会改变原数据的值。**

​	**(4) 可变类型的数据传参时，函数内部可以改变原数据。**

#### 7.函数的递归

- 说明

  （1）递归一定要控制递归的层数，当符合某一条件时要终止递归调用 

  （2）几乎所有的递归都能用while循环来代替

- 实现方法

  ```python
  # 函数直接的调用自身
  def f():
  	 f()  # 调用自己
  	 #此递归达最大深度出错
  # f()
  print("递归完成")
  
   # 函数直接的调用自身
  def fa():
   	fb()
  def fb():
   	fa()
  fa()
  print("递归完成")
  ```

- 实例

  （1）非递归实现阶乘

  ```python
  def factorial(n):
  	 result = 1
   	 i = 1
       while i <= n:
   		result *= i
   		i+=1
   	 return result
  print("factorial: ", factorial(5))
  ```

  （2）递归实现阶乘

  ```python
  # file: factorial.py
  def factorial(n):
   	if n == 1:
   		return 1
   	s = n * factorial(n-1)
   	return s
  print("factorial: ", factorial(5))
  ```

- 递归优缺点

  **优点： 递归可以把问题简单化，让思路更为清晰,代码更简洁** 

  **缺点： 递归因系统环境影响大，当递归深度太大时，可能会得到不可预知的结果**

## Day_04

### Python的类和对象

#### 1.类和对象基础语法

##### 1.1类的定义

- 数据成员：表明事物的特征。  相当于变量 

- 方法成员：表明事物的功能。  相当于函数

-  通过c lass 关键字定义类。

- 方法

  ```python
  class 类名 (继承列表):
  	实例方法(类内的函数method) 定义
  	类变量(class variable) 定义
  	类方法(@classmethod) 定义
  	静态方法(@staticmethod) 定义
  ```

- 说明

  类名必须为标识符(与变量的命名相同,建议首字母大写) 

  类名实质上就是变量，它绑定一个类

- 实例

  ```python
  class Dog:  # 定义一个Dog类
  	pass
  
  class Person:
  	def __init__(self, name, age):
  		self.name = name
  		self.age = age
  	def introduce(self):
  		print(f"My name is {self.name} and I am {self.age} years old.")
  ```

##### 2.2实例化对象

- 语法

  ```python
  变量 = 类名([参数])
  ```

- 说明

  \-- 变量存储的是实例化后的对象地址。

  -- 类名后面的参数按照构造方法的形参传递

  对象是类的实例，具有类定义的属性和方法。 

  通过调用类的构造函数来创建对象。 

  每个对象都有自己的状态，但共享相同的方法定义。

- 实例

  实例有自己的作用域和名字空间,可以为该实例添加实例变量（也叫属性) 

  实例可以调用类方法和实例方法 

  实例可以访问类变量和实例变量

  ```python
  class Dog:
  	pass
  # 创建第一个实例：
  dog1 = Dog()
  print(id(dog1))  # 打印这个对象的ID
   # 创建第二个实例对象
  dog2 = Dog()  # dog2 绑定一个Dog类型的对象
  print(id(dog2))
  
  lst1 = list()
  print(id(lst1))
  lst2 = list()
  print(id(lst2))
  class Person:
   	def __init__(self, name, age):
  	 self.name = name
  	 self.age = age
   	def introduce(self):
  	 print(f"My name is {self.name} and I am {self.age} years old.")
  person1 = Person("Alice", 25)
  person2 = Person("Bob", 30)
  ```

##### 2.3 self说明

- 类实例化后，self即代表着实例（对象）本身

- self 是类方法的第一个参数，用于引用对象本身。

- self 不是Python关键字，但是约定俗成的命名，可以使用其他名称代替，但通常不建议。 示例代码：

- 实例

  ```python
  class Students:
   	# 构造方法
  	def __init__(self,name):
   		self.name = name
   	# 实例方法
  	def study(self,examination_results):
  		 self.examination_results = examination_results
   		 print("同学{}的考试分数是{}".format(self.name,self.examination_results))
   		 print("该实例对象的地址是{}".format(self))
   
  studend_a = Students('studend_a')
  print(studend_a.name)
   
  studend_b = Students('studend_b')
  print(studend_b.name)
  ```

#### 2.属性和方法

##### 2.1属性

- 语法

  ```python
  实例.属性名
  ```

- 说明

  首次为属性赋值则创建此属性. 

  再次为属性赋值则改变属性的绑定关系

- 实例

  属性使用

  ```python
  # file : attribute.py
  class Dog:
  	def eat(self, food):
  		print(self.color, '的', self.kinds, '正在吃', food)
   	pass
   # 创建一个实例：	
  dog1 = Dog()
  dog1.kinds = "京巴"  # 添加属性
  dog1.color = "白色"
  dog1.color = "黄色"  # 改变属性的绑定关系print(dog1.color, '的', dog1.kinds)
  og2 = Dog()
  dog2.kinds = "藏獒"
  dog2.color = "棕色"
  print(dog2.color, '的', dog2.kinds)
  ```

  实例方法与实例属性结合使用

  ```python
  class Dog:
          def eat(self, food):
              print(self.color, '的',
                    self.kinds, '正在吃', food)
      # 创建第一个对象
  dog1 = Dog()
  dog1.kinds = '京巴'  # 添加属性kinds
  dog1.color = '白色'  # 添加属性color
  # print(dog1.color, '的', dog1.kinds)  # 访问属性 
  dog1.eat("骨头")
      
  dog2 = Dog()
  dog2.kinds = '牧羊犬'
  og2.color = '灰色'
  # print(dog2.color, '的', dog2.kinds)  # 访问属性
  dog2.eat('包子')
  ```

##### 2.2方法

- 语法

  ```python
  class 类名(继承列表):
      def 实例方法名(self, 参数1, 参数2, ...):
          "文档字符串"
          语句块
  ```

- 说明

  **用于描述一个对象的行为,让此类型的全部对象都拥有相同的行为**

  **实例方法的实质是函数，是定义在类内的函数** 

  **实例方法至少有一个形参，第一个形参绑定调用这个方法的实例,一般命名为"self"** 

  **实例方法名是类属性**

- 实例

  ```python
  # file: instance_method.py
  class Dog:
      """这是一个种小动物的定义
      这种动物是狗(犬)类，用于创建各种各样的小狗
      """
   	def eat(self, food):
  		 '''此方法用来描述小狗吃东西的行为'''
   		 print("小狗正在吃", food)
   	def sleep(self, hour):
   		 print("小狗睡了", hour, "小时!")
   	def play(self, obj):
   		 print("小狗正在玩", obj)
  dog1 = Dog()
  dog1.eat("骨头")
  dog1.sleep(1)
  dog1.play('球')
  dog2 = Dog()
  dog2.eat("窝头")
  dog2.sleep(2)
  dog2.play('飞盘')
  >>> help(Dog) # 可以看到Dog类的文档信息
  ```

##### 2.3类属性

- 说明

  类属性是类的属性，此属性属于类，不属于此类的实例

  作用： 

  ​		通常用来存储该类创建的对象的共有属性 

  类属性说明 

  ​		类属性,可以通过该类直接访问 

  ​		类属性,可以通过类的实例直接访问

- 实例

  ```python
  class Human:
   	total_count = 0  # 创建类属性
  	def __init__(self, name):
   		self.name = name
  print(Human.total_count)
  h1 = Human("小张")
  print(h1.total_count)
  ```

##### 2.4类方法

- 说明

  类方法需要使用@classmethod装饰器定义 

  类方法至少有一个形参，第一个形参用于绑定类，约定写为'cls' 

  类和该类的实例都可以调用类方法 

  **类方法可以通过访问此类创建的对象的实例属性进行访问**

- 实例

  ```python
   class A:
   	v = 0
   	@classmethod
   	def set_v(cls, value):
   		cls.v = value
   	@classmethod
   	def get_v(cls):
   		return cls.v
   print(A.get_v())
   A.set_v(100)
   print(A.get_v())
   a = A()
   print(a.get_v())
  ```

  ```python
  class MyClass:
      class_attr = 0  # 类属性
      def __init__(self, value):
          self.instance_attr = value  # 实例属性
      @classmethod
      def modify_class_attr(cls, new_value):
          cls.class_attr = new_value
          print(f"类属性已修改为: {cls.class_attr}")
      @classmethod
      def try_modify_instance_attr(cls):
          try:
              cls.instance_attr = 10  # 尝试修改实例属性（失败）
          except AttributeError as e:
              print(f"错误: {e}")
      def show_attrs(self):
          print(f"实例属性: {self.instance_attr}")
          print(f"类属性: {self.class_attr}")
   # 创建类的实例
  obj = MyClass(5)
   # 调用类方法修改类属性
  MyClass.modify_class_attr(20)  # 输出: 类属性已修改为: 20
  obj.show_attrs()
   # 输出:
   # 实例属性: 5
   # 类属性: 20
   # 调用类方法尝试修改实例属性
  MyClass.try_modify_instance_attr()
   # 错误
  obj.try_modify_instance_attr()
   # 错误
  ```

- cls实例

  说明

  ​		在Python中， cls 是一个约定俗成的名称，用于表示类本身。在类方法(使用  @classmethod 装饰的 方法)中，cls 作为第一个参数传递给方法。这使得类方法可以访问和修改类属性以及调用其他类方法，而不需要引用具体的实例。

  作用

  		1. 访问类属性：类方法可以通过  cls 访问和修改类属性。 
  		1.  调用类方法：类方法可以通过  cls 调用其他类方法。
  		1.  创建类实例：类方法可以使用  cls 来创建类的实例。

  实例

  ```python
  class MyClass:
   	class_attr = 0  # 类属性
  	def __init__(self, value):
   		self.instance_attr = value  # 实例属性
  	@classmethod
   	def modify_class_attr(cls, new_value):
   		cls.class_attr = new_value
   		print(f"类属性已修改为: {cls.class_attr}")
   	@classmethod
  	def show_class_attr(cls):
  		print(f"类属性当前值: {cls.class_attr}")
  	@classmethod
   	def create_instance(cls, value):
           # 使用 cls 创建类实例
          return cls(value)
  # 调用类方法修改类属性
  MyClass.modify_class_attr(20)  # 输出: 类属性已修改为: 20
   # 调用类方法显示类属性
  MyClass.show_class_attr()  # 输出: 类属性当前值: 20
   # 使用类方法创建类的实例
  new_instance = MyClass.create_instance(10)
  print(f"新实例的实例属性: {new_instance.instance_attr}")  # 输出: 新实例的实例属性: 10
  print(f"新实例的类属性: {new_instance.class_attr}")  # 输出: 新实例的类属性: 20
  ```

##### 2.5静态方法

- 定义

  @staticmethod

  静态方法是定义在类的内部函数，此函数的作用域是类的内部

- 说明

  静态方法需要使用@staticmethod装饰器定义 

  静态方法与普通函数定义相同，不需要传入self实例参数和cls类参数 

  静态方法只能凭借该类或类创建的实例调用 

  **静态方法可以访问类属性和实例属性**

- 实例

  ```python
  class A:
   	class_attr = 42  # 类属性
  	def __init__(self, value):
   		self.instance_attr = value  # 实例属性
  	@staticmethod
   	def myadd(a, b):
   		# 只能访问传递的参数，不能访问类属性和实例属性
  		return a + b
  # 创建类实例
  a = A(10)
  # 调用静态方法
  print(A.myadd(100, 200))  # 输出: 300
  print(a.myadd(300, 400))  # 输出: 700
  # 尝试在静态方法内访问类属性或实例属性（会导致错误）
  class B:
      class_attr = 42
  	def __init__(self, value):
   		self.instance_attr = value
   	@staticmethod
   	def myadd(a, b):
          # 以下访问会导致错误
          # return a + b + B.class_attr
          # return a + b + self.instance_attr
          return a + b
  # 创建类实例
  b = B(10)
  # 调用静态方法
  print(B.myadd(100, 200))  # 输出: 300
  print(b.myadd(300, 400))  # 输出: 700
  ```

##### 2.6初始化方法

- 语法

  ```python
  class 类名(继承列表):
   	def __init__(self[, 形参列表]):
  		语句块
  # [] 代表其中的内容可省略
  ```

- 说明

  初始化方法名必须为 __init__ 不可改变 

  初始化方法会在构造函数创建实例后自动调用,且将实例自身通过第一个参数self传入 __init__ 方法 

  构造函数的实参将通过 __init__ 方法的参数列表传入到  __init__ 方法中 

  初始化方法内如果需要return语句返回，则必须返回None

- 实例

  ```python
  class Car:
  	def __init__(self, c, b, m):
          self.color = c  # 颜色
          self.brand = b  # 品牌
          self.model = m  # 型号
  	def run(self, speed):
  		print(self.color, "的", self.brand, self.model, "正在以", speed, "公里／小时的速度行驶")
  	def change_color(self, c):
  		self.color = c
  a4 = Car("红色", "奥迪", "A4")
  a4.run(199)
  a4.change_color("黑色")
  a4.run(230)
  ```

##### 2.7魔术方法

- 说明

  Python中的魔术方法（Magic Methods）是一种特殊的方法，它们以双下划线开头和结尾，例如 __init__ ， __str__ ， __add__ 等。这些方法允许您自定义类的行为，以便与内置Python功能（如 +运算符、迭代、字符串表示等）交互。

- 常用魔术方法

  ```py
  1. __init__(self, ...) : 初始化对象，通常用于设置对象的属性。
  
  2. __str__(self) : 定义对象的字符串表示形式，可通过如str(object) 或
  print(object) 调用，例如您可以返回一个字符串，描述对象的属性。
  
  3.__repr__(self) : 定义对象的“官方”字符串表示形式，通常用于调试。可通过repr(object) 调用。
  
  4. __len__(self) : 定义对象的长度，可通过len(object) 调用。通常在自定义容器类中使用。
  
  5.__getitem__(self, key) : 定义对象的索引操作，使对象可被像列表或字典一样索引。例如，object[key]
  
  6.__setitem__(self, key, value): 定义对象的赋值操作，使对象可像列表或字典一样赋值。例如，object[key] = value。
  
  7.__delitem__(self, key) : 定义对象的删除操作，使对象可像列表或字典一样删除元素。例如，del object[key]。
  
  8. __iter__(self) : 定义迭代器，使对象可迭代，可用于for 循环。
  
  9. __next__(self) : 定义迭代器的下一个元素，通常与__iter__ 一起使用。
  
  10. __add__(self, other) : 定义对象相加的行为，使对象可以使用+运算符相加。例如，object1 + object2。
  
  11. __sub__(self, other) : 定义对象相减的行为，使对象可以使用-运算符相减。
  
  12. __eq__(self, other) : 定义对象相等性的行为，使对象可以使用== 运算符比较。
  
  13. __lt__(self, other) : 定义对象小于其他对象的行为，使对象可以使用<运算符比较。
  
  14. __gt__(self, other) : 定义对象大于其他对象的行为，使对象可以使用>运算符比较。
  ```
  
  
  
- 实例

  1.

  ```py
   __init__(self, ...) : 初始化对象
   
   class MyClass:
   	def __init__(self, value):
   		self.value = value
   obj = MyClass(42)
  ```

  2.

  ```py
   __str__(self) : 字符串表示形式
   
  class MyClass:
   	def __init__(self, value):
   		self.value = value
   	def __str__(self):
   		return f"MyClass instance with value: {self.value}"
   obj = MyClass(42)
   print(obj)  # 输出：MyClass instance with value: 42
  ```

  3.

  ```py
  __repr__(self) : 官方字符串表示形式
  
  class MyClass:
   	def __init__(self, value):
   		self.value = value
   	def __repr__(self):
   		return f"MyClass({self.value})"
   obj = MyClass(42)
   print(obj)  # 输出：MyClass(42)
  ```

  4.

  ```py
  __len__(self) : 定义对象的长度
  
  class MyList:
   	def __init__(self, items):
   		self.items = items
   	def __len__(self):
   		return len(self.items)
  my_list = MyList([1, 2, 3, 4])
  print(len(my_list))  # 输出：4
  ```

  5.

  ```py
   __getitem__(self, key) : 索引操作
   
  class MyDict:
   	def __init__(self):
   		self.data = {}
   	def __getitem__(self, key):
   		return self.data.get(key)
  my_dict = MyDict()
  my_dict.data = {'key1': 'value1', 'key2': 'value2'}
  print(my_dict['key1'])  # 输出：value1
  ```

  6.

  ```py
   __setitem__(self, key, value): 赋值操作
   
  class MyDict:
   	def __init__(self):
   		self.data = {}
   	def __setitem__(self, key, value):
   		self.data[key] = value
  my_dict = MyDict()
  my_dict['key1'] = 'value1'
  print(my_dict.data)  # 输出：{'key1': 'value1'}
  ```

  7.

  ```py
   __delitem__(self, key) : 删除操作
   
  class MyDict:
   	def __init__(self):
   		self.data = {}
   	def __delitem__(self, key):
   		del self.data[key]
  my_dict = MyDict()
  my_dict.data = {'key1': 'value1', 'key2': 'value2'}
  del my_dict['key2']
  print(my_dict.data)  # 输出：{'key1': 'value1'}
  ```

  8.

  ```py
   __iter__(self) : 迭代器
   
  class MyIterable:
   	def __init__(self):
   		self.data = [1, 2, 3, 4]
   	def __iter__(self):
   		self.index = 0
   		return self
  	def __next__(self):
   		if self.index >= len(self.data):
   			raise StopIteration
   		value = self.data[self.index]
   		self.index += 1
   		return value
  my_iterable = MyIterable()
  for item in my_iterable:
  print(item)
  # 输出：1, 2, 3, 4
  ```

  

#### 3.继承和派生

- 定义

  ```python
  1.继承是从已有的类中派生出新的类，新类具有原类的数据属性和行为，并能扩展新的能力。
  
  2.派生类就是从一个已有类中衍生出新类，在新的类上可以添加新的属性和行为
  ```

- 目的

  ```python
  1.继承的目的是延续旧的类的功能
  
  2.派生的目地是在旧类的基础上添加新的功能
  ```

- 作用

  ```python
  1.用继承派生机制，可以将一些共有功能加在基类中。实现代码的共享。
  
  2.在不改变基类的代码的基础上改变原有类的功能
  ```

- 方法名

  ```python
  1.基类(base class)/超类(super class)/父类(father class)
  
  2.派生类(derived class)/子类(child class)
  ```

##### 3.1 单继承

- 语法

  ```python
  class 类名(基类名):
  	语句块
  ```

- 说明

  单继承是指派生类由一个基类衍生出来的

- 实例

  ```python
  class Human:  # 人类的共性
      def say(self, what):  # 说话
  		print("说：", what)
   	def walk(self, distance):  # 走路
  		print("走了", distance, "公里")
  class Student(Human):
   	def study(self, subject):  # 学习
  		print("学习:", subject)
  class Teacher(Human):
   	def teach(self, language):
   		print("教:", language)
  h1 = Human()
  h1.say("天气真好!")
  h1.walk(5)
  s1 = Student()
  s1.walk(4)
  s1.say("感觉有点累")
  s1.study("python")
  t1 = Teacher()
  t1.teach("面向对象")
  t1.walk(6)
  t1.say("一会吃点什么好呢")
  ```

##### 3.2 多继承

- 语法

  ```python
   class DerivedClassName(Base1, Base2, Base3):
       <statement-1>
       .
       .
       .
       <statement-N>
  ```

- 说明

  **需要注意圆括号中父类的顺序，若是父类中有相同的方法名，而在子类使用时未指定，python从左至右搜索:即方法在子类中未找到时，从左到右查找父类中是否包含方法**。

- 实例

  ```python
  #类定义
  class people:
      #定义基本属性
      name = ''
      age = 0
      #定义私有属性,私有属性在类外部无法直接进行访问
      __weight = 0
      #定义构造方法
      def __init__(self,n,a,w):
           self.name = n
           self.age = a
           self.__weight = w
   	def speak(self):
   		 print("%s 说: 我 %d 岁。" %(self.name,self.age))
  #单继承示例
  class student(people):
      grade = ''
      def __init__(self,n,a,w,g):
          #调用父类的构函
          people.__init__(self,n,a,w)
          self.grade = g
      #覆写父类的方法
      def speak(self):
          print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))
   
  #另一个类，多继承之前的准备
  class speaker():
      topic = ''
      name = ''
      def __init__(self,n,t):
          self.name = n
          self.topic = t
      def speak(self):
          print("我叫 %s，我是一个演说家，我演讲的主题是 %s"%(self.name,self.topic))
   
  #多继承
  class sample(speaker,student):
      a =''
      def __init__(self,n,a,w,g,t):
          student.__init__(self,n,a,w,g)
          speaker.__init__(self,n,t)
   
  test = sample("Tim",25,80,4,"Python")
  test.speak()   #方法名同，默认调用的是在括号中参数位置排前父类的方法
  ```

##### 3.3 覆盖override

- 定义

  覆盖是指在有继承关系的类中，子类中实现了与基类同名的方法,在子类的实例调用该方法时，实际调用的是子类中的覆盖版本,这种现象叫覆盖

- 作用

  实现和父类同名，但功能不同的方法

- 实例

  ```python
  class A:
      def work(self):
          print("A.work 被调用!")
   class B(A):
      '''B类继承自A类'''
      def work(self):
          print("B.work 被调用!!!")
      pass
  b = B()
  b.work()  # 请问调用谁?  B
  a = A() 
  a.work()  # 请问调用谁?  A
  ```

#### 4.封装 enclosure

- 定义

  封装是指隐藏类的实现细节，让使用者不用关心这些细节

- 作用

  封装的目的是让使用者通过尽可能少的方法(或属性)操作对象

- 特性

  Python的封装是假的（模拟的）封装

- 私有属性和方法

  ```python
  python类中以双下划线(__)开头，不以双下划线结尾的标识符为私有成员,私有成员只能使用方法来进行访问和修改
  	1.以__开头的属性为类的私有属性，在子类和类外部无法直接使用
  	2.以__开头的方法为私有方法，在子类和类外部无法直接调用
  ```

- 私有属性和方法实例

  ```python
  class A:
   	def __init__(self):
   		self.__p1 = 100  # 私有属性
  	def __m1(self):  # 私有方法
  		print("__m1(self) 方法被调用")
   	def showA(self):
   		self.__m1()
   		print("self.__p1 = ", self.__p1)
  class B(A):
   	def __init__(self):
   		super().__init__()
   	def showB(self):
   		self.__m1()  # 出错，不允许调用
  		print("self.__p1 = ", self.__p1)  # 出错，不允许调用
  # self._A__m1()  # 正常调用
  # print("self.__p1 =", self._A__p1)  # 正常访问
  a = A()
  a.showA()
  a.__m1()       # 出错，不允许调用
  v = self.__p1  # 出错，不允许调用
  b = B()
  b.showB()
  # 访问私有属性
  print(a._A__p1)  # 输出: 100
  # 调用私有方法
  a._A__m1()  # 输出: __m1(self) 方法被调用
  # 不推荐了解就行
  ```

#### 5.多态 polymorphic

- 定义

  多态是指在有继承/派生关系的类中，调用基类对象的方法,实际能调用子类的覆盖方法的现象 叫多态

- 状态

  静态(编译时状态) 

  动态(运行时状态)

- 说明

  多态调用的方法与对象相关，不与类型相关 

  Python的全部对象都只有"运行时状态(动态)", 没有"C++语言"里的"编译时状态(静态)"

- 实例

  ```python
  class Shape:
   	def draw(self):
   		print("Shape的draw()被调用")
  class Point(Shape):
   	def draw(self):
   		print("正在画一个点!")
  class Circle(Point):
   	def draw(self):
   		print("正在画一个圆!")
   	def my_draw(s):
   		s.draw()  # 此处显示出多态
  shapes1 = Circle()
  shapes2 = Point()
  my_draw(shapes1)  # 调用Circle 类中的draw
  my_draw(shapes2)  # Point 类中的draw
  ```

- 面向对象编程语言的特征

  继承 

  封装 

  多态

#### 6.方法重写

定义

​	父类方法的功能不能满足你的需求，你可以在子类重写你父类的方法

###### 6.1函数重写

- 对象转字符串函数重写

  ```python
  	str() 函数的重载方法:
   		def __str__(self)
  			如果没有 __str__(self) 方法，则返回repr(obj)函数结果代替
  			
  str/repr函数重写示例：
  class MyNumber:
  	   "此类用于定义一个自定义的类，用于演示str/repr函数重写"
   		def __init__(self, value):
   			"构造函数,初始化MyNumber对象"
  			 self.data = value
   		def __str__(self):
  			 "转换为普通字符串"
   			 return "%s" % self.data
   n1 = MyNumber("一只猫")
   n2 = MyNumber("一只狗")
   print("str(n2) ===>", str(n2))
  ```

- 内建函数重写

  ```python
  1.__abs__ abs(obj) 函数调用
  
  2.__len__ len(obj) 函数调用    
  
  3.__reversed__  reversed(obj) 函数调用
  
  4.__round__  round(obj) 函数调用
  
  # file : len_overwrite.py
   class MyList:
   	def __init__(self, iterable=()):
   		self.data = [x for x in iterable]
   	def __repr_(self):
   		return "MyList(%s)" % self.data
   	def __len__(self):
   		print("__len__(self) 被调用!")
   		return len(self.data)
   	def __abs__(self):
   		print("__len__(self) 被调用!")
   		return MyList((abs(x) for x in self.data))
   myl = MyList([1, -2, 3, -4])
   print(len(myl))
   print(abs(myl))
  ```

###### 6.2运算符重载

- 定义

  运算符重载是指让自定义的类生成的对象(实例)能够使用运算符进行操作

- 作用

  ```python
  1.让自定义类的实例像内建对象一样进行运算符操作
  2.让程序简洁易读
  3.对自定义对象将运算符赋予新的运算规则
  ```

- 说明

  运算符重载方法的参数已经有固定的含义,不建议改变原有的意义

- 算术运算符重载

  | 方法名                    | 运算符和表达式 | 说明       |
  | ------------------------- | -------------- | ---------- |
  | `__add__(self, rhs)`      | self + rhs     | 加法       |
  | `__sub__(self, rhs)`      | self - rhs     | 减法       |
  | `__mul__(self, rhs)`      | self * rhs     | 乘法       |
  | `__truediv__(self, rhs)`  | self / rhs     | 除法       |
  | `__floordiv__(self, rhs)` | self // rhs    | 地板除     |
  | `__mod__(self, rhs)`      | self % rhs     | 取模(求余) |
  | `__pow__(self, rhs)`      | self ** rhs    | 幂         |

  ```python
   rhs (right hand side) 右手边
  ```

- 算术运算符重载实例

  ```python
  class MyNumber:
   	"此类用于定义一个自定义的类，用于演示运算符重载"
   	def __init__(self, value):
   		"构造函数,初始化MyNumber对象"
   		 self.data = value
   	def __str__(self):
  		 "转换为表达式字符串"
   		 return "MyNumber(%d)" % self.data
   	def __add__(self, rhs):
   		"加号运算符重载"
   		 print("__add__ is called")
   		 return MyNumber(self.data + rhs.data)
   	def __sub__(self, rhs):
   		"减号运算符重载"
   		 print("__sub__ is called")
   		 return MyNumber(self.data - rhs.data)
   n1 = MyNumber(100)
   n2 = MyNumber(200)
   print(n1 + n2)
   print(n1 - n2)
  ```

#### 7.super函数

- 定义

  super() 函数是用于调用父类(超类)的一个方法。

- 说明

  super() 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使 用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。

- 语法

  ```python
  1.在子类方法中可以使用super().add()调用父类中已被覆盖的方法
  2.可以使用super(Child, obj).myMethod()用子类对象调用父类已被覆盖的方法
  ```

- 实例

  ```python
  class A:
   	def add(self, x):
   		y = x+1
   		print(y)
  class B(A):
   	def add(self, x):
   		print("子类方法")
   		super().add(x)
  b = B()
  b.add(2)  # 3
  class Parent:  # 定义父类
  	def myMethod(self):
   		print ('调用父类方法')
  class Child(Parent): # 定义子类
  	def myMethod(self):
   		print ('调用子类方法')
  c = Child()# 子类实例
  c.myMethod()# 子类调用重写方法
  super(Child,c).myMethod() #用子类对象调用父类已被覆盖的方法
  ```

- super().init()

  说明

  ```python
  super().__init__() 是 Python 中用于调用父类（基类）构造函数的一种方式。它通常用于子类的构造函数中，以确保父类的构造函数被正确调用和初始化。这在继承（inheritance）中尤为重要，因为父类的初始化代码可能包含设置实例变量或执行其他重要的初始化任务。
  ```

  作用

  ```python
  1.代码重用：避免在子类中重复父类的初始化代码。
  
  2.正确初始化：确保父类的初始化逻辑（如设置属性、分配资源等）被执行。
  
  3.支持多重继承：在多重继承情况下，super() 可以确保所有基类的构造函数都被正确调用。
  class Attention(nn.Module):
   	def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
   	super().__init__()
   	inner_dim = dim_head * heads  # 计算内部维度
  	project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要投影输出
  ```

  实例

  ```python
  class Parent:
   	def __init__(self):
   		print("Parent class constructor called")
   		self.parent_attribute = "I am a parent attribute"
  class Child(Parent):
   	def __init__(self):
   		super().__init__()
   		print("Child class constructor called")
   		self.child_attribute = "I am a child attribute"
  # 创建一个 Child 类的实例
  child_instance = Child()
  # 输出
  # Parent class constructor called
  # Child class constructor called
  """
  注释
  1. Parent 类：
  	定义了一个构造函数 __init__() ，在构造函数中打印了一条消息，并初始化了一个属性 parent_attribute
  	
  2. Child 类：
  继承自 Parent 类。
  
  在其构造函数 __init__() 中，首先调用了 Parent 类的构造函数，确保 
  super().__init__() 。这行代码会调用 Parent 类的初始化逻辑被执行。
  
  然后打印了一条消息，并初始化了一个属性 child_attribute 
  
  3. 实例化 Child 类：
  创建 Child 类的实例时，首先执行Parent 类的构造函数，打印 "Parent class constructorcalled"， 然后执行 Child 类的构造函数，打印"Child class constructor called"。
  
  """
  ```

## day_05

### Python迭代器和生成器

#### 1.迭代器iterator

- 定义

  ```py
  1.迭代器是访问可迭代对象的工具
  
  2.迭代器是指用 iter(obj) 函数返回的对象(实例)
  
  3.迭代器可以用next(it)函数获取可迭代对象的数据
  ```

- 迭代器函数

  | 函数           | 说明                                                         |
  | -------------- | ------------------------------------------------------------ |
  | iter(iterable) | 从可迭代对象中返回一个迭代器,iterable必须是能提供一个迭代器的对象 |
  | next(iterator) | 从迭代器iterator中获取下一个记录，如果无法获取一下条记录，则触发 StopIteration 异常 |

  实例

  ```python
  L = [2, 3, 5, 7]
  it = iter(L)
  # 访问列表中的所有元素
  while True:
   	try:
   		print(next(it))
   	except StopIteration:
   		print("迭代器访问结束")
   		break
  L = [2, 3, 5, 7]
  for x in L:
  	print(x)
  else:
  	print("迭代器访问结束")
  ```

- 说明

  ```python
  1.迭代器只能往前取值,不会后退
  
  2.用iter函数可以返回一个可迭代对象的迭代器
  ```

- 实例

  ```python
  # 示例 可迭代对象
  L = [1, 3, 5, 7]
  it = iter(L)  # 从L对象中获取迭代器
  next(it)  # 1  从迭代器中提取一个数据
  next(it)  # 3
  next(it)  # 5
  next(it)  # 7
  next(it)  # StopIteration 异常
  # 示例2 生成器函数
  It = iter(range(1, 10, 3))
  next(It)  # 1
  next(It)  # 4
  next(It)  # 7
  next(It)  # StopIteration
  ```

#### 2.生成器

- 定义

  生成器是在程序运行时生成数据，与容器不同，它通常不会在内存中保留大量的数据，而是现用现生成。

- 特性

  ```python
  1.yield 是一个关键字，用于定义生成器函数，生成器函数是一种特殊的函数，可以在迭代过程中逐步产生值，而不是一次性返回所有结果。
  
  2.跟普通函数不同的是，生成器是一个返回迭代器的函数，只能用于迭代操作，更简单点理解生成器就是一个迭代器。
  
  3.每次使用 yield 语句生产一个值后，函数都将暂停执行，等待被重新唤醒。
  
  4.yield 语句相比于 return 语句，差别就在于 yield 语句返回的是可迭代对象，而 return 返回的为不可迭代对象。
  
  5.然后，每次调用生成器的 next() 方法或使用 for 循环进行迭代时，函数会从上次暂停的地方继续执行，直到再次遇到 yield 语句。
  ```

- 种类与语法

  ```python
  1. 生成器函数
  	yield 表达式
  	
  2. 生成器表达式
  	( 表达式 for 变量 in 可迭代对象 [if 真值表达式])
  ```

- 实例

  生成器函数

  ```python
  ## 定义一个生成器函数, 有 yield 的函数调用后回返回生成器对象
  def myrange(stop):
   	i = 0
   	while i < stop:
   		yield i    # 为遍历次生产器的for 语句提供数据
  		i += 1
  for x in myrange(5):
   	print('x=', x)
  # 创建一个生成器对象
  gen = myrange(5)
  # 使用 next() 函数迭代生成器
  print(next(gen)) 
  print(next(gen))  
  print(next(gen)) 
  print(next(gen))  
  print(next(gen))
  ```

  ```python
  def Descendorder(n):
   	while n > 0:
   		yield n
   		n -= 1
  # 创建生成器对象
  generator = Descendorder(5)
  # 通过迭代生成器获取值
  print(next(generator))#5
  print(next(generator))#4
  # 使用 for 循环迭代生成器
  for i in generator:
  	print('for循环：', i)#3  2  1
  ```

  生成器表达式

  ```python
  >>> [x ** 2 for x in range(1, 5)]   # 列表解析(列表推导式)
  [1, 4, 9, 16]
  >>> 
  >>> (x ** 2 for x in range(1, 5))  # 生成器表达式
  <generator object <genexpr> at 0x7f41dcd30a40>
  >>> for y in (x ** 2 for x in range(1, 5)):
  ...     print(y)
  ... 
  1
  4
  9
  16
  ```

### Python函数式编程

- 定义

  ```python
  1.函数可以赋值给变量，赋值后变量绑定函数。 
  
  2.允许将函数作为参数传入另一个函数。
  
  3.允许函数返回一个函数。
  ```

- 思想

  ```python
  1.什么时候使用函数式编程思想？
  	很多的逻辑或者说核心点是不变的，大多数就是一致的，这个时候我们就可以使用函数式编程思想，可以很好的去定位这个逻辑【函数 式编程思想相对于面向对象编程思想，它更接近于算法】。
  2.函数式编程思想替代了面向对象思想？
  	如果需求中存在多个逻辑变化点时，可以使用类来进行，因为面向对象中存在继承、重写。而函数式编程思想则是将变化点提取到函数 中，实现简单的逻辑。
  ```

#### 1.函数作为参数

```python
def func01():
 	print("func01执行")
# a = func01
# # print(a)
# a()
def func02():
 	print("func02执行")
# 通用
def func03(func):
 	print("func03执行")
 	func()
func03(func02)
func03(func01)
"""
注释
a = func01:
    变量 a 现在指向func01 函数对象。
	a 不是函数的返回值，而是函数对象本身
print(a) :
打印 a，输出 <function func01 at 0x...>，表示 a 是一个函数对象，并显示其内存地址。赋值语句 a = func01 并不会执行 func01 函数，只是将函数对象赋值给 a,调用 a()或fun01()才会执行函数代码
"""
```

```python
list01 = [4, 54, 56, 65, 67, 7]
 # 需求1：定义函数,在列表中查找所有大于50的数字
def find01():
 	for item in list01:
 		if item > 50:
 			yield item
# 需求2：定义函数,在列表中查找所有小于10的数字
def find02():
 	for item in list01:
 		if item < 10:
 			yield item
# “封装” -- 分
def condition01(item):
 	return item > 50
def condition02(item):
 	return item < 10
# 通用
# “继承” - 隔
def find(func):
 	for item in list01:
 		# if item < 10:
 		# if condition02(item):
 		if func(item):
 			yield item
 			
# "多态" - 做
for item in find(condition01):
 	print(item)
for item in find(condition02):
 	print(item)
```

##### 1.1 lambda表达式

- 定义

  是一种匿名函数

- 作用

  ```py
  -- 作为参数传递时语法简洁，优雅，代码可读性强。 
  
  -- 随时创建和销毁，减少程序耦合度。
  ```

- 语法

  ```python
  # 定义：
  变量 = lambda 形参: 方法体
  
  # 调用：
  变量(实参)
  ```

- 说明

  ```py
  -- 形参没有可以不填 
  
  -- 方法体只能有一条语句，且不支持赋值语句。
  ```

- 实例

  ```python
  # 1. 有参数有返回值
  # def func01(a,b):
  #     return a > b
  #
  # print(func01(10,20))
  func01 = lambda a,b:a > b
  print(func01(10,20))
  
  
  # 2. 无参数有返回值
  # def func02():
  #     return "ok"
  #
  # print(func02())
  func02 = lambda :"ok"
  print(func02())
  
  
  # 3. 无参数无返回值
  # def func03():
  #     print("ok")
  #
  # func03()
  func03 = lambda :print("ok")
  func03()
  
  
  # 4. 有参数无返回值
  # def func03(a):
  #     print(a)
  #
  # func03(100)
  func03 = lambda a:print(a)
  func03(100)
  
  
  # 5. lambda 不支持赋值语句
  # def func05(iterable):
  #     iterable[0] = 100
  #
  # list01 = [1]
  # func05(list01)
  # print(list01)
  # func05 = lambda iterable: iterable[0] = 100 报错
  
  
  # 6. lambda 不支持多条语句
  # def func06(a,b):
  #     print(a)
  #     print(b)
  #
  # func06(10,20)
  # func06 = lambda a,b: print(a);print(b)
  ```

##### 1.2 内置高阶函数

- 定义

  将函数作为参数或返回值的函数。

- 常用内置高阶函数

  ```
  （1）map(函数，可迭代对象) 
  	使用可迭代对象中的每个元素调用函数，将返回值作为新可迭代对象元素；返回值为新可迭代对象。
  	
  （2）filter(函数，可迭代对象) 
  	根据条件筛选可迭代对象中的元素，返回值为新可迭代对象。 
  	
  （3）sorted(可迭代对象, key=函数, reverse=True)
  	排序，返回值为排序后的列表结果。 
  	
  （4）max(可迭代对象, key = 函数) 
  	根据函数获取可迭代对象的最大值。 
  	
  （5）min(可迭代对象，key = 函数) 
  	根据函数获取可迭代对象的最小值
  ```

- 实例

  ```python
  class Girl:
   def __init__(self, name="", face_score=0, age=0, height=0):
   	self.name = name
  	self.face_score = face_score
   	self.age = age
   	self.height = height
   def __str__(self):
   	return "%s-%d-%d-%d" % (self.name, self.face_score, 	self.age, self.height)
   	
  list_girl = [
   Girl("双儿", 96, 22, 166),
   Girl("阿珂", 100, 23, 173),
   Girl("小郡主", 96, 22, 161),
   Girl("方怡", 86, 27, 166),
   Girl("苏荃", 99, 31, 176),
   Girl("建宁", 93, 24, 163),
   Girl("曾柔", 88, 26, 170),
   ]
   
  # 1.  map 映射
  #  在美女列表中获取所有名称
  #  类似于：select
  for element in map(lambda item: item.name, list_girl):
   	print(element)
   
  # 2. filter 过滤器
  # 在美女列表中获取颜值大于90的所有美女
  #  类似于：find_all
  for element in filter(lambda item: item.face_score > 90, list_girl):
   	print(element)
  
  
  # 3. max/min
  # 在美女列表中获取颜值最高的美女
  #  类似于：get_max
  print(max(list_girl,key = lambda item:item.face_score))
  print(min(list_girl,key = lambda item:item.face_score))
   
   
  # 4.sorted 排序
  # 注意：没有改变原有列表,而是返回新的
  # 升序
  for item in sorted(list_girl,key = lambda item:item.height):
   	print(item)
  # 降序
  for item in sorted(list_girl,key = lambda item:item.height,reverse=True):
   	print(item)
  
  ```

#### 2.函数作为返回值

##### 2.1闭包 closure

- 定义

  ```python
  1.闭包是指引用了此函数外部嵌套函数的变量的函数
  
  2.闭包就是能够读取其他函数内部变量的函数。只有函数内部的嵌套函数才能读取局部变量，所以闭包可以理解成“定义在一个函数内部的函数,同时这个函数又引用了外部的变量“。
  
  3.在本质上，闭包是将内部嵌套函数和函数外部的执行环境绑定在一起的对象。
  ```

- 条件

  ```python
  1.必须有一个内嵌函数
  
  2.内嵌函数必须引用外部函数中变量
  
  3.外部函数返回值必须是内嵌函数。
  ```

- 优缺点

  优点

  ```python
  1. 逻辑连续，当闭包作为另一个函数调用参数时，避免脱离当前逻辑而单独编写额外逻辑。
  
  2. 方便调用上下文的局部变量。
  
  3. 加强封装性，是第2点的延伸，可以达到对变量的保护作用。
  ```

  缺点

  ```python
  1. 由于闭包会使得函数中的变量都被保存在内存中，内存消耗很大，所以不能滥用闭包
  
  2. 闭包会在父函数外部，改变父函数内部变量的值。所以，如果你把父函数当作对象（object）使用，把闭包当作它的公用方法（Public Method），把内部变量当作它的私有属性（private value），这时一定小心，不要随便改变父函数内部变量的值。
  ```

- 实例

  ```python
  def give_yasuiqian(money):
   	def child_buy(obj, m):
   		nonlocal money
   		if money > m:
   			money -= m
   			print('买', obj, '花了', m, '元, 剩余', money, '元')
   		else:
   			print("买", obj, '失败')
   	return child_buy
  cb = give_yashuqian(1000)    
  cb('变形金刚', 200)
  cb('漫画三国', 100)
  cb('手机', 1300)
  ```

  ```python
   # file : closure.py
   def make_power(y):
   	def fn(x):
   		return x ** y
   	return fn
   pow2 = make_power(2)
   print("5的平方是:", pow2(5))
   pow3 = make_power(3)
   print("6的立方是:", pow3(6))
  ```

##### 2.2装饰器 decorators

- 定义

  装饰器是一个函数，主要作用是来用包装另一个函数或类

- 作用

  在不修改被装饰的函数的源代码，不改变被装饰的函数的调用方式的情况下添加或改变原函数的功能。

- 语法

  ```python
  def 装饰器函数名(fn):
  	语句块
  	return 函数对象
  @装饰器函数名 <换行>
  def 被装饰函数名(形参列表):
  	语句块
  ```

- 实例

  用函数装饰器替换原函数myfun

  ```python
   def mydeco(fn):
   	fn()
   	print("装饰器函数被调用了,并返回了fx")
   	def fx():
   		print("fx被调用了")
  		# return fn()
   	return fx
   @ mydeco
   def myfun():
   	print("函数myfun被调用")
   	
   myfun()
   myfun()
  """
  当使用@mydeco 语法装饰myfun 函数时，实际上发生的是：
  1. myfun 函数作为参数传递给了mydeco装饰器
  
  2. 在mydeco 内部，首先调用了fn() ，即此时调用了myfun 函数，产生了输出："函数myfun被调用"。
  
  3. 接着，打印了"装饰器函数被调用了,并返回了fx"。
  
  4. 然后，mydeco 装饰器返回了新的函数
  
  因此，此刻myfun 实际上被替换成了新的函数fx 。这样的行为正是Python装饰器的特性之一：装饰器可以修改函数的行为，甚至完全替换被装饰的函数。
  
  """
  ```

###### 2.2.1基本装饰器

- 有参数的函数装饰器(在myfunc外加了一层)

  ```python
  def mydeco(fn):
   	def fx():
   		print("====这是myfunc被调用之前====")
   		ret = fn()
   		print("----这是myfunc被调用之后====")
   		return ret
   	return fx
   @mydeco
   def myfunc():
   	print("myfunc被调用.")
   	
   myfunc()
   myfunc()
   myfunc()
  ```

###### 2.2.2带参数的装饰器

```python
def repeat(num):
 	def decorator(func):
		 def wrapper(*args, **kwargs):
 			for _ in range(num):
 				func(*args, **kwargs)
 		return wrapper
 	return decorator
@repeat(3)  # 应用装饰器，重复执行下面的函数3次
def greet(name):
 	print(f"Hello, {name}!")
greet("Alice")  # 调用被装饰的函数


"""
repeat 是一个接受参数的装饰器工厂函数，它返回一个装饰器。

decorator 是真正的装饰器，它接受一个函数func 作为参数。

wrapper 函数重复执行被装饰的函数 num 次。

使用 @repeat(3) 应用装饰器，使 greet 函数被执行3次。

"""
```

###### 2.2.3装饰器链

```python
def uppercase(func):
 	def wrapper(*args, **kwargs):
		 result = func(*args, **kwargs)
 		 return result.upper()
 	return wrapper
def exclamation(func):
 	def wrapper(*args, **kwargs):
 		result = func(*args, **kwargs)
 		return result + "!"
 	return wrapper
@exclamation
@uppercase
def say_hello(name):
 	return f"Hello, {name}"
greeting = say_hello("Bob")
print(greeting)  # 输出 "HELLO, BOB!"


"""
uppercase 和 exclamation 是两个装饰器，分别将文本转换为大写并添加感叹号。

使用 @exclamation 和 @uppercase 创建装饰器链，它们按照声明的顺序倒着依次应用。

say_hello 函数在执行前被链中的装饰器处理，最终输出 "HELLO, BOB!"。

"""
```

###### 2.2.4类装饰器

```python
class MyDecorator:
 	def __init__(self, func):
 		self.func = func
 	def __call__(self, *args, **kwargs):
 		print("Something is happening before the function is called.")
 		result = self.func(*args, **kwargs)
 		print("Something is happening after the function is called.")
 		return result
@MyDecorator  # 应用类装饰器
def say_hello(name):
 	print(f"Hello, {name}!")
 	
say_hello("Charlie")  # 调用被装饰的函数

"""

MyDecorator 是一个类装饰器，它接受一个函数func 作为参数并在操作。

使用 @MyDecorator 应用类装饰器，它将包装 __call__ 方法中执行额外
say_hello 方法，使其在调用前后执行额外操作。

与基本装饰器类似

"""

```

## day_o6

### Python包和模块

#### 1.模块

##### 1.1模块定义

- 定义

  1.一个.py 文件就是一个模块

  2.模块是含有一系列**数据**，**函数**，**类**等的程序

- 作用

  把相关功能的函数等放在一起有利于管理，有利于多人合作开发

- 模块分类

  ```python
    1. 内置模块（在python3 程序内部，可以直接使用）
    2. 标准库模块(在python3 安装完后就可以使用的 )
    3. 第三方模块（需要下载安装后才能使用）
    4. 自定义模块(用户自己编写)
  
  模块名如果要给别的程序导入，则模块名必须是 标识符
  
  ```

- 实例

  ```python
  ## file: mymod.py
  '''
  小张写了一个模块，内部有两个函数，两个字符串
  ... 此处省略 200字
  '''
  name1 = 'audi'
  name2 = 'tesla'
  
  def mysum(n):
      '''
      此函数用来求和
      by weimingze
      '''
      print("正在计算， 1 + 2 + 3 + .... + n的和 ")
  
  def get_name():
      return "tarena"
  ```

  ```python
  ## file: test_mod.py
  ## 小李写的程序，小李想调用 小张写的 mymod.py 里的两个函数和两个字符串
  
  ## 用import 语句导入模块
  import mymod
  
  print(mymod.name1)  # Audi
  print(mymod.name2)    # tesla
  
  mymod.mysum(100)  # 调用 mymod 模块里的 mysum 函数
  print(mymod.get_name())   # 'tarena'
  ```

##### 1.2导入模块

- 语法

  ```python
  import 模块名  [as 模块新名字1]
  导入一个模块到当前程序
  
  from 模块名 import 模块属性名 [as 属性新名]
  导入一个模块内部的部分属性到当前程序
  
  from 模块名 import *
  导入一个模块内部的全部属性到当前程序
  
  
  import mymod
  mymod.mysum(10)   # 要加模块名
  
  from mymod import get_name
  print(get_name())   # 调用get_name 时不需要加 "模块名."
  
  from mymod import *   
  print(get_name())
  print(name2)
  ```

- 模块内部属性

  ```python
  __file__  绑定模块的路径
  __name__  绑定模块的名称
         如果是主模块（首先启动的模块）则绑定 '__main__'
         如果不是主模块则 绑定 xxx.py 中的 xxx 这个模块名
  ```

  ```python
  模块的 __name__ 属性
  
  每个.py 模块文件都会有 `__name__` 属性
  
  1. 当一个模块是最先运行的模块，则这个模块是主模块, 主模块的`__name__` 属性绑定`'__main__'` 字符串
  
  2. 如果一个模块是用三种 import 语句中的一个导入的模块，则此模块不是主模块。
  
     不是主模块的模块的 __name__ 绑定的是模块名
  ```

  ```python
  主模块 (`__name__` == `'__main__'`)： 当一个模块是直接运行的，即不是通过 `import` 语句导入的，那么它的 `__name__` 属性会被赋值为 `'__main__'`.
  
      
  被导入的模块 (`__name__` == 模块名)： 当一个模块被导入到另一个模块中时，它的 `__name__` 属性会被赋值为它的模块名。
  
  ```

##### 1.3 Python常用内建模块

###### 1.3.1 random模块

- 引入

  ```python
  import random
  ```

- 常用random方法

  | 函数                                | 描述                                                         |
  | ----------------------------------- | ------------------------------------------------------------ |
  | random.choice(seq)                  | 从序列的元素中随机挑选一个元素，比如random.choice(range(10))，从0到9中随机挑选一个整数。 |
  | random.randrange (start, stop,step) | 从指定范围内，按指定基数递增的集合中获取一个随机数，基数默认值为 1 |
  | random.random()                     | 随机生成下一个实数，它在[0,1)范围内。                        |
  | random.shuffle(list)                | 将序列的所有元素随机排序,修改原list                          |
  | uniform(x, y)                       | 随机生成实数，它在[x,y]范围内.                               |

- 实例

  ```python
  >>> import random
  >>> random.randint(1, 6)  # random.randint(a,b) 生产 a~b的随机整数
  3
  >>> random.randint(1, 6)
  4
  >>> random.random()   # random.random  生成包含0 但不包含1 的浮点数
  0.5884109388439075
  >>> random.choice("ABCD")    # 从一个序列中，随机返回一个元素
  'C'
  >>> random.choice("ABCD")
  'B'
  >>> L = [1, 2, 3, 6, 9]
  >>> random.choice(L)
  6
  >>> random.shuffle(L)   # random.shuffer(x)  # 把列表X 打乱
  >>> L
  [1, 6, 2, 9, 3]
  ```

###### 1.3.2 time模块

- 引入

  ```python
  import time
  ```

- 实例

  ```python
  >>> import time
  >>> time.time()   # 返回当前时间的时间戳
  1617117219.0382686
  >>> time.ctime()    #返回当前的UTC 时间的字符串
  'Tue Mar 30 23:14:48 2021'
  >>> t1 = time.localtime()   # 返回当前的本地时间元组
  >>> t1
  time.struct_time(tm_year=2021, tm_mon=3, tm_mday=30, tm_hour=23, tm_min=18, tm_sec=22, tm_wday=1, tm_yday=89, tm_isdst=0)
  >>> t1.tm_year
  2021
  >>> t1.tm_yday
  89
  >>> time.sleep(3)  # time.sleep(n)  # 让程序睡眠 n 秒
  >>> time.strftime("%Y-%m-%d", t1)   # 格式化时间
  '2021-03-30'
  >>> time.strftime("%y-%m-%d", t1)
  '21-03-30'
  >>> time.strftime('%Y-%m-%d %H:%M:%S', t1)
  '2021-07-21 17:37:41'
      # 用时间元组来创建一个自定义的时间
  >>> t2 = time.struct_time ( (2021,1, 1, 10, 11, 20, 0, 0, 0) )
  ```

###### 1.3.3 datetime模块

- 引入

  ```python
  import datetime
  ```

- 实例

  ```python
  >>> import datetime
  >>> d1 = datetime.datetime.now()  # 返回当前的时间
  >>> d1
  datetime.datetime(2021, 3, 30, 23, 32, 7, 342559)
  >>> d1.year
  2021
  >>> d1.year, d1.month, d1.day, d1.hour, d1.minute, d1.second, d1.microsecond  # 用 datetime 的各个属性可以得到 具体的信息
  (2021, 3, 30, 23, 32, 44, 757673)
  >>> d1.strftime("%Y-%m-%d")
  '2021-03-30'
  
  ## 计算时间差
  >>> delta_time = datetime.timedelta(days=2, hours=1)  # 生成 2天1小时后的时间差
  >>> delta_time
  datetime.timedelta(2, 3600)
  >>> t1 = datetime.datetime.now()  # 得到当前时间
  >>> t1
  datetime.datetime(2021, 3, 30, 23, 39, 26, 863109)
  >>> t1 + delta_time  # 计算 未来时间
  datetime.datetime(2021, 4, 2, 0, 39, 26, 863109)
  ```

###### 1.3.4 os模块

- os模块

  定义

  ```python
  `os`模块是Python标准库中的一部分，提供了一种与操作系统进行交互的方法。主要功能包括文件和目录的操作、路径处理、进程管理等
  
  import os
  ```

  实例

  ```python
  #1. os.getcwd(): 获取当前工作目录
  import os
  current_directory = os.getcwd()
  print("当前工作目录:", current_directory)
  
  #2. os.chdir(path): 改变当前工作目录
  import os
  new_directory = "/path/to/new/directory"
  os.chdir(new_directory)
  print("工作目录已更改为:", os.getcwd())
  
  #3. os.listdir(path='.'): 返回指定目录下的所有文件和目录列表
  import os
  directory_path = "."
  files_and_dirs = os.listdir(directory_path)
  print("指定目录下的文件和目录列表:", files_and_dirs)
  
  #4. os.mkdir(path): 创建目录
  import os
  new_directory = "new_folder"
  os.mkdir(new_directory)
  print(f"目录 '{new_directory}' 已创建")
  
  #5. os.rmdir(path): 删除目录
  import os
  directory_to_remove = "new_folder"
  os.rmdir(directory_to_remove)
  print(f"目录 '{directory_to_remove}' 已删除")
  
  #6. os.remove(path): 删除文件
  import os
  file_to_remove = "example.txt"
  os.remove(file_to_remove)
  print(f"文件 '{file_to_remove}' 已删除")
  ```

- os.path模块

  定义

  ```python
  `os.path` 模块是 Python 标准库的一部分，专门用于处理文件和目录路径的操作。它提供了一系列函数，用于操作和处理文件路径，使得路径操作更加方便和跨平台。
  ```

  实例

  ```python
  #1.os.path.basename(path): 返回路径中最后的文件名或目录名
  import os
  path = "/path/to/some/file.txt"
  print(os.path.basename(path))  # 输出: file.txt
  
  #2.os.path.dirname(path): 返回路径中的目录部分
  import os
  path = "/path/to/some/file.txt"
  print(os.path.dirname(path))  # 输出: /path/to/some
  
  #3.os.path.join(*paths): 将多个路径合并成一个路径
  import os
  path1 = "/path/to"
  path2 = "some/file.txt"
  full_path = os.path.join(path1, path2)
  print(full_path)  # 输出: /path/to/some/file.txt
  
  #4.os.path.split(path): 将路径分割成目录和文件名
  import os
  path = "/path/to/some/file.txt"
  print(os.path.split(path))  # 输出: ('/path/to/some', 'file.txt')
  
  #5.os.path.splitext(path): 将路径分割成文件名和扩展名
  import os
  path = "/path/to/some/file.txt"
  print(os.path.splitext(path))  # 输出: ('/path/to/some/file', '.txt')
  
  #6.os.path.exists(path): 检查路径是否存在
  import os
  path = "/path/to/some/file.txt"
  print(os.path.exists(path))  # 输出: True 或 False
  
  #7.os.path.isfile(path): 检查路径是否是文件
  import os
  path = "/path/to/some/file.txt"
  print(os.path.isfile(path))  # 输出: True 或 False
  
  #8.os.path.isdir(path): 检查路径是否是目录
  import os
  path = "/path/to/some/directory"
  print(os.path.isdir(path))  # 输出: True 或 False
  ```

#### 2.包

##### 2.1包的定义和作用

- 定义

  ```python
  - 包是将模块以文件夹的组织形式进行分组管理的方法，以便更好地组织和管理相关模块。
  
  - 包是一个包含一个特殊的`__init__.py`文件的目录，这个文件可以为空，但必须存在，以标识目录为Python包。
  
  - 包可以包含子包（子目录）和模块，可以使用点表示法来导入。
  ```

- 作用

  ```python
  - 将一系列模块进行分类管理,有利于防止命名冲突
  
  - 可以在需要时加载一个或部分模块而不是全部模块
  ```

- 实例

  ```python
   mypack/
          __init__.py
          menu.py            # 菜单管理模块
          games/
              __init__.py
              contra.py      # 魂斗罗
              supermario.py  # 超级玛丽 mario
              tanks.py       # 坦克大作战
          office/
              __init__.py
              excel.py
              word.py
              powerpoint.py
  ```

##### 2.2导入包和子包

```python
    # 同模块的导入规则
    import 包名 [as 包别名]
    import  包名.模块名 [as 模块新名]
    import  包名.子包名.模块名 [as 模块新名]
    
    from 包名 import 模块名 [as 模块新名]
    from 包名.子包名 import 模块名 [as 模块新名]
    from 包名.子包名.模块名 import 属性名 [as 属性新名]
    
    # 导入包内的所有子包和模块
    from 包名 import *
    from 包名.模块名 import *
```

##### 2.3使用包和子包

```python
# 使用包中的模块
import pandas as pd
data_frame = pd.DataFrame()

# 使用子包中的模块
from tensorflow.keras.layers import Dense
```

##### 2.4 __init__.py文件

- 说明

  ```python
  `__init__.py` 文件的主要作用是用于初始化Python包（package）或模块（module），它可以实现以下功能：
  
  1. **标识包目录：** 告诉Python解释器所在的目录应被视为一个包或包含模块的包。没有这个文件，目录可能不会被正确识别为包，导致无法导入包内的模块。
  
  2. **执行初始化代码：** 可以包含任何Python代码，通常用于执行包的初始化操作，如变量初始化、导入模块、设定包的属性等。这些代码在包被导入时会被执行。
  
  3. **控制包的导入行为：** 通过定义 `__all__` 变量，可以明确指定哪些模块可以被从包中导入，从而限制包的公开接口，防止不需要的模块被导入。
  
  4. **提供包级别的命名空间：** `__init__.py` 中定义的变量和函数可以在包的其他模块中共享，提供了一个包级别的命名空间，允许模块之间共享数据和功能。
  
  5. **批量导入模块：** 可以在 `__init__.py` 文件中批量导入系统模块或其他模块，以便在包被导入时，这些模块可以更方便地使用。
  ```

- 实例

  ```python
  # __init__.py 文件示例
  
  # 1. 批量导入系统模块
  import os
  import sys
  import datetime
  
  # 2. 定义包级别的变量
  package_variable = "This is a package variable"
  
  # 3. 控制包的导入行为
  __all__ = ['module1', 'module2']
  
  # 4. 执行初始化代码
  print("Initializing mypackage")
  
  # 注意：这个代码会在包被导入时执行
  
  # 5. 导入包内的模块
  from . import module1
  from . import module2
  
  """
  在这个示例中，`__init__.py` 文件用于批量导入系统模块、定义包级别的变量、控制包的导入行为、执行初始化代码，以及导入包内的模块。这有助于包的组织、初始化和导入管理。
  
  """
  ```

#### 3.第三方包

##### 3.1安装和使用

###### 3.1.1安装

- 使用pip安装包

  ```python
  pip install package-name
  ```

- 安装特定版本的包

  ```
  pip install package-name==version
  ```

- 通过镜像安装

  ```
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package1 package2 package3 ...
  ```

- 从requirements文件安装

  ```
  将要安装的包及其版本记录在一个文本文件中，通常命名为requirements.txt
  pip install -r requirements.txt
  ```

###### 3.1.2使

- 导入包

  ```python
  import package_name
  ```

- 使用包

  ```python
  每个第三方包都有不同的用法和功能，通常伴随着官方文档和示例代码。可以查阅官方文档，或者使用`help()`函数来了解包的功能和方法
  import package_name
  help(package_name)
  ```

- 更新和卸载包

  ```python
  更新包
  pip install --upgrade package-name
  
  卸载包
  pip uninstall package-name
  ```

##### 3.2依赖清单

- 作用

  可以使用pipreqs来维护requirements.txt文件，以便轻松地重建环境。

  ```
  `pipreqs`是一个用于管理Python项目依赖清单的工具，它会自动分析项目代码，并生成`requirements.txt`文件，列出项目所需的所有依赖包及其版本。
  ```

- 说明

  `pipreqs`是一个方便的工具，它可以帮助你自动创建和维护项目的依赖清单。不过，需要记住，生成的依赖清单可能包含一些不必要的依赖，因此你应该仔细检查和编辑`requirements.txt`文件以确保它反映了项目的真实需求。

  安装pipreqs

  ```
  pip install pipreqs  
  ```

  项目中运行pipreqs

  ```
  pipreqs .
  ```

  ```python
  这会分析项目代码，并在当前目录下生成一个名为`requirements.txt`的文件，其中包含了项目所需的所有依赖包及其版本。
  
  如果遇到编码错误UnicodeDecodeError，则将指定编码为utf8：
  pipreqs ./ --encoding=utf8
  pipreqs ./ --encoding=gbk
  pipreqs ./ --encoding='iso-8859-1' 
  ```

  查看生成的requirements.txt文件

  ```python
  打开`requirements.txt`文件，内容如下
  package1==1.0.0
  package2==2.1.3
  ...
  ```

  安装依赖

  ```
  pip install -r requirements.txt
  ```

  定期更新依赖

  ```
  定期使用`pipreqs`重新生成`requirements.txt`文件，以确保依赖清单保持最新。
  
  pipreqs .
  ```

### Python异常

#### 1.try语句

- 语法

  ```python
  try:
      可能发生异常的语句块
  except 错误类型1 [as 变量名1]:
      异常处理语句块1
  except 错误类型2 [as 变量名2]:
      异常处理语句块2
  ...
  except 错误类型n [as 变量名n]:
      异常处理语句块n
  except:
      异常处理语句块other
  else:
      未发生异常的语句
  finally:
      最终的处理语句
  ```

- 作用

  尝试捕获异常，得到异常通知，将程序由异常状态变为正常状态

- 说明

  ```python
  except 子句可以有 1个或多个
  
  except: 不给错误类型，可以匹配全部的错误类型
  
  else 子句里的语句会在 没有错误发生时执行，当处于异常时不执行
  
  finally 子句里的语句，无论何时都执行
  ```

- 实例

  ```python
  try:
      x = int(input("请输入一个整数："))
      print('x=', x)
  except ValueError:
      print('您的输入不能转成整数')
  
  print("程序结束")
  ```

#### 2.raise语句

- 语法

  ```python
  raise 异常类型
  或
  raise 异常对象
  ```

- 作用

  - 抛出一个错误，让程序进入异常状态
  - 发送错误通知给调用者

- 实例

  ```python
  ## 写一个函数, get_score 函数，读取用户输入的整数成绩,
  ## 成绩的正常值是0~100 之间， 要求, 如果不在0~100 之间
  ## 报 ValueError类型的错误
  def get_score():
      x = int(input('请输入成绩:'))
      if 0 <= x <= 100:
          return x
      # raise ValueError
      raise ValueError('用户输入的成绩不在 0～100 之间')
  
  try:
      score = get_score()
      print(score)
  except ValueError as err:
      print("成绩输入有误 err=", err)
  ```

#### 3.Python常见错误类型

| 错误类型                  | 说明                                               |
| ------------------------- | -------------------------------------------------- |
| ZeroDivisionError         | 除(或取模)零 (所有数据类型)                        |
| ValueError                | 传入无效的参数                                     |
| AssertionError            | 断言语句失败                                       |
| StopIteration             | 迭代器没有更多的值                                 |
| IndexError                | 序列中没有此索引(index)                            |
| IndentationError          | 缩进错误                                           |
| OSError                   | 输入/输出操作失败                                  |
| ImportError               | 导入模块/对象失败                                  |
| NameError                 | 未声明/初始化对象 (没有属性)                       |
| AttributeError            | 对象没有这个属性                                   |
|                           | <!-- 以下不常用 -->                                |
| GeneratorExit             | 生成器(generator)发生异常来通知退出                |
| TypeError                 | 对类型无效的操作                                   |
| KeyboardInterrupt         | 用户中断执行(通常是输入^C)                         |
| OverflowError             | 数值运算超出最大限制                               |
| FloatingPointError        | 浮点计算错误                                       |
| BaseException             | 所有异常的基类                                     |
| SystemExit                | 解释器请求退出                                     |
| Exception                 | 常规错误的基类                                     |
| StandardError             | 所有的内建标准异常的基类                           |
| ArithmeticError           | 所有数值计算错误的基类                             |
| EOFError                  | 没有内建输入,到达EOF 标记                          |
| EnvironmentError          | 操作系统错误的基类                                 |
| WindowsError              | 系统调用失败                                       |
| LookupError               | 无效数据查询的基类                                 |
| KeyError                  | 映射中没有这个键                                   |
| MemoryError               | 内存溢出错误(对于Python 解释器不是致命的)          |
| UnboundLocalError         | 访问未初始化的本地变量                             |
| ReferenceError            | 弱引用(Weak reference)试图访问已经垃圾回收了的对象 |
| RuntimeError              | 一般的运行时错误                                   |
| NotImplementedError       | 尚未实现的方法                                     |
| SyntaxError Python        | 语法错误                                           |
| TabError                  | Tab 和空格混用                                     |
| SystemError               | 一般的解释器系统错误                               |
| UnicodeError              | Unicode 相关的错误                                 |
| UnicodeDecodeError        | Unicode 解码时的错误                               |
| UnicodeEncodeError        | Unicode 编码时错误                                 |
| UnicodeTranslateError     | Unicode 转换时错误                                 |
| 以下为警告类型            |                                                    |
| Warning                   | 警告的基类                                         |
| DeprecationWarning        | 关于被弃用的特征的警告                             |
| FutureWarning             | 关于构造将来语义会有改变的警告                     |
| OverflowWarning           | 旧的关于自动提升为长整型(long)的警告               |
| PendingDeprecationWarning | 关于特性将会被废弃的警告                           |
| RuntimeWarning            | 可疑的运行时行为(runtime behavior)的警告           |
| SyntaxWarning             | 可疑的语法的警告                                   |
| UserWarning               | 用户代码生成的警告                                 |

### Python的文件操作

##### 1.打开文件

```python
# 打开一个文本文件以读取内容
file = open("example.txt", "r")
```

##### 2.读取文件

```python
# 读取整个文件内容
content = file.read()

# 逐行读取文件内容
for line in file:  #直接遍历文件对象，每次读取一行。这种方式更内存友好，因为不需要将所有行读入内存。
    print(line)

with open('example.txt', 'r') as file:
    lines = file.readlines() # 读取文件的所有行，并将其作为一个列表返回。
    for line in lines:
        print(line, end='') 代码和file = open("example.txt", "r")for line in file:
    print(line) 代码的区别
```

##### 3.写入文件

```python
# 打开文件以写入内容
file = open("example.txt", "w")

# 写入内容
file.write("这是一个示例文本。")
```

##### 4.关闭文件

```python
file.close()
```

##### 5.使用with

```python
with open("example.txt", "r") as file:
    content = file.read()
    # 文件自动关闭
```

##### 6.检查是否存在

```python
import os

if os.path.exists("example.txt"):
    print("文件存在")
```

##### 7.处理异常

```python
try:
    with open("example.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("文件不存在")
except Exception as e:
    print(f"发生错误：{e}")
```

##### 8.复制文件

```python
import shutil

source_file = "source.txt"
destination_file = "destination.txt"

shutil.copy(source_file, destination_file)
```

##### 9.删除文件

```python
import os

file_to_delete = "file_to_delete.txt"

if os.path.exists(file_to_delete):
    os.remove(file_to_delete)
    print(f"{file_to_delete} 已删除")
else:
    print(f"{file_to_delete} 不存在")
```

##### 10.修改文件

```python
import os

old_name = "old_name.txt"
new_name = "new_name.txt"

if os.path.exists(old_name):
    os.rename(old_name, new_name)
    print(f"文件名已更改为 {new_name}")
else:
    print(f"{old_name} 不存在")
```

### Python Json数据解析

#### 1.导入Json模块

```python
import json
```

#### 2.序列化

```python
import json
data = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

json_str = json.dumps(data)  # json.dumps() 是 Python 的 json 模块中的一个函数，它的作用是将 Python 对象转换为 JSON 格式的字符串。
print(json_str)
```

#### 3.反序列化

```python
json_str = '{"name": "John", "age": 30, "city": "New York"}'

data = json.loads(json_str) # json.loads() 是 Python json 模块中的一个函数，它的作用是将 JSON 格式的字符串转换为 Python 对象。
print(data)
```

#### 4.对象存文件

```python
data = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

with open('data.json', 'w') as json_file:
    json.dump(data, json_file)
```

#### 5.从文件加载

```python
with open('data.json', 'r') as json_file:
    data = json.load(json_file)
    print(data)
```

#### 6.嵌套Json数据

```python
如果JSON数据包含嵌套结构，可以使用递归来访问和修改其中的值
json_data = {
    "name": "Alice",
    "info": {
        "age": 25,
        "location": "Paris"
    }
}

# 获取嵌套的值
age = json_data["info"]["age"]

# 修改嵌套的值
json_data["info"]["location"] = "New York"

# 将更改后的数据转换为JSON字符串
new_json_str = json.dumps(json_data)
```

#### 7.Json中列表

```python
json_data = {
    "fruits": ["apple", "banana", "cherry"]
}

# 获取列表中的第一个水果
first_fruit = json_data["fruits"][0]

# 添加一个新水果到列表
json_data["fruits"].append("orange")
```

#### 8.Json中空值

```python
JSON允许表示空值（null），在Python中，它通常转换为`None`。

json_data = {
    "value": None
}
```

#### 9.**字典和JSON格式不同之处**

1. **数据类型限制**：
   - **JSON**：支持的数据类型包括对象（类似于字典）、数组（类似于列表）、字符串、数字、布尔值和 `null`。JSON 不支持 Python 特有的数据类型如 `tuple`、`set`、`bytes` 等。
   - **Python 字典**：可以包含多种 Python 特有的数据类型，比如 `tuple`、`set`、`bytes` 等。
2. **格式要求**：
   - **JSON**：数据必须以字符串的形式表示，键必须是双引号括起来的字符串，值可以是字符串、数字、布尔值、数组、对象或 `null`。
   - **Python 字典**：键可以是任意不可变的类型（如字符串、数字、元组），值可以是任意类型。键通常用单引号或双引号括起来，但 Python 允许在字典中使用不加引号的键。



### Python正则表达式

|  **模式**   | **描述**                                                     |
| :---------: | :----------------------------------------------------------- |
|      ^      | 匹配字符串的开头                                             |
|      $      | 匹配字符串的末尾。                                           |
|      .      | 匹配任意字符，除了换行符，当re.DOTALL标记被指定时，则可以匹配包括换行符的任意字符。 |
|     […]     | 用来表示一组字符,单独列出：[amk] 匹配 'a'，'m'或'k'          |
|    [^…]     | 不在[]中的字符：abc 匹配除了a,b,c之外的字符。                |
|     re*     | 匹配0个或多个的表达式。                                      |
|     re+     | 匹配1个或多个的表达式。                                      |
|     re?     | 匹配0个或1个由前面的正则表达式定义的片段，非贪婪方式         |
|    re{n}    | 匹配n个前面表达式。例如，"o{2}"不能匹配"Bob"中的"o"，但是能匹配"food"中的两个o。 |
|   re{n,}    | 精确匹配n个前面表达式。例如，"o{2,}"不能匹配"Bob"中的"o"，但能匹配"foooood"中的所有o。"o{1,}"等价于"o+"。"o{0,}"则等价于"o*"。 |
|  re{n, m}   | 匹配 n 到 m 次由前面的正则表达式定义的片段，贪婪方式         |
|     \|      | 匹配a或b                                                     |
|    (re)     | 匹配括号内的表达式，也表示一个组                             |
|   (?imx)    | 正则表达式包含三种可选标志：i, m, 或 x 。只影响括号中的区域。 |
|   (?-imx)   | 正则表达式关闭 i, m, 或 x 可选标志。只影响括号中的区域。     |
|   (?: re)   | 类似 (...), 但是不表示一个组                                 |
| (?imx: re)  | 在括号中使用i, m, 或 x 可选标志                              |
| (?-imx: re) | 在括号中不使用i, m, 或 x 可选标志                            |
|   (?#...)   | 注释.                                                        |
|   (?= re)   | 前向肯定界定符。如果所含正则表达式，以 ... 表示，在当前位置成功匹配时成功，否则失败。但一旦所含表达式已经尝试，匹配引擎根本没有提高；模式的剩余部分还要尝试界定符的右边。 |
|   (?! re)   | 前向否定界定符。与肯定界定符相反；当所含表达式不能在字符串当前位置匹配时成功。 |
|   (?> re)   | 匹配的独立模式，省去回溯。                                   |
|     \w      | 匹配数字字母下划线                                           |
|     \W      | 匹配非数字字母下划线                                         |
|     \s      | 匹配任意空白字符，等价于 [\t\n\r\f]。                        |
|     \S      | 匹配任意非空字符                                             |
|     \d      | 匹配任意数字，等价于 [0-9]。                                 |
|     \D      | 匹配任意非数字                                               |
|     \A      | 匹配字符串开始                                               |
|     \Z      | 匹配字符串结束，如果是存在换行，只匹配到换行前的结束字符串。 |
|     \z      | 匹配字符串结束                                               |
|     \G      | 匹配最后匹配完成的位置。                                     |
|     \b      | 匹配一个单词边界，也就是指单词和空格间的位置。例如， 'er\b' 可以匹配"never" 中的 'er'，但不能匹配 "verb" 中的 'er'。 |
|     \B      | 匹配非单词边界。'er\B' 能匹配 "verb" 中的 'er'，但不能匹配 "never" 中的 'er'。 |
| \n, \t, 等  | 匹配一个换行符。匹配一个制表符, 等                           |
|   \1...\9   | 匹配第n个分组的内容。                                        |
|     \10     | 匹配第n个分组的内容，如果它经匹配。否则指的是八进制字符码的表达式。 |

#### 1.字符匹配

- 常用字符匹配功能

  | **方法**      | **功能**                                                     |
  | ------------- | ------------------------------------------------------------ |
  | match()       | 判断一个正则表达式是否从开始处匹配一个字符串                 |
  | search()      | 遍历字符串，找到正则表达式匹配的第一个位置，返回匹配对象     |
  | **findall()** | 遍历字符串，找到正则表达式匹配的所有位置，并以**列表**的形式返回。如果给出的正则表达式中包含子组，就会把子组的内容单独返回，如果有多个子组就会以元组的形式返回。 |
  | finditer()    | 遍历字符串，找到正则表达式匹配的所有位置，并以迭代器的形式返回 |

  - **hqyj匹配文本中的hqyj**

    ```python
    import re
    text="hqyj牛皮6666,hqyj有个老师也牛皮666"
    data=re.findall("hqyj",text)
    print(data)#['hqyj', 'hqyj']
    ```

  - **[hqyj]匹配h或者q或者y或者j字符**

    ```python
    import re
    text="hqyj牛皮6666,hqyj有个老师也牛皮666"
    data=re.findall("[hqyj]",text)
    print(data)#['h', 'q', 'y', 'j', 'h', 'q', 'y', 'j']
    
    import re
    text="hqyj牛皮6666,hqyj有个老师也牛皮666"
    data=re.findall("[hqyj]牛",text)
    print(data)#['j牛']
    ```

  - **[^hqyj]匹配除了hqyj以外的其他字符**

    ```python
    import re
    text="hqyj牛皮6666,hqyj有个老师也牛皮666"
    data=re.findall("[^hqyj]",text)
    print(data)#['牛', '皮', '6', '6', '6', '6', ',', '有', '个', '老', '师', '也', '牛', '皮', '6', '6', '6']
    ```

  - **[a-z]匹配a~z的任意字符([0-9]也可以)**

    ```python
    import re
    text="hqyj牛皮6666,hqyj有个老师abchqyj也牛皮666"
    data=re.findall("[a-z]hqyj",text)
    print(data)#['chqyj']
    ```

  - **.匹配除了换行符以外的任意字符**

    ```python
    import re
    text="hqyj牛皮6666,hqyj有个老师abchqyj也牛皮666"
    data=re.findall(".hqyj",text)
    print(data)#[',hqyj', 'chqyj']
    
    import re
    text="hqyj牛皮6666,hqyj有个老师abchqyj也牛皮666"
    data=re.findall(".+hqyj",text) #贪婪匹配(匹配最长的)
    print(data)#['hqyj牛皮6666,hqyj有个老师abchqyj']
    
    import re
    text="hqyj牛皮6666,hqyj有个老师abchqyj也牛皮666"
    data=re.findall(".?hqyj",text)
    print(data)#['hqyj', ',hqyj', 'chqyj']
    ```

- 特殊字符

  | **特殊字符** | **含义**                                                     |
  | ------------ | ------------------------------------------------------------ |
  | \d           | 匹配任何十进制数字；相当于类 [0-9]                           |
  | \D           | 与 \d 相反，匹配任何非十进制数字的字符；相当于类 0-9         |
  | \s           | 匹配任何空白字符（包含空格、换行符、制表符等）；相当于类 [  \t\n\r\f\v] |
  | \S           | 与 \s 相反，匹配任何非空白字符；相当于类   \t\n\r\f\v        |
  | \w           | 匹配任意一个文字字符，包括大小写字母、数字、下划线，等价于表达式[a-zA-Z0-9_] |
  | \W           | 于 \w 相反  (注：re.ASCII 标志使得 \w 只能匹配 ASCII 字符)   |
  | \b           | 匹配单词的开始或结束                                         |
  | \B           | 与 \b 相反                                                   |

  - **\w 匹配字母数字下划线(汉字)**

    ```python
    import re
    text="华清_远见abc 华清hqyj远见 华清牛皮远见"
    data=re.findall("华清\w+远见",text)
    print(data)#['华清_远见', '华清hqyj远见', '华清牛皮远见']
    ```

  - **\d匹配数字**

    ```python
    import re
    text="hqyj66d6 a1h43d3fd43s43d4 "
    data=re.findall("d\d",text) # 只匹配一个数字
    print(data)#['d6', 'd3', 'd4', 'd4']
    
    import re
    text="hqyj66d6 a1h43d3fd43s43d4 "
    data=re.findall("d\d+",text)
    print(data)#['d6', 'd3', 'd43', 'd4']
    ```

  - **\s匹配任意空白符 包括空格,制表符等等**

    ```python
    import re
    text="hqyj666  jack karen 666"
    data=re.findall("\sj\w+\s",text)
    print(data)#[' jack '
    ```

#### 2.数量控制

- ***重复0次或者更多次**

  ```python
  import re
  text="华清远见 华清666远见"
  data=re.findall("华清6*远见",text)
  print(data)#['华清远见', '华清666远见']
  ```

- **+重复1次或者更多次**

  ```python
  import re
  text="华清远见 华清666远见 华清6远见"
  data=re.findall("华清6+远见",text)
  print(data)#['华清666远见', '华清6远见']
  ```

- **?重复1次或者0次**

  ```python
  import re
  text="华清远见 华清666远见 华清6远见"
  data=re.findall("华清6?远见",text)
  print(data)#['华清远见', '华清6远见']
  ```

- **{n}重复n次,n是数字**

  ```python
  import re
  text="华清远见 华清666远见 华清6远见"
  data=re.findall("华清6{3}远见",text)
  print(data)#['华清666远见']
  ```

- **{n,}重复n次或者更多次**

  ```python
  import re
  text="华清远见 华清666远见 华清6远见 华清66远见"
  data=re.findall("华清6{2,}远见",text)
  print(data)#['华清666远见', '华清66远见']
  ```

- **{n,m}重复n到m次**

  ```python
  import re
  text="华清远见 华清666远见 华清6远见 华清66远见"
  data=re.findall("华清6{0,2}远见",text)
  print(data)#['华清远见', '华清6远见', '华清66远见']
  ```

#### 3.分组

- **()提取兴趣区域**

  ```python
  import re
  text="谢帝谢帝,我要迪士尼,我的电话号码18282832341,qq号码1817696843"
  data=re.findall("号码(\d{10,})",text)
  print(data)#['18282832341', '1817696843']
  
  import re
  text="谢帝谢帝,我要迪士尼,我的电话号码18282832341,qq号码1817696843"
  data=re.findall("(\w{2}号码(\d{10,}))",text)
  print(data)#['18282832341', '1817696843']
  ```

- **(|)提取兴趣区域(| = or)**

  ```python
  import re
  text="第一名张三 第一名物理149分 第一名数学150分 第一名英语148分 第一名总分740分"
  data=re.findall("第一名(\w{2,}|\w{2,}\d{2,}分)",text)
  print(data)#['张三', '物理149分', '数学150分', '英语148分', '总分740分']
  ```

#### 4.开始和结束

- **^开始**

  ```python
  import re
  text = "hqyj66abc hqyj123"
  data = re.findall("^hqyj\d+", text)
  print(data)  #['hqyj66']
  ```

- **$结尾**

  ```python
  import re
  text = "hqyj66abc hqyj123"
  data = re.findall("hqyj\d+$", text)
  print(data)  #['hqyj123']
  ```

#### 5.特殊字符

```python
由于正则表达式中* . \  {} () 等等符号具有特殊含义,如果你指定的字符正好就是这些符号,需要用\进行转义

import re
text = "数学中集合的写法是{2}"
data = re.findall("\{2\}", text)
print(data)  #['{2}']
```

#### 6.re模块的常用方法

###### 1.re.findall

```python
获取匹配到的所有数据

import re
text="hqyj66d6 a1h43d3fd43s43d4 "
data=re.findall("d\d+",text)
print(data)#['d6', 'd3', 'd43', 'd4']
```

###### 2.re.match

从字符串的起始位置匹配，成功返回一个对象否则返回none

| **方法** | **功能**                               |
| -------- | -------------------------------------- |
| group()  | 返回匹配的字符串                       |
| start()  | 返回匹配的开始位置                     |
| end()    | 返回匹配的结束位置                     |
| span()   | 返回一个元组表示匹配位置（开始，结束） |

```python
import re

# 在起始位置匹配，并返回一个包含匹配 (开始,结束) 的位置的元组
print(re.match('www', "www.python.com").span())#(0, 3)
print(re.match('www', "www.python.com").start())#0
print(re.match('www', "www.python.com").end())#3
# 不在起始位置匹配
print(re.match('com', "www.python.com"))# None
```

###### 3.re.search

```python
扫描整个字符串并返回第一个成功匹配的字符串。成功返回一个对象否则返回none

import re

# 在起始位置匹配 
print(re.search('www', 'www.hqyj.com').span())#(0, 3)
# 不在起始位置匹配
print(re.search('com', 'www.hqyj.com').span())#(9, 12)
```

###### 4.re.sub

```python
替换匹配成功的字符

import re
text = "以前华清远见在四川大学旁边,现在华清远见在西南交大旁边"
data = re.sub("华清远见","北京华清远见科技集团成都中心", text)
print(data)#以前北京华清远见科技集团成都中心在四川大学旁边,现在北京华清远见科技集团成都中心在西南交大旁边
```

###### 5.re.split

```python
根据匹配成功的位置对字符串进行分割

import re
text = "python is   very easy"
data = re.split("\s{1,}", text)
print(data)#['python', 'is', 'very', 'easy']
```

###### 6.re.finditer

```python
类似findall 但是不会全部返回出来 而是返回迭代器(比如匹配成功了10万个  全部返回就很吃内存了)

import re
text = "python is   very easy"
data = re.findall("\w+", text)
print(data)#['python', 'is', 'very', 'easy']

import re
text = "python is   very easy"
data = re.finditer("\w+", text)
print(data)
for el in data:
    print(el.group())
```

#### 7.常见的一些正则

```python
QQ号：[1 - 9][0 - 9]{4, }(腾讯QQ号从10000开始)
帐号(字母开头，允许5-16字节，允许字母数字下划线)：^[a-zA-Z][a-zA-Z0-9_]{4,15}$
手机号码：^(13[0-9]|14[5|7]|15[0|1|2|3|5|6|7|8|9]|18[0|1|2|3|5|6|7|8|9])\d{8}$
Email地址：^\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$
密码(以字母开头，长度在6~18之间，只能包含字母、数字和下划线)：^[a-zA-Z]\w{5,17}$
身份证号(15位、18位数字)：^\d{15}|\d{18}$
短身份证号码(数字、字母x结尾)：^([0-9]){7,18}(x|X)?$ 或 ^\d{8,18}|[0-9x]{8,18}|[0-9X]{8,18}?$
```

### Python网络

Python requests 是一个常用的 HTTP 请求库，可以方便地向网站发送 HTTP 请求，并获取响应结果。

使用 requests 发送 HTTP 请求需要先下载并导入 requests 模块：

```python
# 导入 requests 包
import requests
response = requests.get('https://www.baidu.com')# 发送请求
print(response.content)  # 获取响应内容


response2 = requests.get('http://localhost:7001/test')# 发送请求
print(response2.json())  # 获取json数据并解析
```

每次调用 requests 请求之后，会返回一个 response 对象，该对象包含了具体的响应信息，如状态码、响应头、响应内容等：

| response的属性或方法  | 说明                                                         |
| --------------------- | ------------------------------------------------------------ |
| apparent_encoding     | 编码方式                                                     |
| close()               | 关闭与服务器的连接                                           |
| **content**           | 返回响应的内容，以字节为单位                                 |
| cookies               | 返回一个 CookieJar 对象，包含了从服务器发回的 cookie         |
| elapsed               | 返回一个 timedelta 对象，包含了从发送请求到响应到达之间经过的时间量，可以用于测试响应速度。比如 r.elapsed.microseconds 表示响应到达需要多少微秒。 |
| encoding              | 解码 r.text 的编码方式                                       |
| headers               | 返回响应头，字典格式                                         |
| history               | 返回包含请求历史的响应对象列表（url）                        |
| is_permanent_redirect | 如果响应是永久重定向的 url，则返回 True，否则返回 False      |
| is_redirect           | 如果响应被重定向，则返回 True，否则返回 False                |
| iter_content()        | 迭代响应                                                     |
| iter_lines()          | 迭代响应的行                                                 |
| json()                | 返回结果的 JSON 对象 (结果需要以 JSON 格式编写的，否则会引发错误) |
| links                 | 返回响应的解析头链接                                         |
| next                  | 返回重定向链中下一个请求的 PreparedRequest 对象              |
| ok                    | 检查 "status_code" 的值，如果小于400，则返回 True，如果不小于 400，则返回 False |
| raise_for_status()    | 如果发生错误，方法返回一个 HTTPError 对象                    |
| reason                | 响应状态的描述，比如 "Not Found" 或 "OK"                     |
| request               | 返回请求此响应的请求对象                                     |
| status_code           | 返回 http 的状态码，比如 404 和 200（200 是 OK，404 是 Not Found） |
| text                  | 返回响应的内容，unicode 类型数据                             |
| url                   | 返回响应的 URL                                               |

**requests的方法**

| 方法                             | 描述                            |
| -------------------------------- | ------------------------------- |
| delete(*url*, *args*)            | 发送 DELETE 请求到指定 url      |
| get(*url*, *params, args*)       | 发送 GET 请求到指定 url         |
| head(*url*, *args*)              | 发送 HEAD 请求到指定 url        |
| patch(*url*, *data, args*)       | 发送 PATCH 请求到指定 url       |
| post(*url*, *data, json, args*)  | 发送 POST 请求到指定 url        |
| put(*url*, *data, args*)         | 发送 PUT 请求到指定 url         |
| request(*method*, *url*, *args*) | 向指定的 url 发送指定的请求方法 |

**requests.get(url, params ={key: value}, args)**

- **url** 请求 url。

- **params ** 参数为要发送到指定 url 的 JSON 对象。

- **args** 为其他参数，比如 cookies、headers、verify等。

  ```python
  import requests
  
  # 图片URL地址
  image_url = 'http://localhost:7001/public/1.png'
  
  # 发送GET请求获取图片数据
  response = requests.get(image_url)
  
  # 检查请求是否成功（HTTP状态码为200）
  if response.status_code == 200:
      # 将图片数据写入本地文件
      with open('image.jpg', 'wb') as f:
          f.write(response.content)
      print("图片已成功下载并保存为 image.jpg")
  else:
      print(f"无法下载图片，响应状态码：{response.status_code}")
  ```

**requests.post(url, data={key: value}, json={key: value}, args)**

- **url** 请求 url。

- **data** 参数为要发送到指定 url 的字典、元组列表、字节或文件对象。

- **json** 参数为要发送到指定 url 的 JSON 对象。

- **args** 为其他参数，比如 cookies、headers、verify等。

  ```python
  import requests
  headers = {'User-Agent': 'Mozilla/5.0'}  # 设置请求头
  params = {'key1': 'value1', 'key2': 'value2'}  # 设置查询参数
  data = {'username': 'jack', 'password': '123456'}  # 设置请求体
  response = requests.post('http://localhost:7001/test', headers=headers, params=params, data=data)
  print(response.text)
  ```

  