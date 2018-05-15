- remove()方法适用于知道要删除的具体的值，pop(),del 适合不知道值，但是知道位置的情况。
```
list.remove(具体的值)
list.pop(位置)
del list[位置]
```
- 列表切片，起始索引的值包含在结果中，最后的不包括在结果中
`list[1:4] #最后结果包括123`,可以认为切片索引并不指向元素本身，而指向元素边缘。倒着输出`list[::-1]`
- 元组中数据不可更改，所以没有append(),insert()的操作，**字符串可以看作特殊的元组，所以字符串的值不可变**，二维tuple第二维如果是list则第二维可以更改，保证指向元素不变就可以。
- zip产生结对元素，与for循环一起用，给字典赋值
```
for key,value in zip(list a,list b):
    字典[key]=value
```
字典的索引号并非自动生成，而是键。`dict.get(键值)`，返回value,`key.has_key(键值)`返回bool类型值。</br>
如果想获得所有键值`dict.keys()`,`dict.values()`,获取所有元组对`dict.items()`,删除字典元素
```
del dict[1]
dict.pop(2)
dict.clear() #全删了
```
- 创建空集合，必须用set(),不能用{},赋值用{}，set可以去除重复元素
- 读取文件</br>
```
f=open('文件名','r')
content=f.read()
f.close()

with open('文件名','r') as f:
    content=f.read()
```
- 用分割符提取文本文件的关键信息
```
f=open('文件名'，'r')
content=f.readlines()
f.close()

content_new=[]
for con in content:
    tmp=con.strip()
    tmp=con.split('\t')
    content_new.append(tmp)
```
csv读取文件可以`import csv`调用 `csv.reader(f)`方法,写入文件可以`writer=csv.writer(f)`,使用`writer.writerow()`逐行写入
- python中list的apply方法，`apply(test, userInput)`
表示把userInput作为test的参数，也就是说比如在程序运行时，userInput得到的值是[1,2,3]，那么就相当于test(1,2,3)。如果userInput得到的实际值是[1,2]，那么就相当于test(1,2)。
- np.digitize() 如果满足 bins[i-1]<=a<bins[i],那么就保存i
http://blog.csdn.net/weixin_38358654/article/details/78997769 

- X=iris.drop("Class",axis=1) 不要忘了后面的axis=1

- 朴素贝叶斯中处理文本文件,nlp,然后有一个有趣的包，叫jieba，好神奇，安装教程：<!https://www.cnblogs.com/xuqiulin/p/6623182.html>

- 不能在动态遍历循环时删元素
- 类中变量前加__变成私有变量
- 继承类用变量需要加self
```
class Female_student(High_school_student):
    def __init__(self):
        High_school_student.__init__(self)
        self.female_teenager_sns=[]
        for i in range(len(self.teenager_sns)):
            if(self.teenager_sns[i][1]=='F'):
                self.female_teenager_sns.append(self.teenager_sns[i])
```
- 
```
def read_csv(path):
    f=open(path,'r')
    reader=csv.reader(f)
    content=[]
    for row in reader:
        content.append(row)
    f.close()
    return content
```

 在函数中用`global length=***`可以变成全局变量
- enumerate(list) 产生一个字典，对应在列表中的序号+列表中的值
- re.search(pattern,string,flag)可以匹配任意位置的字符，re.match(pattern,string,flag)只能从头匹配
- re.sub(pattern,repl,string,count=0,flags=0) 把string中符合pattern的替换成repl
- 正则式匹配[]内的元素表示或匹配[]+表示多次匹配
- datetime.strptime(date_string,format),data_string日期，format日期格式
- 判断一个key在不在字典中` if str(month) not in stat_dict.keys():`
- 打印numpy数组的前五行 array[:5,:]，前五行是0-4，写五是因为最后一个数字不包括进去
- array.astype("float_") 将array里的数据转换成float型,**转换成float之前改变类型，如果有空值，换成np.nan**
-
```
---------------------------------------------------------------------------ValueError Traceback (most recent call last) in ()
1 #请在下面作答#
2 import numpy as np
----> 3 sh_ndarray=np.ndarray(sh_content)
4 sh_part_ndarray=sh_ndarray[1:,1:5]
5 print sh_part_ndarray[:5,:]
ValueError: sequence too large; cannot be greater than 32
```
出现这个错误，把ndarray换成array
- 按布尔值索引方法找到缺省值并将缺省值替换为np.nan`np_str[np_str=='']=np.nan`
- np.log(array) 求array的对数，np.diff(array，n=1,axis=n)，求的是后一个数减前一个数的list，n是0是纵向的，n是1是横向的
- numpy.amax(array,axis)返回某行或某列的最大值，最小值，numpy.argmax(array,axis,out)返回最大值所在位置，记得万一有表头，把表头那行加上
- 防止某个变量出错用try...except Exception,e
- 带命名数组，相当于加表头
```
content_array=[tuple(row) for row in content[1:]]

dtype=[('Date',object),('Location',object),('Operator',object),('Type', object),('Aboard', object),('Fatalities', object),('Summary', object)]

content_name=np.array(content_array,dtype=dtype)
```
- sorted函数对字典排序，`a=sorted(stat_operator.items(),key=lambda item:item[1])`</br>实现按照字典的value对整个字典排序，返回一个list，list的元素是元组，每个元组由key和value组成，还可以只对value,或者key排序，`a=sorted(dict.keys())`，这时返回的就是一个由key组成的list，默认从小到大，从大到小把reverse设置为TRUE
