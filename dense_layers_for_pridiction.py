import ...

# 在线下载汽车效能数据集
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/autompg.data")
# 利用 pandas 读取数据集，字段有效能（公里数每加仑），气缸数，排量，马力，重量
# 加速度，型号年份，产地
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()


# 查看部分数据
dataset.head() #原始数据中的数据可能含有空字段(缺失值)的数据项，需要清除这些记录项：
dataset.isna().sum() # 统计空白数据
dataset = dataset.dropna() # 删除空白数据项
dataset.isna().sum() # 再次统计空白数据


# 处理类别型数据，其中 origin 列代表了类别 1,2,3,分布代表产地：美国、欧洲、日本
# 先弹出(删除并返回)origin 这一列
origin = dataset.pop('Origin')
# 根据 origin 列来写入新的 3 个列
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()



# 按着 8:2 的比例切分训练集和测试集：
# 切分为训练集和测试集
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# 将 MPG 字段移出为标签数据：
# 移动 MPG 油耗效能这一列为真实标签 Y
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')



#统计训练集的各个字段数值的均值和标准差，并完成数据的标准化：
# 查看训练集的输入 X 的统计数据
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()


# 标准化数据
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


#打印出训练集和测试集的大小：
print(normed_train_data.shape,train_labels.shape)
print(normed_test_data.shape, test_labels.shape)
(314, 9) (314,) # 训练集共 314 行，输入特征长度为 9,标签用一个标量表示
(78, 9) (78,) # 测试集共 78 行，输入特征长度为 9,标签用一个标量表示

#利用切分的训练集数据构建数据集对象：
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values,
train_labels.values)) # 构建 Dataset 对象
train_db = train_db.shuffle(100).batch(32) # 随机打散，批量化

###############################################################################

class Network(keras.Model):
    # 回归网络
	def __init__(self):
	    super(Network, self).__init__()
		# 创建 3 个全连接层
		self.fc1 = layers.Dense(64, activation='relu')
		self.fc2 = layers.Dense(64, activation='relu')
		self.fc3 = layers.Dense(1)
		
		
	def call(self, inputs, training=None, mask=None):
	    # 依次通过 3 个全连接层
		x = self.fc1(inputs)
		x = self.fc2(x)
		x = self.fc3(x)
	    return x


model = Network() # 创建网络类实例
# 通过 build 函数完成内部张量的创建，其中 4 为任意的 batch 数量， 9 为输入特征长度
model.build(input_shape=(4, 9))
model.summary() # 打印网络信息
optimizer = tf.keras.optimizers.RMSprop(0.001) # 创建优化器，指定学习率


#接下来实现网络训练部分。通过 Epoch 和 Step 的双层循环训练网络，共训练 200 个 epoch:

for epoch in range(200): # 200 个 Epoch
    for step, (x,y) in enumerate(train_db): # 遍历一次训练集
    # 梯度记录器
	with tf.GradientTape() as tape:
	    out = model(x) # 通过网络获得输出
		loss = tf.reduce_mean(losses.MSE(y, out)) # 计算 MSE
		mae_loss = tf.reduce_mean(losses.MAE(y, out)) # 计算 MAE
		if step % 10 == 0: # 打印训练误差
		    print(epoch, step, float(loss))
        # 计算梯度，并更新
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
