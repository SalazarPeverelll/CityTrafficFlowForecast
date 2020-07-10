import torch
import pandas
import matplotlib.pyplot as plt
import numpy as np
"""
注：
    这里仅仅是对一个训练集进行反复训练得出的结果，并无太大意义。
    文件中定义了RNN LSTM两种网络
    LSTM效果更好
"""


# 超参数定义
TIME_STEP = 140
# 把这个目录改成dataSet的目录
data = pandas.read_csv("../DataSets/01-12/chongzhi_beier-east-01-12.csv", header=None)
# data.plot()
# plt.show()
# 读取数据

left_data = data.iloc[0:140, 3].to_numpy()
index = np.arange(left_data.size)
size = left_data.size


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(1, 64, 2, batch_first=True)
        self.out = torch.nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x, None)
        x = self.out(x)
        x = x.reshape(-1, TIME_STEP, 1)
        return x


# 构建网络
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.out = torch.nn.Linear(64, 1)

    def forward(self, x, h_state):
        # reshape to [batch_size, time_step, input_size]
        x = x.reshape((-1, TIME_STEP, 1))
        x, h_state = self.rnn(x, h_state)
        x = self.out(x)
        # reshape to [batch_size, time_step, output_size]
        out = x.reshape((-1, TIME_STEP, 1))
        return out, h_state


rnn = LSTM()
h_state = None
optimizer = torch.optim.Adam(rnn.parameters(), 0.003)
loss_func = torch.nn.MSELoss()

# 开始训练
for i in range(1000):
    index_input = torch.Tensor(index.reshape((-1, TIME_STEP, 1)))
    left_data = torch.Tensor(left_data.reshape((-1, TIME_STEP, 1))).type(torch.FloatTensor)
    # output, h_state = rnn(index_input, h_state)
    output = rnn(index_input)
    # h_state = h_state.data
    optimizer.zero_grad()
    loss = loss_func(output, left_data)
    loss.backward()
    optimizer.step()

# test_index = np.arange(size)
test_index0 = np.arange(size)
test_index1 = np.arange(0, 2*size)
test_index0 = torch.Tensor(test_index0.reshape((-1, TIME_STEP, 1))).type(torch.FloatTensor)
test_index1 = torch.Tensor(test_index1.reshape((-1, TIME_STEP, 1))).type(torch.FloatTensor)
# output, _ = rnn(test_index, h_state)
output = rnn(test_index1)
test_index1 = test_index1.reshape((-1))
test_index0 = test_index0.reshape((-1))
left_data = left_data.reshape((-1))
output = output.reshape((-1)).data.numpy()
print(output)
# plt.subplot(121), plt.plot(test_index, output, 'g')
# plt.subplot(122), plt.plot(test_index, left_data, 'g')
plt.plot(test_index1, output, 'r')
plt.plot(test_index0, left_data, 'g')
plt.show()
