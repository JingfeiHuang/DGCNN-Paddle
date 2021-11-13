
import numpy as np
import paddle
import paddle.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.reshape([-1])

    if smoothing:
        eps = 0.2
        n_class = pred.shape[1]

        one_hot_array = np.zeros_like(pred.detach().cpu().numpy())
        index = gold.cpu().numpy()
        for i in range(pred.shape[0]):
            one_hot_array[i][index[i]]=1
        one_hot = paddle.to_tensor(one_hot_array)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, axis=1)

        loss = -(one_hot * log_prb).sum(axis=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
