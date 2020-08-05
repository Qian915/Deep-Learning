import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
data = pd.read_csv('data.csv', sep=';')
train_data, val_data = train_test_split(data, test_size=0.2)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train_set = ChallengeDataset(train_data, 'train')
val_set = ChallengeDataset(val_data, 'val')
train_dl = t.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)  # set batch_size = 8
val_dl = t.utils.data.DataLoader(val_set, batch_size=8, num_workers=2)


# create an instance of our ResNet model
# TODO
model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
crit = t.nn.BCELoss()
optim = t.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
trainer = Trainer(model, crit, optim, train_dl, val_dl, True, 100)  # set early stopping persistence

# go, go, go... call fit on trainer
# TODO
res = trainer.fit(600)  # set epochs

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig('losses.png')
