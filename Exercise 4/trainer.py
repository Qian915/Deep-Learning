import torch as t
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import binarize
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        # TODO

        if self._cuda:
            x = x.clone().detach().cuda()
            y = y.clone().detach().cuda().squeeze()

            #x = t.tensor(x, dtype=t.float).cuda()
            #y = t.tensor(y, dtype=t.float).cuda().squeeze()

        self._optim.zero_grad()
        y_pred = self._model(x)

        y_predTmp = y_pred.clone()
        y_predTmp = binarize(y_predTmp.cpu().detach().numpy(), threshold=0.5)  # numpy array w/o grad
        y_pred.data = t.tensor(y_predTmp, dtype=t.float).cuda()

        loss = self._crit(y_pred, y.float())
        loss.backward()
        self._optim.step()
        return loss

    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        # TODO

        if self._cuda:
            x = x.clone().detach().cuda()
            y = y.clone().detach().cuda().squeeze()
            #x = t.tensor(x, dtype=t.float).cuda()
            #y = t.tensor(y, dtype=t.float).cuda().squeeze()

        y_pred = self._model(x)
        y_pred = binarize(y_pred.cpu().detach().numpy(), threshold=0.5)  # numpy array
        y_pred = t.tensor(y_pred, dtype=t.float).cuda()

        loss = self._crit(y_pred, y.float())

        return loss, y_pred

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        # TODO

        self._model.train()

        loss = 0
        step = 0

        for i, (batch_x, batch_y) in enumerate(self._train_dl):
            step = i+1
            if self._cuda:
                batch_x = batch_x.clone().detach().cuda()
                batch_y = batch_y.clone().detach().cuda()
                #batch_x = t.tensor(batch_x, dtype=t.float).cuda()
                #batch_y = t.tensor(batch_y, dtype=t.float).cuda()

            batch_loss = self.train_step(batch_x, batch_y)
            loss += batch_loss.item()

        avg_loss = loss / step
        return avg_loss
    
    def val_test(self):
        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        # TODO

        self._model.eval()

        loss = 0
        f1 = 0
        step = 0

        with t.no_grad():
            for i, (batch_x, batch_y) in enumerate(self._val_test_dl):
                step = i+1
                if self._cuda:
                    batch_x = batch_x.clone().detach().cuda()
                    batch_y = batch_y.clone().detach().cuda()
                    #batch_x = t.tensor(batch_x, dtype=t.float).cuda()
                    #batch_y = t.tensor(batch_y, dtype=t.float).cuda()

                batch_loss, batch_pred = self.val_test_step(batch_x, batch_y)

                f1_batch = f1_score(batch_y.squeeze().cpu().detach().numpy(), batch_pred.cpu().detach().numpy(), average='micro')

                f1 += f1_batch
                loss += batch_loss.item()

        # f1_score
        avg_f1_score = f1 / step
        print('F1_score:',  avg_f1_score)

        # average loss
        avg_loss = loss / step

        return avg_loss
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        # TODO
        train_loss = []
        val_loss = []
        epoch = -1
        counter = 0
        best_loss = None

        while True:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            # TODO

            epoch += 1
            if epoch < epochs:
                print(f'Epoch: {epoch}')
                train_loss.append(self.train_epoch())
                val_loss.append(self.val_test())

                if best_loss is None:
                    best_loss = val_loss[epoch]
                    continue
                elif val_loss[epoch] < best_loss:
                    best_loss = val_loss[epoch]
                    self.save_checkpoint(epoch)
                    counter = 0
                    continue
                else:
                    counter += 1
                    print(f'Epoch:{epoch} EarlyStopping counter: {counter} out of {self._early_stopping_patience}')
                    if counter >= self._early_stopping_patience:
                        print('Early Stopping!')
                        break
                    else:
                        continue
        print(train_loss)
        print(val_loss)
        return train_loss, val_loss



