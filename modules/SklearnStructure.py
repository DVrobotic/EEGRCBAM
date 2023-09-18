import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.tensorboard import SummaryWriter

from modules.utils import setup_tensors, get_minibatches, setup_training_tensors


class SklearnStructure:
    def __init__(self):
        pass

    def predict_proba(self, X, device='cuda'):
        self.model.eval()

        X_tensor, _ = setup_tensors(X, device=device)

        pred, loss = self.model(X_tensor, None)

        self.model.train()

        return F.softmax(pred, dim=-1).detach().cpu().numpy()

    def predict(self, X, device='cuda'):
        return np.argmax(self.predict_proba(self, X, device=device), axis=-1)

    def get_model_size():
        print('Missing Function')
        return 0

    def train_model(self, X, y, batchsize=32, lr='auto', iterations=2000, device='cuda', validation=0, track=None, patience=500, verbose=False):
        # validation can be 0, float(0-1), (X, y)
        self.model.to(device=device)

        if lr == 'auto':
            lr = 0.001 if validation != 0 else 0.001

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        X_tensor, y_tensor, X_val_tensor, y_val_tensor = setup_training_tensors(validation, X, y, device)
        X_track, y_track = track
        X_track_tensor, y_track_tensor = setup_tensors(X_track, y_track)

        if validation != 0:
            best_loss = np.inf
            best_fit = self.model.state_dict()
            optimizer_data = optimizer.state_dict()

        old_lr = lr
        for it in range(iterations):
            X_temp, y_temp = get_minibatches(X_tensor, y_tensor, batchsize)
            pred, loss = self.model(X_temp, y_temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if validation != 0:
                with torch.no_grad():
                    self.model.eval()
                    pred_val, loss_val = self.model(X_val_tensor, y_val_tensor)
                    self.model.train()

                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_fit = self.model.state_dict()
                        stopped_iterations = 0
                    else:
                        stopped_iterations += 1

            if track is not None:
                with torch.no_grad():
                    self.model.eval()
                    pred_track, loss_track = self.model(X_track_tensor, y_track_tensor)
                    self.model.train()

                    acc_track = accuracy_score(y_track_tensor.detach().cpu().numpy(), pred_track.argmax(1).detach().cpu().numpy())
                    print('BATCH loss: %.4f, Track loss: %.4f, Track acc: %.4f, iter: %d                         ' % (loss, loss_track, acc_track, it), end='\r')
            else:
                print('loss: %.4f, iter: %d' % (loss, it), end='\r')

        if validation != 0:
            self.model.load_state_dict(best_fit)

        print("final result: \n")
        self.model.eval()
        pred_track, loss_track = self.model(X_track_tensor, y_track_tensor)
        self.model.train()
        acc_track = accuracy_score(y_track_tensor.detach().cpu().numpy(), pred_track.argmax(1).detach().cpu().numpy())
        print(f"model final result: loss: {loss_track}, acc: {acc_track}\n")
        return acc_track


