import numpy as np
from comm_fw import *
import torch
import torch.nn as nn
import optuna
from neuralnet_framework import measure_error, train_decision_nn

L = 10**5
sps=8 # samples per symbol / oversampling factor
symbol_rate = 100e9 # symbol rate: 100 GBd
T = 1/symbol_rate # symbol period
Ts = T/sps # sample period

roll_off = 0.3

B = (1+roll_off)*symbol_rate/2
t = time_vector(sps, T, n_T=4)
g = rrc_filter_v2(t, T, roll_off=roll_off)
sync = sync_fun(g)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)



# Defined per model/file
class DecisionNN(nn.Module):
    def __init__(self, hidden_nodes, classes, activation=nn.Tanh()):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, hidden_nodes),
            activation,
            nn.Linear(hidden_nodes, classes)
        )
        self.device = device
        self.use_bessel = False
        self.double()
    def forward(self, x):
        x = np.convolve(x, g, mode='valid')
        x = x[::sps]
        x = torch.tensor(x, device=device).reshape(-1,1)
        x = self.linear_stack(x)
        return x

    def decision_nn(self, x):
        x = self.forward(x).argmax(dim=1)
        x = torch.Tensor.numpy(x.cpu())
        return x



# Optuna objective function for DecisionNN model
def obj_DecisionNN(trial, M, train_snr, max_snr, epochs, batch_size=2**12):
    # batch_size = 2**trial.suggest_int("batch_size", 9, 12)
    lr = trial.suggest_float("learning_rate", 1e-4,1e-1, log=True)
    # hidden_nodes = trial.suggest_int("hidden_nodes", 1,10)
    # train_snr = trial.suggest_int("SNR", 22, 32, step=2)
    print(trial.params)
    model = DecisionNN(hidden_nodes=M, classes=M).to(device)
    model = train_decision_nn(model, M, 
                              n_symbols=2**20, 
                              split=0.9, 
                              SNRdB=train_snr, 
                              batch_size=batch_size, 
                              eta=lr, epochs=epochs, earlystop=False)[0]
    return measure_error(model, M, max_snr=max_snr)[2]


PAM  = [2,4,8,16]
SNR_dict = dict(zip(PAM, [6.79, 14.11, 20.46, 26.58]))
max_snr_dict = dict(zip(PAM, [9, 16, 23, 29]))
bs_dict = dict(zip(PAM, [2**9, 2**10, 2**11, 2**12]))
lr_dict = dict.fromkeys(PAM)

for M in PAM:
    study = optuna.create_study()
    study.optimize(lambda trial: obj_DecisionNN(trial, M=M, 
                                                train_snr=SNR_dict[M], 
                                                max_snr=max_snr_dict[M], 
                                                epochs=5*M, 
                                                batch_size=bs_dict[M]), 
                                                n_trials=15)
    lr_dict[M] = study.best_params['learning_rate']

np.save("decision_models/learning_rates.npy", np.fromiter(lr_dict.values(), dtype=float))

model_dict = dict.fromkeys(PAM)
train_loss = dict.fromkeys(PAM)
test_loss = dict.fromkeys(PAM)

for M in PAM:
    # if M not in {4, 8,16}: continue
    print(f"M = {M}")
    model = DecisionNN(hidden_nodes=M, classes=M).to(device)
    model_dict[M], train_loss[M], test_loss[M] = train_decision_nn(model, M, 
                                                                   n_symbols=2**20, 
                                                                   split=0.9, 
                                                                   SNRdB=SNR_dict[M], 
                                                                   batch_size=bs_dict[M], 
                                                                   eta=lr_dict[M], 
                                                                   epochs=10*M, 
                                                                   earlystop=False)
    np.save(f'decision_models/loss/train_loss_{M}.npy', train_loss[M])
    np.save(f'decision_models/loss/test_loss_{M}.npy', test_loss[M])
    torch.save(model_dict[M], f"decision_models/model_{M}pam.pth")


for M in PAM:
    # if M not in {4,8,16}: continue
    print(f"M = {M}")
    print("Calculating SER...")
    print(f"lr = {lr_dict[M]}")
    err, Pe, _ = measure_error(model_dict[M], M)
    np.save(f"decision_models/SER/Pe_{M}.npy", Pe)
    np.save(f"decision_models/SER/SER_{M}.npy", err)


if __name__ == "__main__":
    import plot_decision