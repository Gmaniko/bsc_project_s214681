import numpy as np
from comm_fw import *
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy import signal


L = int(1e5)

sps=8 # samples per symbol / oversampling factor
symbol_rate = 100e9 # symbol rate: 100 GBd
T = 1/symbol_rate # symbol period
Ts = T/sps # sample period

roll_off = 0.3 # Roll-off factor for RC filter

B = (1+roll_off)*symbol_rate/2 #Bandwidth
t = time_vector(sps, T, n_T=4)
g = rrc_filter_v2(t, T, roll_off=roll_off) # RRC filter
sync = sync_fun(g)
bessel_sos = signal.bessel(5, 0.75*B, output='sos', norm='mag', fs=1/Ts)



def forward_symbols(symbols, SNRdB, use_bessel=False):
    symbols_up = upsample(symbols, sps)
    x = np.convolve(symbols_up, g)
    x = signal.sosfilt(bessel_sos, x) if use_bessel else x
    r = awgn_channel(x, SNRdB)
    r = signal.sosfilt(bessel_sos, r) if use_bessel else r
    return r


def train_loop(train_symb, train_label, batch_size, SNRdB, model, loss_fn, optimizer):
    # Unnecessary in this situation but added for best practices
    model.train()

    symbol_batches = torch.tensor(train_symb).split(batch_size)
    label_batches = torch.tensor(train_label, dtype=torch.long, device=model.device).split(batch_size)
    correct = 0
    train_loss = 0
    for X,y in zip(symbol_batches, label_batches):
        r = forward_symbols(X, SNRdB, use_bessel=model.use_bessel)
        pred = model(r)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        correct += (pred.argmax(1) == y).type(torch.long).sum().item()
        train_loss += loss.item()
    train_loss = train_loss/len(symbol_batches)
    print(f"Training Error: \n Accuracy: {(100*correct/train_symb.size):0.1f}, Avg loss: {train_loss:>8f}")
    return train_loss


def test_loop(test_symb, test_label, batch_size, SNRdB, model, loss_fn):
    # Unnecessary in this situation but added for best practices
    model.eval()
    test_loss, correct = 0, 0
    symbol_batches = torch.tensor(test_symb).split(batch_size)
    label_batches = torch.tensor(test_label, dtype=torch.long, device=model.device).split(batch_size)
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for X,y in zip(symbol_batches, label_batches):
            r = forward_symbols(X, SNRdB, use_bessel=model.use_bessel)
            pred = model(r)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.long).sum().item()

    test_loss = test_loss/len(symbol_batches)
    print(f"Test Error: \n Accuracy: {(100*correct/test_symb.size):0.1f}, Avg loss: {test_loss:>8f} \n")
    return test_loss



def train_decision_nn(model, M, n_symbols=10**6, split=0.9, SNRdB=30, eta=0.05, batch_size=32, epochs=5, earlystop=True):
    symbols = pam_symb_gen_norm(M, n_symbols)
    train_symb, test_symb = np.split(symbols, [int(split*n_symbols)])
    labels = symbol2label(symbols, M)
    train_label, test_label = np.split(labels, [int(split*n_symbols)])

    loss_fn = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=0.9)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=eta)
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss[t] = train_loop(train_symb, train_label, batch_size, SNRdB, model, loss_fn, optimizer)
        if split != 1:
            test_loss[t] = test_loop(test_symb, test_label, batch_size, SNRdB, model, loss_fn)
            scheduler.step(test_loss[t])
            print(f"lr= {scheduler.get_last_lr()}")
            if earlystop and (scheduler.get_last_lr()[0] < eta): break
        

    return model, train_loss, test_loss


def measure_error(model, M, max_snr=33):
    SNRdB = np.arange(0,33)
    Pe = symbol_err_prob(M, SNRdB)
    err = np.zeros(SNRdB.shape)
    for _ in tqdm(range(10)):
        symbols = pam_symb_gen_norm(M, L)
        symbols_up = upsample(symbols, sps)
        x = np.convolve(symbols_up, g)
        x = signal.sosfilt(bessel_sos, x) if model.use_bessel else x
        for i, snr_db in enumerate(SNRdB):
            r = awgn_channel(x, snr_db)
            r = signal.sosfilt(bessel_sos, r) if model.use_bessel else r
            a_gen = model.decision_nn(r)
            err[i] += np.mean(symbol2label(symbols, M) != a_gen)
    err = err/10
    return err, Pe, np.mean(err/Pe, where=(SNRdB<=max_snr))

