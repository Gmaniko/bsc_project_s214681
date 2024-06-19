import numpy as np
import matplotlib.pyplot as plt
import torch
from base_system import g
from neuralnet_receiver import ReceiverNN
from comm_fw import *

PAM = [2, 4, 8, 16]

SNRdB = np.arange(0,33)

fig, ax = plt.subplots()
col_dict = dict(zip(PAM, ['C0','C1','C2','C3']))

lr_dict = dict(zip(PAM, np.load("receiver_models/learning_rates.npy").tolist()))


for M in PAM:
    # if M not in {4,8,16}: continue
    print(f"lr = {lr_dict[M]}")
    # err, Pe, _ = measure_error(model_dict[M], M)
    Pe = np.load(f"receiver_models/SER/Pe_{M}.npy")
    err = np.load(f"receiver_models/SER/SER_{M}.npy")
    ax.semilogy(SNRdB, err, color=col_dict[M], label=f"{M}-PAM")
    ax.semilogy(SNRdB, Pe,'--', color=col_dict[M])

ax.set_ylim([1e-4,1])
ax.set_xlim([0,32])
ax.set_xlabel("SNR [dB]")
ax.set_ylabel("SER")
ax.set_xticks(np.arange(0,33,2))
ax.legend()
ax.grid(True, which='both')
fig.suptitle("Receiver NN")
fig.set_figheight(6)
fig.set_figwidth(7)
plt.tight_layout()



fig, axs = plt.subplots(2,2)
fig.suptitle("Loss for Receiver NN")
fig.supxlabel('Number of epochs')
fig.supylabel('Loss')
for M, ax in zip(PAM, axs.ravel()):
    p_train, = ax.semilogy(np.load(f'receiver_models/loss/train_loss_{M}.npy'), label="Training")
    p_test,  = ax.semilogy(np.load(f'receiver_models/loss/test_loss_{M}.npy'), label="Validation")
    ax.set_title(f"{M}-PAM")

fig.legend((p_train, p_test), ('Training', 'Validation'), loc='upper right')
fig.set_figheight(6)
fig.set_figwidth(7.5)
plt.tight_layout()



fig, ax = plt.subplots()
for M in PAM:
    model = torch.load(f"receiver_models/model_{M}pam.pth")
    print(model.conv_layer[0].bias.numpy(force=True).flatten())
    fir = np.flip(model.conv_layer[0].weight.numpy(force=True).flatten())
    ax.plot(fir, label=f"{M}-PAM")

ax.plot(g, label="RRC", color="black")
ax.legend()
fig.set_figheight(5)
fig.set_figwidth(7)
plt.tight_layout()

fig, ax = plt.subplots()
for M in PAM:
    model = torch.load(f"receiver_models/model_{M}pam.pth")
    fir = np.flip(model.conv_layer[0].weight.numpy(force=True).flatten())
    impulse_response = np.convolve(g,fir)
    ax.plot(impulse_response, label=f"{M}-PAM")
    if M==2:
        for i in range(-4,5):
            ax.axvline(np.argmax(np.abs(impulse_response))+i*8, linestyle='--', color='red')

ax.plot(np.convolve(g,g), label="RC", color="black")
ax.legend()
ax.set_xlim(20,100)
fig.set_figheight(5)
fig.set_figwidth(7)
plt.tight_layout()



plt.show()


