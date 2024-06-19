import numpy as np
import matplotlib.pyplot as plt

PAM = [2, 4, 8, 16]

SNRdB = np.arange(0,33)

fig, ax = plt.subplots()
col_dict = dict(zip(PAM, ['C0','C1','C2','C3']))

lr_dict = dict(zip(PAM, np.load("decision_models/learning_rates.npy").tolist()))


for M in PAM:
    # if M not in {4,8,16}: continue
    print(f"lr = {lr_dict[M]}")
    # err, Pe, _ = measure_error(model_dict[M], M)
    Pe = np.load(f"decision_models/SER/Pe_{M}.npy")
    err = np.load(f"decision_models/SER/SER_{M}.npy")
    ax.semilogy(SNRdB, err, color=col_dict[M], label=f"{M}-PAM")
    ax.semilogy(SNRdB, Pe,'--', color=col_dict[M])

ax.set_ylim([1e-4,1])
ax.set_xlim([0,32])
ax.set_xlabel("SNR [dB]")
ax.set_ylabel("SER")
ax.set_xticks(np.arange(0,33,2))
fig.suptitle("Decision NN")
fig.set_figheight(6)
fig.set_figwidth(7)
ax.grid(True, which='both')
plt.tight_layout()
ax.legend()



fig, axs = plt.subplots(2,2)
fig.suptitle("Loss for Decision NN")
fig.supxlabel('Number of epochs')
fig.supylabel('Loss')
for M, ax in zip(PAM, axs.ravel()):
    p_train, = ax.semilogy(np.load(f'decision_models/loss/train_loss_{M}.npy'), label="Training")
    p_test,  = ax.semilogy(np.load(f'decision_models/loss/test_loss_{M}.npy'), label="Validation")
    ax.set_title(f"{M}-PAM")

fig.legend((p_train, p_test), ('Training', 'Validation'), loc='upper right')
fig.set_figheight(6)
fig.set_figwidth(7.5)
plt.tight_layout()
plt.show()


