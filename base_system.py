import numpy as np 
from comm_fw import *
from scipy import signal
import matplotlib.pyplot as plt


M = 4 # PAM order (number of distinct symbols)
L = 1e5 # number of total symbols 
# symbols = pam_symb_gen(M,L)
symbols = pam_symb_gen_norm(M, L)

sps=8 # samples per symbol / oversampling factor
symbol_rate = 100e9 # symbol rate: 100 GBd
T = 1/symbol_rate # symbol period
Ts = T/sps # sample period
roll_off = 0.3
t = time_vector(sps, T, n_T=4)
g = rrc_filter_v2(t, T, roll_off=roll_off)
B = (1+roll_off)*symbol_rate/2
sync = sync_fun(g)

bessel_sos = signal.bessel(5, 0.75*B, output='sos', norm='mag', fs=1/Ts)

if __name__=="__main__":
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(t,g)
    ax[0].set_title("RRC impulse response")
    ax[0].set_xticks(np.arange(-4,5)*T, [r"$-4T$",r"$-3T$",r"$-2T$",r"$-T$",r"$0$",
                           r"$T$",r"$2T$",r"$3T$",r"$4T$"])
    ax[0].spines['bottom'].set_position('zero')
    ax[0].spines['left'].set_position('zero')
    ax[0].spines['right'].set_color('none')
    ax[0].spines['top'].set_color('none')
    ax[0].yaxis.tick_left()
    rc_ir = np.convolve(g,g, mode='same')
    ax[1].plot(t,rc_ir)
    ax[1].set_title("RC impulse response")
    ax[1].set_xticks(np.arange(-4,5)*T, [r"$-4T$",r"$-3T$",r"$-2T$",r"$-T$",r"$0$",
                           r"$T$",r"$2T$",r"$3T$",r"$4T$"])
    ax[1].spines['bottom'].set_position('zero')
    ax[1].spines['left'].set_position('zero')
    ax[1].spines['right'].set_color('none')
    ax[1].spines['top'].set_color('none')
    ax[1].yaxis.tick_left()
    fig.set_figheight(6)
    fig.set_figwidth(7)
    plt.tight_layout()
    
    unit_impulse = np.zeros(8*sps+1)
    unit_impulse[unit_impulse.size//2] = 1

    fig, ax = plt.subplots()
    f_bessel, h_bessel = signal.sosfreqz(bessel_sos, worN=2048, fs=1/Ts)
    f_bessel = np.arange(len(f_bessel)) / (2*len(f_bessel) * Ts)
    ax.plot(f_bessel, 20*np.log10(np.abs(h_bessel)))
    ax.grid(True, which='both')
    ax.axvline(0.75*B, ymax=0.87, color='red', linestyle='--')
    ax.axhline(-3, xmax=0.25, color='red', linestyle='--', label=r"$f_{-3dB}=0.75B$")
    ax.set_xticks([0, 0.75*B, 1e11, 1.5e11], [r"$0$",r"$0.75B$",r"$100$",r"$150$"])
    ax.set_yticks([0,-3,-10,-15,-20,-25,-30])
    ax.set_xlabel(r"Frequency [GHz]")
    ax.set_ylabel(r"Attenuation [dB]")
    ax.set_ylim(-30, 1)
    ax.set_xlim(0,3*B)
    ax.set_title("Bessel Filter")
    ax.legend()
    fig.set_figheight(5)
    fig.set_figwidth(6)
    plt.tight_layout()


    fig, ax = plt.subplots()
    fig.suptitle("System response")
    system_response = np.convolve(g,unit_impulse)
    system_response = signal.sosfilt(bessel_sos, system_response)
    system_response = signal.sosfilt(bessel_sos, system_response)
    system_response = np.convolve(g, system_response, 'valid')
    ax.plot(t, system_response, label="with Bessel")
    ax.plot(t, rc_ir, label="without Bessel")
    ax.set_xticks(np.arange(-4,5)*T, [r"$-4T$",r"$-3T$",r"$-2T$",r"$-T$",r"$0$",
                                      r"$T$",r"$2T$",r"$3T$",r"$4T$"])
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.tick_left()
    ax.legend()
    fig.set_figheight(5)
    fig.set_figwidth(7)
    plt.tight_layout()

    # plt.show()
    SNRdB = np.arange(0,33)
    PAM  = [2,4,8,16]
    fig, ax = plt.subplots()
    colors = ['C0','C1','C2','C3']
    for k,M in enumerate(PAM):
        Pe = symbol_err_prob(M, SNRdB)
        # ax.semilogy(SNRdB, Pe, linestyle='--', color=colors[k], label=f"{M}-PAM")
        ax.semilogy(SNRdB, symbol_err_prob_upper(M, SNRdB), linestyle='--', color=colors[k], label=f"{M}-PAM")

    ax.set_ylim([1e-4,1])
    ax.set_xlim([0,32])
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Probability of symbol error")
    ax.set_xticks(np.arange(0,33,2))
    ax.legend()
    fig.set_figheight(5)
    fig.set_figwidth(6)
    plt.tight_layout()
    plt.show()