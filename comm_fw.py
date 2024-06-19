# Nikolai Stephansen Hansen
# Communication system framework

import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import erfc
# from scipy.signal import bessel, lfilter, freqz
# from scipy.fft import fft, fftshift

# Symbol values for average symbol energy of Es = 1
def alphabet_norm(M):
    d = np.sqrt(12/(M**2-1))
    pam_symb = np.linspace(-d*(M-1)/2, d*(M-1)/2, M)
    return pam_symb

# Encodes symbol values to indices suitable for use in PyTorch
def symbol2label(s, M):
    a = alphabet_norm(M)
    a2l = np.arange(M)
    idx = [np.where(s==symbol) for symbol in a]
    labels = np.zeros(len(s))
    for i, label in enumerate(idx):
        labels[label] = a2l[i]
    return labels

def pam_symb_gen(M, L):
    "Randomly generates L M-PAM symbols"
    pam_symb = np.arange(-(M-1),M,2)
    s = np.random.choice(pam_symb, int(L))
    return s

# Generates L symbols where the average symbol energy is Es=1
def pam_symb_gen_norm(M, L):
    # pam_symb = normalize(np.arange(-(M-1),M,2))
    pam_symb = alphabet_norm(M)
    s = np.random.choice(pam_symb, int(L))
    return s

def qam_symb_gen(M, L):
    a = pam_symb_gen(int(np.sqrt(M)), L)
    b = pam_symb_gen(int(np.sqrt(M)), L)
    s = a+1j*b
    return s

# Upsampling by zero-padding sps-1 zeroes between each symbol
def upsample(x, sps):
    "Creates a sequence comprising the original samples separated by L-1 zeros."
    x_up = np.zeros(sps*len(x))
    x_up[::sps] = x
    return x_up

# Sample incoming signal. might be unecessary
def downsample(x, sps, start_offset, end_offset):
    return x[start_offset:-end_offset:sps]


# generates a time vector for the RRC filter and plotting
def time_vector(sps, T, n_T):
    "Generates a time vector in units [s] from (-n_T)T to (n_T)T given samples per symbol (sps) and symbol period (T)."
    return np.linspace(-n_T*T,n_T*T,int(sps*2*n_T+1))


def rrc_filter(t, T, roll_off=0.3):
    "Impulse response of root-raised-cosine filter"
    return (np.cos((1+roll_off)*np.pi*t/T) + np.pi*(1-roll_off)/(4*roll_off)*np.sinc((1-roll_off)*t/T)) / (1-(4*roll_off*t/T)**2)

# RRC filter normalized to unit energy
def rrc_filter_v2(t_vec, T, roll_off=0.3):
    h = np.zeros_like(t_vec)
    for i, t in enumerate(t_vec):
        if t == 0.0:
            h[i] = (1+roll_off*(4/np.pi-1))
        elif t == np.abs(T/(4*roll_off)):
            h[i] = roll_off/(np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*roll_off)) + (1-2/np.pi)*np.cos(np.pi/(4*roll_off)))
        else:
            h[i] = (np.sin(np.pi*t/T*(1-roll_off)) + 4*roll_off*t/T*np.cos(np.pi*t/T*(1+roll_off))) / (np.pi*t/T*(1-(4*roll_off*t/T)**2))

    return h/np.linalg.norm(h)




def gauss_filter(sd, t):
    "Impulse response of Gaussian filter"
    return 1/np.sqrt(2*np.pi*sd**2) * np.exp(-0.5/sd**2 * t**2)

def sinc_filter(t, T):
    return np.sinc(t/T)


# For use in carrier modulation
def carrier(fc, l, Ts):
    "Carrier wave. 'samples' is the number of samples, or length of signal."
    tc = np.linspace(-l*Ts/2,l*Ts/2, l)
    return np.sqrt(2)*np.cos(2*np.pi*fc*tc), np.sqrt(2)*np.sin(2*np.pi*fc*tc)

def modulator(s, g, fc, step):
    x = np.convolve(s.real,g)
    y = np.convolve(s.imag,g)
    
    c = carrier(fc, len(x), step)
    v = x * c[0] + y * c[1]
    return v


# Average Power. might be unecessary
def Pavg(v):
    return np.mean(np.abs(v)**2)

def normalize(x):
    return x/np.sqrt(Pavg(x))


# AWGN channel
def awgn_channel(v: np.ndarray, SNRdB):
    "AWGN channel given average energy of signal is normalized"
    noise_power = 10**(-SNRdB/10)
    sd = np.sqrt(noise_power/2)
    noise = np.random.normal(0, sd, v.shape)
    v_n = v + noise
    return v_n



def demodulator(v, g, fc, step):
    c = carrier(fc, len(v), step)
    I = np.convolve(v * c[0], g)
    Q = np.convolve(v * c[1], g)
    s = I+1j*Q
    return s

# Find synchronization point for distortionless channel
def sync_fun(g):
    x = np.convolve(g,g)
    sync = np.argmax(x)
    # E = np.sum(np.abs(g)**2)
    return sync


def iq_plot(sig, M, mode='qam'):
    if mode == 'qam': M = int(np.sqrt(M))
    _, ax = plt.subplots()
    ax.plot(sig.real, sig.imag, 'r.', alpha=0.2)
    ax.grid()
    ax.axhline(0, color='black', linewidth=.5)
    ax.axvline(0, color='black', linewidth=.5)
    ax.set_xlabel('In-phase')
    ax.set_ylabel('Quadrature')
    ticks = np.arange(-(M-1),M,2)
    labels = []
    for tick in ticks:
        s = '$'
        if np.abs(tick) == 1:
            s += '-' if tick < 0 else ''
        else:
            s += str(tick)
        labels.append(s + 'E$')    
    # ax.set_xticks(normalize(ticks))
    # ax.set_xticklabels(labels)

    # ax.set_yticks(normalize(ticks))
    # ax.set_yticklabels(labels)

        
    return


def eyediagram(r, sync, offset, M, sps, mode='qam'):
    """ 
    r : Real input signal (either in-phase or quadrature component)
    g : Pulse-shaping filter impulse response
    M : Number of distinct symbols (M-QAM or M-PAM)
    sps : Samples per symbol
    n_T : Number of periods before and after 0.
    """
    if mode == 'qam': M = int(np.sqrt(M))
    t = np.linspace(-sps,sps, 2*sps+1)
    # t = time_vector(sps, 1, n_T)
    sig = downsample(r, 1, sync +offset, sync-offset)
    _, ax = plt.subplots()

    ax.plot(t, sig[0:2*sps+1], color='yellow', alpha=0.9)
    for i in range(1, np.floor_divide(len(sig)-(2*sps+1), 2*sps)):
        ax.plot(t, sig[i*2*sps:(i+1)*2*sps+1], color='yellow', alpha=0.1)
    
    ax.set_facecolor('black')
    ticks = np.arange(-(M-1),M,2)
    labels = []
    for tick in ticks:
        s = '$'
        if np.abs(tick) == 1:
            s += '-' if tick < 0 else ''
        else:
            s += str(tick)
        labels.append(s + 'E$')   
    ax.set_yticks(normalize(ticks))
    ax.set_yticklabels(labels)
    return


def selection(x, M, mode = 'qam'):
    if mode == 'qam': M = int(np.sqrt(M))
    # sync, E = sync_fun(g)
    # x = x[sync:-sync:m]
    x = x*(0.027989830917147428*M)
    symbols = np.arange(-(M-1),M,2)
    cond = []
    for i in range(len(symbols)-1):
        cond.append(x < (symbols[i] + 1))
    cond.append(x > (symbols[-2]+1))

    return np.select(cond,symbols,0)

def decision(x, M, mode = 'qam'):
    if mode == 'qam': M = int(np.sqrt(M))
    # symbols = np.arange(-(M-1),M,2)
    pam_symb = alphabet_norm(M)
    idx = np.argmin(np.subtract.outer(x, pam_symb)**2, axis=1)

    return pam_symb[idx]

def symbol_err_prob(M, SNRdB):
    "Calculates the symbol error probability for M-PAM given SNR [dB]."
    SNR = 10**(SNRdB/10)
    Pe = 2*(M-1)/M * Qfunc(np.sqrt(6/(M**2-1)*SNR))

    return Pe

def symbol_err_prob_upper(M, SNRdB):
    SNR = 10**(SNRdB/10)
    Pe = 2*(1-1/M**2)*Qfunc(np.sqrt((np.pi/4)**2 *6/(M**2-1)*SNR ))
    return Pe



def Qfunc(x):
    "Q-function"
    return 0.5*erfc(x/np.sqrt(2))



# def plot_fft_ab_response(fb: np.ndarray, fa: np.ndarray,
#                          ax: plt.Axes, Ts: float, plot_label=None, n_fft=2048):
#     """
#         Plots the filter response of a digital filter, given its transfer function coefficients (fb and fa)
#     """
#     eps = 1e-16

#     # Do freqz of the filter and redefine fq axis (double sided)
#     fqs_freqz, H = freqz(fb, fa, fs=1/Ts, whole=True, worN=n_fft)
#     fqs_freqz = np.arange(-len(fqs_freqz)//2, len(fqs_freqz)//2) / (len(fqs_freqz) * Ts)

#     # Plot ampltidue response in dB domain
#     ax.plot(fqs_freqz, 20.0 * np.log10(fftshift(np.absolute(H)) + eps), label=plot_label)
#     return


# def plot_fft(input_signal: np.ndarray, ax: plt.Axes, Ts: float,
#              plot_label=None):
#     """
#         Plots the spectrum of a signal
#     """
#     eps = 1e-16

#     N_fft = len(input_signal)
#     H = fft(input_signal) * Ts
#     fqs = np.arange(-N_fft//2, N_fft//2) / (N_fft * Ts)
#     Hshifted = fftshift(H)

#     ax.plot(fqs, 20.0 * np.log10(np.absolute(Hshifted) + eps), label=plot_label)
#     return