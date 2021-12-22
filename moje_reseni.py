import numpy as np
import soundfile as sf
from scipy.io import wavfile
import matplotlib.pyplot as plt
import IPython
from scipy.signal import spectrogram, lfilter, freqz, butter

# Načítanie signálu a normalizácia
sig, fs = sf.read("xsehno01.wav")

### Úloha 4.1
pocet_vzorkov = len(sig)
cas = pocet_vzorkov / fs
print("Vzorkovacia frekvencia signálu: {} [Hz]".format(fs))
print("Dĺžka signálu vo vzorkách:", pocet_vzorkov)
print("Dĺžka signálu v sekundách: {} [s]".format(cas))
print("Minimálna hodnota:", sig.min())
print("Maximálna hodnota:", sig.max())

t = np.arange(sig.size) / fs
plt.figure(figsize=(6,3))
plt.plot(t, sig)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Vstupný zvukový signál')
plt.tight_layout()

#TODO: vymazat
IPython.display.display(IPython.display.Audio(sig, rate=fs))
sig_old = sig

plt.savefig("uloha4-1.png")
plt.show()


### Úloha 4.2
pos = 0
frames = []
for i in range(pocet_vzorkov // 512):
    frame = sig[pos:pos+1024]
    frames.append(frame)
    pos += 512

t = np.arange(1024) / fs

plt.figure(figsize=(6, 3))
plt.plot(t, frames[24])
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title("Znelý rámec")
plt.savefig("uloha4-2.png")
plt.show()


### Úloha 4.3
def frame_dft(s_input):
    from_n = 50 * 512 # zaciatok segmentu
    to_n = from_n + 1024 # koniec segmentu

    s_seg = s_input[from_n:to_n]

    N = s_seg.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    s_dft = np.dot(e, s_seg)

    return s_dft

s_dft = frame_dft(sig)
fs_frame = fs // 2 / 512
time = np.arange(512)*fs_frame
plt.figure(figsize=(6, 4))
plt.plot(time, abs(s_dft[:512]))
plt.gca().set_title("DFT")
plt.gca().set_xlabel('Frekvence [Hz]')
plt.savefig("uloha4-3.png")
plt.show()

### Úloha 4.4
f, t, sgr = spectrogram(sig, fs)
# prevod na PSD
# (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
sgr_log = 10 * np.log10(sgr+1e-20)
plt.figure(figsize=(6, 3))
plt.pcolormesh(t, f, sgr_log) #sqr_log
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig("uloha4-4.png")
plt.show()

### Úloha 4.5 - Určenie rušivých frekvencií f1, f2, f3, f4
# rušivé frekvencie: 920 1840, 2760, 3680

### Úloha 4.6 - Generovanie signálu
casove_useky = [i/fs for i in range(pocet_vzorkov)]
casove_useky = np.array(casove_useky)

cos1 = np.cos(2 * np.pi * 920 * casove_useky)
cos2 = np.cos(2 * np.pi * 1840 * casove_useky)
cos3 = np.cos(2 * np.pi * 2760 * casove_useky)
cos4 = np.cos(2 * np.pi * 3680 * casove_useky)

cos_merge = cos1 + cos2 + cos3 + cos4
wavfile.write("4cos.wav", fs, cos_merge.astype(np.float32))

"""
# Kontrola vygenerovaného signálu
wavfile.write("4cos.wav", fs, cos_merge.astype(np.float32))
sig_cos, fs_cos = sf.read("xsehno01.wav")
pocet_vzorkov = len(sig_cos)
cas = pocet_vzorkov / fs_cos
print("Vzorkovacia frekvencia signálu: {} [Hz]".format(fs))
print("Dĺžka signálu vo vzorkách:", pocet_vzorkov)
print("Dĺžka signálu v sekundách: {} [s]".format(cas))
print("Minimálna hodnota:", sig.min())
print("Maximálna hodnota:", sig.max())

f, t, sgr = spectrogram(sig_cos, fs_cos)
# prevod na PSD
# (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
sgr_log = 10 * np.log10(sgr+1e-20)
plt.figure(figsize=(6, 3))
plt.pcolormesh(t, f, sgr_log) #sqr_log
plt.gca().set_title("Spektogram generovaného signálu")
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig("uloha4-6.png")
plt.show()

# Dalsi sposob zobrazenia spektogramu
plt.specgram(sig_cos, Fs=fs_cos)
cbar = plt.colorbar()
plt.tight_layout()
plt.show()
"""

### Úloha 4.7 - Čislicový filter

# 3. návrh 4 pásmových zádrží
low = 890 / (0.5 * fs)
high = 950 / (0.5 * fs)
b1, a1 = butter(4, [low, high], btype="bandstop")
z1, p1, k1 = butter(4, [low, high], btype="bandstop", output="zpk")

low = 1810 / (0.5 * fs)
high = 1870 / (0.5 * fs)
b2, a2 = butter(4, [low, high], btype="bandstop")
z2, p2, k2 = butter(4, [low, high], btype="bandstop", output="zpk")

low = 2730 / (0.5 * fs)
high = 2790 / (0.5 * fs)
b3, a3 = butter(4, [low, high], btype="bandstop")
z3, p3, k3 = butter(4, [low, high], btype="bandstop", output="zpk")

low = 3650 / (0.5 * fs)
high = 3710 / (0.5 * fs)
b4, a4 = butter(4, [low, high], btype="bandstop")
z4, p4, k4 = butter(4, [low, high], btype="bandstop", output="zpk")

# impulsni odezva
imp = [1, *np.zeros(pocet_vzorkov-1)]
h1 = lfilter(b1, a1, imp[:50])
h2 = lfilter(b2, a2, imp[:50])
h3 = lfilter(b3, a3, imp[:50])
h4 = lfilter(b4, a4, imp[:50])

fig, ax = plt.subplots(2, 2, figsize=(10, 5))

fig.gca().set_title('Impulsní odezva $h[n]$')

ax[0][0].stem(np.arange(50), h1, basefmt=' ')
ax[0][0].set_xlabel('$n$')
ax[0][0].grid(alpha=0.5, linestyle='--')

ax[1][0].stem(np.arange(50), h2, basefmt=' ')
ax[1][0].set_xlabel('$n$')
ax[1][0].grid(alpha=0.5, linestyle='--')

ax[0][1].stem(np.arange(50), h3, basefmt=' ')
ax[0][1].set_xlabel('$n$')
ax[0][1].grid(alpha=0.5, linestyle='--')

ax[1][1].stem(np.arange(50), h4, basefmt=' ')
ax[1][1].set_xlabel('$n$')
ax[1][1].grid(alpha=0.5, linestyle='--')

ax[1][1].set_title('Impulsní odezva $h[n]$')

plt.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
plt.show()

### Úloha 4.8 - Nulové body a póly

plt.figure(figsize=(4,3.5))

# jednotkova kruznice
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))

# nuly, poly
plt.scatter(np.real(z1), np.imag(z1), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p1), np.imag(p1), marker='x', color='g', label='póly')
plt.scatter(np.real(z2), np.imag(z2), marker='o', facecolors='none', edgecolors='r')
plt.scatter(np.real(p2), np.imag(p2), marker='x', color='g')
plt.scatter(np.real(z3), np.imag(z3), marker='o', facecolors='none', edgecolors='r')
plt.scatter(np.real(p3), np.imag(p3), marker='x', color='g')
plt.scatter(np.real(z4), np.imag(z4), marker='o', facecolors='none', edgecolors='r')
plt.scatter(np.real(p4), np.imag(p4), marker='x', color='g')

plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginární složka $\mathbb{I}\{$z$\}$')

plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig("uloha4-8.png")
plt.show()

### Úloha 4.9 - Frekvenčná charakteristika
freq1, h1 = freqz(b1, a1)
freq2, h2 = freqz(b2, a2)
freq3, h3 = freqz(b3, a3)
freq4, h4 = freqz(b4, a4)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(freq1 / 2 / np.pi * fs, np.abs(h1), color='blue')
ax[0].plot(freq2 / 2 / np.pi * fs, np.abs(h2), color='red')
ax[0].plot(freq3 / 2 / np.pi * fs, np.abs(h3), color='yellow')
ax[0].plot(freq4 / 2 / np.pi * fs, np.abs(h4), color='green')
ax[0].set_title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')
ax[0].set_xlabel("Frekvencia [Hz]")
ax[0].grid(alpha=0.5, linestyle='--')
ax[1].plot(freq1 / 2 / np.pi * fs, np.angle(h1), color='blue')
ax[1].plot(freq2 / 2 / np.pi * fs, np.angle(h2), color='red')
ax[1].plot(freq3 / 2 / np.pi * fs, np.angle(h3), color='yellow')
ax[1].plot(freq4 / 2 / np.pi * fs, np.angle(h4), color='green')
ax[1].set_title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')
ax[1].set_xlabel("Frekvencia [Hz]")
ax[1].grid(alpha=0.5, linestyle='--')
plt.tight_layout()
plt.savefig("uloha4-9.png")
plt.show()

### Úloha 4.10 - Filtrácia

sig = lfilter(b1, a1, sig) #filtfilt
sig = lfilter(b2, a2, sig)
sig = lfilter(b3, a3, sig)
sig = lfilter(b4, a4, sig)

wavfile.write("clean_bandstop.wav", fs, sig.astype(np.float32))

# TODO: vymazat, iba kontrola
f, t, sgr = spectrogram(sig, fs)
# prevod na PSD
# (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)
sgr_log = 10 * np.log10(sgr+1e-20)
plt.figure(figsize=(6, 3))
plt.pcolormesh(t, f, sgr_log) #sqr_log
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()

# TODO: vymazat, iba kontrola
s_dft = frame_dft(sig)
fs_frame = fs // 2 / 512
time = np.arange(512)*fs_frame
plt.figure(figsize=(6, 4))
plt.plot(time, abs(s_dft[:512]))
plt.gca().set_title("DFT")
plt.gca().set_xlabel('Frekvence [Hz]')
plt.show()





# TODO: kontrola, vymazat
t = np.arange(sig.size) / fs
plt.figure(figsize=(6,3))
plt.plot(t, sig_old)
plt.plot(t, sig)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Zvukový signál')
plt.tight_layout()
plt.show()


