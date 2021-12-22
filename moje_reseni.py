import numpy as np
import soundfile as sf
from scipy.io import wavfile
import matplotlib.pyplot as plt
import IPython
from scipy.signal import spectrogram, filtfilt, lfilter, freqz, tf2zpk, butter

# Načítanie signálu a normalizácia
#fs, sig = wavfile.read("xsehno01.wav")
sig, fs = sf.read("xsehno01.wav")
#sig = sig / (2 ** 15)

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
plt.gca().set_title('Zvukový signál')
plt.tight_layout()
IPython.display.display(IPython.display.Audio(sig, rate=fs))

# TODO: odkomentovat savefig, zakomentovat show()
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


# TODO: vymazat (iba kontrola)
"""
from_n = 50 * 512 # zaciatok segmentu
to_n = from_n + 1024 # koniec segmentu
s_seg = sig[from_n:to_n]

dftArray = np.fft.fft(s_seg)
time = np.arange(s_seg[:512].size)/fs
plt.figure(figsize=(6, 3))
plt.plot(time, abs(dftArray[:512]))
plt.gca().set_title("DFT")
plt.show()
"""

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
# rušivé frekvencie: 900, 1800, 2700, 3600

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

### Úloha 4.7 - Čistiaci filter

# 3. návrh 4 pásmových zádrží
low = 870 / (0.5 * fs)
high = 970 / (0.5 * fs)
b1, a1 = butter(4, [low, high], btype="bandstop")
z1, p1, k1 = butter(4, [low, high], btype="bandstop", output="zpk")

low = 1790 / (0.5 * fs)
high = 1890 / (0.5 * fs)
b2, a2 = butter(4, [low, high], btype="bandstop")
z2, p2, k2 = butter(4, [low, high], btype="bandstop", output="zpk")


low = 2710 / (0.5 * fs)
high = 2810 / (0.5 * fs)
b3, a3 = butter(4, [low, high], btype="bandstop")
z3, p3, k3 = butter(4, [low, high], btype="bandstop", output="zpk")


low = 3630 / (0.5 * fs)
high = 3730 / (0.5 * fs)
b4, a4 = butter(4, [low, high], btype="bandstop")
z4, p4, k4 = butter(4, [low, high], btype="bandstop", output="zpk")

sig = lfilter(b1, a1, sig) #filtfilt
sig = lfilter(b2, a2, sig)
sig = lfilter(b3, a3, sig)
sig = lfilter(b4, a4, sig)

# Koeficienty filtru
print(a1, b1)
print(a2, b2)
print(a3, b3)
print(a4, b4)

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


# Frequency response
freq, h = freqz(b1, a1, fs=fs)
# Plot
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
ax[0].set_title("Frequency Response")
ax[0].set_ylabel("Amplitude (dB)", color='blue')
ax[0].set_xlim([0, 8000])
ax[0].set_ylim([-100, 10])
ax[0].grid()
ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
ax[1].set_ylabel("Angle (degrees)", color='green')
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_xlim([0, 8000])
#ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[1].set_ylim([-90, 90])
ax[1].grid()
plt.show()


### Úloha 4.8 - Nulové body a póly

plt.figure(figsize=(4,3.5))

# jednotkova kruznice
ang = np.linspace(0, 2*np.pi,100)
plt.plot(np.cos(ang), np.sin(ang))

# nuly, poly
plt.scatter(np.real(z2), np.imag(z2), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p2), np.imag(p2), marker='x', color='g', label='póly')
# TODO: nejako vela ich tam je
print(z2)
print(p2)

plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginární složka $\mathbb{I}\{$z$\}$')

plt.grid(alpha=0.5, linestyle='--')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
### Úloha 4.9 - Frekvenčná charakteristika
### Úloha 4.10 - Filtrácia