import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_res = pd.read_csv('Resonance.csv')
data_field = pd.read_csv('RF_MagE_Freq.csv', delimiter = '\t')

freq_res = data_res['Freq [GHz]']
S21 = data_res['dB(St(wguide_T2,wguide_T1)) []']

freq_field = data_field['Frequency (GHz)']
max_field = data_field['Maximum MagE (V/m)']
max_phase = data_field['Maximum MagE at phase (DEG)']

plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = (16, 12)

fig, ax1 = plt.subplots(1, 1)
ax1.axvline(x=1.2460, linestyle='--', color='pink')
ax1.axvline(x=1.7840, linestyle='--', color='pink')
ax1.plot(freq_res, S21, 'r-')
ax1.set_ylabel('S21 (dB)', color='r')
ax1.set_xlabel('Frequency (GHz)')
ax1.set_title('Resonator Resonance and Center cube maximum MagE')
ax1.tick_params(axis='y', labelcolor='r')
ax1.spines['left'].set_color('red')

ax2 = ax1.twinx()
ax2.plot(freq_field, max_field, 'b-')
ax2.set_ylabel('Maximum E Field (V/m)', color='b')
ax2.tick_params(axis='y', labelcolor='b')

plt.show()