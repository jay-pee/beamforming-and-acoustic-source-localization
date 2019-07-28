#%% [markdown]
# ## Basic of Beamforming and Source Localization with Steered response Power
# ![](2019-07-28-14-30-39.png)
# Image shows a filter and sum beamformer.	Microphone signals $\underline{X}(\Omega)$ are multiplied with the beamformer weights $\underline{W}(\Omega)$ and then accumulated to the beamformer output signal $Y(\Omega)$.
# Model Description
# $$\underline{X}(\Omega) = \underline{A}^{\text{ff}}(\Omega) \cdot S(\Omega)$$
# Beamforming
# $$Y(\Omega) = \underline{W}^\text{H}(\Omega) \cdot \underline{X}(\Omega)$$

#%% 
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(precision=3)
def H(A, **kwargs):
    return np.transpose(A,**kwargs).conj()

#%%
# Parameter
varphi = 45 / 180 * np.pi # Angle of attack of the Source S(\Omega) in relation to the mic array 
c = 343000 # Velocity of sound in mm/s
mic = 6 # count of mics
d = 20 # distance in mm
fs = 16000 # Sample rate

n_fft = 512 # Fourier Transform length
n_spec = 257 # Number of frequency bins 
n_dir = 72 # Number of directions which the steering vector is steered to

#%%
# Calculate the delay vectors to each microphone to the source in the frequency domain.

# pos_y and pos_x are the microphone positions. It is a Uniform Linear Array (ULA) type (like seen in the Figure below)
pos_y = np.zeros((1,mic))
pos_x = np.r_[0.:mic]*d
tau = (pos_x*np.cos(varphi)+pos_y*np.sin(varphi))/c #calculating delay vector tau (in the time domain) depending on the array geometry.
tau = tau.reshape([mic,1,1])
Omega_array = np.r_[0.:n_spec].T*np.pi/n_fft*2
Omega_array = Omega_array.reshape([1,1,n_spec])
A_ff = np.exp(-1j*Omega_array*fs*tau)

#%%
# Plot Microphone Positions
fig, ax = plt.subplots()
ax.scatter(pos_x, pos_y, c='tab:red', alpha=1, edgecolors='white')
plt.ylabel('Y Position [mm]')
plt.xlabel('X Position [mm]')
plt.ylim((-50, 50))

#%%
# Calculate the steering vectors W_H for the beamforming/localization
angle_array = np.c_[0:360:5]/180*np.pi
tau_steering = (pos_x*np.cos(angle_array)+pos_y*np.sin(angle_array))/c
tau_steering = tau_steering.T.copy()
tau_steering = tau_steering.reshape([mic,1,1,n_dir])
W = np.exp(-1j*Omega_array.reshape([1,1,n_spec,1])*fs*tau_steering)
W.shape
W_H = W.reshape([1,mic,n_spec,n_dir]).conj()
W_H.shape

#%%
tmp = np.squeeze(np.round(np.angle(A_ff[:,:,:])/np.pi*180))
plt.plot(tmp.T)
plt.ylabel("Angle [Deg]")
plt.xlabel("Frequency [Bin]")

#%%
A_ff_H = A_ff.reshape([1,mic,n_spec]).copy()
A_ff_H = A_ff_H.conj()
phi_xx = A_ff_H * A_ff

#%%
df = pd.DataFrame(phi_xx[:,:,50])
df.style.format('{:,.2f}'.format)

#%%
power_steered = np.zeros((n_spec,n_dir))
for iDir in range(n_dir):
    for iF in range(n_spec):
        tmp = np.dot(W_H[:,:,iF,iDir], phi_xx[:,:,iF])
        power_steered[iF,iDir] = np.abs(np.dot(tmp, W[:,:,iF,iDir]))

#%%
plt.figure(1)
plt.imshow(power_steered, aspect='auto', origin='lower')
plt.show()

# with pd.option_context('display.precision', 3):
# pd.set_option('precision', 0)
# pd.set_option('display.float_format', lambda x: '%.0f' % x)
# df = pd.DataFrame(power_steered)
# df.style.format('{:,.2f}'.format)
# print(df)


