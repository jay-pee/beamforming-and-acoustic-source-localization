#%% [markdown]
# # Basic of Beamforming and Source Localization with Steered response Power
# ## Motivation
# Beamforming is a technique to spatially filter out desired signal and surpress noise. This is applied in many different domains, like for example radar, mobile radio, hearing aids, speech enabled IoT devices.
# 
# ## Signal Model
# ![](2019-07-28-19-25-21.png)
# Model Description:
# $$\underline{X}(\Omega) = \underline{A}^{\text{ff}}(\Omega) \cdot S(\Omega)$$
# ## Beamforming 
# Beamforming or spatial filtering is an array processing technique used to improve the quality of the desired signal in the presence of noise. This filtering is accomplished by a linear combination of the recorded signals $X_m(\Omega)$ and the beamformer weights $W_m(\Omega)$. In other words, the filtered microphone signals are summed together (compare with figure below). When the filter weights are configured correctly, the desired signal is superimposed constructively.
# ![](2019-07-28-14-30-39.png)
# Image shows a filter and sum beamformer.	Microphone signals $\underline{X}(\Omega)$ are multiplied with the beamformer weights $\underline{W}(\Omega)$ and then accumulated to the beamformer output signal $Y(\Omega)$.
# $$Y(\Omega) = \underline{W}^\text{H}(\Omega) \cdot \underline{X}(\Omega)$$

#%% 
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(precision=3)

#%%[markdwon]
# ## Parameter

#%%
varphi = 45 / 180 * np.pi # Angle of attack of the Source S(\Omega) in relation to the mic array 
c = 343000 # Velocity of sound in mm/s
mic = 6 # count of mics
d = 20 # distance in mm
fs = 16000 # Sample rate

n_fft = 512 # Fourier Transform length
n_spec = 257 # Number of frequency bins 
n_dir = 180 # Number of directions which the steering vector is steered to

#%%[markdown]
# ##Microphone Positions
# `pos_y` and `pos_x` are the microphone positions. It is a Uniform Linear Array (ULA) type (like seen in the Figure below)
#%%
pos_y = np.zeros((1,mic))
pos_x = np.r_[0.:mic]*d

fig, ax = plt.subplots()
ax.scatter(pos_x, pos_y, c='tab:red', alpha=1, edgecolors='white')
plt.ylabel('Y Position [mm]')
plt.xlabel('X Position [mm]')
plt.ylim((-50, 50))

#%%[markdown]
# ## Free Field model and delay vectors
# ...
#$$\underline A_q^{\text{ff}}(\Omega) = \exp\big(-j\Omega f_s \Delta\underline \tau(\varphi_q)\big),$$
# Calculate the delay vectors to each microphone to the source $q$ in the frequency domain:
#%%
tau = (pos_x*np.cos(varphi)+pos_y*np.sin(varphi))/c #calculating delay vector tau (in the time domain) depending on the array geometry.
tau = tau.reshape([mic,1,1])
Omega_array = np.r_[0.:n_spec].T*np.pi/n_fft*2
Omega_array = Omega_array.reshape([1,1,n_spec])
A_ff = np.exp(-1j*Omega_array*fs*tau)

#%%
tmp = np.squeeze(np.round(np.angle(A_ff[:,:,:])/np.pi*180))
plt.plot(tmp.T)
plt.ylabel("Angle [Deg]")
plt.xlabel("Frequency [Bin]")


#%%[markdown]
# The plot shows the angle of the complex spectral time delays from the desired signal between reference microphone 1 and the others. for higher frequencys you see that the angle is growing due to the faster swinging of the signal. This means for the same time delay different frequencys have different phase differences between two microphones. 


# ## Delay and Sum Beamformer
# ...
# ## Calculate the steering vectors W_H for the localization:

#%%
angle_array = np.c_[0:360:360/n_dir]/180*np.pi
tau_steering = (pos_x*np.cos(angle_array)+pos_y*np.sin(angle_array))/c
tau_steering = tau_steering.T.copy()
tau_steering = tau_steering.reshape([mic,1,1,n_dir])
W = np.exp(-1j*Omega_array.reshape([1,1,n_spec,1])*fs*tau_steering)
W.shape
W_H = W.reshape([1,mic,n_spec,n_dir]).conj()
W_H.shape

#%%[markdown]
# ## Spatial Convariance
# Another important signal property is the covariance that describes the interdependencies between the microphone signals $\underline X(\Omega)$. To obtain this covariance, it is presumed that the signals are stochastic. When only considering one source ($Q=1$),

# the spatial covariance matrix can be denoted as

# $$\mathbf \Phi_{xx}(\Omega) = \text{E}\{\underline X(\Omega)\underline X^H(\Omega)\}$$
# $$ = \underline A(\Omega) \text{E} \{ S'(\Omega)  S'^*(\Omega)\}\underline A^H(\Omega) + \text{E}\{\underline V(\Omega)\underline V^H(\Omega)\}$$
# $$ = \mathbf \Phi_{ss}(\Omega) + \mathbf \Phi_{vv}(\Omega),$$

# where $E\{\cdot\}$ represents the expectation value operator, $^*$ denotes the complex conjugate operator, $\mathbf \Phi_{ss}(\Omega)$ represents the source correlation matrix, $\mathbf \Phi_{vv}(\Omega)$ the noise correlation matrix and $(\cdot)^H$ the Hermitean operator. 
# If we consider noise not present $V=0$ and the expectation value of the signal $\text{E}{S(\Omega)}=1$ then the formular for the spatial covariance matrix $\mathbf \Phi_{xx}(\Omega)$ reduces to
# $$\mathbf \Phi_{xx}(\Omega) =  \underline A(\Omega) \underline A^H(\Omega) $$
#%%
A_ff_H = A_ff.reshape([1,mic,n_spec]).copy()
A_ff_H = A_ff_H.conj()
phi_xx = A_ff_H * A_ff

#%%
df = pd.DataFrame(phi_xx[:,:,50])
df.style.format('{:,.2f}'.format)

# ## Acoustic Sound Localization
# Acoustic sound localization is the task of locating a sound source given measurements of the sound field. ...
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


