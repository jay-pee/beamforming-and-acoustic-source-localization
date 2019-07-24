#%% [markdown]
# ## Basic of Beamforming and Source Localization with Steered response Power
# 
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

#%% [markdown]
# What is A^ff?
# A^ff are the Delay Vector to each microphone to the Source in the Frequency Domain.
# But first you need the delays tau:

#%%
# Parameter
varphi = 45 / 180 * np.pi # Angle of attack of the Source S(\Omega) in relation to the mic array 
c = 343 # Velocity of sound in m/s
mic = 6 # count of mics
d = 20 # distance in mm
fs = 16000 # Sample rate
# mictype = 'ULA' # Uniform Linear Array type (like seen in the Figure)
n_fft = 512
n_spec = 257
n_dir = 72

#%%
pos_y = np.zeros((1,mic))
pos_x = np.r_[0.:mic]*d
tau = (pos_x*np.cos(varphi)+pos_y*np.sin(varphi))/c
tau = tau.reshape([mic,1,1])
Omega_array = np.r_[0.:n_spec].T*np.pi/n_fft*2
Omega_array = Omega_array.reshape([1,1,n_spec])
A_ff = np.exp(-1j*Omega_array*fs*tau)

#%%
angle_array = np.c_[0:360:5]/180*np.pi
tau_steering = (pos_x*np.cos(angle_array)+pos_y*np.sin(angle_array))/c
tau_steering = tau_steering.T.copy()
tau_steering = tau_steering.reshape([mic,1,1,n_dir])
W = np.exp(-1j*Omega_array.reshape([1,1,n_spec,1])*fs*tau_steering)
W.shape
W_H = W.reshape([1,mic,n_spec,n_dir]).conj()
W_H.shape
# angle_array.shape
# pos_x.shape
#%% [markdown]
#  How to calculate A^ff(\Omega)? you need the Array geometry for that.

#%%
A_ff_H = A_ff.reshape([1,mic,n_spec]).copy()
A_ff_H = A_ff_H.conj()
phi_xx = A_ff_H * A_ff


#%%
# with pd.option_context('display.precision', 3):
# pd.set_option('precision', 0)
# pd.set_option('display.float_format', lambda x: '%.0f' % x)
df = pd.DataFrame(phi_xx[:,:,50])
df.style.format('{:,.2f}'.format)

#%%
tmp = np.sum(W_H * phi_xx[:,:,:,np.newaxis],0, keepdims=True)
tmp.shape
tmp = np.abs(np.squeeze(np.sum(tmp*W_H.conj(),1)))
tmp.shape

#%%
plt.figure(1)
# pd.DataFrame(tmp)
plt.imshow(tmp.T)
plt.show()

df = pd.DataFrame(tmp)
df.style.format('{:,.2f}'.format)
print(df)