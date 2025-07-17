import scipy
import numpy as np
import acoustics
scipy.io.wavfile.write('white_noise.wav', 16000, np.array(((acoustics.generator.noise(16000*60, color='white'))/3) * 32767).astype(np.int16))