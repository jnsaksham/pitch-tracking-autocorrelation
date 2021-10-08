# Define functions

###### Computing F0 ######

def block_audio(x,blockSize,hopSize,fs):
    
    # returns a matrix xb (dimension NumOfBlocks X blockSize) and a vector timeInSec (dimension NumOfBlocks) 
    # for blocking the input audio signal into overlapping blocks. 
    # timeInSec will refer to the start time of each block
    
    t = 0
    timeInSec = np.array([])
    xb = []
    samples = len(x)
    
    while t < samples:#-blockSize:
        if t <= samples-blockSize:
            block = x[t:t+blockSize]
        if t>samples-blockSize and t<samples:
            block = np.append(x[t:], np.zeros(blockSize-len(x[t:])))
        
        timeInSec = np.append(timeInSec, t/fs)
        xb.append(block)
        t += hopSize
    xb = np.array(xb)
    
    return xb, timeInSec

def comp_acf(inputVector, bIsNormalized = True):
    
    x1 = inputVector
    r = np.array([])

    # for pos position, two vectors to be multiplied are x1[:len(x1)-pos] and x1[pos:]
    for pos in np.arange(len(x1)):
        vector1 = x1
        vector2 = np.append(x1[pos:], np.zeros(pos))

        ac = np.dot(vector1, vector2)

        # Normalization
        # lambda = 1/{sigma (x(i)^2).{sigma y(i)^2}^}^(1/2). i = from ie to is

        if bIsNormalized is True:
            normalize = (np.dot(vector1, vector1)*np.dot(vector2, vector2))**(1/2)
            ac_norm = ac/normalize
            r = np.append(r, ac_norm)
            
        else:
            r = np.append(r, ac)
            print ('ac: ', ac)
    
    return r

def get_f0_from_acf(r, fs):
    # Return fundamental frequency in Hz
    
# Approach 1: find maximas and take argmax

    peaks_indices = np.array([]).astype(np.int16)
    for i in np.arange(1, len(r)-1):
        if r[i] >= r[i-1] and r[i] >= r[i+1]:
            peaks_indices = np.append(peaks_indices,int(i))
            
    peaks_amplitude = r[peaks_indices]
    second_max_arg = np.argmax(peaks_amplitude)
    second_max = peaks_indices[second_max_arg]

    frequency = fs/second_max
    
    
# Approach 2: Identify the maximum after the first minimum

#     first_max = np.argmax(r)
# #     print ('First max: ', first_max)

#     # Finding first minima to detect the second maxima after that
#     first_min = np.argmin(r)
# #     print ('First min: ', first_min)

#     # Finding second maximua
#     second_max = np.argmax(r[first_min:]) + first_min
# #     print ('Second max: ', second_max)

#     frequency = fs/(second_max-first_max)
    
    return frequency

def track_pitch_acf(x,blockSize,hopSize,fs):
    """calls the three functions above and returns two vectors f0 and timeInSec."""
    
    blocks, timeInSec = block_audio(x,blockSize,hopSize,fs)
    f0 = np.array([])
    index = 0
    for inputVector in blocks:
        print ('Block_num: ', index)
        r = comp_acf(inputVector, bIsNormalized = True)
        f0_block = get_f0_from_acf(r, fs)
        f0 = np.append(f0, f0_block)
        index += 1
        
    return f0, timeInSec

###### Evaluation ######

def convert_freq2midi(freqInHz): 

    """Returns a variable pitchInMIDI of the same dimension as freqInHz. 
    Note that the dimension of freqInHz can be a scalar, a vector, or a matrix. 
    The conversion is described in Textbook Section: 7.2.3. Assume f(A4) = 440Hz."""
    
    pitchInMIDI  =  12*np.log2(freqInHz/440) + 69

    
    return pitchInMIDI


def eval_pitchtrack(estimateInHz, groundtruthInHz):
    """
    estimateInHz: acf frequency vector
    groundtruthInHz: Ground truth of pitch
    
    Returns: the RMS of the error in Cent (Textbook Section: 7.2.3)
    in the pitch domain (not frequency) and returns this as errCentRms. 

    Note: exclude blocks with annotation = 0
    """

    # ignore annotated 0 values
    groundtruthInHz_final = np.array([])
    estimateInHz_final = np.array([])

    for i in np.arange(len(estimateInHz)):
        if groundtruthInHz[i] != 0:
            groundtruthInHz_final = np.append(groundtruthInHz_final, groundtruthInHz[i])
            estimateInHz_final = np.append(estimateInHz_final, estimateInHz[i])
    

    # Convert non-zero frequencies to MIDI
    estimateInMIDI = convert_freq2midi(estimateInHz_final)
    groundtruthInMIDI = convert_freq2midi(groundtruthInHz_final)
    

    # f2/f1 = 2**(cents/1200)
    #log2(f2) - log2(f1) = cents/1200  
    error = groundtruthInMIDI-estimateInMIDI
    errorCents = error*100
    errCentRms = (np.dot(errorCents, errorCents)/len(errorCents))**(1/2)
    
    return errCentRms


def run_evaluation(complete_path_to_data_folder):

    # Move to dataset folder
    os.chdir(complete_path_to_data_folder)
    
    # Append relevant files in an array

    txt = np.array([])
    wav = np.array([])
    for filename in os.listdir():
        if filename[-3:] == "txt":
            txt = np.append(txt, filename)
        if filename[-3:] == "wav":
            wav = np.append(wav, filename)

    # Sort the arrays by file name
    txt = np.sort(txt)
    wav = np.sort(wav)
    print ('Text files: ', txt)
    print ('Wav files: ', wav)
    
    f0_all = np.array([])

    for audio in wav:
        rate, x = sp.read(audio)
        x = x.astype(np.float64)
        f0, timeInSec = track_pitch_acf(x, blockSize, hopSize, rate)
        f0_all = np.append(f0_all, f0)
    
    print ('Total blocks detected: ', len(f0_all))
    
    # Read txt file and compile data in an array

    onset_seconds_all = np.array([])
    pitch_frequency_all = np.array([])

    for txtfile in txt:
        data = np.loadtxt(txtfile)
        onset_seconds = data[:, 0]
        pitch_frequency = data[:, 2]
    
        onset_seconds_all = np.append(onset_seconds_all, onset_seconds)
        pitch_frequency_all = np.append(pitch_frequency_all, pitch_frequency)
    
    print ('Total ground truth pitches detected: ', len(pitch_frequency_all))
    
    # Check if there are any inf frequencies
    inf_f0 = np.isinf(f0_all)

    # Check # of inf values
    inf_values = np.count_nonzero(inf_f0)
    print (inf_values)

    # Arguments of inf values
    inf_indices = np.array(np.nonzero(inf_f0)[0])
    print (inf_indices, inf_indices.shape)
    
    # Evaluate pitch track
    errCentRms = eval_pitchtrack(f0_all, pitch_frequency_all)
    
    return errCentRms


## Import packages

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile as sp
import glob, os
import IPython.display as ipd
import time
    
## Generate a sine wave

f1 = 441
f2 = 882
fs = 44100
A = 1
t1 = np.arange(0, 1, 1./fs)
t2 = np.arange(1,2,1./fs)

# 441 Hz sine wave from 0-1s
wave1 = A*np.sin(2*np.pi*f1*t1)

# 882 Hz sine wave from 1-2s
wave2 = A*np.sin(2*np.pi*f2*t2)

# Final wave
test_wave = np.append(wave1, wave2)
print ('Total samples: ', len(test_wave))
print ('Total duration: ', len(test_wave)/fs, 's')

# Play audio
ipd.Audio(test_wave, rate=fs)

## Evaluate on test signal

# get blocks from the wave
blockSize = 1024

# hopSize calculation
hopSize = blockSize//2

freqInHz_test, timeInSec_test = track_pitch_acf(test_wave,blockSize,hopSize,fs)

# Compute ground truth
predictedFreqInHz_test = np.array([])
for timestamp in timeInSec_test:
    if timestamp < 1:
        predictedFreqInHz_test = np.append(predictedFreqInHz_test, 441)
    if timestamp >= 1 and timestamp < 2:
        predictedFreqInHz_test = np.append(predictedFreqInHz_test, 882)
        
# Compute error in Hz
errorInHz_test = freqInHz_test - predictedFreqInHz_test

# plt.plot(errorInHz_test, '.')
plt.figure(figsize = (17, 5))
plt.plot(timeInSec_test, freqInHz_test)
plt.plot(timeInSec_test, errorInHz_test, color = 'red')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.legend(['Frequency', 'Error'])
plt.title('Frequency and Error in Hz')
plt.show()

# Convert frequency to MIDI to calculate error
pitchInMIDI_test = convert_freq2midi(freqInHz_test)

# Calculate rmse error
rmse_test = eval_pitchtrack(freqInHz_test, predictedFreqInHz_test)
print ('RMSE - test wave: ', rmse_test)


## Test on real signals

# Print current directory
print ('Current directory: ', os.getcwd())

# Input folder_name. Default = trainData
folder_name = 'trainData'

# Final folder path
complete_path_to_data_folder = os.getcwd() + '/' + folder_name
print ('Destination path: ', complete_path_to_data_folder)

## Read files, compute f0, track pitch and evaluate the error

# Define blockSize and hopSize
blockSize = 1024
hopSize = blockSize//2

rmse_dataset_cents = run_evaluation(complete_path_to_data_folder)

print ('RMSE in cents for the whole dataset: ', rmse_dataset_cents)
