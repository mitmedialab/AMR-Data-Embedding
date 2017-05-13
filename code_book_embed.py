######################################################
# code_book_embed.py
# Methods that allow for easy data embedding using
#   the "code book" method, where speech samples 
#   are used as bit or numeric markers and recovered
#   by matched filtering post AMR compression.
#
# Author: Ishwarya Ananthabhotla
######################################################

#######################
# Imports
#######################

import numpy as np
# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms

# and IPython.display for audio output
import IPython.display
from IPython.display import Audio

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display

import subprocess
import time
import os
import peakutils

from scipy import signal

#######################
# Visualization Tools
#######################

def view_spectrogram(path_to_sample, n_fft=2048):
    sample, sr = librosa.load(path_to_sample)
    S = librosa.stft(sample, n_fft=n_fft)
    plt.title(path_to_sample)
    librosa.display.specshow(librosa.core.amplitude_to_db(S,ref=np.max), y_axis='log', x_axis='time')


def view_sample(path_to_sample):
    sample, sr = librosa.load(path_to_sample)
    plt.figure()
    plt.title(path_to_sample)
    plt.plot(sample)
    plt.show()

#######################
# Embed Waveforms
#######################

class Embed:
    # source: cover speech audio
    # sample_path_list: takes a list of paths to the sample original waveforms,
    # digit_list: digits that correspond to waveform samples
    # embed_sequence: sequence in which those digits should be embedded
    def __init__(self, path_to_source, sample_path_list, digit_list, embed_sequence):
        self.all_idx = range(len(sample_path_list))
        self.source, sr = librosa.load(path_to_source)
        self.samples = []
        self.digit_sample_map = {}
        for i, path in enumerate(sample_path_list):
            sample, sr = librosa.load(path)
            self.samples.append(sample)
            self.digit_sample_map[digit_list[i]] = self.samples[i]

        self.digit_list = digit_list

        # self.samples = np.array(self.samples)
        print [len(l) for l in self.samples]

        self.embed_sequence = embed_sequence

        #self. sr = sr
        self.sr = 22050 # default

    # return the timeseries waveforms of specified indexes
    # default returns all
    def get_data_timeseries(self, idx_list=None):
        if idx_list == None:
            idx_list = self.all_idx
        ts = []
        for i in idx_list:
            ts.append(self.samples[i])
        return ts

    # return truncated version of specified timeseries waveforms
    def truncate(self, size_frac, idx_list=None):
        if idx_list == None:
            idx_list = self.all_idx

        for i in idx_list:
            s = self.samples[i].copy()
            s_t = s[:int(len(s) * size_frac)]
            self.samples[i] = s_t

    # return energy-scaled version of specified timeseries waveforms
    def energy(self,  energy_frac, idx_list=None):
        if idx_list == None:
            idx_list = self.all_idx

        for i in idx_list:
            s = self.samples[i].copy()
            s_t = s * energy_frac
            self.samples[i] = s_t

    # return pitch shifted version of specified timeseries waveforms
    def pitch_shift(self,  num_steps, idx_list=None):
        if idx_list == None:
            idx_list = self.all_idx

        for i in idx_list:
            s = self.samples[i].copy()
            s_t = librosa.effects.pitch_shift(s, self.sr, n_steps=num_steps)
            self.samples[i] = s_t

    def remap(self):
        for i, dig in enumerate(self.digit_list):
            self.digit_sample_map[dig] = self.samples[i]

    # return pre-compression data embedded cover speech audio
    def get_embedded_audio(self, plot=False):
        self.remap()

        # add together sequence repeatedly to match length of cover file

        for num in self.embed_sequence:
            try:
                seq = np.concatenate((seq, self.digit_sample_map[num]))
            except UnboundLocalError:
                seq = self.digit_sample_map[num]

        repeat_seq = seq

        while len(repeat_seq) < len(self.source):
            repeat_seq = np.concatenate((repeat_seq, seq))

        repeat_seq = repeat_seq[:len(self.source)]

        print "Length of Sequence to be Embedded", len(repeat_seq)

        if plot:
            plt.figure()
            plt.plot(repeat_seq)
            plt.title("Sequence to be Embedded")
            plt.show()

            plt.figure()
            plt.plot(self.source)
            plt.title("Cover Speech Audio")
            plt.show()

            plt.figure()
            plt.plot(repeat_seq + self.source)
            plt.title("Embedded Audio")
            plt.show()

        # add the sequence waveform to the cover waveform
        return self.source + repeat_seq


#############################
# Compression/ Decompression
#   CLI Pipeline
#############################

def compress_and_decompress(embedded_signal, save_dir, plot=False, sample_rate=22050, n_fft=2048):
    time_stamp = str(time.time())
    
    librosa.output.write_wav(save_dir + "embedded_" + time_stamp + ".wav", embedded_signal, sample_rate)
    
    # compression  
    compression_call = "ffmpeg -i " + save_dir + "embedded_" + time_stamp + ".wav -ar 8000 -ab 12.2k -ac 1 " + save_dir + "compressed_" + time_stamp + ".amr"
    subprocess.call(compression_call, shell=True)

    # decompression   
    decompression_call = "ffmpeg -i " + save_dir + "compressed_" + time_stamp + ".amr -ar 22050 " + save_dir + "decompressed_" + time_stamp + ".wav"
    subprocess.call(decompression_call, shell=True)
    
    decompressed_signal, sr = librosa.load(save_dir + "decompressed_" + time_stamp + ".wav")

    if plot:
        plt.figure()
        E = librosa.stft(embedded_signal, n_fft=n_fft)
        plt.title("Embedded Signal")
        librosa.display.specshow(librosa.core.amplitude_to_db(E,ref=np.max), y_axis='log', x_axis='time')

        plt.figure()
        D = librosa.stft(decompressed_signal, n_fft=n_fft)
        plt.title("Compressed and Decompressed Signal")
        librosa.display.specshow(librosa.core.amplitude_to_db(D,ref=np.max), y_axis='log', x_axis='time')
    
    return decompressed_signal, sr

#######################
# Recover Waveforms
#######################

class Recover:
    # d_embed_timeseries: post compression embedded cover speech audio
    # data_timeseries: timeseries of all waveforms used for embedding
    # digit_list: digits that correspond to waveform samples
    # embed_sequence: sequence in which those digits should have been embedded
    def __init__(self, d_embed_timeseries, data_timeseries, digit_list, embed_sequence, full_seq_length):
        self.all_idx = range(len(data_timeseries))
        self.d_embed_timeseries = d_embed_timeseries
        self.samples = data_timeseries
        self.digit_list = digit_list

        self.embed_sequence = embed_sequence
        self.full_seq_length = full_seq_length

    # return bit sequence estimation based on cross-correlation and time interpolation
    def get_bit_sequence(self, thres=0.5, plot=False):
        all_bits = []
        for i, w in enumerate(self.samples):
            # cross correlate each waveform with the source audio
            corr = signal.correlate(self.d_embed_timeseries, w, mode='same')

            # find peaks and collect timestamps (sample numbers)
            indexes = peakutils.indexes(corr, thres=thres, min_dist=len(w)/2)

            if plot:
                plt.figure()
                plt.plot(corr)
                plt.plot(indexes, corr[indexes], 'ro')
                plt.title("Cross Correlation Post Compression for " + str(self.digit_list[i]))
                plt.show()

            digit = self.digit_list[i]
            rec_digits = np.column_stack((indexes, np.repeat(digit, len(indexes))))
            try:
                all_bits = np.vstack((all_bits, rec_digits))
            except:
                all_bits = rec_digits

        # time interpolation - stack individual arrays and sort by time column
        all_bits = all_bits[np.argsort(all_bits[:, 0])]

        if plot:
            # this will let us know about missing bits based on time intervals
            plt.figure()
            plt.stem(all_bits[:, 0], all_bits[:, 1])
            plt.title("Recovered Sequence: Raw")
            plt.show()
        #clean_bits = self.check_corr_errors(all_bits)
        # conf = self.assign_confidence(all_bits, plot=True)
        # clean_bits = self.clean_sequence(all_bits, conf)

        clean_bits = self.clean_by_distance_confidence(all_bits)


        if plot:
            # this will let us know about missing bits based on time intervals
            plt.figure()
            plt.stem(clean_bits[:, 0], clean_bits[:, 1])
            plt.title("Recovered Sequence: Cleaned")
            plt.show()

        return clean_bits[:,1]



    # assign confidence values based on density of points
    # NOTE: assume that we know length of sequence
    def clean_by_distance_confidence(self, time_bits, thres=0.3):

        def compute_dists(time_bits):
            dists = []
            for i in range(1, len(time_bits[:,0])):
                dists.append([i,time_bits[i,0] - time_bits[i-1,0]])
            return dists        

        del_list = []
        clean_bits = time_bits
        for i in range(len(time_bits[:,1]) - self.full_seq_length):
            dists_idxs = compute_dists(clean_bits)
            s_dists_idxs = sorted(dists_idxs, key = lambda x: x[1])
            #print "sorted dist_idxs", s_dists_idxs
            del_idx = s_dists_idxs.pop(0)[0]
            #print "del idx", del_idx
            clean_bits = np.delete(clean_bits, del_idx, axis=0)
            #print "clean_bits", clean_bits

        return clean_bits

    def get_recovery_estimate(self, predict_bit_seq):
        ref_seq = self.embed_sequence
        while len(ref_seq) < len(predict_bit_seq):
            ref_seq = np.tile(ref_seq, 2)
        ref_seq = ref_seq[:len(predict_bit_seq)]

        print "ref sequence = ", ref_seq
        print "predicted sequence = ", predict_bit_seq

        matches = np.equal(predict_bit_seq, ref_seq)
        return np.sum(matches) / float(len(matches))

if __name__ == "__main__":
    path_to_source = "audio_samples/man2_orig.wav"
    sample_path_list = ['speech_samples/pronunciation_en_zero2.mp3','speech_samples/pronunciation_en_one.mp3']
    # E = Embed(path_to_source, sample_path_list, [1], [1,1])

    # # mess around with the codebook waveforms before embedding
    # E.truncate(0.7, idx_list=[0])
    # print "Sample Lengths: "
    # for samp in E.samples:
    #     print len(samp)

    # print "Source Length: "
    # print len(E.source)
    # E.energy(0.2, idx_list=[0])
    # E.pitch_shift(-15, idx_list=[0])

    # embed = E.get_embedded_audio(plot=True)
    # d_embed, sr = compress_and_decompress(embed, "compression_samples/", plot=True)

    # # get the timeseries of the the original waveforms and recover
    # w = E.get_data_timeseries()
    # R = Recover(d_embed, w, [1], [1,1])
    # final_sequence = R.get_bit_sequence(thres=0.7, plot=True)
    # print final_sequence

    E2 = Embed(path_to_source, sample_path_list, [0,1], [0,1,1,1,0])

    # mess around with the codebook waveforms before embedding
    E2.truncate(0.4, idx_list=[0,1])
    E2.energy(0.2, idx_list=[0])
    E2.energy(0.3, idx_list=[1])
    E2.pitch_shift(-15, idx_list=[1])
    E2.pitch_shift(-15, idx_list=[0])

    embed2 = E2.get_embedded_audio(plot=False)
    d_embed2, sr = compress_and_decompress(embed2, "compression_samples/", plot=False)

    # get the timeseries of the the original waveforms and recover
    w2 = E2.get_data_timeseries()
    # embedded sequence length
    seq_len = 28
    R2 = Recover(d_embed2, w2, [0,1], [0,1,1,1,0], seq_len)
    final_sequence2 = R2.get_bit_sequence(thres=0.75, plot=True)
    print "raw ", final_sequence2
    print R2.get_recovery_estimate(final_sequence2)


