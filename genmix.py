#conda create --name cypherbot
#conda install tensorflow
#move to dir...
#pip install -r requirements.txt or manual install...

#conda install spyder
#pip install pyloudnorm

#pip install librosa
#conda install -c conda-forge ffmpeg






#https://github.com/csteinmetz1/pyloudnorm
#https://github.com/teticio/Deej-AI



#%% get file list
import os

def file_name_pickle(file_dir):  
    L=[]
    L_full=[]
    for root, dirs, files in os.walk(file_dir): 
        for file in files: 
            if os.path.splitext(file)[1] == '.p': 
                L.append(os.path.splitext(file)[0])
                L_full.append(os.path.join(root, file))
    return L , L_full

def file_name(file_dir):  
    L=[]
    L_full=[]
    for root, dirs, files in os.walk(file_dir): 
        for file in files: 
            if os.path.splitext(file)[1] == '.mp3' or os.path.splitext(file)[1] == '.wav': 
                L.append(os.path.splitext(file)[0])
                L_full.append(os.path.join(root, file))
    return L , L_full

L, L_full = file_name('./_raw/')

#%% normalize samplerate/channel/lufs & save to wav


import librosa
import soundfile as sf
import pyloudnorm as pyln
# os.mkdir('./_normalized/')
for i in range(len(L)):
    print('processing: ' + L_full[i])
    data, sr = librosa.load(L_full[i], mono=False, sr=44100)
    assert data.shape[0]==2, 'must be 2ch input'
    data = data.T
    # measure the loudness first 
    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(data)
    # loudness normalize audio to -14 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -14.0)
    sf.write('./_normalized/' + L[i] + '.wav', loudness_normalized_audio, sr, 'PCM_24')


#%%
os.system("python MP3ToVec.py Pickles mp3tovec --scan ./_normalized/")

# import keras
# from keras.models import load_model

# model = load_model('speccy_model', compile=False)
# model.summary()

#%%
import pickle
import numpy as np

mp3tovec = pickle.load(open('./Pickles/mp3tovecs/mp3tovec.p', 'rb'))
print(f'{len(mp3tovec)} MP3s')

def most_similar(positive=[], negative=[], topn=5, noise=0):
    if isinstance(positive, str):
        positive = [positive] # broadcast to list
    if isinstance(negative, str):
        negative = [negative] # broadcast to list
    mp3_vec_i = np.sum([mp3tovec[i] for i in positive] + [-mp3tovec[i] for i in negative], axis=0)
    mp3_vec_i += np.random.normal(0, noise * np.linalg.norm(mp3_vec_i), len(mp3_vec_i))
    similar = []
    for track_j in mp3tovec:
        if track_j in positive or track_j in negative:
            continue
        mp3_vec_j = mp3tovec[track_j]
        cos_proximity = np.dot(mp3_vec_i, mp3_vec_j) / (np.linalg.norm(mp3_vec_i) * np.linalg.norm(mp3_vec_j))
        similar.append((track_j, cos_proximity))
    return sorted(similar, key=lambda x:-x[1])[:topn]

def make_playlist(seed_tracks, size=10, lookback=3, noise=0):
    max_tries = 10
    playlist = seed_tracks
    while len(playlist) < size:
        similar = most_similar(positive=playlist[-lookback:], topn=max_tries, noise=noise)
        candidates = [candidate[0] for candidate in similar if candidate[0] != playlist[-1]]
        for candidate in candidates:
            if not candidate in playlist:
                break
        playlist.append(candidate)
    return playlist

for key in mp3tovec:
    # this works, but I don't have the values
    print(key)
print("start from " + list(mp3tovec.keys())[0])
playlist = make_playlist([list(mp3tovec.keys())[0]], size=len(mp3tovec))



# for i in range(len(playlist)):
    
# playlist[0]

import wave
data = []
timecode = 0
print('==========\n')
print('Beat tape compiled by BMT cypherbot\n')
for i in range(len(playlist)):
    w = wave.open(playlist[i], 'rb')
        
    mstart, sstart = divmod(timecode/44100, 60)
    timecode+=w.getnframes()
    mend, send = divmod(timecode/44100, 60)
    print('{:02d}:{:02d} ~ {:02d}:{:02d}: {}'.format(int(mstart), int(sstart), int(mend), int(send), (os.path.splitext(os.path.basename(playlist[i]))[0])))
    data.append( [w.getparams(), w.readframes(w.getnframes())] )
    w.close()

w = wave.open('tag.wav', 'rb')
data.append( [w.getparams(), w.readframes(w.getnframes())] )
w.close()
print('\n==========\n')
output = wave.open('mixdown.wav', 'wb')
output.setparams(data[0][0])
for i in range(len(data)):
    output.writeframes(data[i][1])
output.close()

if os.path.exists('mixdown.mp3'):
    os.remove('mixdown.mp3')
os.system("ffmpeg -i mixdown.wav -b:a 320k mixdown.mp3")

os.remove("mixdown.wav")

# os.remove('./_normalized/')
del_L, del_L_full = file_name('./_normalized/')
for i in range(len(del_L)):
    os.remove(del_L_full[i])
    
del_L, del_L_full = file_name_pickle('./Pickles/')
for i in range(len(del_L)):
    os.remove(del_L_full[i])