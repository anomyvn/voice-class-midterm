import os
import threading
import tkinter
import pyaudio
import wave
import numpy as np
import math
import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
from time import time
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import log_loss
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
import pickle



TITLE = "Word Reconigtion"
RESOLUTION = "300x150"
BUTTON_CONFIG = {
    'height': 1,
    'width': 15
}
LABEL_CONFIG = {
    'wraplength': 500
}

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
FRAME_PER_BUFFER = 1024

RECORDING_FILE = "temp.wav"

def clustering(X, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    print("centers", kmeans.cluster_centers_.shape)
    return kmeans  

def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix

def one_hot_vec(y):
    return [1 if s==y else 0 for s in stt ]
stt = ["tôi","không","một","người","cáchly"]

def train(data_train,state_num):
    models = {}
    dataset = data_train.copy()
    for cname in dataset.keys():
        n = state_num[cname]
        startprob = np.zeros(n)
        startprob[0] = 1
        transmat=np.diag(np.full(n,1))
        
        
        hmm = hmmlearn.hmm.MultinomialHMM(
            n_components=n, random_state=0, n_iter=1000, verbose=False,
            startprob_prior=startprob,
            transmat_prior=transmat,
        )
        
        X = np.concatenate(dataset[cname])
        lengths = list([len(x) for x in dataset[cname]])
        hmm.fit(X, lengths=lengths)
        models[cname] = hmm    
    return models

def get_result(data_test,models):
    result = {}
    dataset = data_test.copy()
    for true_cname in data_test.keys():
        true = 0 
        for O in dataset[true_cname]:
    
            score = {cname : model.score(O, [len(O)]) for cname, model in models.items()}
            
            label = max(score.keys(),key=lambda x:score[x])
            if label == true_cname:
                true += 1
        result[true_cname] = true/len(dataset[true_cname])  
        print(true,len(dataset[true_cname]))
    return result

def predict(X_test,models):
    y_predict = []
    for x in X_test:
        score = {}
        for cname in models.keys():
            score[cname] = models[cname].score(x, [len(x)])
        label = max(score.keys(),key=lambda x:score[x])
        y_predict.append(label)
    return y_predict

def predict_pro(X_test,models):
    y_pro = []
    
    for x in X_test:
        score = {}
        for cname in models.keys():
            score[cname] = models[cname].score(x, [len(x)])
        pro = softmax([score[s] for s in stt])
        y_pro.append(pro)
    return y_pro

def pro_to_label(y):
    return stt[max(range(len(y)),key=lambda x:y[x])]



class Model:
    def __init__(self,dataset_train,k,models_state):
        self.dataset_train = dataset_train
        self.k = k
        self.kmeans = self.get_kmeans()
        self.data_train = self.convert_data_train()
        self.models_state = models_state
        self.ensemble = create_emsemble(self.models_state,self.data_train)
        
    def get_kmeans(self):
        all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in self.dataset_train.items()], axis=0)
        kmeans = clustering(all_vectors,self.k)
        return kmeans
    
    def convert_data_train(self):
        X_train = []
        y_train = []

        for cname in self.dataset_train.keys():
            self.dataset_train[cname] = list([self.kmeans.predict(v).reshape(-1,1) for v in self.dataset_train[cname]])
            X_train += [d for d in self.dataset_train[cname]]
            y_train += [cname for d in self.dataset_train[cname]]

        data_train = defaultdict(list)
        for t in zip(X_train,y_train):
            data_train[t[1]].append(t[0])
        return data_train
    def predict(self,X_test):
        X_test = [self.kmeans.predict(v).reshape(-1,1) for v in X_test]
        res = emsemble_predict(self.ensemble,X_test)
        
        voting_result = np.zeros(np.array(res['model1']['y_predict_pro']).shape)
        for k in res.keys():
            voting_result += np.array(res[k]['y_predict_pro'])
        return res,voting_result
def create_emsemble(models_state,data_train):
    model_emsemble = []
    for s in models_state:
        state_num = {"tôi":s[0],
            "không":s[1],
            "một":s[2],
            "người":s[3],
            "cáchly":s[4]
                }
        
        model = train(data_train,state_num)
        model_emsemble.append(model)
    return model_emsemble

def emsemble_predict(model_ensemble,X_test):
    result = {}
    count = 0
    for m in model_ensemble:
        count += 1
        result[f'model{count}'] = {'y_predict':predict(X_test,m),
                                  'y_predict_pro':predict_pro(X_test,m)}
    
    return result 

class Recorder:
    def __init__(self):
        self.start_button = tkinter.Button(
            root,
            text="Start Recording",
            command=self.start_recording,
            **BUTTON_CONFIG
        )
        self.start_button.pack()
        self.start_lock = False

        self.stop_button = tkinter.Button(
            root,
            text="Stop Recording",
            command=self.stop_recording,
            **BUTTON_CONFIG
        )
        self.stop_button.pack()
        self.stop_lock = True

        self.status = tkinter.Label(
            root,
            text="No recording"
        )
        self.status.pack()

        self.recognize_button = tkinter.Button(
            root,
            text="Recognize Word",
            command=self.recognize,
            **BUTTON_CONFIG
        )
        self.recognize_button.pack()
        self.recognize_lock = True

        self.is_recording = False


    def start_recording(self):
        if self.start_lock:
            return

        self.start_lock = True

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            frames_per_buffer=FRAME_PER_BUFFER,
            input=True
        )

        self.frames = []

        self.is_recording = True
        self.status.config(text="Recording")

        self.recognize_lock = True
        self.stop_lock = False

        thread = threading.Thread(target=self.record)
        thread.start()

    def stop_recording(self):
        if self.stop_lock:
            return

        self.stop_lock = True

        self.is_recording = False

        wave_file = wave.open("temp.wav", "wb")

        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(self.audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)

        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()

        self.status.config(text="Recorded")

        self.recognize_lock = False
        self.start_lock = False

    def record(self):
        while (self.is_recording):
            data = self.stream.read(FRAME_PER_BUFFER)
            self.frames.append(data)

    def recognize(self):
        mfcc = [get_mfcc(RECORDING_FILE)]
        file = open('model', 'rb')
        m = pickle.load(file)
        file.close()
        full_result,result= m.predict(mfcc)
        
        y_predict = [pro_to_label(y_pro) for y_pro in result][0]
        self.status.config(text=f"This is \"{y_predict}\"")


root = tkinter.Tk()
root.title(TITLE)
root.geometry(RESOLUTION)
app = Recorder()
root.mainloop()