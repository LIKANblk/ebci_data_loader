from scipy.io import loadmat
import os
import numpy as np
import pickle
from random import shuffle,seed
import math
from mne.filter import resample
EEG_SAMPLE_RATE = 500 #Hz

def to_onehot(labels):
    unique_labels = list(set(labels))
    corrected_labels = map(lambda x: unique_labels.index(x),labels)
    y = np.zeros((len(labels),len(unique_labels)))
    y[range(len(labels)),corrected_labels] = 1
    return y

def data_shuffle(x, y,subj_indices=None):
    seed(1)
    d_len = len(y)
    sh_data = list(range(d_len))
    shuffle(sh_data)
    new_y = np.zeros_like(y)
    for i in range(d_len):
        new_y[i,...] = y[sh_data[i],...]
    new_x = np.zeros(x.shape)
    for i in range(d_len):
        new_x[i,...] = x[sh_data[i],...]
    if subj_indices is not None:
        new_subj_indices=np.zeros_like(subj_indices)
        for i in range(d_len):
            new_subj_indices[i, ...] = subj_indices[sh_data[i], ...]
        return new_x, new_y,subj_indices
    return new_x,new_y

def get_topography():
    '''
    Function returenss informantion about channel order and topography
    :return: Dict {Ch_number: [(x,y), Ch_name]}
    '''
    with open('/home/likan_blk/Yandex.Disk/eyelinesOnlineNew/data/order&locations.info', 'r') as f:
        topo_list = [line.split() for line in f][1:]
    topo_dict = {}
    ch_coordinates = []
    ch_names = []
    for elem in topo_list:
        alpha, r = float(elem[1]), float(elem[2])
        alpha = math.pi * alpha / 180.  # to radians
        x, y = r * math.sin(alpha), r * math.cos(alpha)
        name = str(elem[3])
        # topo_dict[int(elem[0])] = [(x, y), name]
        ch_coordinates.append((x,y))
        ch_names.append(name)
    return ch_coordinates,ch_names

class Data(object):
    def __init__(self,path_to_data,start_epoch,end_epoch,sample_rate=500):
        self.start_epoch = start_epoch  # seconds
        self.end_epoch = end_epoch
        self.sample_rate = sample_rate
        self.path_to_data = path_to_data
        if len(path_to_data.split('/')[-1]) == 0:
            self.exp_name = path_to_data.split('/')[-2]
        else:
            self.exp_name = path_to_data.split('/')[-1]

    def _resample(self, X, resample_to):
        '''
        Resample OVER 1-st axis
        :param X: eeg trials x Time x Channels
        :param resample_to:
        :return:
        '''

        # duration = X.shape[1]*self.sample_rate
        # downsample_factor = X.shape[1] / (resample_to * duration)
        downsample_factor = self.sample_rate / resample_to
        return resample(X, up=1., down=downsample_factor, npad='auto', axis=1)

# class DataProcessExperiment(Data):
#     def __init__(self,path_to_data):
#         start_epoch = -1.5 #seconds
#         end_epoch = 1 #seconds
#         Data.__init__(self,path_to_data,start_epoch)
#
#     def get_data(self,shuffle=False,start_window=0.200,end_window=0.500):
#         """
#         Returns:
#             A tuple of 2 numpy arrays: data (Trials x Channels x Time) and labels
#         """
#         data_info = loadmat(os.path.join(self.path_to_data,'events.mat'))['events']['field_type'][0][0]
#         start_window_ind = int((start_window - self.start_epoch)*self.sample_rate)
#         end_window_ind = int((end_window - self.start_epoch)*self.sample_rate)
#         indexes = [(i,(str[0][0] in ['ball','field'])) for i,str in enumerate(data_info) if str[0][0] in ['ball','ball_nT','field','field_nT']]
#         indexes,labels = map(lambda x:list(x),zip(*indexes))
#         data = loadmat(os.path.join(self.path_to_data,'eeg_epochs.mat'))['eeg_epochs'].transpose(2,0,1)
#         if shuffle:
#             return data_shuffle(data[indexes,start_window_ind:end_window_ind,:],labels)
#         return data[indexes,start_window_ind:end_window_ind,:],labels

class DataBuildClassifier(Data):
    def __init__(self,path_to_data):
        '''
        :param path_to_data: string, path to folder with all experiments
        '''
        start_epoch = -0.5 #seconds
        end_epoch = 1#seconds
        super(DataBuildClassifier, self).__init__( path_to_data,start_epoch,end_epoch)
        # Data.__init__(self, path_to_data,start_epoch,end_epoch)



    def _baseline_normalization(self,X,baseline_window=()):
        bl_start = int((baseline_window[0] - self.start_epoch) * self.sample_rate)
        bl_end = int((baseline_window[1] - self.start_epoch) * self.sample_rate)
        baseline = np.expand_dims(X[:, bl_start:bl_end, :].mean(axis=1), axis=1)
        X = X - baseline
        return X
        # return X[:,bl_start:bl_end,:].mean(axis=1)

    def get_data(self,subjects,shuffle=True,windows=None,baseline_window=(),resample_to=None):
        '''

        :param subjects: list subject's numbers, wich data we want to load
        :param shuffle: bool
        :param windows: list of tuples. Each tuple contains two floats - start and end of window in seconds
        :param baseline_window:
        :param resample_to: int, Hz - new sample rate
        :return: Dict. {Subject_number:tuple of 2 numpy arrays: data (Trials x Time x Channels) and labels}
        '''
        res={}
        for subject in subjects:

            eegT = loadmat(os.path.join(self.path_to_data,str(subject),'eegT.mat'))['eegT']
            eegNT = loadmat(os.path.join(self.path_to_data,str(subject),'eegNT.mat'))['eegNT']
            X = np.concatenate((eegT,eegNT),axis=-1).transpose(2,0,1)
            if len(baseline_window):
                X = self._baseline_normalization(X,baseline_window)
            y = np.hstack((np.ones(eegT.shape[2]),np.zeros(eegNT.shape[2])))
            #y = np.hstack(np.repeat([[1,0]],eegT.shape[2],axis=0),np.repeat([[0,1]],eegT.shape[2],axis=0))

            if (resample_to is not None) and (resample_to != self.sample_rate):
                X = self._resample(X, resample_to)
            else:
                resample_to = self.sample_rate

            time_indices=[]
            if windows is not None:
                for win_start,win_end in windows:
                    start_window_ind = int((win_start - self.start_epoch)*resample_to)
                    end_window_ind = int((win_end - self.start_epoch)*resample_to)
                    time_indices.extend(range(start_window_ind,end_window_ind))
                X,y = X[:,time_indices,:],y

            if shuffle:
                X,y = data_shuffle(X,y)
            res[subject]=(X,y)
        return res

# class AllEventsDataBuildClassifier(DataBuildClassifier):
#     def __init__(self,path_to_data,target_events=None,nontarget_events=None):
#         '''
#         :param path_to_data: string, path to folder with all experiments
#         :param target_events: list of strings (target events). Possible events "ClickedToUnlock" (button press),
#   #                   "ballSelect", "ballMove"
#         :param nontarget_events: string, path to folder with all experiments
#         '''
#         self.target_events = target_events
#         self.nontarget_events = nontarget_events
#         super(AllEventsDataBuildClassifier, self).__init__( path_to_data)
#
#     def manage_timestamps(self,timestamps):
#
#
#
#     def get_data(self,subjects,shuffle=True,windows=None,baseline_window=(),resample_to=None):
#         '''
#
#         :param subjects: list subject's numbers, wich data we want to load
#         :param shuffle: bool
#         :param windows: list of tuples. Each tuple contains two floats - start and end of window in seconds
#         :param baseline_window:
#         :param resample_to: int, Hz - new sample rate
#         :return: Dict. {Subject_number:tuple of 2 numpy arrays: data (Trials x Time x Channels) and labels}
#         '''
#
#         res = {}
#         for subject in subjects:
#             for event in self.target_events:
#             eegT = np.concatenate(
#                 [loadmat(os.path.join(self.path_to_data, str(subject), 'eegT_%s.mat' %event))['epochs'] for event in self.target_events],
#                 axis=2)
#
#
#             eegNT = np.concatenate(
#                 [loadmat(os.path.join(self.path_to_data, str(subject), 'eegNT_%s.mat' %event))['epochs'] for event in self.nontarget_events],
#                 axis=2)
#
#
#             X = np.concatenate((eegT, eegNT), axis=-1).transpose(2, 0, 1)
#             if len(baseline_window):
#                 baseline = self._baseline_normalization(X, baseline_window)
#                 baseline = np.expand_dims(baseline, axis=1)
#                 X = X - baseline
#             y = np.hstack((np.ones(eegT.shape[2]), np.zeros(eegNT.shape[2])))
#             # y = np.hstack(np.repeat([[1,0]],eegT.shape[2],axis=0),np.repeat([[0,1]],eegT.shape[2],axis=0))
#
#             if (resample_to is not None) and (resample_to != self.sample_rate):
#                 X, y = self._resample(X, y, resample_to)
#
#             time_indices = []
#             if windows is not None:
#                 for win_start, win_end in windows:
#                     start_window_ind = int((win_start - self.start_epoch) * self.sample_rate)
#                     end_window_ind = int((win_end - self.start_epoch) * self.sample_rate)
#                     time_indices.extend(range(start_window_ind, end_window_ind))
#                 X, y = X[:, time_indices, :], y
#
#             if shuffle:
#                 X, y = data_shuffle(X, y)
#             res[subject] = (X, y)
#         return res


class OldData(DataBuildClassifier):
    def __init__(self,path_to_data,start_epoch = -0.5,end_epoch = 1.5,target_events=None,nontarget_events=None):
        #Data already fileterd and baseline corrected
        #Correct for experiments sbj = { 'e401','e402','e406','e407','e408','e409','e410','e411'};
        # Experiment indices { '1','2','3','4','5','6','7','8'};

        all_channels = \
            ['FZ','F3','F4','Cz','C3','C4','PZ','P3','P4','P1','P2','PO7','PO8','PO3','PO4','Oz','O1', 'O2','POz','vEOG','hEOG']

        Data.__init__(self, path_to_data,start_epoch,end_epoch)
        self._process_mat_file(target_events,nontarget_events)

    def _process_mat_file(self,target_events=None,nontarget_events=None):
        data_mat = loadmat(os.path.join(self.path_to_data, 'e4epochs.mat'))['alldata']
        # L and R - button position
        # BP - button pressed, BC - ball choosen, BM - ball moved, BB - ball klicked in blocked mode
        if target_events is None:
            self.target_events = ['L500BP','L500BC','L500BM','R500BP','R500BC','R500BM']
        else:
            self.target_events = target_events

        if nontarget_events is None:
            self.nontarget_events = ['L500BB','R500BB']
        else:
            self.nontarget_events = nontarget_events


        field_indices = {'L500BP':0, 'L500BC':1,'L500BM':2,'L500BB':3, 'R500BP':8, 'R500BC':9, 'R500BM':10, 'R500BB':11}
        self.subj_data = {}
        #Loading data omitting EOG channels
        for i in range(8):
            self.subj_data[i] = \
                {field_name:data_mat[0,i][field_indices[field_name]].transpose(2,1,0)[:,:,:19] \
                 for field_name in self.target_events + self.nontarget_events}


    def get_data(self,subjects,shuffle=False,windows=None,baseline_window=(),resample_to=None):
        '''

        :param subjects: list subject's numbers, wich data we want to load
        :param shuffle: bool
        :param windows: list of tuples. Each tuple contains two floats - start and end of window in seconds
        :param baseline_window:
        :param resample_to: int, Hz - new sample rate
        :return: Dict. {Subject_number:tuple of 2 numpy arrays: data (Trials x Time x Channels) and labels}
        '''
        res={}
        for subject in subjects:

            eegT = np.concatenate([self.subj_data[subject][field_name] for field_name in self.target_events],axis=0)
            eegNT = np.concatenate([self.subj_data[subject][field_name] for field_name in self.nontarget_events],axis=0)
            X = np.concatenate((eegT,eegNT),axis=0)
            if len(baseline_window):
                baseline = self._baseline_normalization(X,baseline_window)
                baseline = np.expand_dims(baseline,axis=1)
                X = X - baseline
            y = np.hstack((np.ones(eegT.shape[0]),np.zeros(eegNT.shape[0])))
            #y = np.hstack(np.repeat([[1,0]],eegT.shape[2],axis=0),np.repeat([[0,1]],eegT.shape[2],axis=0))

            if (resample_to is not None) and (resample_to != self.sample_rate):
                X, y = self._resample(X, y, resample_to)

            time_indices=[]
            if windows is not None:
                for win_start,win_end in windows:
                    start_window_ind = int((win_start - self.start_epoch)*self.sample_rate)
                    end_window_ind = int((win_end - self.start_epoch)*self.sample_rate)
                    time_indices.extend(range(start_window_ind,end_window_ind))
                X,y = X[:,time_indices,:],y

            if shuffle:
                X,y = data_shuffle(X,y)
            res[subject]=(X,y)
        return res


if __name__ == '__main__':
    data = DataBuildClassifier('/home/likan_blk/BCI/NewData/').get_data([33],shuffle=True,
                                                                               windows=[(0.2, 0.5)],baseline_window=(0.2, 0.3))
    
