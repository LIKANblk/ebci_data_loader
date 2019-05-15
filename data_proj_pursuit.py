from data import Data
import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
import pickle

EEG_SAMPLE_RATE = 500 #Hz


def max_artifactfree_segment(eeg, track_time, thr, trigger_time):
    corrected_track_time = track_time.copy()


    for trial in range(eeg.shape[0]):
        t = 1
        maxs = eeg[trial, trigger_time, :].copy()
        mins = eeg[trial, trigger_time, :].copy()
        while t < track_time[trial]:
            current_sample = eeg[trial, trigger_time - t, :]
            mins_ind = current_sample < mins
            mins[mins_ind] = current_sample[mins_ind]

            maxs_ind = current_sample > maxs
            maxs[maxs_ind] = current_sample[maxs_ind]
            if np.any((maxs[maxs_ind] - mins[maxs_ind]) > thr) or np.any((maxs[mins_ind] - mins[mins_ind]) > thr):
                corrected_track_time[trial] = t - 1
                break
            t+=1
    return corrected_track_time



class DataProjPursuit(Data):
    def __init__(self,path_to_data,start_epoch = -1.2,end_epoch=1.2):
        super(DataProjPursuit, self).__init__(path_to_data,start_epoch,end_epoch)

    def _baseline_normalization(self,X,baseline_window=()):
        bl_start = int((baseline_window[0] - self.start_epoch) * self.sample_rate)
        bl_end = int((baseline_window[1] - self.start_epoch) * self.sample_rate)
        baseline = np.expand_dims(X[:, bl_start:bl_end, :].mean(axis=1), axis=1)
        X = X - baseline
        return X


    def train_test_split_event_eeg(
            self,
            subj,event,
            rej_thrs,
            resample_to,
            window=(-0.4,0),
            eeg_ch = range(19),
            baseline_window=(-0.1,0),
            augment=False,
            aug_step=0.05,
            lambd_delta = 100,
            shuffle=True,
            test_size = 0.2,
    ):
        '''

        :param test_size:
        :return:
        '''
        trigger_time = int((0 - self.start_epoch) * EEG_SAMPLE_RATE) + 1
        data_mat = loadmat('%s/subj %d/%s.mat' % (self.path_to_data, subj, event))
        eeg = data_mat['EEG']
        track_time = np.squeeze(data_mat['trackTime']).astype(int)-lambd_delta
        corrected_track_time = max_artifactfree_segment(eeg, track_time, rej_thrs, trigger_time)
        eeg = eeg[:, :, eeg_ch]
        eeg = eeg[corrected_track_time > (window[1] - window[0]), :, :]
        corrected_track_time = corrected_track_time[corrected_track_time > (window[1] - window[0])]

        if len(baseline_window):
            eeg = self._baseline_normalization(eeg, baseline_window)

        eeg_tr_ind,eeg_tst_ind = train_test_split(range(eeg.shape[0]),test_size=test_size,shuffle=shuffle)
        # eeg_tst_raw = eeg[eeg_tst_ind].copy()
        aug_eeg=np.empty((0,int(window[1]-window[0]*EEG_SAMPLE_RATE)+1,eeg.shape[2]))
        if augment:
            aug_eeg,_ = self.get_augmented_epochs(eeg[eeg_tr_ind], corrected_track_time[eeg_tr_ind], window, step=aug_step)

        if window is not None:
            start_window_ind = int((window[0] - self.start_epoch) * self.sample_rate)
            end_window_ind = int((window[1] - self.start_epoch) * self.sample_rate)
            eeg = eeg[:, start_window_ind:end_window_ind, :]

        if (resample_to is not None) and (resample_to != self.sample_rate):
            eeg = self._resample(eeg, resample_to)
            aug_eeg = self._resample(aug_eeg, resample_to)

        eeg_tr = eeg[eeg_tr_ind]
        eeg_tst = eeg[eeg_tst_ind]
        if shuffle:
            eeg_tr = sklearn.utils.shuffle(eeg_tr)

        return eeg_tr,aug_eeg,eeg_tst

    def get_event_data(self,subj,event,rej_thrs,resample_to,window=(-0.4,0),eeg_ch = range(19), baseline_window=(-0.1,0), shuffle=True):
        '''

        :param subject: subjects's index to load
        :param shuffle: bool
        :param windows: list of tuples. Each tuple contains two floats - start and end of window in seconds
        :param baseline_window:
        :param resample_to: int, Hz - new sample rate
        :return: numpy array trials x time x channels
        '''
        trigger_time = int((0-self.start_epoch) * EEG_SAMPLE_RATE) + 1
        data_mat = loadmat('%s/subj %d/%s.mat' % (self.path_to_data,subj, event))
        eeg = data_mat['EEG']
        track_time = np.squeeze(data_mat['trackTime']).astype(int)
        corrected_track_time = max_artifactfree_segment(eeg, track_time, rej_thrs, trigger_time)
        eeg = eeg[:,:,eeg_ch]
        eeg = eeg[corrected_track_time > (window[1] - window[0]),:,:]

        if len(baseline_window):
            eeg = self._baseline_normalization(eeg, baseline_window)

        if window is not None:
            start_window_ind = int((window[0] - self.start_epoch) * self.sample_rate)
            end_window_ind = int((window[1] - self.start_epoch) * self.sample_rate)
            eeg = eeg[:, start_window_ind:end_window_ind, :]


        if (resample_to is not None) and (resample_to != self.sample_rate):
            eeg = self._resample(eeg, resample_to)

        if shuffle:
            np.random.shuffle(eeg)

        return eeg

    # def get_data(self,subjects,shuffle=True,windows=None,baseline_window=(),resample_to=None,nt_events=('cl3.mat'),t_events = ('cl1.mat','cl4.mat')):
    #     '''
    #
    #     :param subjects: list subject's numbers, wich data we want to load
    #     :param shuffle: bool
    #     :param windows: list of tuples. Each tuple contains two floats - start and end of window in seconds
    #     :param baseline_window:
    #     :param resample_to: int, Hz - new sample rate
    #     :return: Dict. {Subject_number:tuple of 2 numpy arrays: data (Trials x Time x Channels) and labels}
    #     '''
    #     res={}
    #     for subject in subjects:
    #
    #         eegT = np.concatenate([loadmat(os.path.join(self.path_to_data, str(subject), '%s.mat' %t_ev))['data'] for t_ev in t_events],
    #                               axis=0)
    #
    #         eegNT = np.concatenate([loadmat(os.path.join(self.path_to_data, str(subject), '%s.mat' % nt_ev))['data'] for nt_ev in nt_events],
    #             axis=0)
    #
    #         X = np.concatenate((eegT,eegNT),axis=0)
    #         if len(baseline_window):
    #             X = self._baseline_normalization(X,baseline_window)
    #         y = np.hstack((np.ones(eegT.shape[0]),np.zeros(eegNT.shape[0])))
    #         #y = np.hstack(np.repeat([[1,0]],eegT.shape[2],axis=0),np.repeat([[0,1]],eegT.shape[2],axis=0))
    #         time_indices=[]
    #         if windows is not None:
    #             for win_start,win_end in windows:
    #                 start_window_ind = int((win_start - self.start_epoch)*self.sample_rate)
    #                 end_window_ind = int((win_end - self.start_epoch)*self.sample_rate)
    #                 time_indices.extend(range(start_window_ind,end_window_ind))
    #             X,y = X[:,time_indices,:],y
    #
    #         if (resample_to is not None) and (resample_to != self.sample_rate):
    #             X, y = self._resample(X, y, resample_to)
    #
    #         if shuffle:
    #             X,y = data_shuffle(X,y)
    #         res[subject]=(X,y)
    #     return res

    def get_augmented_epochs(self, eeg, track_times, window=(-0.4, 0), step=0.05):

        start_window_ind = int((window[0] - self.start_epoch) * self.sample_rate)
        end_window_ind = int((window[1] - self.start_epoch) * self.sample_rate)
        step = int(step * self.sample_rate)

        # data_mat = loadmat(os.path.join(self.path_to_data, str(subject), '%s.mat' % t_event))
        # eeg = data_mat['data']
        # track_time = data_mat['durations'] # measured in samples
        aug_epochs = []
        inds = []
        for ind, track_time in enumerate(track_times):
            delta = track_time - (end_window_ind - start_window_ind)
            if delta > 0:
                for shift in range(0, delta, step):
                    inds.append(ind)
                    aug_epochs.append(eeg[ind, (start_window_ind - shift):(end_window_ind - shift), :])
        return np.stack(aug_epochs, axis=0),inds


    def get_ampl_features_from_eeg(self,eeg,times_beg=None):
        trials = eeg.shape[0]
        chan_num = eeg.shape[2]
        if times_beg is None:
            times_beg = np.arange(-0.3, -0.05, 0.02)
        times_end = times_beg + 0.05
        ts_beg = np.round((times_beg - self.start_epoch) * self.sample_rate).astype(np.int32) - 1
        ts_end = np.round((times_end - self.start_epoch) * self.sample_rate).astype(np.int32)
        x = np.zeros((trials, chan_num,len(times_beg)))

        for t in range(len(ts_beg)):
            x[:,:,t] = np.mean(eeg[:, ts_beg[t]:ts_end[t], :], axis=1)
        return x.reshape((x.shape[0],-1))

if __name__ == '__main__':
    data = DataProjPursuit('/home/likan_blk/BCI/DataProjPursuit/')
    # all_subjects = ['subj%d' % i for i in range(1, 7)]
    subj = 10
    # cl='cl3'
    # data_mat_cl3 = loadmat('/home/likan_blk/BCI/DataProjPursuit/subj %d/%s.mat' % (subj, cl))
    # eeg_cl3 = data_mat_cl3['EEG']
    # eeg_cl3 = data._baseline_normalization(eeg_cl3, baseline_window=(-0.1, 0))
    # track_time_cl3 = data_mat_cl3['trackTime'].squeeze().astype(int)-100
    # corrected_track_time_cl3 = max_artifactfree_segment(eeg_cl3, track_time_cl3, thr=100, trigger_time=601)
    # aug_epoch_cl3,inds_cl3 = data.get_augmented_epochs(eeg_cl3, corrected_track_time_cl3, window=(-0.4, 0), step=0.05)
    # plt.plot(aug_epoch_cl3[:,:,12].mean(axis=0))
    #
    # cl = 'cl1'
    # data_mat_cl1 = loadmat('/home/likan_blk/BCI/DataProjPursuit/subj %d/%s.mat' % (subj, cl))
    # eeg_cl1 = data_mat_cl1['EEG']
    # eeg_cl1 = data._baseline_normalization(eeg_cl1, baseline_window=(-0.1, 0))
    # track_time_cl1 = data_mat_cl1['trackTime'].squeeze().astype(int)
    # corrected_track_time_cl1 = max_artifactfree_segment(eeg_cl1, track_time_cl1, thr=100, trigger_time=601)
    # aug_epoch_cl1,inds_cl1 = data.get_augmented_epochs(eeg_cl1, corrected_track_time_cl1, window=(-0.4, 0), step=0.05)
    # plt.plot(aug_epoch_cl1[:,:,12].mean(axis=0))
    #
    # cl = 'cl4'
    # data_mat_cl4 = loadmat('/home/likan_blk/BCI/DataProjPursuit/subj %d/%s.mat' % (subj, cl))
    # eeg_cl4 = data_mat_cl4['EEG']
    # eeg_cl4 = data._baseline_normalization(eeg_cl4, baseline_window=(-0.1, 0))
    # track_time_cl4 = data_mat_cl4['trackTime'].squeeze().astype(int)
    # corrected_track_time_cl4 = max_artifactfree_segment(eeg_cl4, track_time_cl4, thr=100, trigger_time=601)
    # aug_epoch_cl4, inds_cl4 = data.get_augmented_epochs(eeg_cl4, corrected_track_time_cl4, window=(-0.4, 0), step=0.05)
    # plt.plot(aug_epoch_cl4[:, :, 12].mean(axis=0))
    # print(1)
    cl3_eeg_tr,cl3_eeg_aug_tr,cl3_eeg_tst= \
        data.train_test_split_event_eeg(subj, 'cl3', rej_thrs=100, resample_to=None, window=(-0.4, 0), eeg_ch=range(19),
                               baseline_window=(-0.1, 0), augment=False, aug_step=0.05, lambd_delta=100, shuffle=True,
                               test_size=0.2)
    # cl3_eeg = data.get_event_data(subj, 'cl3', rej_thrs=100, resample_to=None, window=(-0.4, 0),
    #                               baseline_window=(-0.1, 0), augment=True, aug_step=0.05, shuffle=True)
    # plt.plot(aug_epoch_cl3[:,:,12].mean(axis=0))
    plt.plot(cl3_eeg[:,:,12].mean(axis=0))
    plt.show()