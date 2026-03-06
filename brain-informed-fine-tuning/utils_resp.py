import numpy as np
import os
from utils_ridge.utils import load_data

class ResponseUtils:
    def __init__(self) -> None:
        self.stories_for_encoding = ['odetostepfather', 'souls', 'myfirstdaywiththeyankees']
        
        # Test story
        self.test_story = ['wheretheressmoke']

        #Stories for fine-tuning
        self.stories = {'en': ['alternateithicatom', 'avatar', 'howtodraw',  'legacy', 'life', 'naked' , 'undertheinfluence']}

    def zscore(self, a, mean=None, std=None, return_info=False):
        print(a.shape)
        # Z-score normalization
        # a is [TRs x voxels]
        EPSILON = 0.000000001
        if type(mean) != np.ndarray:
            mean = np.nanmean(a, axis=0)
        if type(std) != np.ndarray:
            std = np.nanstd(a, axis=0)
        if not return_info:
            return (a - np.expand_dims(mean, axis=0)) / (np.expand_dims(std, axis=0) + EPSILON)
        
        return (a - np.expand_dims(mean, axis=0)) / (np.expand_dims(std, axis=0) + EPSILON), mean, std
        
    def load_subject_fMRI(self, fdir, subject, modality):
        """Loads fMRI data for a subject
        Returns training and test data
        """
        fname_tr5 = os.path.join(fdir, 'subject{}_{}_fmri_data_trn.hdf'.format(subject, modality))
        trndata5 = load_data(fname_tr5)
        print(trndata5.keys())

        fname_te5 = os.path.join(fdir, 'subject{}_{}_fmri_data_val.hdf'.format(subject, modality))
        tstdata5 = load_data(fname_te5)
        print(tstdata5.keys())
        
        trim = 5
        zRresp = np.vstack([self.zscore(trndata5[story][5+trim:-trim-5,:]) for story in trndata5.keys()])
        zPresp = np.vstack([self.zscore(tstdata5[story][0][5+trim:-trim-5,:]) for story in tstdata5.keys()])
        
        return zRresp, zPresp