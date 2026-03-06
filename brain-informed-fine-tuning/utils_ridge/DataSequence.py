import numpy as np

class DataSequence(object):
    """DataSequence class provides a nice interface for handling data that is both continuous
    and discretely chunked. For example, semantic projections of speech stimuli must be
    considered both at the level of single words (which are continuous throughout the stimulus)
    and at the level of TRs (which contain discrete chunks of words).
    """
    def __init__(self, data, split_inds, data_times=None, tr_times=None):
        """Initializes the DataSequence with the given [data] object (which can be any iterable)
        and a collection of [split_inds], which should be the indices where the data is split into
        separate TR chunks.
        """
        self.data = data
        self.split_inds = split_inds
        self.data_times = data_times
        self.tr_times = tr_times
    
    @classmethod
    def from_grid(cls, grid_transcript, trfile):
        """Creates a new DataSequence from a [grid_transript] and a [trfile].
        grid_transcript should be the product of the 'make_simple_transcript' method of TextGrid.
        """
        data_entries = list(zip(*grid_transcript))[2]
        if isinstance(data_entries[0], str):
            data = list(map(str.lower, list(zip(*grid_transcript))[2]))
        else:
            data = data_entries
        word_starts = np.array(list(map(float, list(zip(*grid_transcript))[0])))
        word_ends = np.array(list(map(float, list(zip(*grid_transcript))[1])))
        word_avgtimes = (word_starts + word_ends)/2.0
        
        tr = trfile.avgtr
        trtimes = trfile.get_reltriggertimes()
        
        split_inds = [(word_starts<(t+tr)).sum() for t in trtimes][:-1]
        return cls(data, split_inds, word_avgtimes, trtimes+tr/2.0)