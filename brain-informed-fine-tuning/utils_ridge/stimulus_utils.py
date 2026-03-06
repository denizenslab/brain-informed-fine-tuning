from utils_ridge.textgrid import TextGrid
import os
import numpy as np
import logging as logger
import re

def load_textgrids(stories, data_dir: str):
    grids = {}
    for story in stories:
        grid_path = os.path.join(data_dir, "%s.TextGrid" % story)
        if not os.path.exists(grid_path): 
            grid_path = os.path.join(data_dir, "%s.TextGrid" % story)
        grids[story] = TextGrid(open(grid_path).read())
    return grids


class TRFile(object):
    def __init__(self, trfilename, expectedtr=2.0045):
        """Loads data from [trfilename], should be output from stimulus presentation code.
        """
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr
        self.frametimes = []
        self.trdata_aslist = None

        if trfilename is not None:
            self.load_from_file(trfilename)
        

    def load_from_file(self, local_filepath):
        """Loads TR data from report with given [trfilename].

        Parameters:

        s3_filepath: str
            Path to trfilename on cloud.
        """

        trdata_aslist = []
        for ll in open(local_filepath, encoding="utf-8"):
            trdata_aslist.append(ll)
        self.trdata_aslist = trdata_aslist
        logger.info(f"TRFile {local_filepath} is loaded from local path")

        # Read the report file and populate the datastructure
        for idx, ll in enumerate(trdata_aslist):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)
            if label.startswith("word") or label.startswith("START: word"):
                self.frametimes.append((time, label))
            if re.match(r"^-?\d+(?:\.\d+)$", ll.strip()):
                continue  # because some timestamps incorrectly on newline.
            if label in ("init-trigger", "trigger") or label.startswith("first"):
                self.trtimes.append(time)

            elif (
                label == "sound-start"
                or label.startswith("start")
                or label.startswith("START")
            ):
                self.soundstarttime = time
            elif label.split()[0] == "word" and self.soundstarttime == -1:
                self.soundstarttime = time

            elif label == "sound-stop" or label.startswith("END"):
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))

        # Fix weird TR times
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes > (itrtimes.mean() * 1.5))[0]
        newtrs = []
        for btr in badtrtimes:
            logger.info("badtrtimes are fixed")
            # Insert new TR where it was missing..
            newtrtime = self.trtimes[btr] + self.expectedtr
            newtrs.append((newtrtime, btr))

        for ntr, btr in newtrs:
            self.trtimes.insert(btr + 1, ntr)

    def simulate(self, ntrs):
        """Simulates [ntrs] TRs that occur at the expected TR.
        """
        self.trtimes = list(np.arange(ntrs)*self.expectedtr)
    
    def get_reltriggertimes(self):
        """Returns the times of all trigger events relative to the sound.
        """
        return np.array(self.trtimes)-self.soundstarttime

    @property
    def avgtr(self):
        """Returns the average TR for this run.
        """
        return np.diff(self.trtimes).mean()

def load_generic_trfiles(stories, root="data/trfiles", language="en"):
    """Loads a dictionary of generic TRFiles (i.e. not specifically from the session
    in which the data was collected.. this should be fine) for the given stories.
    """
    trdict = dict()

    for story in stories:
        try:
            trf_path = os.path.join(root, f"{story}.report")
            if not os.path.exists(trf_path): 
                if language == "en":
                    trf_path = os.path.join(root, f"{story}.report")
            trf = TRFile(trf_path)   
            trdict[story] = [trf]
        except Exception as e:
            print (e)

    return trdict
