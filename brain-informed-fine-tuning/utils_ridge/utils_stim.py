from utils_ridge.dsutils import make_word_ds
from utils_ridge.stimulus_utils import load_textgrids, load_generic_trfiles

def get_story_wordseqs(stories, GRIDS_DIR, TRFILES_DIR):
    """loads words and word times of stimulus stories
    """
    textgrids_dir = GRIDS_DIR
    trfiles_dir = TRFILES_DIR
    
    grids = load_textgrids(stories, textgrids_dir)
    trfiles = load_generic_trfiles(stories, trfiles_dir)
    wordseqs = make_word_ds(grids, trfiles)
    return wordseqs