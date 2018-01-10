import glob
import os
import xml.etree.ElementTree as ET


BNC_ROOT = os.path.expanduser('../data/raw/BNC')
DATA_DIR_ACA = os.path.join(BNC_ROOT, 'aca')
DATA_DIR_DEM = os.path.join(BNC_ROOT, 'dem')
DATA_DIR_FIC = os.path.join(BNC_ROOT, 'fic')
DATA_DIR_NEWS= os.path.join(BNC_ROOT, 'news')
DATA_DIRS  = [ DATA_DIR_ACA, DATA_DIR_DEM, DATA_DIR_FIC, DATA_DIR_NEWS ]

# DATA_DIRS = [DATA_DIR_ACA]


OUT_ROOT = os.path.expanduser('../data/raw/processed')
DATA_OUT = os.path.join(OUT_ROOT, 'bnc.txt')
def GetBNC():
    out = "../input/train/word_emb/BNC.txt"
    gf = open(out, 'w', encoding ='utf8', errors='ignore')

    for data_dir in DATA_DIRS:
        all_files = [ (data_dir, x) for x in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, x)) ]
        for a_file in all_files:
            file = os.path.join(a_file[0], a_file[1])
            Corpus = glob.glob(file)
            for c in Corpus:
                tree = ET.parse(c)
                root = tree.getroot()
                for sen in root.iter('s'):
                    s = ""
                    for word in sen:
                        w = word.text
                        if w != None:
                            gf.write(w)
                    gf.write("\n")
GetBNC()
