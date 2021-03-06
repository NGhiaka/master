# http://www.nltk.org/howto/corpus.html#corpus-reader-classes
# Xử lý bộ ngữ liệu semcor3.0
# đầu tiên chuyển về dạng file xml
# Chuyển về dạng file text, chia từng câu, lấy các nhãn
#============THAM KHẢO===============
# https://github.com/letuananh/pysemcor/blob/master/semcortk.py


import os
import sys
import argparse
import lxml
from lxml import etree
import xml.etree.ElementTree as ET
from collections import namedtuple, Counter
from timeit import Timer
from bs4 import BeautifulSoup  #pip install bs4
import nltk
import random


from  chirptext.leutile import *   #https://github.com/letuananh/chirptext

TokenInfo = namedtuple("TokenInfo", ['text', 'sk'])
SEMCOR_ROOT = os.path.expanduser('../data/raw/semcor3.0')
DATA_DIR_1 = os.path.join(SEMCOR_ROOT, 'brown1', 'tagfiles')
DATA_DIR_2 = os.path.join(SEMCOR_ROOT, 'brown2', 'tagfiles')
DATA_DIR_V = os.path.join(SEMCOR_ROOT, 'brownv', 'tagfiles')
DATA_DIRS  = [ DATA_DIR_1, DATA_DIR_2, DATA_DIR_V ]

#print(DATA_RAW_DIR_1)
SEMCOR_FIXED_ROOT = os.path.expanduser('../data/xml/semcor3.0')
DATA_DIR_1_FIXED  = os.path.join(SEMCOR_FIXED_ROOT, 'brown1', 'tagfiles')
DATA_DIR_2_FIXED  = os.path.join(SEMCOR_FIXED_ROOT, 'brown2', 'tagfiles')
DATA_DIR_V_FIXED  = os.path.join(SEMCOR_FIXED_ROOT, 'brownv', 'tagfiles')

OUTPUT_DIRS = {
    DATA_DIR_1 : SEMCOR_FIXED_ROOT
    ,DATA_DIR_2 : SEMCOR_FIXED_ROOT
    ,DATA_DIR_V : SEMCOR_FIXED_ROOT
}
XML_DIR = SEMCOR_FIXED_ROOT
SEMCOR_RAW = os.path.expanduser('../data/processed/semcor/semcor.txt')
SEMCOR_TAB = os.path.expanduser('../data/processed/semcor/semcor_wn30.txt')
SEMCOR_TAG = os.path.expanduser('../data/processed/semcor/semcor_wn30.tag')
SEMCOR_LLTP = os.path.expanduser('../data/processed/semcor/semcor_lltp.txt')
SEMCOR_LLDEV = os.path.expanduser('../data/processed/semcor/semcor_lldev.txt') # subset of the full Semcor for LeLesk development
SS_SK_MAP = os.path.expanduser('../data/processed/semcor/sk_map_ss.txt')
SK_NOTFOUND = os.path.expanduser('../data/processed/semcor/sk_notfound.txt')
SEMCOR_TXT = os.path.expanduser('../data/processed/semcor/semcor_wn30_tokenized.txt')
multi_semcor_aligned = ["br-a01", "br-a11", "br-a12", "br-a13", "br-a14", "br-b13", "br-b20", "br-c01", "br-c02", "br-c04", "br-d01", "br-d02", "br-d03", "br-e01", "br-e04", "br-e23", "br-e24", "br-e27", "br-e28", "br-e29", "br-e30", "br-f03", "br-f10", "br-f14", "br-f15", "br-f16", "br-f19", "br-f22", "br-f23", "br-f24", "br-f25", "br-f43", "br-g11", "br-g12", "br-g14", "br-g15", "br-g16", "br-g17", "br-g18", "br-g21", "br-g22", "br-g23", "br-g39", "br-g43", "br-h01", "br-h13", "br-h14", "br-h16", "br-h17", "br-h18", "br-h21", "br-j01", "br-j03", "br-j04", "br-j05", "br-j10", "br-j17", "br-j22", "br-j23", "br-j29", "br-j30", "br-j31", "br-j33", "br-j34", "br-j35", "br-j37", "br-j38", "br-j41", "br-j42", "br-j52", "br-j53", "br-j55", "br-j57", "br-j58", "br-j60", "br-k01", "br-k02", "br-k03", "br-k05", "br-k08", "br-k10", "br-k11", "br-k13", "br-k15", "br-k18", "br-k19", "br-k21", "br-k22", "br-k24", "br-k26", "br-k29", "br-l08", "br-l10", "br-l11", "br-l12", "br-l14", "br-l16", "br-l18", "br-m01", "br-m02", "br-n05", "br-n09", "br-n12", "br-n15", "br-n17", "br-n20", "br-p01", "br-p07", "br-p09", "br-p10", "br-p12", "br-p24", "br-r04", "br-r06", "br-r07", "br-r08", 'br-r09']

def fix_malformed_xml_file(filepathchunks, postfix='.xml'):
    file_name = filepathchunks[1]
    input_file_path = os.path.join(*filepathchunks)
    output_dir = OUTPUT_DIRS[filepathchunks[0]]
    output_file_path = os.path.join(output_dir, file_name + postfix)

    print('Fixing the file: %s ==> %s' % (input_file_path ,output_file_path))
    soup = BeautifulSoup(open(input_file_path).read())

    # Create output dir if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file_path, 'w') as output_file:
        output_file.write(soup.prettify())

def convert_file(file_name, semcor_txt, semcor_raw=None, semcor_tag=None, semcor_tab=None):
    #print('Loading %s' %file_name)

    tree = etree.iterparse(file_name)
    for event, element in tree:
        if event == 'end' and element.tag == 's':
            fcode = os.path.basename(file_name)
            fcode = os.path.basename(fcode)[:-4] if fcode.endswith('.xml') else fcode
            scode = fcode + '-' + str(element.get('snum'))
            #print("Found a sentence (length = %s) - sid = %s" % (len(element), scode,) )

            # Generate TAB file with tags
            tokens = []
            for token in element:
                if token.tag == 'wf':
                    lemma = StringTool.strip(token.get('lemma'))
                    lexsn = StringTool.strip(token.get('lexsn'))
                    sk = lemma + '%' + lexsn if lemma and lexsn else ''
                    sk = StringTool.strip(sk.replace('\t', ' ').replace('|', ' '))
                    text = StringTool.strip(token.text.replace('\t', ' ').replace('|', ' ').replace('_', ' '))
                    tokens.append(TokenInfo(text, sk))
                elif token.tag == 'punc':
                    tokens.append(TokenInfo(token.text.strip(), ''))
            element.clear()

            tokens_text = '\t'.join([ x.text + '|' + x.sk for x in tokens])
            semcor_txt.write(tokens_text + '\n')

            # Generate raw file
            if semcor_tab:
                sentence_text = ' '.join([ x.text for x in tokens ])
                sentence_text = sentence_text.replace(" , , ", ", ")
                sentence_text = sentence_text.replace(' , ', ', ').replace('`` ', ' “').replace(" ''", '”')
                sentence_text = sentence_text.replace(' ! ', '! ').replace(" 'll ", "'ll ").replace(" 've ", "'ve ").replace(" 're ", "'re ").replace(" 'd ", "'d ")
                sentence_text = sentence_text.replace(" 's ", "'s ")
                sentence_text = sentence_text.replace(" 'm ", "'m ")
                sentence_text = sentence_text.replace(" ' ", "' ")
                sentence_text = sentence_text.replace(" ; ", "; ")
                sentence_text = sentence_text.replace("( ", "(")
                sentence_text = sentence_text.replace(" )", ")")
                sentence_text = sentence_text.replace(" n't ", "n't ")
                sentence_text = sentence_text.replace("Never_mind_''", "Never_mind_”")
                sentence_text = sentence_text.replace("327_U._S._114_''", "327_U._S._114_”")
                sentence_text = sentence_text.replace("``", "“")
                sentence_text = sentence_text.replace("''", "”")
                if sentence_text[-2:] in (' .', ' :', ' ?', ' !'):
                    sentence_text = sentence_text[:-2] + sentence_text[-1]
                sentence_text = sentence_text.strip()

                # Generate mapping file
                if semcor_tag:
                    cfrom = 0
                    cto = len(sentence_text)
                    stags = []
                    strace = []
                    previous_token = ''
                    for token in tokens:
                        tokentext = token.text.replace('``', '“').replace("''", '”')
                        if ',' == tokentext and ',' == previous_token:
                            print("WARNING: Duplicate punc (,) at %s" % ("('%s', %d)" % (scode, cfrom),))
                            continue
                        strace.append("looking for '%s' from '%s' - %s" % (tokentext, cfrom, "('%s', %d)" % (scode, cfrom)))
                        tokenfrom = sentence_text.find(tokentext, cfrom)
                        if cfrom == 0 and tokenfrom != 0:
                            print("WARNING: Sentence starts at %s instead of 0 - sid = %s [sent[0] is |%s|]" % (tokenfrom, scode, sentence_text[0]))
                        if tokenfrom == -1:
                            print("WARNING: Token not found (%s) in %s from %s |%s|" % (tokentext, scode, cfrom, sentence_text))
                            for msg in strace[-4:]:
                                print(msg)
                            return
                        else:
                            cfrom = tokenfrom + len(tokentext)
                            if token.sk:
                                stags.append((scode, tokenfrom, cfrom,token.sk,tokentext,))
                        # Finished processing this token
                        previous_token = tokentext
                    if cfrom != cto:
                        print("WARNING: Sentence length is expected to be %s but found %s lasttoken=|%s| (sid = %s)" % (cto, cfrom, tokentext, scode))
                        print("Debug info: %s" % (stags,))
                    for tag in stags:
                        semcor_tag.write('\t'.join([ str(x) for x in tag]) + '\n')
                # Done!
                semcor_tab.write(scode + '\t')
                semcor_tab.write(sentence_text + '\n')
            if semcor_raw:
                semcor_raw.write(sentence_text + '\n')
                pass

def fix_data():
    t = Timer()
    c = Counter()
    for data_dir in DATA_DIRS:
        all_files = [ (data_dir, x) for x in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, x)) ]
        for a_file in all_files:
            fix_malformed_xml_file(a_file)
            c.count('file')

def gen_text(only_multi_semcor=False):
    with open(SEMCOR_TAG, 'w') as semcor_tag, open(SEMCOR_RAW, 'w') as semcor_raw, open(SEMCOR_TXT, 'w') as semcor_txt, open(SEMCOR_TAB, 'w') as semcor_tab:
        all_files = [ os.path.join(XML_DIR, x) for x in os.listdir(XML_DIR) if os.path.isfile(os.path.join(XML_DIR, x)) ]
        if only_multi_semcor:
            all_files = [ x for x in all_files if os.path.splitext(os.path.basename(x))[0] in multi_semcor_aligned ]
        print("Processing %s file(s) ..." % len(all_files))
        for file_name in all_files:
            convert_file(file_name, semcor_txt, semcor_raw, semcor_tag, semcor_tab)

def sk_to_ss():
    """Update sensekey in tag file to synsetID (offset-pos)"""
    all_sk = set()
    print("Reading tag file ...")
    with open(SEMCOR_TAG, 'r') as semcor_tag:
        lines = [ x.split() for x in semcor_tag.readlines() ]
    for line in lines:
        sk = line[3]
        scloc = sk.find(';')
        if scloc > -1:
            sk = sk[:scloc] # only consider the first sensekey
        all_sk.add(sk)
    print(len(all_sk))

    print("Loading WordNet ...")
    from nltk.corpus import wordnet as wn
    all_sk_notfound = set()
    with open(SS_SK_MAP, 'w') as mapfile:
        for sk in all_sk:
            try:
                if sk not in all_sk_notfound:
                    ss = wn.lemma_from_key(sk).synset()
                    sid = '%s-%s' % (ss.offset(), ss.pos())
                    mapfile.write('%s\t%s\n' % (sk, sid))
            except nltk.corpus.reader.wordnet.WordNetError:
                all_sk_notfound.add(sk)
            except ValueError:
                print("Invalid sk: %s" % (sk,))
                all_sk_notfound.add('[INVALID]\t' + sk)
    with open(SK_NOTFOUND, 'w') as notfoundfile:
        for sk in all_sk_notfound:
            notfoundfile.write(sk)
            notfoundfile.write('\n')
    print("Map file has been created")

def multi_semcor():
    gen_text(True)
    sk_to_ss()

def generate_lelesk_test_profile():
    '''Generate test profile for lelesk (new format 31st Mar 2015)
    '''
    # Read all senses
    sensemap = {}
    with open(SS_SK_MAP, 'r') as mapfile:
        for mapitem in mapfile:
            # format: hotshot%1:18:00:: 9762509-n
            parts = [ x.strip() for x in mapitem.split('\t') ]
            if len(parts) == 2:
                sk, sid = parts
                sensemap[sk] = sid

    # Read all tags
    tagmap = {} # sentence ID > tags list
    TagInfo = namedtuple('TagInfo', 'sentid cfrom cto sk word'.split())
    with open(SEMCOR_TAG, 'r') as tags:
        for tag in tags:
            # br-k22-1  8   11  not%4:02:00::   not
            parts = [ x.strip() for x in tag.split('\t') ]
            if len(parts) == 5:
                tag = TagInfo(*parts)
                if tag.sk in sensemap:
                    # there is synset id for this sensekey ...
                    if tag.sentid in tagmap:
                        tagmap[tag.sentid].append(tag)
                    else:
                        tagmap[tag.sentid] = [tag]

    # build test profile
    sentences = []
    with open(SEMCOR_RAW, 'r') as lines:
        for line in lines:
            if line.startswith('#'):
                continue
            else:
                parts = [ x.strip() for x in line.split('\t') ]
                if len(parts) == 2:
                    sid, sent = parts
                    if sid in tagmap:
                        print(sent)
                        # found tags
                        for tag in tagmap[sid]:
                            sentences.append((tag.word, sensemap[tag.sk], sent))
                            # print("%s - %s" % (tag.word, sensemap[tag.sk]))

    # write to file
    with open(SEMCOR_LLTP, 'w') as outputfile:
        for sentence in sentences:
            outputfile.write("%s\t%s\t%s\n" % sentence)

    # write dev profile
    random.seed(31032015)
    itemids = sorted(random.sample(range(114341), 1000))
    with open(SEMCOR_LLDEV, 'w') as outputfile:
        for itemid in itemids:
            outputfile.write("%s\t%s\t%s\n" % sentences[itemid])
    pass

def main():
    fix_data()
    multi_semcor()


if __name__ == "__main__":
    main()
    print("All done!")
