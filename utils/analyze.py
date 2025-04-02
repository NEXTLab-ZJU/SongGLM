import glob
import music21
import os
from utils.melody_skeleton_extractor_v8 import Melody_Skeleton_Extractor
import pprint
from tqdm import tqdm
from multiprocessing.pool import Pool
import datetime
import pickle


RHYTHM_TYPE = ["DownBeat","Long","Split","DownBeat_Long","Long_Split","Normal"]
NOTE_KEY_TYPE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

class Note:
    def __init__(self, start, end, pitch, velocity, is_downbeat=False, is_long=False, is_split=False, is_peak = False, is_non_peak = False):
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.is_downbeat = is_downbeat
        self.is_long = is_long
        self.is_split = is_split
        self.key_scale = {0:"C",1:"C#",2:"D",3:"D#",4:"E",5:"F",6:"F#",7:"G",8:"G#",9:"A",10:"A#",11:"B"}
        self.key_name = self.key_scale[self.pitch % 12]
        self.is_peak = is_peak
        self.is_non_peak = is_non_peak
    
    def __repr__(self):
        return f'start:{self.start:<6} end:{self.end:<6} pitch:{self.pitch:<6} velocity:{self.velocity:<6} '\
               f'downbeat:{self.is_downbeat:<6} long:{self.is_long:<6} split:{self.is_split:<6} name:{self.key_name:<6}'\
               f'is_peak:{self.is_peak:<6} is_non_peak:{self.is_non_peak:<6}'

    @ property
    def type_v1(self):   # ["DownBeat_Long C"]
        # downbeat , long , split
        v1_dict = {(False,False,False):["Normal"],
                   (True,False,False):["DownBeat"],
                   (True,False,True):["Normal"],
                   (True,True,False):["DownBeat_Long"],
                   (True,True,True):["Normal"],
                   (False,True,False):["Long"],
                   (False,True,True):["Long_Split"],
                   (False,False,True):["Split"]}
        return [f'{feat} {self.key_name}' for feat in v1_dict[(self.is_downbeat,self.is_long,self.is_split)]]

    @ property
    def type_v2(self): # ["Long A","Split A","Long_Split A"]
        # downbeat , long , split
        v1_dict = {(False,False,False):["Normal"],
                   (True,False,False):["DownBeat"],
                   (True,False,True):["Normal"],
                   (True,True,False):["DownBeat","Long","DownBeat_Long"],
                   (True,True,True):["Normal"],
                   (False,True,False):["Long"],
                   (False,True,True):["Long","Split","Long_Split"],
                   (False,False,True):["Split"]}
        return [f'{feat} {self.key_name}' for feat in v1_dict[(self.is_downbeat,self.is_long,self.is_split)]]
    

# key_mode of a midi file
def key_analyze(midi_path):
    score = music21.converter.parse(midi_path)
    key = score.analyze('key')
    return key.tonic.name, key.mode


def is_melodic_peak(note_list):
    peak_list = []
    non_peak_list = []

    non_peak_list.append((note_list[0].start, note_list[0].end))

    for i in range(1, len(note_list)-1):
        window = note_list[i-1:i+2]
        if window[1].pitch > window[0].pitch and window[1].pitch >= window[2].pitch:
            peak_list.append((window[1].start,window[1].end))
        else:
            non_peak_list.append((window[1].start,window[1].end))
    
    if len(note_list) > 1:
        non_peak_list.append((note_list[-1].start, note_list[-1].end))

    return peak_list,non_peak_list

# 统计一个midi的音符特征分布
def note_analyze_per_midi(midi):
    # print(midi)
    # state
    # midi_state = {'key_mode':'','note_list':[]}
    midi_state = {'note_list':[]}
    # skeleton
    m = Melody_Skeleton_Extractor(midi)
    split_dict,_,_,_ = m._get_split()     # 切分音
    heavy_dict = m._get_stress()    # 节拍重音
    long_dict = m._get_long()       # 长音

    # key_mode
    # _,key_mode = key_analyze(midi)

    # rhythm visit_list
    split_vis = []
    heavy_vis = []
    long_vis = []


    for nl in split_dict.values():
        for note in nl:
            split_vis.append((note.start,note.end))
    for nl in heavy_dict.values():
        for note in nl:
            heavy_vis.append((note.start,note.end))
    for nl in long_dict.values():
        for note in nl:
            long_vis.append((note.start,note.end))
    
    # note_list
    notes = m.notes
    peak_list,non_peak_list = is_melodic_peak(notes)

    note_list = []
    for note in notes:
        ct = (note.start,note.end)
        note_list.append(Note(start=note.start, 
                              end=note.end, 
                              pitch=note.pitch, 
                              velocity=note.velocity, 
                              is_downbeat=ct in heavy_vis, 
                              is_long=ct in long_vis, 
                              is_split=ct in split_vis,
                              is_peak = ct in peak_list,
                              is_non_peak = ct in non_peak_list)
                              )
    
    # update
    # midi_state['key_mode'] = key_mode
    midi_state['note_list'] = note_list

    return midi_state


# 统计一个文件夹的音符特征分布，每个文件夹输出2张图(需要指定节奏分割的版本V1，V2)
# [1] Major [2] Minor
def note_analyze_per_dir(dir,algorithm_version = 1,tonal='auto'):

    print(f'>>>>>>>>>>>>>>>>> {dir}')

    # Initial
    exp_name = os.path.basename(dir)
    if algorithm_version not in [1,2]:
        raise

    # Statistic 
    rhythm_type = ["DownBeat","Long","Split","DownBeat_Long","Long_Split","Normal"]
    rhythm_type_simplify = ["D","L","S","DL","LS"]
    note_key_type = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    note_type = [f'{rt} {nt}' for rt in rhythm_type for nt in note_key_type]
    detail = {'song_num':0,'song_major_num':0,'song_minor_num':0,'note_num':0,'note_major_num':0,'note_minor_num':0}
    dict_dir = {'name':os.path.basename(dir),'major':{nt:0 for nt in note_type},'minor':{nt:0 for nt in note_type},'detail':detail}  

    # Loop - Multiprocess -> midi_states
    midi_files = glob.glob(f'{dir}/**/*.mid', recursive=True)
    pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    # pool = Pool(int(os.getenv('N_PROC', 1)))
    futures = [pool.apply_async(note_analyze_per_midi, args=[midi, algorithm_version]) for midi in midi_files]
    pool.close()
    midi_states = [x.get() for x in tqdm(futures)]  # [dict1, dict2 ,dictn]
    pool.join()

    # Analyze
    for ms in midi_states:
        key_mode = ms['key_mode'] if tonal=='auto' else tonal
        note_list = ms['note_list']

        # statistic & [update note info]
        for note in note_list:
            cur_type = note.type_v1 if algorithm_version==1 else note.type_v2
            for t in cur_type:
                dict_dir[key_mode][t] += 1
        
        # statistic & [update detail info]
        dict_dir['detail']['song_num'] += 1
        dict_dir['detail']['song_major_num'] += 1 if key_mode == 'major' else 0
        dict_dir['detail']['song_minor_num'] += 1 if key_mode == 'minor' else 0
        dict_dir['detail']['note_num'] += len(note_list)
        dict_dir['detail']['note_major_num'] += len(note_list) if key_mode == 'major' else 0 
        dict_dir['detail']['note_minor_num'] += len(note_list) if key_mode == 'minor' else 0

    return dict_dir

def note_analyze_multi_dir(dir_list,save_root,algorithm_version=1,tonal_list=None):

    # analyze per dir
    tonal_list = ['auto'] * len(dir_list) if tonal_list == None else tonal_list 
    results = []
    for d,tonal in zip(dir_list,tonal_list):
        results.append(note_analyze_per_dir(d,algorithm_version,tonal))

    # save
    current_datetime = datetime.datetime.now()

    day = current_datetime.day
    hour = current_datetime.hour
    minute = current_datetime.minute

    exp_dir = os.path.join(save_root,f'EXPv{algorithm_version}_{day}-{hour}-{minute}')
    os.mkdir(exp_dir)
    
    # [pkl]
    pkl_name = 'record.pkl'
    with open(os.path.join(exp_dir,pkl_name),'wb') as f:
        pickle.dump(results,f)

    # [txt]
    txt_name = 'record.txt'
    formatted_data = [pprint.pformat(d) for d in results]
    with open(os.path.join(exp_dir,txt_name),'w') as f:
        #dir
        f.write('>>>>>>>>>>>>>> Dir Input <<<<<<<<<<<<<\n')
        for d in dir_list:
            f.write(f'* {os.path.basename(d)}\n')

        for d in formatted_data:
            f.write('-'*50)
            f.write('\n')
            f.write(d)
            f.write('\n')

    print('>>>>>>>>>>> Finish <<<<<<<<<<<')
    print(f'* Save Dir: {exp_dir}')
