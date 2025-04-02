import os
import argparse
import numpy as np
from collections import Counter
from math import gcd
import numpy as np
from typing import List, Union
from multiprocessing.pool import Pool
import miditoolkit
from miditoolkit import Note, Lyric, MidiFile
from functools import reduce
from dtw import accelerated_dtw
from tqdm import tqdm
import statistics
import sklearn
from scipy import integrate, stats
from sklearn.model_selection import LeaveOneOut


class Metrics(object):

    def __init__(self, original_midi_files: List[str], generated_midi_files: List[str]) -> None:
        self.original_midi_files = original_midi_files
        self.generated_midi_files = generated_midi_files
        self.alignment_max = 6

    # ------------------------------------------------------------------------------------------------------------------------------
    # OA Computer
    # ------------------------------------------------------------------------------------------------------------------------------
    def c_dist(self, A, B, mode="None", normalize=0):
        '''Calculate the distance between array A and each element in array B.'''
        c_dist = np.zeros(len(B))
        for i in range(0, len(B)):
            if mode == "None":
                # Euclidean distance
                c_dist[i] = np.linalg.norm(A - B[i])
            elif mode == "EMD":
                # Wasserstein distance
                if normalize == 1:
                    A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm="l1")[0]
                    B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm="l1")[0]
                else:
                    A_ = A.reshape(1, -1)[0]
                    B_ = B[i].reshape(1, -1)[0]

                c_dist[i] = stats.wasserstein_distance(A_, B_)

            elif mode == "KL":
                if normalize == 1:
                    A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm="l1")[0]
                    B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm="l1")[0]
                else:
                    A_ = A.reshape(1, -1)[0]
                    B_ = B[i].reshape(1, -1)[0]

                B_[B_ == 0] = 0.00000001
                c_dist[i] = stats.entropy(A_, B_)
        return c_dist

    def cross_valid(self, A: np.ndarray, B: np.ndarray):
        loo = LeaveOneOut()
        num_samples = len(A)
        loo.get_n_splits(np.arange(num_samples))
        result = np.zeros((num_samples, num_samples))
        for _, test_index in loo.split(np.arange(num_samples)):
            result[test_index[0]] = self.c_dist(A[test_index], B)
        return result.flatten()

    def overlap_area(self, A, B):
        """Calculate overlap between the two PDF"""
        pdf_A = stats.gaussian_kde(A)
        pdf_B = stats.gaussian_kde(B)
        return integrate.quad(
            lambda x: min(pdf_A(x), pdf_B(x)), np.min((np.min(A), np.min(B))), np.max((np.max(A), np.max(B)))
        )[0]

    def compute_oa(self, generated_metrics: np.ndarray, test_metrics: np.ndarray):
        inter = self.cross_valid(generated_metrics, test_metrics)
        intra_generated = self.cross_valid(generated_metrics, generated_metrics)
        oa = self.overlap_area(intra_generated, inter)
        return oa


    ### 1. Generated Melody V.S. Ground Truth
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OA(A) - Alignment Distribution Similarity
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    def cal_A_job(self, midi_file):
        midi = miditoolkit.MidiFile(midi_file)
        notes = midi.instruments[0].notes
        lyrics = midi.lyrics
        histogram = [0] * self.alignment_max
        mixed_tokens: List[Union[Note, Lyric]] = []
        mixed_tokens.extend(notes)
        for lyric in lyrics:
            for token_index, token in enumerate(mixed_tokens):
                if type(token) != Note:
                    continue
                if token.start < lyric.time:
                    continue
                else:
                    mixed_tokens.insert(token_index, lyric)
                    break
        
        note_cnt = 0
        for idx, token in enumerate(mixed_tokens):
            if idx == 0:
                continue
            if type(token) == Note:
                note_cnt += 1
            elif type(token) == Lyric:
                if note_cnt == 0:
                    continue
                elif note_cnt >= self.alignment_max:
                    note_cnt = self.alignment_max
                histogram[note_cnt - 1] += 1
                note_cnt = 0
        if note_cnt != 0:
            if note_cnt >= self.alignment_max:
                note_cnt = self.alignment_max
            histogram[note_cnt - 1] += 1
            note_cnt = 0
        
        return np.array(histogram)
    
    def cal_A(self, midi_files):
        pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
        futures = [pool.apply_async(self.cal_A_job, args=[midi_file]) for midi_file in midi_files]
        pool.close()
        alignment_list = np.stack([x.get() for x in futures])  # 显示进度
        pool.join()
        return alignment_list

    def compute_oa_A(self):
        print(f"OA_A metrics in {os.path.dirname(self.generated_midi_files[0])} and {os.path.dirname(self.original_midi_files[0])} for {len(self.generated_midi_files)} and {len(self.original_midi_files)} files")
        
        ai_song_a_list = self.cal_A(self.generated_midi_files)
        human_song_a_list = self.cal_A(self.original_midi_files)
        oa_a = self.compute_oa(ai_song_a_list, human_song_a_list)
        
        return oa_a

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OA(PD) / OA(PCH) - Pitch Distribution Similarity
    # the average overlapped area of the Pitch Class Histogram (PCH) distribution between AI-generated and human-composed melodies. 
    # PCH is a pitch-based feature to evaluate the overall tonal distribution of a piece of music.
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    def cal_PCH_job(self, midi_file):
        midi = miditoolkit.MidiFile(midi_file)
        notes = midi.instruments[0].notes
        notes = sorted(notes, key=lambda x:x.start)
        pitch_classes = [note.pitch % 12 for note in notes]
        frequency = Counter(pitch_classes)
        histogram = [frequency[i]/len(notes) for i in range(12)]
        return np.array(histogram)

    def cal_PCH(self, midi_files):
        pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
        futures = [pool.apply_async(self.cal_PCH_job, args=[midi_file]) for midi_file in midi_files]
        pool.close()
        pch_list = np.stack([x.get() for x in futures])  # 显示进度
        pool.join()
        return pch_list
    
    def compute_oa_PCH(self):
        print(f"OA_PCH metrics in {os.path.dirname(self.generated_midi_files[0])} and {os.path.dirname(self.original_midi_files[0])} for {len(self.generated_midi_files)} and {len(self.original_midi_files)} files")
        
        ai_song_pch_list = self.cal_PCH(self.generated_midi_files)
        human_song_pch_list = self.cal_PCH(self.original_midi_files)
        oa_pch = self.compute_oa(ai_song_pch_list, human_song_pch_list)
        
        return oa_pch


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OA(IOI) - Rhythm Distribution Similarity
    # the average overlapped area of the Inter-Onset-Interval (IOI) distribution between AI-generated and human-composed melodies. 
    # IOI measures the time interval between the onsets of consecutive notes, which provides insights into the overall rhythmic patterns of a piece of music.
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    def cal_IOI_job(self, midi_file):
        """
        avg_IOI (Average inter-onset-interval):
        To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.
        """
        midi = miditoolkit.MidiFile(midi_file)
        notes = midi.instruments[0].notes
        notes = sorted(notes, key=lambda x:x.start)
        onsets = [note.start for note in notes]
        intervals = [t - s for s, t in zip(onsets, onsets[1:])]
        avg_IOI = sum(intervals) / len(intervals) if intervals else 0
        return avg_IOI


    def cal_IOI(self, midi_files):
        pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
        futures = [pool.apply_async(self.cal_IOI_job, args=[ midi_file]) for midi_file in midi_files]
        pool.close()
        result_list = np.stack([x.get() for x in futures])  # 显示进度
        pool.join()
        return result_list


    def compute_oa_IOI(self):
        print(f"OA_IOI metrics in {os.path.dirname(self.generated_midi_files[0])} and {os.path.dirname(self.original_midi_files[0])} for {len(self.generated_midi_files)} and {len(self.original_midi_files)} files")

        ai_song_ioi_list = self.cal_IOI(self.generated_midi_files)
        human_song_ioi_list = self.cal_IOI(self.original_midi_files)
        oa_IOI = self.compute_oa(ai_song_ioi_list, human_song_ioi_list)

        return oa_IOI


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OA(DD)
    # the average overlapped area of the Duration Distribution between AI-generated and human-composed melodies. 
    # To extract the note duration histogram, we quantize the duration into 32 classes 
    # corresponding to 32 duration attributes in note symbol and compute the distribution of classes.
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    def cal_DD_job(self, midi_file):
        """
        avg_IOI (Average inter-onset-interval):
        To calculate the inter-onset-interval in the symbolic music domain, we find the time between two consecutive notes.
        """
        double_duration = set([i * 30 for i in range(1, 65)])
        triplet_duration = set([40, 80, 160, 320, 640])
        duration_bins = list(sorted(double_duration | triplet_duration))

        midi = miditoolkit.MidiFile(midi_file)
        notes = midi.instruments[0].notes
        notes = sorted(notes, key=lambda x:x.start)
        durations = [note.end-note.start for note in notes]
        frequency = Counter(durations)
        histogram = [frequency[bin]/len(durations) for bin in duration_bins]

        return np.array(histogram)


    def cal_DD(self, midi_files):
        pool = Pool(int(os.getenv('N_PROC', os.cpu_count())))
        futures = [pool.apply_async(self.cal_DD_job, args=[midi_file]) for midi_file in midi_files]
        pool.close()
        result_list = np.stack([x.get() for x in futures])  # 显示进度
        pool.join()
        return result_list
    
    def compute_oa_DD(self):
        print(f"OA_DD metrics in {os.path.dirname(self.generated_midi_files[0])} and {os.path.dirname(self.original_midi_files[0])} for {len(self.generated_midi_files)} and {len(self.original_midi_files)} files")

        ai_song_dd_list = self.cal_DD(self.generated_midi_files)
        human_song_dd_list = self.cal_DD(self.original_midi_files)
        oa_dd = self.compute_oa(ai_song_dd_list, human_song_dd_list)

        return oa_dd


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Melody Distance - SongMASS / TeleMelody
    # To evaluate the pitch trend of the melody, we spread out the notes into a time series of pitch according to the duration, 
    # with a granularity of 1/16 note. We subtract each pitch with the average pitch of the entire sequence for normalization. 
    # To measure the similarity between the generated and ground-truth time series with different lengths, we use dynamic time warping to measure their distance.
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    def cal_MD_job(self, midi_file, granularity):
        midi = MidiFile(midi_file)
        pitch_arr = []
        pitches = [note.pitch for note in midi.instruments[0].notes]
        duration = [note.end - note.start for note in midi.instruments[0].notes]
        repeat_time = [d // granularity for d in duration]
        for i, r_t in enumerate(repeat_time):
            pitch_arr += [pitches[i]] * r_t
        return pitch_arr
    
    def cal_MD(self, midi_files):
        double_duration = set([i * 30 for i in range(1, 65)])
        triplet_duration = set([40, 80, 160, 320, 640])
        duration_bins = list(sorted(double_duration | triplet_duration))
        
        granularity = reduce(gcd, duration_bins)
        md_list = []
        for midi_file in midi_files:
            md_list.append(self.cal_MD_job(midi_file, granularity))
        return md_list
    
    def compute_dtw_MD(self):
        print(f"dtw_MD metrics in {os.path.dirname(self.generated_midi_files[0])} and {os.path.dirname(self.original_midi_files[0])} for {len(self.generated_midi_files)} and {len(self.original_midi_files)} files")

        ai_song_md_list = self.cal_MD(self.generated_midi_files)
        human_song_md_list = self.cal_MD(self.original_midi_files)

        assert len(ai_song_md_list) == len(human_song_md_list), "generated melody and GT must be one-to-one"

        dtw_mean = []
        for ai_song_md, human_song_md in (pbar := tqdm(zip(ai_song_md_list, human_song_md_list))):
            ai_song_md = np.array(ai_song_md, dtype=np.float64).reshape(-1, 1)
            human_song_md = np.array(human_song_md, dtype=np.float64).reshape(-1, 1)

            ai_song_md -= np.mean(ai_song_md)
            human_song_md -= np.mean(human_song_md)
            d, _, _, _ = accelerated_dtw(human_song_md, ai_song_md, dist="euclidean")
            dtw_mean.append(d / len(ai_song_md))

            pbar.set_description(f"{sum(dtw_mean) / len(dtw_mean)}")

        dtw_mean_MD = sum(dtw_mean) / len(dtw_mean)

        return dtw_mean_MD
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_midi_path", type=str, required=True)
    parser.add_argument("--original_midi_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=10)

    args = parser.parse_args()

    generated_midi_path = args.generated_midi_path
    original_midi_path = args.original_midi_path
    output_path = args.output_path
    batch_size = args.batch_size
    
    oa_a_list = []
    oa_pch_list = []
    oa_ioi_list = []
    oa_dd_list = []
    md_list = []

    for bz in range(batch_size):
        generated_midi_batch_path = os.path.join(generated_midi_path, f"batch_{bz}")
        original_midi_files = []
        generated_midi_files = []
    
        for file in os.listdir(generated_midi_batch_path):
            generated_midi_files.append(os.path.join(generated_midi_batch_path, file))
            original_midi_files.append(os.path.join(original_midi_path, file))

        assert len(original_midi_files) == len(generated_midi_files)

        if len(original_midi_files) == 0:
            continue

        m = Metrics(original_midi_files=original_midi_files, generated_midi_files=generated_midi_files)
        
        try:
            oa_a_list.append(m.compute_oa_A())
            oa_pch_list.append(m.compute_oa_PCH())
            oa_ioi_list.append(m.compute_oa_IOI())
            oa_dd_list.append(m.compute_oa_DD())
            md_list.append(m.compute_dtw_MD())
        except Exception as e:
            continue

    print(f'Evaluation | OA(A) = {round(statistics.mean(oa_a_list),4)}±{round(statistics.stdev(oa_a_list),4)}')
    print(f'Evaluation | OA(PCH) = {round(statistics.mean(oa_pch_list),4)}±{round(statistics.stdev(oa_pch_list),4)}')
    print(f'Evaluation | OA(IOI) = {round(statistics.mean(oa_ioi_list),4)}±{round(statistics.stdev(oa_ioi_list),4)}')
    print(f'Evaluation | OA(DD) = {round(statistics.mean(oa_dd_list),4)}±{round(statistics.stdev(oa_dd_list),4)}')    
    print(f'Evaluation | MD = {round(statistics.mean(md_list),4)}±{round(statistics.stdev(md_list),4)}')
    