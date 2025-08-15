# pip install scapy==2.4.4

from IPython.display import display
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.utils import RawPcapReader
from tqdm import tqdm
from scipy.stats import skew

import os
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.utils.data import ConcatDataset

# 현재 파일(data_utils.py)의 위치를 기준으로 dataset 폴더의 절대 경로를 계산
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'dataset'

# 0. Seed setting
def seed_everything(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

# 1. TimeSeriesGenerator
class TimeseriesGenerator:
    def __init__(self, data, length, sampling_rate=1, stride=1,
                 start_index=0, end_index=None,
                 shuffle=False, reverse=False, batch_size=128, label=None):
        self.data = data
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        if end_index is None:
            end_index = len(data)
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size
        self.label = label if label is None else np.array(label)
        if self.start_index > self.end_index:
            raise ValueError(
                "`start_index+length=%i > end_index=%i` "
                "is disallowed, as no part of the sequence "
                "would be left to be used as current step."
                % (self.start_index, self.end_index)
            )

    def __len__(self):
        return (self.end_index - self.start_index + self.batch_size * self.stride) // (self.batch_size * self.stride)

    def __getitem__(self, index):
        rows = self.__index_to_row__(index)
        samples, y = self.__compile_batch__(rows)
        return samples, y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __index_to_row__(self, index):  # Returns a list of rows that will compose a given batch (index). len(rows) is equal to the batch size.
        if self.shuffle:
            rows = np.random.randint(self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size * self.stride, self.end_index + 1), self.stride)
        return rows

    def __compile_batch__(self, rows):  # Generate time series features for each given row.
        samples = np.array([self.data[row - self.length: row: self.sampling_rate] for row in rows])
        if self.reverse:
            samples = samples[:, ::-1, ...]
        if self.length == 1:
            samples = np.squeeze(samples)

        if self.label is None:
            return samples, samples
        else:
            return samples, self.label[rows - self.length]

    @property
    def output_shape(self):
        x, y = self[0]
        return x.shape, y.shape

    @property
    def num_samples(self):
        count = 0
        for x, y in self:
            count += x.shape[0]
        return count

    def __str__(self):
        return '<TimeseriesGenerator data.shape={} / num_batches={:,} / output_shape={}>'.format(
            self.data.shape, len(self), self.output_shape,
        )

    def __repr__(self):
        return self.__str__()

# [CHANGED] --- Helpers for protocol parsing & offsets ---
def _ethertype(frame: np.ndarray, offset: int = 12) -> int:
    # EtherType at bytes 12-13, but if VLAN (0x8100), actual EtherType at 16-17
    if len(frame) < offset + 2:
        return -1
    et = int.from_bytes(frame[offset:offset+2], 'big', signed=False)
    if et == 0x8100 and len(frame) >= 18:
        et = int.from_bytes(frame[16:18], 'big', signed=False)
    return et

def _l2_len(frame: np.ndarray) -> int:
    # Ethernet header 14B (+4B VLAN tag if present)
    if len(frame) >= 14 and int.from_bytes(frame[12:14], 'big') == 0x8100:
        return 18
    return 14

def _is_ipv4_udp(frame: np.ndarray) -> bool:
    et = _ethertype(frame)
    if et != 0x0800:  # IPv4
        return False
    # IPv4 header: protocol at byte 23 (assuming no VLAN; VLAN already handled by _ethertype)
    if len(frame) < 24:
        return False
    ip_proto = frame[23]
    return ip_proto == 17  # UDP

def _udp_dst_port(frame: np.ndarray) -> int:
    # IPv4 header length: IHL at byte 14 (lower 4 bits) * 4
    # But if VLAN, L2 is 18. Compute dynamically.
    l2 = _l2_len(frame)
    if len(frame) < l2 + 20:  # minimal IPv4 header
        return -1
    ihl = (frame[l2] & 0x0F) * 4
    ip_start = l2
    udp_start = ip_start + ihl
    if len(frame) < udp_start + 4:
        return -1
    dst_port = int.from_bytes(frame[udp_start+2:udp_start+4], 'big', signed=False)
    return dst_port

def _detect_protocol(frame: np.ndarray) -> str:
    et = _ethertype(frame)
    if et == 0x22F0:   # AVTP
        return 'AVTP'
    if et == 0x88F7:   # gPTP
        return 'PTP'
    if et == 0x0800 and _is_ipv4_udp(frame):
        return 'UDP'
        # dp = _udp_dst_port(frame)
        # if 17220 <= dp <= 17230:
        #     return 'UDP'  # CAN/UDP
    return ''  # unknown/other

# 2. Load Dataset
class PktDataset: # 기존의 Dataset 클래스명의 충돌 방지를 위해 PktDataset으로 이름 변경
    def __init__(self, df: pd.DataFrame, trim_etc_protocols=True):
        if trim_etc_protocols:
            self.df = df[df['ProtocolType'] != ''].copy()
        else:
            self.df = df
        assert self.df['abstime'].is_monotonic_increasing
        assert self.df['monotime'].is_monotonic_increasing

    @classmethod
    def _load_towids_dataset(cls, path_pcap, usec_unit, path_csv=None, **kwargs):
        reader = RawPcapReader(str(path_pcap))
        list_output = list()
        for idx, (payload, metadata) in tqdm(enumerate(reader), desc='Parsing the pcap file...'):
            sec, usec, wirelen, caplen = metadata
            list_output.append((sec, usec, wirelen, caplen, payload))
        df_pcap = pd.DataFrame(list_output, columns=['sec', 'usec', 'wirelen', 'caplen', 'payload'])

        if path_csv:
            df_label = pd.read_csv(path_csv, header=None, names=['idx', 'label', 'y_desc'])
            assert df_pcap.shape[0] == df_label.shape[0], \
                f'Record count mismatch. {df_pcap.shape=}, {df_label.shape=}'
            assert (df_label['idx'].diff().bfill() == 1).all(), 'Field `idx` does not increase sequentially.'
            df_label['y'] = df_label['label'].map({'Normal': 0, 'Abnormal': 1})
        else:
            df_label = pd.DataFrame(index=df_pcap.index)
            df_label['y'] = 0
            df_label['y_desc'] = 'Normal'

        abstime = pd.to_datetime(df_pcap['sec'], unit='s') + pd.to_timedelta(df_pcap['usec'], unit=usec_unit)
        dupcounts = abstime.duplicated(keep=False).sum()

        if dupcounts > 0:
            print(f'There were {dupcounts} distinct timestamps.', end=' ')
            for _ in range(100):
                duplicated = abstime.duplicated()
                if duplicated.sum() == 0:
                    break
                abstime[duplicated] += pd.Timedelta(milliseconds=1)
            else:
                raise ValueError('Something went wrong.')
            print(f'-> {_} correction(s).')

        monotime = (abstime - abstime.min()).dt.total_seconds()
        df_pcap['payload'] = df_pcap['payload'].map(lambda x: np.frombuffer(x, dtype='uint8'))

        df: pd.DataFrame = pd.concat([
            abstime.rename('abstime'),
            monotime.rename('monotime'),
            df_pcap[['wirelen', 'caplen', 'payload']],
            df_label[['y', 'y_desc']]
        ], axis=1)

        df = df.sort_values('abstime')
        assert df['abstime'].is_monotonic_increasing
        assert df['monotime'].is_monotonic_increasing

        # [CHANGED] Protocol specification by parsing, not wirelen
        protos = []
        for arr, ydesc in zip(df['payload'].values, df['y_desc'].values):
            proto = _detect_protocol(arr)
            # special case: keep P_I as PTP if detection missed (defensive)
            if ydesc == 'P_I' and proto == '':
                proto = 'PTP'
            protos.append(proto)
        df['ProtocolType'] = protos

        return cls(df, **kwargs)

    @classmethod
    def towids_train(cls, **kwargs):
        return cls._load_towids_dataset(
            DATASET_DIR / 'Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap',
            'ns',
            DATASET_DIR / 'y_train.csv',
            **kwargs
        )

    @classmethod
    def towids_test(cls, **kwargs):
        return cls._load_towids_dataset(
            DATASET_DIR / 'Automotive_Ethernet_with_Attack_original_10_17_20_04_test.pcap',
            'ns',
            DATASET_DIR / 'y_test.csv',
            **kwargs
        )

    def do_label(self, window_size) -> np.ndarray:
        y = self.df.rolling(window=window_size)['y'].max().dropna().astype('int32').values
        assert isinstance(y, np.ndarray)
        return y

    def trim(self, time_start=None, time_end=None, is_absolute=None):
        assert is_absolute is not None
        monotime_min = self.df['monotime'].min()
        monotime_max = self.df['monotime'].max()

        if time_start is not None:
            if is_absolute is False:
                time_start = monotime_min + time_start
            assert monotime_min < time_start
        else:
            time_start = monotime_min

        if time_end is not None:
            if is_absolute is False:
                time_end = monotime_max - time_end
            assert time_end < monotime_max
        else:
            time_end = monotime_max

        df = self.df.query(f'{time_start} <= monotime <= {time_end}').copy()
        return PktDataset(df)
        
    # 2-1. Feature generator 1 (FG1)
    # [CHANGED] add stride argument and use it for generator
    def do_fg1_transition_matrix(self, window_size=2048, stride=1) -> np.array:
        df = self.df
        idx = {'AVTP': 0, 'PTP': 1, 'UDP': 2}
        N = len(idx)

        proto_seq = df['ProtocolType'].map(idx).values

        def seq_to_transition_matrix(seq):
            T = np.zeros((N, N), dtype=np.float32)
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i+1]
                if a >= 0 and b >= 0:
                    T[a, b] += 1
            denom = max(1, (len(seq) - 1))
            T /= denom
            return T

        if len(proto_seq) < window_size:
            raise ValueError(f"Insufficient data length ({len(proto_seq)}) for window_size {window_size}")

        generator = TimeseriesGenerator(proto_seq, length=window_size, sampling_rate=1, stride=stride, batch_size=1, shuffle=False)

        result = []
        for X, _ in generator:
            seq = X[0]
            result.append(seq_to_transition_matrix(seq))

        return np.stack(result)

    # 2-2. Feature generator 2 (FG2)
    # [CHANGED] adjust slice start with L2 length (VLAN-aware)
    def do_fg2_payload(self, window_size=2048, byte_start=0x22, byte_end=0x22 + 9) -> np.array:
        '''
        - Take 9 bytes from the 34th byte offset within protocol/payload region (paper setting).
        - Short payloads are padded with 0x00.
        - Returns shape (n, 9) normalized to [0, 1].
        '''
        assert byte_start < byte_end
        num_bytes = byte_end - byte_start

        payloads = []
        for frame in self.df['payload'].values:
            l2 = _l2_len(frame)  # 14 or 18 (VLAN)
            start = l2 + byte_start
            end = start + num_bytes

            segment = np.zeros(num_bytes, dtype=np.uint8)
            if start < len(frame):
                segment_len = max(0, min(end, len(frame)) - start)
                if segment_len > 0:
                    segment[:segment_len] = frame[start:start+segment_len]
            payloads.append(segment / 255.0)

        return np.array(payloads, dtype=np.float32)

    # 2-3. Feature generator 3 (FG3)
    # [CHANGED] add stride argument and use it for generator
    def do_fg3_statistics(self, window_size=2048, methods=('mean', 'std', 'skew'), stride=1) -> np.array:
        '''
        - Returns shape=(num_windows, 3, 3).
        - Applies log10 scaling with rules from the paper.
        '''
        df = self.df
        idx = {'AVTP': 0, 'PTP': 1, 'UDP': 2}
        N = len(idx)

        monotime = df['monotime'].values
        protos = df['ProtocolType'].map(idx).values

        generator = TimeseriesGenerator(
            np.stack([monotime, protos], axis=1),
            length=window_size,
            sampling_rate=1,
            stride=stride,
            batch_size=1,
            shuffle=False
        )

        result = []
        for X, _ in generator:
            x_window = X[0]
            t = x_window[:, 0]
            p = x_window[:, 1].astype(int)

            stat_matrix = np.full((N, 3), 1e+7, dtype=np.float32)

            for i in range(N):
                ti = t[p == i]
                if len(ti) >= 2:
                    diffs = np.diff(ti)
                    stat_matrix[i, 0] = np.mean(diffs)
                    if len(diffs) >= 2:
                        stat_matrix[i, 1] = np.std(diffs)
                    if len(diffs) >= 3:
                        stat_matrix[i, 2] = np.abs(skew(diffs))

            stat_matrix = np.where(stat_matrix == 0, 1e-7, stat_matrix)
            stat_matrix = np.log10(stat_matrix)

            result.append(stat_matrix)

        return np.stack(result, dtype=np.float32)

dataset_train = PktDataset.towids_train()
dataset_test = PktDataset.towids_test()

# 3. Create train/validation/test sets
args = [
    [dataset_train, 'Train', 5, 60, False],
    [dataset_train, 'Validation', 60, 71.11, False],
    [dataset_train, 'Test', 71.11, None, True],
    [dataset_test, 'Train', 5, 80, False],
    [dataset_test, 'Validation', 80, 91.88, False],
    [dataset_test, 'Test', 91.89, None, True],
]

def do(dataset, purpose, time_start, time_end, trim_last_5sec):
    name = 'Packet dump 1' if dataset is dataset_train else 'Packet dump 2'

    dataset = dataset.trim(time_start, time_end, is_absolute=True)
    if trim_last_5sec:
        dataset = dataset.trim(time_end=5, is_absolute=False)
        time_end = dataset.df['monotime'].max()
    a = dataset.df['y'].value_counts()
    a.name = name
    a['Purpose'] = purpose
    a['Time range'] = '[{:.2f}, {:.2f}]'.format(time_start, time_end)
    a = a.rename({0: 'Benign', 1: 'Intrusion'})
    a = a.reindex(['Purpose', 'Time range', 'Benign', 'Intrusion'], fill_value=0)
    return a, dataset

# 4. Define AEGenerator
class AEGenerator(TorchDataset): # Dataset 변수명 충돌 해결
    def __init__(self, T, P, S, labels=None, window_size=2048, stride=1, sampling_rate=1, shuffle=False, reverse=False): 
        '''
        T : nparray (num_windows, 3, 3)
        P : nparray (num_packets, 9)
        S : nparray (num_windows, 3, 3)
        '''
        self.T = T
        self.P = P
        self.S = S
        self.n = T.shape[0]
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.shuffle = shuffle
        self.reverse = reverse

        # index list
        self.indices = np.arange(window_size, len(P) + 1, stride)
        if len(self.indices) > len(T):
            self.indices = self.indices[:len(T)] # align with T, S

        if self.labels is not None:
            assert len(self.indices) == len(self.labels), f'Label number {len(self.labels)} ≠ Index number {len(self.indices)}'

        # [CHANGED] To avoid misalignment when shuffle=True, keep it off by default (already default).
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]  # packet-level end index for the window

        # T, S window: use the same sequential window order as constructed (index-th window)
        t = torch.from_numpy(self.T[index].astype('float32')).flatten()
        s = torch.from_numpy(self.S[index].astype('float32')).flatten()
        
        # P window by end idx
        start_idx = max(0, idx - self.window_size)
        end_idx = idx
        p_window = self.P[start_idx : end_idx : self.sampling_rate]

        if self.reverse:
            p_window = p_window[::-1]

        if p_window.shape[0] < self.window_size:
            pad_size = self.window_size - p_window.shape[0]
            padding = np.zeros((pad_size, self.P.shape[1]), dtype=np.float32)
            p_window = np.vstack((padding, p_window))

        p = torch.from_numpy(p_window.astype('float32'))

        x = (t, p, s)
        y = self.labels[index] if self.labels is not None else (t, p, s)

        return x, y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def output_shape(self):
        x, y = self[0]
        t, p, s = x
        return (t.shape, p.shape, s.shape)

    @property
    def num_samples(self):
        return len(self)

    def __str__(self):
        return f'<AEGenerator num_samples={self.num_samples} / output_shape={self.output_shape}>'

    def __repr__(self):
        return self.__str__()

# 5. Concat train/validation/test dataset and generate final dataloader for the model
def get_cache_paths(cache_dir, split_name, idx, window_size=2048, stride=1):
    base = Path(cache_dir) / split_name
    base.mkdir(parents=True, exist_ok=True)
    t_path = base / f'T_idx[{idx}]_ws{window_size}_st{stride}.pkl'
    p_path = base / f'P_idx[{idx}]_ws{window_size}_st{stride}.pkl'
    s_path = base / f'S_idx[{idx}]_ws{window_size}_st{stride}.pkl'
    return t_path, p_path, s_path

def process_split(indices, split_name, window_size, stride, cache_dir):
    datasets = []

    # 1. load raw sliced datasets
    list_dataset_sub = list()
    for arg in tqdm(args, desc=f'Loading args for {split_name}'):
        _, dataset_sub = do(*arg)
        list_dataset_sub.append(dataset_sub)
    
    # [CHANGED] Ensure Train/Valid contain only benign (defensive)
    if split_name in ('train','valid'):
        for i in indices:
            assert list_dataset_sub[i].df['y'].max() == 0, f"{split_name} split idx={i} contains abnormal labels."

    # 2. For each dataset slice
    for idx in tqdm(indices, desc=f'Creating AEGenerators for {split_name} indices'):
        dataset = list_dataset_sub[idx]
        t_path, p_path, s_path = get_cache_paths(cache_dir, split_name, idx, window_size, stride) 

        # Load or compute T
        if t_path.exists():
            with t_path.open('rb') as f:
                T = pickle.load(f)
        else:
            print(f"    Generating T for idx={idx}...")
            T = dataset.do_fg1_transition_matrix(window_size=window_size, stride=stride)  # [CHANGED]
            with t_path.open('wb') as f:
                pickle.dump(T, f)
            
        # Load or compute P
        if p_path.exists():
            with p_path.open('rb') as f:
                P = pickle.load(f)
        else:
            print(f"    Generating P for idx={idx}...")
            P = dataset.do_fg2_payload(window_size=window_size)  # keep API consistent
            with p_path.open('wb') as f:
                pickle.dump(P, f)

        # Load or compute S
        if s_path.exists():
            with s_path.open('rb') as f:
                S = pickle.load(f)
        else:
            print(f"    Generating S for idx={idx}...")
            S = dataset.do_fg3_statistics(window_size=window_size, stride=stride)  # [CHANGED]
            with s_path.open('wb') as f:
                pickle.dump(S, f)

        # 3. Labels for test set
        labels = None
        if split_name == 'test':
            y_seq = dataset.df['y'].values
            all_indices = np.arange(window_size, len(P) + 1, stride)
            if len(all_indices) > len(T):
                all_indices = all_indices[:len(T)]

            labels = []
            for i_end in all_indices:
                start_idx = max(0, i_end - window_size)
                end_idx = i_end
                window_label = y_seq[start_idx : end_idx]
                labels.append(int(window_label.max()))
            labels = np.array(labels)

        datasets.append(AEGenerator(T, P, S, labels=labels, window_size=window_size, stride=stride))
    
    return ConcatDataset(datasets)

def get_processed_dataloader(window_size=2048, stride=1, batch_size=64, cache_dir='./cache'):
    splits = {'train': [0, 3], 'valid': [1, 4], 'test': [2, 5]}
    dataloaders = {}
    for split_name, indices in splits.items():
        concat_dataset = process_split(indices, split_name, window_size, stride, cache_dir)
        dataloader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        dataloaders[split_name] = dataloader
    return dataloaders['train'], dataloaders['valid'], dataloaders['test']
