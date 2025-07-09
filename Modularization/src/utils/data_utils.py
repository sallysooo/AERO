# pip install scapy==2.4.4

from IPython.display import display
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.utils import RawPcapReader
from tqdm import tqdm
from scipy.stats import skew

# 현재 파일(data_utils.py)의 위치를 기준으로 dataset 폴더의 절대 경로를 계산
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'dataset'

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

# 2. Load Dataset
class PktDataset:
    def __init__(self, df: pd.DataFrame, trim_etc_protocols=True):
        if trim_etc_protocols:
            self.df = df[df['ProtocolType'] != ''].copy()
        else:
            self.df = df
        assert self.df['abstime'].is_monotonic_increasing
        assert self.df['monotime'].is_monotonic_increasing

    @classmethod
    def _load_towids_dataset(cls, path_pcap, usec_unit, path_csv=None, **kwargs):
        # assert scapy.__version__ == '2.4.4', 'scapy version mismatch.'

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

        # Protocol specification
        df['ProtocolType'] = ''
        df.loc[df['wirelen'] == 60, 'ProtocolType'] = 'UDP'
        df.loc[df['wirelen'].isin([68, 90]), 'ProtocolType'] = 'PTP'
        df.loc[df['wirelen'].isin([82, 434]), 'ProtocolType'] = 'AVTP'
        # special treatment
        df.loc[df['y_desc'] == 'P_I', 'ProtocolType'] = 'PTP'

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
        # print('Before [{} ~ {}] / Required [{} ~ {}] / After [{} ~ {}]'.format(
        #     monotime_min, monotime_max,
        #     time_start, time_end,
        #     df['monotime'].min(), df['monotime'].max()
        # ))
        return PktDataset(df)
        
    # 2-1. Feature generator 1 (FG1)
    def do_fg1_transition_matrix(self, window_size=2048) -> np.array:
        # When the number of collected packets is n, a numpy array of shape = (n, 3, 3) should be the output
        df = self.df
        # proto_types = sorted(df['ProtocolType'].unique()) # ex) ['AVTP', 'PTP', 'UDP']
        idx = {'AVTP': 0, 'PTP': 1, 'UDP': 2} # ex) {'AVTP': 0, 'PTP': 1, 'UDP': 2}
        N = len(idx) # 3

        # 1. ProtocolType sequence -> integer index
        proto_seq = df['ProtocolType'].map(idx).values # [2, 0, 0, 1, 2]

        # 2. generate T
        def seq_to_transition_matrix(seq):
          T = np.zeros((N, N), dtype=np.float32)
          for i in range(len(seq) - 1):
            a, b = seq[i], seq[i+1]
            T[a, b] += 1
          T /= (len(seq)-1) # normalization
          return T

        if len(proto_seq) < window_size:
          raise ValueError(f"Insufficient data length ({len(proto_seq)}) for window_size {window_size}")

        # checkpoint
        print("Data shape:", proto_seq.shape)
        print("Window size:", window_size)

        # 3. sliding window using TimeseriesGenerator
        generator = TimeseriesGenerator(proto_seq, length=window_size, sampling_rate=1, stride=1, batch_size=1, shuffle=False)

        print("Generator length:", len(generator))
        # if len(generator) == 0:
        #   print("Warning: Generator is empty! Check window_size and data length.")
        #   return np.zeros((0, N, N))

        result = []
        for X, _ in generator:
          seq = X[0] # (window_size, )
          T = seq_to_transition_matrix(seq)
          result.append(T)

        return np.stack(result) # (num_windows, N, N)


    # 2-2. Feature generator 2 (FG2)
    def do_fg2_payload(self, window_size=2048, byte_start=0x22, byte_end=0x22 + 9) -> np.array:
        '''
        - The paper's strategy is to take 9 bytes from the 0x22th byte for the payload loaded in each packet.  
        - Short payloads should be padded with 0x00.
        - When the number of collected packets is n, a numpy array with shape = (n, 9) should be generated. 
        - FG2 does not need to apply TimeseriesGenerator.
        '''
        assert byte_start < byte_end
        num_bytes = byte_end - byte_start # 9

        payloads = []
        for arr in self.df['payload'].values:
          segment = np.zeros(num_bytes, dtype=np.uint8) # [0, 0, 0, ..., 0]
          arr_len = len(arr)
          for i in range(num_bytes): # 9
            if byte_start + i < arr_len:
              segment[i] = arr[byte_start + i]
          payloads.append(segment / 255.0)

        return np.array(payloads) # (n ,9)


    # 2-3. Feature generator 3 (FG3)
    def do_fg3_statistics(self, window_size=2048, methods=('mean', 'std', 'skew')) -> np.array:
        '''
        - When the number of collected packets is n, a numpy array of shape=(n, 3, 3) should be generated.
        - The <feature normalization strategy> described at the bottom right of page 5 of the paper must be implemented.
        '''
        df = self.df
        # proto_types = sorted(df['ProtocolType'].unique()) # ex) ['AVTP', 'PTP', 'UDP']
        idx = {'AVTP': 0, 'PTP': 1, 'UDP': 2} # ex) {'AVTP': 0, 'PTP': 1, 'UDP': 2}
        N = len(idx) # ex) 3

        monotime = df['monotime'].values
        protos = df['ProtocolType'].map(idx).values

        # each window is constructed as [window_size * 2]
        generator = TimeseriesGenerator(
            np.stack([monotime, protos], axis=1), # (n, 2)
            length = window_size,
            sampling_rate = 1,
            stride = 1,
            batch_size = 1,
            shuffle = False
            )

        # checkpoint
        print("Data shape:", np.stack([monotime, protos], axis=1).shape)
        print("Window size:", window_size)

        result = []
        for X, _ in generator:
          x_window = X[0] # (window_size, 2)
          t = x_window[:, 0] # first column of 'monotime' [1.0, 1.2, 1.3, 2.0, ...]
          p = x_window[:, 1].astype(int) # second column of 'protos(protocol index)' [0, 0, 1, 0]

          stat_matrix = np.full((N, 3), 1e+7, dtype=np.float32) # Initialize default value to 1e+7

          for i in range(N):
            t_i = t[p == i] # time sequence of the ith protocol / t : [1.0, 1.2, 1.3, 2.0, ...] / p==i : [True, True, False, True, ...] / t[p==i] : [1.0, 1.2, 2.0] => select protocol by this workflow
            if len(t_i) >= 2:
                diffs = np.diff(t_i)
                mean_val = np.mean(diffs)
                stat_matrix[i, 0] = mean_val

                if len(diffs) >= 2:
                    std_val = np.std(diffs)
                    stat_matrix[i, 1] = std_val
                if len(diffs) >= 3:
                    skew_val = np.abs(skew(diffs))
                    stat_matrix[i, 2] = skew_val
            
          stat_matrix = np.where(stat_matrix == 0, 1e-7, stat_matrix)
          stat_matrix = np.log10(stat_matrix)

          result.append(stat_matrix)

        return np.stack(result) # (num_windows, N, 3)

dataset_train = PktDataset.towids_train()
dataset_test = PktDataset.towids_test()


# 3. Create train/validation/test sets by dividing the two packet dump datasets (dataset_train, dataset_test) into different time ranges.
# Organize the number of malicious traffic (intrusion) and normal traffic (benign) in these sets into a table as below.
# Arguments to be passed to the do function: [dataset, purpose, start time, end time, whether to remove the last 5 seconds of noise]

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

    dataset = dataset.trim(time_start, time_end, is_absolute=True) # slice [time_start, time_end] part only in the entire dataset
    if trim_last_5sec: # Remove noise remaining after the data collection step
        dataset = dataset.trim(time_end=5, is_absolute=False)
        time_end = dataset.df['monotime'].max() # Since 'time_end' has changed after removing noise, update it again with the actual maximum time
    a = dataset.df['y'].value_counts() 
    a.name = name 
    a['Purpose'] = purpose 
    a['Time range'] = '[{:.2f}, {:.2f}]'.format(time_start, time_end)
    a = a.rename({0: 'Benign', 1: 'Intrusion'}) # 0 as benign, 1 as intrusion
    a = a.reindex(['Purpose', 'Time range', 'Benign', 'Intrusion'], fill_value=0)
    return a, dataset



# 4. Define new Dataset Class : AEGenerator with sliding window
'''
- Converting the shape of each FG1-3, to match the dimension as an input value of the Autoencoder model afterwards
    - x = (T, P, S)

- dataset[i][0] → ((9,), (2048, 9), (9,))
- dataset[i][1] → ((9,), (2048, 9), (9,)) ⇒ x == y since it's an autoencoder model
- dataloader[i][0] → ((b, 9), (b, 2048, 9), (b, 9))
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset as TorchDataset, DataLoader


class AEGenerator(TorchDataset): # Dataset 변수명 충돌 해결
    def __init__(self, T, P, S, window_size=2048, stride=1, sampling_rate=1, shuffle=False, reverse=False): 
        '''
        T : nparray (n, 3, 3) window 단위
        P : nparray (m, 9) packet 단위
        S : nparray (n, 3, 3) window 단위

        window_size : P sliding window 크기 (default : 2048)
        stride : sliding window stride (default : 1)
        sampling_rate : window 내 sample 간 간격
        shuffle : window의 index 순서를 랜덤화할지 여부
        reverse : window 내부 순서를 뒤집을지 여부
        '''
        self.T = T
        self.P = P
        self.S = S
        self.n = T.shape[0]
        self.window_size = window_size
        self.stride = stride
        self.sampling_rate = sampling_rate
        self.shuffle = shuffle
        self.reverse = reverse

        '''
        P 데이터에서 window 끝의 위치 index 리스트를 생성
        - window 끝 index = window_size 이상 위치부터 sliding 시작
        - stride 단위로 이동
        - T, S 데이터 수에 맞게 index list 조정    
        '''
        # index list
        self.indices = np.arange(window_size, len(P) + 1, stride)
        if len(self.indices) > len(T):
            self.indices = self.indices[:len(T)] # T, S 크기에 맞춤
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    
    def __len__(self): # index list 길이 = 총 샘플 수
        return len(self.indices)

    def __getitem__(self, index): # 하나의 샘플(x, y)를 리턴
        idx = self.indices[index] # idx : 현재 window의 끝 index
        
        # T, S : (3,3) → (9,) flatten
        t = torch.from_numpy(self.T[index].astype('float32')).flatten()
        s = torch.from_numpy(self.S[index].astype('float32')).flatten()
        
        # P window
        start_idx = max(0, idx - self.window_size)
        end_idx = idx
        p_window = self.P[start_idx : end_idx : self.sampling_rate]

        if self.reverse:
            p_window = p_window[::-1]

        # zero padding
        if p_window.shape[0] < self.window_size:
            pad_size = self.window_size - p_window.shape[0]
            padding = np.zeros((pad_size, self.P.shape[1]), dtype=np.float32)
            p_window = np.vstack((padding, p_window))

        # 최종 window를 torch tensor로 변환
        p = torch.from_numpy(p_window.astype('float32'))

        x = y = (t, p, s)
        
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
    def num_samples(self): # 총 샘플 수 반환
        return len(self)

    def __str__(self):
        return f'<AEGenerator num_samples={self.num_samples} / output_shape={self.output_shape}>'

    def __reper__(self):
        return self.__str__()



def get_processed_dataset(window_size=2048, stride=1, batch_size=64):
    list_output = list()
    list_dataset_sub = list() ######### From here, you can retrieve the Dataset instance as needed.
    for arg in args:
        output, dataset_sub = do(*arg)
        list_output.append(output)
        list_dataset_sub.append(dataset_sub)

    original_dataset = list_dataset_sub[0] # 'train' part in dataset_train

    T = original_dataset.do_fg1_transition_matrix() # (95668, 3, 3)
    P = original_dataset.do_fg2_payload() # (97715, 9)
    S = original_dataset.do_fg3_statistics() # (95668, 3, 3)

    dataset = AEGenerator(T, P, S, window_size=2048, stride=1)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False) # n = 64

    for x, _ in dataloader:
        t, p, s = x 
        # t : (64, 9) | p : (64, 2048, 9) | s : (64, 9)
        assert t.shape[0] == p.shape[0] == s.shape[0], "T, P, S must have same number of samples!"
        break

    return T, P, S, dataset, dataloader


