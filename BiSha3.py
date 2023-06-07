import struct
import numpy as np
from scipy import signal, interpolate


class Read_datafiles(object):
    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
            return False

    def readRaw_minut(self, filepath, startTime, L, post):
        # 读取L分钟的数据，从startTime分钟开始
        with open(filepath, 'rb') as f:
            # 40000为文档中包含的数字个数，而一个16bit有符号数占2个字节
            # 16bit有符号bit 对应 struct里format为h
            f.seek(startTime * 2 * 60 * 250, post)
            data_int16 = struct.unpack("h" * L * 7500 * 6, f.read(L * 2 * 6 * 7500))
        re_data_int16 = np.asarray(data_int16).reshape(L * 6 * 2500, 3)
        re_data_int16 = np.transpose(re_data_int16)
        return re_data_int16

    def readRaw(self, filepath, startTime):
        # 从startTime分钟开始，读取到文件最后
        with open(filepath, 'rb') as f:
            # 40000为文档中包含的数字个数，而一个16bit有符号数占2个字节
            # 16bit有符号bit 对应 struct里format为h
            file_data = f.read()[startTime * 2 * 3 * 60 * 250:]
            file_length = int(len(file_data) / 2)
            data_int16 = struct.unpack('h' * file_length, file_data)
        re_data_int16 = np.asarray(data_int16).reshape(int(file_length / 3), 3)
        re_data_int16 = np.transpose(re_data_int16)
        return re_data_int16

    def intercept_after_filter_ecg_(RAW_path, minute):

        RF = Read_datafiles()

        A = RF.readRaw(RAW_path, minute)  # 截取ECG段  从0 开始

        ecg = A[1, :] * 0.0244  #::-1

        [a, b] = signal.butter(3, [0.004, 0.4], 'pass')

        ecg = signal.filtfilt(a, b, ecg)

        return ecg