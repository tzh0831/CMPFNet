import os
import sys
import logging
import random
import time
# from utils import pyt_utils
# from utils.pyt_utils import ensure_dir

_default_level_name = os.getenv('ENGINE_LOGGING_LEVEL', 'INFO')
_default_level = logging.getLevelName(_default_level_name.upper())

class LogFormatter(logging.Formatter):
    log_fout = None
    date_full = '[%(asctime)s %(lineno)d@%(filename)s:%(name)s] '
    date = '%(asctime)s '
    msg = '%(message)s'

    def format(self, record):
        if record.levelno == logging.DEBUG:
            mcl, mtxt = self._color_dbg, 'DBG'
        elif record.levelno == logging.WARNING:
            mcl, mtxt = self._color_warn, 'WRN'
        elif record.levelno == logging.ERROR:
            mcl, mtxt = self._color_err, 'ERR'
        else:
            mcl, mtxt = self._color_normal, ''

        if mtxt:
            mtxt += ' '

        if self.log_fout:
            self.__set_fmt(self.date_full + mtxt + self.msg)
            formatted = super(LogFormatter, self).format(record)
            # self.log_fout.write(formatted)
            # self.log_fout.write('\n')
            # self.log_fout.flush()
            return formatted

        self.__set_fmt(self._color_date(self.date) + mcl(mtxt + self.msg))
        formatted = super(LogFormatter, self).format(record)

        return formatted

    if sys.version_info.major < 3:
        def __set_fmt(self, fmt):
            self._fmt = fmt
    else:
        def __set_fmt(self, fmt):
            self._style._fmt = fmt

    @staticmethod
    def _color_dbg(msg):
        return '\x1b[36m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_warn(msg):
        return '\x1b[1;31m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_err(msg):
        return '\x1b[1;4;31m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_omitted(msg):
        return '\x1b[35m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_normal(msg):
        return msg

    @staticmethod
    def _color_date(msg):
        return '\x1b[32m{}\x1b[0m'.format(msg)

def ensure_dir(path):
    if not os.path.isdir(path):
        try:
            sleeptime = random.randint(0, 3)
            time.sleep(sleeptime)
            os.makedirs(path)
        except:
            print('conflict !!!')

def get_logger(log_dir=None, log_file=None, formatter=LogFormatter):
    # getLogger():返回一个名为 root 的全局日志记录器。
    logger = logging.getLogger()
    # 设置日志记录器的日志级别
    logger.setLevel(_default_level)
    # 删除日志记录器上已有的所有处理器
    del logger.handlers[:]
    if log_dir and log_file:
        ensure_dir(log_dir)
        # 将日志输出到文件
        LogFormatter.log_fout = True
        # 设置一个文件处理器，用于将日志写入文件，设置为追加模式('a')
        file_handler = logging.FileHandler(log_file, mode='a')
        # 设置文件处理器的日志级别和格式化器
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        # 将文件处理器添加到日志记录器
        logger.addHandler(file_handler)

    # 控制台处理器，用于将日志输出到控制台
    stream_handler = logging.StreamHandler()

    stream_handler.setFormatter(formatter(datefmt='%d %H:%M:%S'))
    # 设置控制台处理器的日志级别为 0（最低级别，用于捕获所有级别的日志）
    stream_handler.setLevel(0)
    # 将控制台处理器添加到日志记录器
    logger.addHandler(stream_handler)
    return logger
