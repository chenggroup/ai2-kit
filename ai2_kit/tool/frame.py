from ai2_kit.core.util import expand_globs, slice_from_str, SAMPLE_METHOD, list_sample, ensure_dir
from ai2_kit.core.log import get_logger
import re


logger = get_logger(__name__)


class FrameTool:
    """
    This tool is design to sampling frames from large trajectory file without parsing them.
    You can use this tool to merge, sample frames from multiple files, and write them to a new file.

    A frame file is a file that contains multiple frames, each frame is separated by a fixed number of lines.
    For example, jsonl data file, or trajectory files in LAMMPS, xyz, etc format.
    """

    def __init__(self):
        self.header = []
        self.frames = []

    def read(self, *path_or_glob: str, frame_size: int = 0, rp = None, header_size:int = 0):
        """
        Load trajectory files from multiple paths, support glob pattern

        :param path_or_glob: path or glob pattern to locate data path
        :param frame_size: line number of each frame
        :param rp: repeated pattern, can be string or regex, e.g. 'ITEM: TIMESTEP', 'Lattice.+'
        """
        files = expand_globs(path_or_glob)
        if len(files) == 0:
            raise FileNotFoundError(f'No file found in {path_or_glob}')
        all_lines = []
        for file in files:
            with open(file) as f:
                lines = f.readlines()
                if header_size > 0:
                    self.header = lines[:header_size]
                    lines = lines[header_size:]
                all_lines.extend(lines)

        if frame_size <= 0:
            if rp is None:
                raise ValueError('either frame_size or rp (repeat pattern) should be set')
            frame_size = detect_frame_size(all_lines, rp)

        if len(all_lines) % frame_size > 0:
            raise ValueError(f'Invalid frame lines {frame_size}, cannot divide {len(all_lines)} lines into frames')
        self.frames.extend(all_lines[i: i + frame_size] for i in range(0, len(all_lines), frame_size))
        return self

    def slice(self, expr: str):
        """
        slice frame by python slice expression, for example
        `10:`, `:10`, `::2`, etc

        :param start: start index
        :param stop: stop index
        :param step: step
        """
        s = slice_from_str(expr)
        self.frames = self.frames[s]
        return self

    def sample(self, size: int, method: SAMPLE_METHOD='even', **kwargs):
        """
        sample frame by different method

        :param size: size of sample, if size is larger than data size, return all data
        :param method: method to sample, can be 'even', 'random', 'truncate', default is 'even'
        :param seed: seed for random sample, only used when method is 'random'

        Note that by default the seed is length of input list,
        if you want to generate different sample each time, you should set random seed manually
        """
        self.frames = list_sample(self.frames, size, method, **kwargs)
        return self

    def size(self):
        """
        size of loaded frames
        """
        print(len(self.frames))
        return self

    def write(self, out_file: str, keep_header=False, **kwargs):
        ensure_dir(out_file)
        with open(out_file, 'w', **kwargs) as f:
            if keep_header and self.header:
                f.writelines(self.header)
            for frame in self.frames:
                f.writelines(frame)


def detect_frame_size(l: list, rp: str, max_lines = 4096):
    """
    detect frame size of a file by repeating pattern
    :param rp: repeated pattern, can be string or regex, e.g. 'ITEM: TIMESTEP', 'Lattice.+'
    """
    pattern = re.compile(rp)
    lno = -1
    for i, line in enumerate(l):
        if i > max_lines:
            break
        if pattern.search(line):
            logger.info(f'Detected pattern: {line} at line {i} by pattern {rp}')
            if lno < 0:
                lno = i
            else:
                ret = i - lno
                logger.info(f'Detected frame lines: {ret}')
                return ret
    raise ValueError(f'Cannot detect frame lines in {max_lines} lines')
