import argparse
import h5py as h5
import os, re, sys, glob
import numpy as np
import matplotlib.image as mpimg
from collections import defaultdict


class wellStitcher:
    """Stitch well frames images sequentially into a mosaiq.
    Outputs one file per well across timepoints.

    :param well_regex: string regex matching well in filename
    :param time_regex: string regex matching time in filename
    :param frame_regex: string regex matching frame in filename
    :param timepoints: int representing number of timepoints in data
    :param file_ext: reges string specifying included extension type for files
    """

    def __init__(
        self,
        well_regex="(\D)(\d\d)",
        time_regex="_TimePoint_(\d+)",
        frame_regex="_s(\d+)",
        timepoints=12,
        file_regex=".[tT][iI][fF]*",
    ):
        self.well_regex = well_regex
        self.time_regex = time_regex
        self.frame_regex = frame_regex
        self.timepoints = timepoints
        self.file_regex = file_regex
        self._dir = defaultdict(lambda: defaultdict(dict))

    # loads a directory into the dictionary structure of the class
    def read_dir(self, input_dir):
        regex_well = re.compile(self.well_regex)
        regex_frame = re.compile(self.frame_regex)
        regex_time = re.compile(self.time_regex)
        regex_file = re.compile(self.file_regex)
        files = glob.glob(input_dir + "/**/**" + self.file_regex, recursive=True)
        # TODO: correctly implement file ext regex
        # files = filter(lambda x: regex_file.search(x), files)

        for file in files:
            file_title = os.path.splitext(os.path.split(file)[1])[0]
            cell, frame, time = (
                regex_well.search(file_title),
                regex_frame.search(file_title),
                regex_time.search(file_title),
            )
            self._dir[cell.group()][int(time.group(1))][int(frame.group(1))] = file

    # TODO: parametrize img_map and img_size
    def stich_well(self, well, time, img_map=(5, 5), img_size=(1040, 1392)):
        # stack of frames in left-to-right, up-down order
        img_stack = np.stack(
            [
                mpimg.imread(self._dir[well][time][file])
                for file in sorted(self._dir[well][time])
            ]
        )
        # insert blank frames
        img_stack = np.insert(img_stack, (0, 3, 18, 21), 0, axis=0)
        # reshape row_frame,row_well,column_frame,column_well
        img_grid = np.reshape(
            img_stack, (img_map[0], img_map[1], img_size[0], img_size[1])
        )
        img_grid = np.moveaxis(img_grid, 1, 2)
        # collapse dims into final rows, columns
        img_final = np.reshape(img_grid, (img_grid.shape[0] * img_grid.shape[1], -1))
        return img_final

    def write_well(self, well, out_file):
        well_array = [
            self.stich_well(well, time) for time in range(1, self.timepoints + 1)
        ]
        with h5.File(out_file, "w") as f:
            dset = f.create_dataset("stiched_well", data=well_array)


# TODO: implement main function for command line usage
def __main__(arg):
    pass


if __name__ == "__main__":
    main()
