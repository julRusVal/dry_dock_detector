# %%
import typing
import os

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


# %%
class PreprocessData:
    def __init__(self, sss_data_path: str, nadir_annotations_path: str, wall_annotations_path: str,
                 start_arr_index: int = 0, end_arr_index: int = 0,
                 per_channel_range: int = 0):
        """
        Prepare data for training and ...
        [ ] - Load date and the wall and nadir annotations
        [ ] - Trim the data according to the start and end indices
        [ ] - Split the data by channel and truncate according to the per channel range


        :param sss_data_path: Path to the sss data
        :param nadir_annotations_path: path to the nadir annotations
        :param wall_annotations_path: path to the wall annotations
        :param start_arr_index:
        :param end_arr_index:
        :param per_channel_range:
        """

        # Initialize class attributes, from parameters
        self.data_paths = {"sss": sss_data_path,
                           "nadir": nadir_annotations_path,
                           "wall": wall_annotations_path}

        self.start_arr_index = start_arr_index
        self.end_arr_index = end_arr_index
        self.per_channel_range = per_channel_range

        # Initialize class attributes, other
        self.sss_arr_orig = None  # Originals contain the originals: pre truncating and filtering
        self.nadir_arr_orig = None
        self.wall_arr_orig = None

        self.sss_arr = None
        self.nadir_arr = None
        self.wall_arr = None

        self.grad_arr = None

        self.status_limit_checked = False
        self.status_truncated = False
        self.status_filtered = False

        self.operations_list = []

        # output
        self.output_path = ""

        # Plotting parameters
        # Plotting colors are given in RGB (Damn you openCV!!)
        self.nadir_color = np.array([0, 255, 0], dtype=np.uint8)
        self.wall_color = np.array([255, 0, 0], dtype=np.uint8)

        # Start preprocessing data
        # Load data: sss, nadir, wall
        self.sss_arr_orig = self.load_image(self.data_paths["sss"])
        self.nadir_arr_orig = self.load_image(self.data_paths["nadir"])
        self.wall_arr_orig = self.load_image(self.data_paths["wall"])

        # Perform some basic checks on the data and on the provided truncating limits
        self.check_data()
        self.check_truncation_limits()

        # Perform truncation
        self.sss_arr, self.nadir_arr, self.wall_arr = self.truncate_data()

        # self.show_annotation()

        self.grad_arr = self.down_range_gradient(remove_ringdown=6, show=False)

    # Function to load an image and convert it to a NumPy array
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image not found or the path is incorrect")
        return image

    def check_truncation_limits(self):
        """
        Check that the truncation respects the size of the data and makes sense.
        Truncation is performed on the originals.

        :return:
        """

        shape = self.sss_arr_orig.shape

        # Check truncating of pings
        if self.start_arr_index < 0 or self.start_arr_index >= shape[0]:
            self.start_arr_index = 0

        if self.end_arr_index >= shape[0] or self.end_arr_index <= 0:
            self.end_arr_index = shape[0]

        if self.start_arr_index >= self.end_arr_index:
            raise ValueError("Improperly set start and end")

        # Check truncation of range
        if self.per_channel_range <= 1 or self.per_channel_range > shape[1] // 2:
            self.per_channel_range = shape[1] // 2

        self.status_limit_checked = True

    def check_data(self):
        """
        Basic check that the data and the annotations have the same sizes
        :return:
        """

        if any(element is None for element in [self.sss_arr_orig, self.nadir_arr_orig, self.wall_arr_orig]):
            raise ValueError("Incomplete data")

        if (self.sss_arr_orig.shape != self.nadir_arr_orig.shape or
                self.sss_arr_orig.shape != self.wall_arr_orig.shape):
            raise ValueError("Size mismatch")

    def truncate_data(self):
        """

        :param return_truncated_original_data:
        :return:
        """
        if not self.status_limit_checked:
            self.check_truncation_limits()

        # track: refers to the ping index, height
        # Straight forward

        # range: refers to the range, width
        # Note that for now the two channels are being processed together

        start_range_ind = self.sss_arr_orig.shape[1] // 2 - self.per_channel_range
        end_range_ind = self.sss_arr_orig.shape[1] // 2 + self.per_channel_range

        trun_sss_arr_orig = self.sss_arr_orig[self.start_arr_index:self.end_arr_index,
                            start_range_ind:end_range_ind]
        trun_nadir_arr_orig = self.nadir_arr_orig[self.start_arr_index:self.end_arr_index,
                              start_range_ind:end_range_ind]
        trun_wall_arr_orig = self.wall_arr_orig[self.start_arr_index:self.end_arr_index,
                             start_range_ind:end_range_ind]

        return trun_sss_arr_orig, trun_nadir_arr_orig, trun_wall_arr_orig

    def set_working_to_truncated_original(self):
        """
        :return:
        """

        self.sss_arr, self.nadir_arr, self.wall_arr = self.truncate_data()

        self.operations_list = []

    def show_annotation(self):
        """
        Show the annotation. This can be modified or refactored to show the detector output  as well
        :return:
        """

        sss_arr_cpy = np.copy(self.sss_arr)
        img_color = np.dstack((sss_arr_cpy, sss_arr_cpy, sss_arr_cpy))
        img_combined = np.copy(img_color)

        # Add nadir
        img_combined[self.nadir_arr > 127] = self.nadir_color
        img_combined[self.wall_arr > 127] = self.wall_color

        plt.imshow(img_combined)
        plt.title("SSS returns with annotations")
        plt.axis('off')
        plt.show()

    def filter_median(self, kernel_size=5, show=False, save=False):
        if kernel_size not in [3, 5, 7, 9]:
            return

        # Before filter image
        img_before = np.copy(self.sss_arr)

        # Perform median filter
        self.sss_arr = cv.medianBlur(self.sss_arr, kernel_size)

        if show:
            med_fig, (ax1, ax2) = plt.subplots(1, 2)
            med_fig.suptitle(f'Median filter, Kernel: {kernel_size}\n'
                             f'Previous Operations: {self.operations_list}')

            ax1.title.set_text('Before filtering')
            ax1.imshow(img_before)

            ax2.title.set_text('After filtering')
            ax2.imshow(self.sss_arr)
            med_fig.show()

        # Record the operation
        self.operations_list.append(f'm_{kernel_size}')

        # Save output
        if save:
            data_label = self.sss_arr.shape[0]
            output_file_path = os.path.join(self.output_path, f'{data_label}_med.png')
            cv.imwrite(output_file_path, self.sss_arr)

    def filter_gaussian(self, kernel_size=5, show=False, save=False):
        if kernel_size not in [3, 5, 7, 9]:
            return

        # Before filter image
        img_before = np.copy(self.sss_arr)

        # Perform Gaussian filter
        self.sss_arr = cv.GaussianBlur(self.sss_arr, (kernel_size, kernel_size), 0)

        if show:
            med_fig, (ax1, ax2) = plt.subplots(1, 2)
            med_fig.suptitle(f'Gaussian filter, Kernel: {kernel_size}\n'
                             f'Previous Operations: {self.operations_list}')

            ax1.title.set_text('Before filtering')
            ax1.imshow(img_before)

            ax2.title.set_text('After filtering')
            ax2.imshow(self.sss_arr)
            med_fig.show()

        # Record the operation and save the output
        self.operations_list.append(f'gauss_{kernel_size}')

        if save:
            output_file_path = os.path.join(self.output_path, f'{self.data_label}_gauss.png')
            cv.imwrite(output_file_path, self.sss_arr)

    def down_range_gradient(self, m_size=5, g_size=5, s_size=5, remove_ringdown=0, show=True, save=False):
        """
        Performs a down range gradient for both channels. Allows for some amount of pre-filtering.

        :param s_size: size of sobel kernel, [3, 5, 7, 9]
        :param g_size: size of gaussian filter, [3, 5, 7, 9]
        :param m_size: size of median filter, [3, 5, 7, 9]
        :param remove_ringdown:
        :param save:
        :param show:
        :return:
        """

        self.set_working_to_truncated_original()

        if m_size in [3, 5, 7, 9]:
            self.filter_median(m_size)

        if g_size in [3, 5, 7, 9]:
            self.filter_gaussian(g_size)

        # The down range gradient requires the data to be split by channel
        channel_size = self.sss_arr.shape[1] // 2
        img_port = np.fliplr(self.sss_arr[:, 0:channel_size])
        img_starboard = self.sss_arr[:, channel_size:]

        ringdown_ind = np.clip(remove_ringdown, 0, channel_size)

        if s_size in [3, 5, 7, 9]:
            dx_port = cv.Sobel(img_port, cv.CV_16S, 1, 0, ksize=s_size)
            dx_star = cv.Sobel(img_starboard, cv.CV_16S, 1, 0, ksize=s_size)

            # - Gradients along each ping -
            # Combined gradient
            # Used for visualizing
            dx_port[:, :ringdown_ind] = 0  # Remove ringdown, mostly needed for negative gradient
            dx_star[:, :ringdown_ind] = 0

            dx = np.hstack((np.fliplr(dx_port), dx_star))  # .astype(np.int16)

            # Negative gradient
            # Used for visualizing
            dx_neg = np.copy(dx)
            dx_neg[dx_neg > 0] = 0  # remove positive gradients
            dx_neg = np.abs(dx_neg)
            dx_neg = dx_neg.astype(np.int16)

            # Positive gradient
            # Used for edge detection
            dx_pos = np.copy(dx)
            dx_pos[dx_pos < 0] = 0
            dx_pos = dx_pos.astype(np.int16)

            # dy = cv.Sobel(self.img, cv.CV_16S, 0, 1, ksize=s_size)
            dy = np.zeros_like(dx)

            if show:
                downrange_grad_fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
                downrange_grad_fig.suptitle(f'Down Range Gradient - '
                                            f'm_size: {m_size}  g_size: {g_size}, s_size: {s_size}')

                ax1.title.set_text('Input image')
                ax1.imshow(self.sss_arr)

                ax2.title.set_text('Combined dx')
                ax2.imshow(dx)

                ax3.title.set_text('Positive dx')
                ax3.imshow(dx_pos)

                ax4.title.set_text('Negative dx')
                ax4.imshow(dx_neg)

                # custom_canny_fig.show()
                plt.gcf().set_dpi(300)
                plt.show()

            if save:
                output_file_path = os.path.join(self.output_path, 'down_range_gradient.png')
                cv.imwrite(output_file_path, dx)

            return dx, dx_pos, dx_neg

    # # Function to preprocess the image
    # def preprocess_image(image, target_size=(256, 256)):
    #     # Resize the image
    #     resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    #
    #     # Convert to grayscale
    #     grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    #
    #     # Normalize pixel values to range [0, 1]
    #     normalized_image = grayscale_image / 255.0
    #
    #     return normalized_image

    # # Function to display the image
    # def display_image(self.image, title='Image'):
    #     plt.imshow(image, cmap='gray')
    #     plt.title(title)
    #     plt.axis('off')
    #     plt.show()


class ReformatData:
    def __init__(self, sss_data: np.ndarray, nadir_annotations: np.ndarray, wall_annotations: np.ndarray,
                 slice_size: int = 10, step_size: int = 5, flipped_copies: bool = True):
        """
            Reformat the data into sub images and annotations.
            Operations:
                - flip (left/right) the port channel, as only one channel at a time will be used for training
                - slice input images and annotations into sub images and annotations, [N,h,w]
                    - N: number of sub images/annotations, determined by slice_size
                    - h: height of sub images/annotations, h = slice_size
                    - w: width of sub images/annotations, w = length of each channel, len(input data) // 2
                - Produce flipped (up/down) images and annotations for more training and testing data

            :param sss_data:
            :param nadir_annotations:
            :param wall_annotations:
            :param slice_size:
        """

        self.sss_input = sss_data
        self.nadir_input = nadir_annotations
        self.wall_input = wall_annotations
        self.slice_size = slice_size
        self.step_size = step_size
        self.flipped_copies = flipped_copies

        self.sss_formatted = None
        self.nadir_formatted = None
        self.wall_formatted = None

        # Check that all the sizes are in agreement
        if sss_data.shape != nadir_annotations.shape or sss_data.shape != wall_annotations.shape:
            raise ValueError("Shape mismatch")

        self.orig_h, self.orig_w = self.sss_input.shape

        orig_h, orig_w = self.sss_input.shape

        # Check provided slice_size value
        if self.slice_size < 1:
            self.slice_size = 1
            print("Invalid slice size -> setting slice size to 1")
        elif self.slice_size > orig_h:
            self.slice_size = 1
            print("Invalid slice size -> setting slice size to 1")

        self.slice_count = (orig_h - slice_size + 1)

        # Check provided step_size value
        if self.step_size < 1:
            self.step_size = 1

        self.step_count = (self.slice_count - 1) // self.step_size + 1

        self.sss_formatted = self.SplitChannelsSliceAndCombine(array=sss_data)
        self.nadir_formatted = self.SplitChannelsSliceAndCombine(array=nadir_annotations)
        self.wall_formatted = self.SplitChannelsSliceAndCombine(array=wall_annotations)

        if self.flipped_copies:
            self.sss_formatted = self.AddFlippedCopies(self.sss_formatted)
            self.nadir_formatted = self.AddFlippedCopies(self.nadir_formatted)
            self.wall_formatted = self.AddFlippedCopies(self.wall_formatted)

    def SliceArray(self, input_array):
        """
        Slices the array into a sub images.
        For an NxM input array and a slice_size of S
        output will be (N-S)+1xSxM
        Where (N-S)+1 is the number of slices of size S one gets from an array of length N, in the relevant dimension

        :param input_array:
        :return:
        """

        input_height, input_width = input_array.shape

        output_array = np.zeros((self.step_count, self.slice_size, input_width))  # .astype(int)

        for out_i, in_i in enumerate(range(0, self.slice_count, self.step_size)):
            output_array[out_i, :, :] = input_array[in_i:in_i + self.slice_size, :]

        return output_array

    def SplitChannelsSliceAndCombine(self, array: np.ndarray):
        """
        Splits the port and starboard, flips port, slices, and then recombines
        :param array:
        :return:
        """
        array_width = array.shape[1]

        # sss data
        array_port = array[:, 0:array_width // 2]
        array_port[:] = array_port[:, ::-1]  # Port data is flipped
        array_star = array[:, array_width // 2:]

        array_port_sliced = self.SliceArray(array_port)
        array_star_sliced = self.SliceArray(array_star)

        array_sliced = np.vstack((array_port_sliced, array_star_sliced))

        return array_sliced

    @staticmethod
    def AddFlippedCopies(array: np.ndarray):
        """
        Input is an [N,h,w] numpy array.
        Output is an [2*N,h,w] numpy array
        :param array:
        :return:
        """
        # Construct the flipped array
        array_flipped = np.copy(array)
        array_flipped[:] = array_flipped[:, ::-1, :]

        output_array = np.vstack((array, array_flipped))

        return output_array

    def PlotFormattedDataDebug(self, frame_interval=0.1):
        """
        Plots the formatted data, the first dimension represents the image number -> [N,h,w]
        :param frame_interval:
        :param formatted_array:
        :return:
        """
        # Create a figure and axis
        fig, ax = plt.subplots()

        frame_count = self.sss_formatted.shape[0]

        for frame_idx in range(frame_count):
            ax.clear()  # Clear the previous frame
            ax.set_title(f"Frame: {frame_idx}")
            ax.imshow(self.sss_formatted[frame_idx, :, :], cmap='viridis')  # Display the current frame
            plt.draw()  # Update the figure
            plt.pause(frame_interval)  # Pause for the interval

        # Optionally keep the last frame displayed
        plt.show()


class TorchDataset(Dataset):
    def __init__(self, sss_array, annotation_array, transform=None, annotation_transform=None):
        self.sss_array = sss_array
        self.annotation_array = annotation_array
        self.transform = transform
        self.target_transform = annotation_transform

    def __len__(self):
        return self.sss_array.shape[0]

    def __getitem__(self, idx):
        sss = torch.from_numpy(self.sss_array[idx, :, :]).float()
        annotation = torch.from_numpy(self.sss_array[idx, :, :]).float()

        # if self.transform:
        #     sss = self.transform(sss)
        # if self.target_transform:
        #     annotation = self.target_transform(annotation)

        return sss, annotation


def main():
    sss_path = "data/sss_data_6436.jpg"
    nadir_path = "data/sss_data_6436_nadir.jpg"
    wall_path = "data/sss_data_6436_wall.jpg"

    # Perform preprocessing
    preprocessed = PreprocessData(sss_data_path=sss_path,
                                  nadir_annotations_path=nadir_path,
                                  wall_annotations_path=wall_path,
                                  start_arr_index=0,
                                  end_arr_index=0,
                                  per_channel_range=500)

    # Display original and preprocessed images

    # Data selection
    sss_preproc = preprocessed.sss_arr  # Gradient: preprocessed.sss_arr
    nadir_preproc = preprocessed.nadir_arr
    wall_preproc = preprocessed.wall_arr

    # Reformatted
    formatted = ReformatData(sss_data=sss_preproc,
                             nadir_annotations=nadir_preproc,
                             wall_annotations=wall_preproc,
                             slice_size=10,
                             step_size=10,
                             flipped_copies=True)

    # Plot for debugging
    formatted.PlotFormattedDataDebug()

    dataset = TorchDataset(sss_array=formatted.sss_formatted,
                           annotation_array=formatted.wall_formatted)


    print("Success!")




if __name__ == "__main__":
    main()
