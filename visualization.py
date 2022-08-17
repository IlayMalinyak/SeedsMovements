# __author:IlayK
# data:02/05/2022

# __author:IlayK
# data:28/04/2022
from matplotlib import  pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go
from abc import ABC, abstractmethod
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class BasicViewer(ABC):

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def update_slice(self, val):
        pass


class DicomViewer(BasicViewer):
    """
    Dicom viewer class
    """
    def __init__(self, array, canvas, name, mask=None):
        self.name = name
        self.array = array
        self.canvas = canvas
        self.mask = mask
        # self.canvas.bind_all("<MouseWheel>", self.update_slice)
        self.fig, self.ax = plt.subplots(1, 1)
        self.figure_canvas_agg = FigureCanvasTkAgg(self.fig, self.canvas)
        # self.figure_canvas_agg.get_tk_widget().forget()
        if len(array.shape) == 3:
            rows, cols, self.slices = array.shape
            self.channels = 0
        else:
            rows, cols, self.channels, self.slices = self.array.shape
        self.ind = self.slices//2
        self.axSlice = plt.axes([0.1, 0.18, 0.05, 0.63])
        self.slice_slider = Slider(
            ax=self.axSlice,
            label='Slice',
            valmin=1,
            valmax=self.slices,
            valinit=self.ind,
            valstep=1, color='magenta',
            orientation="vertical"
        )
        self.slice_slider.on_changed(self.update_slice)
        plt.subplots_adjust(left=0.25, bottom=0.1)

    def show(self):
        self.im = self.ax.imshow(self.array[:,:,self.ind - 1], cmap='gray')
        self.struct = self.ax.imshow(self.mask[..., self.ind - 1], cmap='cool', interpolation='none',
                                alpha=1) if self.mask is not None else None
        self.figure_canvas_agg.draw()
        self.figure_canvas_agg.get_tk_widget().pack(side='top', fill='both')

    def update_slice(self, val):
        # print("scroll event on ", self.name)
        # self.ind = int(abs(self.slice_slider.val - self.slices))
        self.ind = self.slice_slider.val
        # print(self.ind, event.y)
        try:
            self.im.set_data(self.array[..., self.ind - 1])
        except IndexError:
            pass
        if self.mask is not None:
            self.struct.set_data(self.mask[..., self.ind - 1])
        self.im.axes.figure.canvas.draw()
        # self.show()

    def set_array(self, array):
        # self.axSlice.clear()
        # self.ax.clear()
        self.array = array
        # if len(array.shape) == 3:
        #     rows, cols, self.slices = array.shape
        #     self.channels = 0
        # else:
        #     rows, cols, self.channels, self.slices = self.array.shape
        # self.ind = self.slices//2
        # self.axSlice = plt.axes([0.1, 0.18, 0.05, 0.63])
        # self.slice_slider = Slider(
        #     ax=self.axSlice,
        #     label='Slice',
        #     valmin=1,
        #     valmax=self.slices,
        #     valinit=self.ind,
        #     valstep=1, color='magenta',
        #     orientation="vertical"
        # )
        # self.slice_slider.on_changed(self.update_slice)
        # plt.subplots_adjust(left=0.25, bottom=0.1)


    def set_struct(self, array):
        self.mask = array

    def set_ind(self, ind):
        self.ind = ind

    def clear(self):
        self.figure_canvas_agg.get_tk_widget().forget()


class OverlayViewer(DicomViewer):
    """
    dicoms overlay viewer class
    """
    def __init__(self, array1, array2, canvas, name, mask1=None, mask2=None):
        super(OverlayViewer, self).__init__(array1, canvas, name, mask1)
        self.array2 = array2
        self.alpha = 0.3
        # self.mask1 = mask1
        self.mask2 = mask2
        axAlpha = plt.axes([0.25, 0.02, 0.63, 0.05])
        self.alpha_slider = Slider(
            ax=axAlpha,
            label='Alpha',
            valmin=0,
            valmax=1,
            valinit=self.alpha,
            color='magenta',
        )
        self.alpha_slider.on_changed(self.update_alpha)
        plt.subplots_adjust(left=0.25, bottom=0.1)

    def show(self):
        self.im = self.ax.imshow(self.array[:, :, self.ind - 1], cmap='gray')
        self.struct = self.ax.imshow(self.mask[..., self.ind - 1], cmap='cool', interpolation='none',
                                     alpha=1) if self.mask is not None else None
        ind2 = self.ind % self.array2.shape[2]
        # if self.ind < self.array2.shape[-1]:
        self.overlay = self.ax.imshow(self.array2[..., ind2 - 1], cmap='gist_heat', interpolation='none',
                                 alpha=self.alpha)
        self.struct2 = self.ax.imshow(self.mask2[..., self.ind - 1], cmap='gist_yarg', interpolation='none',
                                      alpha=1) if self.mask2 is not None else None
        # else:
        #     self.overlay = self.ax.imshow(self.array2[..., -1], cmap='gist_heat', interpolation='none',
        #                                   alpha=self.alpha)
        self.figure_canvas_agg.draw()
        self.figure_canvas_agg.get_tk_widget().pack(side='top', fill='both')

    def update_slice(self, val):
        self.ind = int(self.slice_slider.val)
        # print(self.ind, event.y)
        self.im.set_data(self.array[..., self.ind - 1])
        if self.mask is not None:
            self.struct.set_data(self.mask[..., self.ind - 1])
        ind2 = self.ind % self.array2.shape[2]
        self.overlay.set_data(self.array2[..., ind2 - 1])
        if self.mask2 is not None:
            self.struct2.set_data(self.mask2[..., ind2 - 1])
        self.im.axes.figure.canvas.draw()
        # self.show()

    def update_alpha(self, val):
        self.alpha = self.alpha_slider.val
        self.overlay.set_alpha(self.alpha)

    def set_arrays(self, arr1, arr2):
        self.array = arr1
        self.array2 = arr2

        # self.ind = self.array.shape[2] // 2

    def set_masks(self, mask1, mask2):
        self.mask = mask1
        self.mask2 = mask2


def parse_lists_from_file(path):
    with open(path, 'r') as f:
        list1 = f.readline().strip().split(',')
        list2 = f.readline().strip().split(',')
    return list1, list2


class viewer():
    """
    class to view DICOMS
    """
    def __init__(self, ax, X, mask=None, aspect=1.0):
        self.ax = ax
        ax.set_title('Scroll to Navigate through the DICOM Image Slices')
        self.X = X
        self.mask = mask
        self.alpha = 0.4
        self.aspect = aspect
        if len(X.shape) == 3:
            rows, cols, self.slices = X.shape
            self.channels = 0
        else:
            rows, cols, self.channels, self.slices = X.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.X[..., self.ind], cmap='gray')
        self.overlay = ax.imshow(self.mask[..., self.ind], cmap='Reds', interpolation='none',
                                 alpha=self.alpha) if self.mask is not None else None
        # self.ax.set_position([0.25, 0,1,1])
        if self.mask is not None:
            axAlph = plt.axes([0.2, 0.02, 0.65, 0.03])
            self.alpha_slider = Slider(
                ax=axAlph,
                label='Alpha',
                valmin=0,
                valmax=1,
                valinit=0.4,
            )
            self.alpha_slider.on_changed(self.update_alpha)
        ax.set_aspect(aspect)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def onkey(self, event):
        if event.key == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def update_alpha(self, val):
        self.alpha = self.alpha_slider.val
        self.overlay.set_alpha(self.alpha)

    def update(self):
        self.im.set_data(self.X[..., self.ind])
        if self.mask is not None:
            self.overlay.set_data(self.mask[:, :, self.ind])
        self.ax.set_ylabel('Slice Number: %s' % self.ind)
        # cid = self.ax.canvas.mpl_connect('key_press_event', self.on_key)
        self.im.axes.figure.canvas.draw()



def show(arr, plane='axial'):
    """
    display axial plane interactively
    :param arr: array of data
    :param contours: array of contours
    :param seeds: array of seeds
    :param aspect: aspect ratio
    :param ax: matplotlib ax object (if None, a new one wil be created)
    """
    if plane == 'coronal':
        arr = np.swapaxes(arr, 0, 2)
    if plane == "sagittal":
        arr = np.swapaxes(arr, 0, 1)
    aspect = arr.shape[1]/arr.shape[0]
    fig, ax = plt.subplots(1, 1)
    # masked_seed_arr = np.ma.masked_where(seeds == 255, seeds) if seeds is not None else None
    tracker = viewer(ax, arr, aspect=aspect)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.onkey)

    return plt.gcf()


def overlay_images(fixed_image, moving_image):
    """
    overlay two image in the same plot
    :param fixed_image: first image
    :param moving_image: second image
    """
    fig, ax = plt.subplots(1, 1)
    aspect = fixed_image.shape[1]/fixed_image.shape[0]
    # masked_seed_arr = np.ma.masked_where(seeds == 255, seeds) if seeds is not None else None
    tracker = viewer(ax, fixed_image, mask=moving_image, aspect=aspect)
    # fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.onkey)

    plt.show()


def display_slices(moving_image, fixed_image):
  """
  Displaying our two image tensors to register
  :param moving_image: [IM_SIZE_0, IM_SIZE_1, 3]
  :param fixed_image:  [IM_SIZE_0, IM_SIZE_1, 3]
  """
  # Display
  idx_slices = [int(5+x*5) for x in range(int(fixed_image.shape[3]/5)-1)]
  nIdx = len(idx_slices)
  plt.figure()
  for idx in range(len(idx_slices)):
      axs = plt.subplot(nIdx, 2, 2*idx+1)
      axs.imshow(moving_image[0,...,idx_slices[idx]], cmap='gray')
      axs.axis('off')
      axs = plt.subplot(nIdx, 2, 2*idx+2)
      axs.imshow(fixed_image[0,...,idx_slices[idx]], cmap='gray')
      axs.axis('off')
  plt.suptitle('Moving Image - Fixed Image', fontsize=200)
  plt.show()


def plot_individual_moves(case, dists, error, save=True):
    """
    plot individuals movements
    :param case: study id
    :param dists: array if distances
    :param error: array of errors (same length as knn_dists)
    :return: None
    """
    fig = plt.figure()
    # ax = fig.add_subplot()
    x = np.arange(1, len(dists) + 1)
    y = dists
    # plt.scatter(x, y, color="red")
    plt.errorbar(x, y, np.average(error, axis=0), fmt="o", color="b")

    plt.axhline(y=np.average(y), xmax=max(x), linestyle="--", label="Average", color="black")
    plt.title("%s individual moves" % case)
    plt.xlabel("# seed")
    plt.ylabel("movement (mm)")
    # plt.ylim((0,15))
    plt.legend()
    if save:
        plt.savefig("./movement_output/%s/movements.png" % case)
    plt.close()


def plot_pairs(seeds1, seeds2, case, save=True):
    """
    plot seeds pairs
    :param seeds1: (3,3,N) array of first seeds. the first axis represent 3 tips (start, middle,end). the second axis represent
            3 coordinates (x,y,z)
    :param seeds2: (3,3,N) array of second seeds.
    :param case: case name
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    n = seeds1.shape[-1]
    for i in range(n):
        ax.plot([seeds1[0,0, i], seeds1[2,0, i]], [seeds1[0,1, i], seeds1[2,1, i]], [seeds1[0,2, i], seeds1[2,2, i]], color='b')
        ax.plot([seeds2[0,0, i], seeds2[2,0, i]], [seeds2[0,1, i], seeds2[2,1, i]], [seeds2[0,2, i], seeds2[2,2, i]], color='orange')
        ax.plot([seeds1[1,0, i], seeds2[1,0, i]], [seeds1[1,1, i], seeds2[1,1, i]], [seeds1[1,2, i], seeds2[1,2, i]], color='green', alpha=0.5,
                linestyle='--')
    ax.set_title(case)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if save:
        plt.savefig("./movement_output/%s/pairs.png" % (case))
    plt.close()


def display_dists(seeds1, seeds2, title, fname, save=True):
    """
    plot matchs and distances
    :param seeds1: 3XN array of first seeds
    :param seeds2: 3XN array of second seeds
    :param title: title for plot
    :param fname: file name for saving
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(seeds1.shape[-1]):
        if i == 0:
            ax.scatter(seeds1[0,i], seeds1[1,i], seeds1[2,i], color='b', label='postop')
            ax.scatter(seeds2[0,i], seeds2[1,i], seeds2[2,i], color='orange', label='removal')
        else:
            ax.scatter(seeds1[0, i], seeds1[1, i], seeds1[2, i], color='b')
            ax.scatter(seeds2[0, i], seeds2[1, i], seeds2[2, i], color='orange')
        ax.plot([seeds1[0, i], seeds2[0, i]], [seeds1[1, i], seeds2[1, i]], [seeds1[2, i],seeds2[2, i]], color='r')
    plt.legend()
    plt.title(title)
    if save:
        plt.savefig("us/graphs/%s" % fname)
    plt.close()


def plot_seeds(seeds_tips, ax: object = None, title: object = None, color: object = None) -> object:
    """
    plot seeds
    :param seeds_tips: (3,3,N) array of seeds tips
    :param ax: matplotlib axes object (if None, a new one will be created)
    :param title: title of the plit
    :param color: color of seeds (if not specified, each seeds will be in different color)
    :return: ax object contains seeds plot. this can be showed using plt.show() command
    """
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    for i in range(seeds_tips.shape[-1]):
        x = seeds_tips[:, 0, i]
        y = seeds_tips[:, 1, i]
        z = seeds_tips[:, 2, i]
        if color is None:
            ax.plot(x, y, z, label="seed number %d" % (i + 1))
        else:
            ax.plot(x, y, z, color=color)

        if title is not None:
            plt.title(title)
        # plt.legend()
    return ax


def overlay_contours(ctr1, ctr2, path, save=True):
    """
    plot 2 contours together
    :param ctr1: np array (N,3) of physical coordinates
    :param ctr2: np array (N,3) of physical coordinates
    :param path: path for saving
    :param save: flag for saving
    """
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ctr1[:,0], ctr1[:,1], ctr1[:,2], alpha=0.01, label='1')
    ax.scatter(ctr2[:,0], ctr2[:,1], ctr2[:,2], alpha=0.01, color='lime',label='2')
    ax.set_xlabel("X")
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("contours overlay")
    handles = []
    patch = mpatches.Patch(color='blue', label='fixed')
    handles.append(patch)
    patch = mpatches.Patch(color='lime', label='moving')
    handles.append(patch)
    plt.legend(handles=handles, loc='upper left')
    if save:
        plt.savefig(path)
    plt.close()


def overlay_contours_interactive(ctrs1, ctrs2):
    """
    plotly interactive plot of 2 contours
    :param ctr1: np array (N,3) of physical coordinates
    :param ctr2: np array (N,3) of physical coordinates
    """
    trace1 = go.Scatter3d(x=ctrs1[:, 0],y=ctrs1[:, 1],z=ctrs1[:, 2],name="fixed", opacity=0.005)
    trace2 = go.Scatter3d(x=ctrs2[:, 0],y=ctrs2[:, 1],z=ctrs2[:, 2],name='moving', opacity=0.005)
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig['layout'].update(height=600, width=800, title="contours overlay")
    print("sohwing plotly figure")
    fig.write_html('contours_overlay.html', auto_open=True)
    # fig.show()


def plot_rmse(rmse, path, save=True):
    """
    plot rmse from icp registration
    :param rmse: array of rmse
    :param path: path to save
    :param save: flag for saving
    """
    fig = plt.figure()
    plt.plot(np.arange(len(rmse)), rmse)
    plt.title( "ICP convergence")
    if save:
        plt.savefig(path)
    plt.close()


def plot_rmse_and_contours(rmse, ctr1, ctr2, path, save=True):
    """
    plot rmse and contour overlay in the same plot
    :param rmse: array of rmse
    :param ctr1: np array (N,3) of physical coordinates
    :param ctr2: np array (N,3) of physical coordinates
    :param path: path to save
    :param save: flag for saving
    """
    plt.close("all")
    fig = plt.figure(figsize=plt.figaspect(0.66))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(np.arange(len(rmse)), rmse)
    ax.set_title("ICP convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Inliers Root Mean Squared Error (mm)")
    # Second subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(ctr1[:, 0], ctr1[:, 1], ctr1[:, 2], alpha=0.1, label='fixed', color='gray')
    ax.scatter(ctr2[:, 0], ctr2[:, 1], ctr2[:, 2], alpha=0.1, color='red', label='moving')
    ax.set_title("Contours Overlay")
    if save:
        plt.savefig(path)
    plt.close()






