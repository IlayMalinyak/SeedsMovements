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
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec



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
        self.rect = Rectangle((0, 0), 1, 1, fill=False, color='red')
        self.x0 = 0
        self.y0 = 0
        self.x1 = self.array.shape[1]
        self.y1 = self.array.shape[0]
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
        self.draw_rect()
        # self.show()

    def set_array(self, array):
        self.array = array

    def set_struct(self, array):
        self.mask = array

    def set_ind(self, ind):
        self.ind = ind

    def clear(self):
        self.figure_canvas_agg.get_tk_widget().forget()

    def on_press(self, event):
        if event.xdata > 3:
            self.x0 = event.xdata
            self.y0 = event.ydata

    def on_release(self, event):
        if event.xdata > 3:
            self.x1 = event.xdata
            self.y1 = event.ydata
            print(self.x1, self.y1, self.x0, self.y0)
            self.draw_rect()

    def draw_rect(self):
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()

    def get_rect(self):
        return self.y0, self.x0, self.y1, self.x1

    def annotate(self):
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)


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
    """
    split a 2 lines with comma
    :param path: path to file
    :return: two lists
    """
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
            print("up")
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def onkey(self, event):
        if event.key == 'up':
            print("up")

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
    :param save: flag for saving. if True - .png file will be saved on './movement_output/<case>/movements.png
    :return: None
    """
    plt.figure()
    x = np.arange(1, len(dists) + 1)
    y = dists
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


def plot_individual_moves_outliers(case, dists, error, outliers_idx, save=True):
    """
    plot individual moves with outliers marked as red
    :param case: study id
    :param dists: array if distances
    :param error: array of errors (same length as knn_dists)
    :param outliers_idx: indexes of outliers
    :param save: flag for saving. if True - .png file will be saved on './movement_output/<case>/movements_outliers.png
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    x = np.arange(1, len(dists) + 1)
    # y = dists
    outliers_bool = np.array([False]*len(dists))
    outliers_bool[outliers_idx] = True
    outliers = dists[outliers_idx]
    outliers_idx = x[outliers_idx]
    inliers = dists[np.logical_not(outliers_bool)]
    inliers_idx = x[np.logical_not(outliers_bool)]
    inliers_error = error[np.logical_not(outliers_bool)]
    ax.errorbar(inliers_idx, inliers, np.average(inliers_error, axis=0), fmt="o", color="b")
    average = np.average(inliers)
    ax.axhline(y=average, xmax=max(x), linestyle="--", color="black", label='Average: {0:.2f}'.format(average))
    ax.scatter(outliers_idx, outliers, color='red')
    ax.set_title("%s individual moves" % case)
    ax.set_xlabel("# seed")
    ax.set_ylabel("movement (mm)")
    handles, labels = ax.get_legend_handles_labels()
    outliers_patch = mpatches.Patch(color="red", label='Outliers')
    handles.append(outliers_patch)
    # plt.ylim((0,15))
    plt.legend(handles=handles)
    if save:
        plt.savefig("./movement_output/%s/movements_outliers.png" % case)
    plt.close()


def plot_pairs(seeds1, seeds2, case, save=True):
    """
    plot seeds pairs
    :param seeds1: (3,3,N) array of first seeds. the first axis represent 3 tips (start, middle,end). the second axis represent
            3 coordinates (x,y,z)
    :param seeds2: (3,3,N) array of second seeds.
    :param case: case name
    :param save: flag for saving. if True - .png file will be saved on './movement_output/<case>/pairs.png

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
    fixed_patch = mpatches.Patch(color='b', label='Fixed Seeds')
    moving_patch = mpatches.Patch(color='orange', label='Moving Seeds')
    plt.legend(handles=[fixed_patch, moving_patch])

    if save:
        plt.savefig("./movement_output/%s/pairs.png" % (case))
    plt.close()

def plot_pairs_with_dose(seeds1, seeds2, dose1, dose2, case, alpha=0.01, save=True):
    """
    plot seeds pairs
    :param seeds1: (3,3,N) array of first seeds. the first axis represent 3 tips (start, middle,end). the second axis represent
            3 coordinates (x,y,z)
    :param seeds2: (3,3,N) array of second seeds.
    :param case: case name
    :param save: flag for saving. if True - .png file will be saved on './movement_output/<case>/pairs.png

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    n = seeds1.shape[-1]
    for i in range(n):
        ax.plot([seeds1[0,0, i], seeds1[2,0, i]], [seeds1[0,1, i], seeds1[2,1, i]], [seeds1[0,2, i], seeds1[2,2, i]], color='b')
        ax.plot([seeds2[0,0, i], seeds2[2,0, i]], [seeds2[0,1, i], seeds2[2,1, i]], [seeds2[0,2, i], seeds2[2,2, i]], color='orange')
        ax.plot([seeds1[1,0, i], seeds2[1,0, i]], [seeds1[1,1, i], seeds2[1,1, i]], [seeds1[1,2, i], seeds2[1,2, i]], color='green', alpha=0.5,
                linestyle='--')
        ax.scatter(dose1[:, 0], dose1[:, 1], dose1[:, 2], alpha=alpha, color='b', label='1')
        ax.scatter(dose2[:, 0], dose2[:, 1], dose2[:, 2], alpha=alpha, color='orange', label='2')
    ax.set_title(case)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fixed_patch = mpatches.Patch(color='b', label='Fixed Seeds')
    moving_patch = mpatches.Patch(color='orange', label='Moving Seeds')
    plt.legend(handles=[fixed_patch, moving_patch])

    if save:
        plt.savefig("./movement_output/%s/pairs_dose.png" % (case))
    plt.close()


def plot_pairs_with_outliers(seeds1, seeds2, outliers_idx, case, save=True):
    """

    :param seeds1: (3,3,N) array of first seeds. the first axis represent 3 tips (start, middle,end). the second axis represent
            3 coordinates (x,y,z)
    :param seeds2: (3,3,N) array of second seeds.
    :param outliers_idx: indexes of outliers
    :param case: case name
    :param save: flag for saving. if True - .png file will be saved on './movement_output/<case>/pairs_outliers.png
    :return: None
    """
    fig = plt.figure(figsize=(8,8 ))
    ax = fig.add_subplot(projection='3d')
    n = seeds1.shape[-1]
    for i in range(n):
        color1 = 'r' if i in outliers_idx else 'b'
        color2 = 'magenta' if i in outliers_idx else 'orange'
        ax.plot([seeds1[0, 0, i], seeds1[2, 0, i]], [seeds1[0, 1, i], seeds1[2, 1, i]],
                [seeds1[0, 2, i], seeds1[2, 2, i]], color=color1)
        ax.plot([seeds2[0, 0, i], seeds2[2, 0, i]], [seeds2[0, 1, i], seeds2[2, 1, i]],
                [seeds2[0, 2, i], seeds2[2, 2, i]], color=color2)
        # ax.plot([seeds1[1, 0, i], seeds2[1, 0, i]], [seeds1[1, 1, i], seeds2[1, 1, i]],
        #         [seeds1[1, 2, i], seeds2[1, 2, i]], color='green', alpha=0.5,
        #         linestyle='--')
    ax.set_title(case)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fixed_patch = mpatches.Patch(color='b', label='Fixed Seeds')
    moving_patch = mpatches.Patch(color='orange', label='Moving Seeds')
    fixed_out_patch = mpatches.Patch(color='r', label='Fixed Outliers')
    moving_out_patch = mpatches.Patch(color='magenta', label='Moving Outliers')
    plt.legend(handles=[fixed_patch, moving_patch, fixed_out_patch, moving_out_patch])

    if save:
        plt.savefig("./movement_output/%s/pairs_outliers.png" % (case))
    plt.close()


def display_dists(seeds1, seeds2, title, case, save=True):
    """
    plot matchs and distances
    :param seeds1: 3XN array of first seeds
    :param seeds2: 3XN array of second seeds
    :param title: title for plot
    :param case: case name
    :param save: flag for saving. if True - .png file will be saved on './movement_output/<case>/dists.png
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
        plt.savefig("./movement_output/%s/dists.png" % (case))
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


def overlay_contours(ctr1, ctr2, path, title = '', save=True, alpha=0.01):
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
    ax.scatter(ctr1[:,0], ctr1[:,1], ctr1[:,2], alpha=alpha, label='1')
    ax.scatter(ctr2[:,0], ctr2[:,1], ctr2[:,2], alpha=alpha, color='lime',label='2')
    ax.set_xlabel("X")
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"contours overlay {title}")
    handles = []
    patch = mpatches.Patch(color='blue', label='fixed')
    handles.append(patch)
    patch = mpatches.Patch(color='lime', label='moving')
    handles.append(patch)
    plt.legend(handles=handles, loc='upper left')
    if save:
        plt.savefig(path)
    plt.close()


def overlay_contours_interactive(ctrs1, ctrs2, title=''):
    """
    plotly interactive plot of 2 contours
    :param ctr1: np array (N,3) of physical coordinates
    :param ctr2: np array (N,3) of physical coordinates
    """
    trace1 = go.Scatter3d(x=ctrs1[:, 0],y=ctrs1[:, 1],z=ctrs1[:, 2],name="fixed", mode='markers', opacity=0.1)
    trace2 = go.Scatter3d(x=ctrs2[:, 0],y=ctrs2[:, 1],z=ctrs2[:, 2],name='moving', mode='markers', opacity=0.1)
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig['layout'].update(height=600, width=800, title="contours overlay")
    print("sohwing plotly figure")
    fig.write_html(f'contours_overlay_{title}.html', auto_open=True)
    # fig.show()


def plot_seeds_interactive(fig, seeds, title, color):
    """
    plot interactive seeds plot using plotly
    :param fig: plotly go.Figure object
    :param seeds: 3X3XN array of seeds
    :param title: plot title
    :param color: color of seeds
    :return: plotly go.Figure object
    """
    for i in range(seeds.shape[-1]):
        legend = True if i == 0 else False
        trace = go.Scatter3d(x=seeds[:,0,i],y=seeds[:,1,i], z=seeds[:,2,i], mode="lines", name=title,
                              marker={'color': color}, showlegend=legend)
        fig.add_trace(trace)
    return fig


def plot_seeds_and_contour_interactive(seeds1, seeds2, ctr):
    """
    plot interactively seeds inside contour
    :param seeds1: 3x3xN array
    :param seeds2: 3x3xN array
    :param ctr: Nx3 array
    :return: None
    """
    fig = go.Figure()
    trace1 = go.Scatter3d(x=ctr[:, 0], y=ctr[:, 1], z=ctr[:, 2], name="fixed", opacity=0.001, marker={'color':'cyan'},
                          showlegend=False)
    fig.add_trace(trace1)
    fig = plot_seeds_interactive(fig, seeds1, "Seeds 1", "blue")
    fig = plot_seeds_interactive(fig, seeds2, "Seeds 2", "orange")
    fig['layout'].update(height=600, width=800, title="contours overlay")
    print("sohwing plotly figure")
    fig.write_html('seeds_and_contour.html', auto_open=True)
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


def plot_loss_and_contours(arr_rmse, ctr1, ctr2, path, alpha=0.1, num_graphs=1, save=True):
    """
    plot rmse and contour overlay in the same plot
    :param arr_rmse: array of rmse (N,M) N = number of points, M = num_graphs
    :param ctr1: np array (N,3) of physical coordinates
    :param ctr2: np array (N,3) of physical coordinates
    :param path: path to save
    :param save: flag for saving
    """
    plt.close("all")
    fig = plt.figure(figsize=plt.figaspect(0.66), constrained_layout=True)
    gs = GridSpec(num_graphs, 2, figure=fig)
    # ax = fig.add_subplot(num_graphs, 2, 1)
    for i in range(num_graphs):
        ax = fig.add_subplot(gs[i, 0])
        rmse = arr_rmse[i] if num_graphs > 1 else arr_rmse
        ax.plot(np.arange(len(rmse)), rmse)
        ax.set_title(f"Convergence plot {i}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("metric")
    # Second subplot
    ax = fig.add_subplot(gs[:,1], projection='3d')
    ax.scatter(ctr1[:, 0], ctr1[:, 1], ctr1[:, 2], alpha=alpha, label='fixed', color='gray')
    ax.scatter(ctr2[:, 0], ctr2[:, 1], ctr2[:, 2], alpha=alpha, color='red', label='moving')
    ax.set_title("Contours Overlay")
    if save:
        plt.savefig(path)
    plt.close()






