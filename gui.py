# __author:IlayK
# data:03/05/2022

import PySimpleGUI as sg
from utils import *
from registration import register_sitk, warp_image_sitk, ContourRegistration, read_image, get_inverse_transform
import SimpleITK as sitk
import pandas as pd
from visualization import *
from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform
from scipy.ndimage import rotate, shift
import time
import dill as pickle


px = 1/plt.rcParams['figure.dpi']
plt.rcParams["figure.figsize"] = (500*px,500*px)
font_header = ("Arial, 18")
font_text = ("Ariel, 12")
sg.theme("SandyBeach")

OPTIMIZERS = ['GD', 'LBFGS2']

NO_REGISTRATION_ERR = "No registration has been saved on this case yet. please choose registrarion" \
                      " type and run registrarion"

BASE_ERROR = 1.3

PLANE_DICT = {'XY':2,'XZ':1, 'YZ':0}
SHIFT_DICT = {'X':0,"Y":1,"Z":2}

GLOBAL_PARAM_INIT = [0.01, 100, 1e-4]



class App():
    """
    GUI class
    """

    def title_bar(self, title, text_color, background_color):
        """
        Creates a "row" that can be added to a layout. This row looks like a titlebar
        :param title: The "title" to show in the titlebar
        :type title: str
        :param text_color: Text color for titlebar
        :type text_color: str
        :param background_color: Background color for titlebar
        :type background_color: str
        :return: A list of elements (i.e. a "row" for a layout)
        :rtype: List[sg.Element]
        """
        bc = background_color
        tc = text_color
        font = 'Helvetica 12'

        return [sg.Col([[sg.T(title, text_color=tc, background_color=bc, font=font, grab=True)]], pad=(0, 0),
                       background_color=bc),
                sg.Col([[sg.T('_', text_color=tc, background_color=bc, enable_events=True, font=font, key='-MINIMIZE-'),
                         sg.Text('❎', text_color=tc, background_color=bc, font=font, enable_events=True, key='Exit')]],
                       element_justification='r', key='-C-', grab=True,
                       pad=(0, 0), background_color=bc)]

    def __init__(self):
        # background_layout = [self.title_bar('This is the titlebar', sg.theme_text_color(), sg.theme_background_color()),
        #                      [sg.Image('logo.PNG')]]
        # self.window_background = sg.Window('Background', background_layout, no_titlebar=True, finalize=True, margins=(0, 0),
        #                               element_padding=(0, 0), right_click_menu=[[''], ['Exit', ]])
        #
        # self.window_background['-C-'].expand(True, False,
        #                                 False)  # expand the titlebar's rightmost column so that it resizes correctly
        sg.theme('Tan Blue')

        data_column = [
            [sg.Text('Experiment name ', size=(15, 1), font=font_header)],
            [sg.InputText(key='-NAME-', enable_events=True)],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text('Upload files', font=font_header)],
            [sg.Text('DICOM dir 1', font=font_text), sg.In(size=(10, 1), enable_events=True, key='-FIXED_FOLDER-'),
             sg.FolderBrowse(font=font_text), sg.Text('seeds dir 1', font=font_text, key='-SEEDS_TEXT_1-',visible=False),
             sg.In(size=(10, 1), enable_events=True, key='-SEEDS_INPUT_1-', visible=False),
             sg.FolderBrowse(font=font_text, key='-SEEDS_BROWSER_1-', visible=False),
             sg.Button("Draw Domain", font=font_text, key="-DOMAIN-", visible=False),
             sg.Text('z start', font=font_text, visible=False,key="-DOMAIN_START_TEXT-"),
             sg.In(size=(6, 1), enable_events=True, key='-DOMAIN_START-', visible=False, do_not_clear=False),
             sg.Text('z end', font=font_text, visible=False, key="-DOMAIN_END_TEXT-"),
             sg.In(size=(6, 1), enable_events=True, key='-DOMAIN_END-' , visible=False, do_not_clear=False)],
            [sg.Text('DICOM dir 2', font=font_text), sg.In(size=(10, 1), enable_events=True, key='-MOVING_FOLDER-'),
             sg.FolderBrowse(font=font_text),
             sg.Text('seeds dir 2', font=font_text, key='-SEEDS_TEXT_2-', visible=False),
             sg.In(size=(10, 1), enable_events=True, key='-SEEDS_INPUT_2-', visible=False),
             sg.FolderBrowse(font=font_text, key='-SEEDS_BROWSER_2-', visible=False)
             ],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text('Registration', font=font_header)],
            [sg.Text("Registration Type", font=font_text), sg.Combo(['Use saved Registration', 'Affine', 'Bspline', "Affine+Bspline",
                                                                     "ICP", "CPD"], enable_events=True, font=font_text, key="-REG_MENU-")],
            [sg.Text('Optimizer', key='-OPT_TEXT-', visible=False),
             sg.Combo(['LBFGS2', 'Gradient Decent'], key='-OPT_MENU-', visible=False, enable_events=True),
             sg.Text('Metric', key='-METRIC_TEXT-', visible=False),
             sg.Combo(['Mean Squares', 'Mutual information'], key='-METRIC_MENU-', visible=False, enable_events=True)],
             [sg.Text('Optimizer 2', key='-OPT_TEXT_2-', visible=False),
              sg.Combo(['LBFGS2', 'Gradient Decent'], key='-OPT_MENU_2-', visible=False, enable_events=True),
              sg.Text('Metric 2', key='-METRIC_TEXT_2-', visible=False),
              sg.Combo(['Mean Squares', 'Mutual information'], key='-METRIC_MENU_2-', visible=False, enable_events=True),],
             [sg.Text('Sampling percentage (0-1)', key='-GLOBAL_PARAM_TEXT_1-', visible=False),
             sg.InputText(size=(6, 1), visible=False, key='-GLOBAL_PARAM_1-', enable_events=True),
             sg.Text('Number of Iterations', key='-GLOBAL_PARAM_TEXT_2-', visible=False),
             sg.InputText(size=(6, 1), visible=False, key='-GLOBAL_PARAM_2-', enable_events=True),
             ],
            [

             sg.Text('Sampling 2', key='-SAMPLE_TEXT_2-', visible=False),
              sg.InputText(size=(6, 1), visible=False, key='-SAMPLE_2-', enable_events=True),
             sg.Text('Number of Iterations 2', key='-NUM_ITER_2_TEXT-', visible=False),
             sg.InputText(size=(6, 1), visible=False, key='-NUM_ITER_2-', enable_events=True),
            sg.Text('Bspline res (cm)', key='-BSPLINE_RES_TEXT-', visible=False),
            sg.InputText(size=(6, 1), visible=False, key='-BSPLINE_RES-', enable_events=True)
            ],
            [
             sg.Text('Convergence Tolerance', key='-GLOBAL_PARAM_TEXT_3-', visible=False),
             sg.InputText(size=(6, 1), visible=False, key='-GLOBAL_PARAM_3-', enable_events=True)
             ,sg.Text('Solution accuracy', key='-LBFGS2_PARAM_TEXT_1-', visible=False),
             sg.InputText(size=(6, 1), visible=False, key='-LBFGS2_PARAM_1-', enable_events=True),
             sg.Text('Learning Rate', key='-GD_PARAM_TEXT_1-', visible=False),
             sg.InputText(size=(6, 1), visible=False, key='-GD_PARAM_1-', enable_events=True),
             ],
            [sg.Text("Contours Source", key='-GET_CONTOURS_TEXT-', visible=False),
             sg.Combo(['dcm file', 'csv file', 'Auto Segmentation'], key='-CONTOURS_MENU-', visible=False, enable_events=True),
             sg.Text("% min correspondence", key='-ICP_THRESH_TEXT-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-ICP_THRESH-', enable_events=True)],
             [sg.Text("Contours path 1", key='-FIXED_CONTOURS_TEXT-', visible=False), sg.In(size=(15, 1), enable_events=True, key='-FIXED_CONTOURS_INPUT-', visible=False),
             sg.FileBrowse(font=font_text, key='-FIXED_CONTOURS_BROWS-', visible=False),
             sg.Text("Contours path 2", key='-MOVING_CONTOURS_TEXT-', visible=False),
             sg.In(size=(15, 1), enable_events=True, key='-MOVING_CONTOURS_INPUT-', visible=False),
             sg.FileBrowse(font=font_text, key='-MOVING_CONTOURS_BROWS-', visible=False)
             ],
            [sg.Text("Iterations", key='-CPD_ITER_TEXT-', visible=False),
             sg.InputText(size=(6, 1), visible=False, key='-CPD_ITER-', enable_events=True),
             sg.Text("Iterations non-rigid", key='-CPD_ITER_TEXT_2-', visible=False),
             sg.InputText(size=(6, 1), visible=False, key='-CPD_ITER_2-', enable_events=True),
             sg.Text("Outliers freq", key='-CPD_W_TEXT-', visible=False),
             sg.InputText(size=(6, 1), visible=False, key='-CPD_W-', enable_events=True)
             ],

            [sg.In(size=(15, 1), enable_events=True, visible=False, key='-TFM_INPUT-'),
             sg.FileBrowse(font=font_text, visible=False, key='-TFM_UPLOADER-')],
            [sg.Button("Run Registration", font=font_text, key='-REGISTER-'), sg.Button("Manual refinement",
            font=font_text, key='-MANUAL_REGISTRATION-'), sg.Button("Save registration",
            font=font_text, key='-SAVE_REGISTRATION-'), sg.Button("Clear registrations",
            font=font_text, key='-CLEAR_REGISTRATION-')],
            [sg.Text("xy(axial) rotation", key='-XY_ROT_TEXT-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-XY_ROT-', enable_events=True),
             sg.Text("xz(coronal) rotation", key='-XZ_ROT_TEXT-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-XZ_ROT-', enable_events=True),
             sg.Text("yz(sagittal) rotation", key='-YZ_ROT_TEXT-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-YZ_ROT-', enable_events=True)],
             [sg.Text("x shift", key='-X_SHIFT_TEXT-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-X_SHIFT-', enable_events=True),
             sg.Text("y shift", key='-Y_SHIFT_TEXT-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-Y_SHIFT-', enable_events=True),
             sg.Text("z shift", key='-Z_SHIFT_TEXT-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-Z_SHIFT-', enable_events=True)],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Button("Show Registration Metric", font=font_text, key='-PLOT_REG-')],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Button("Show overlay", font=font_text, key='-OVERLAY-')],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text("Assign Pairs", font=font_header)],
            [sg.Combo(['Munkres', 'Upload List'], enable_events=True, font=font_text,
                                                                    key="-ASSIGN_MENU-")],
            [sg.Text('Upload assignment file', font=font_text, key='-ASSIGN_TEXT-', visible=False),
             sg.In(size=(15, 1), enable_events=True, key='-ASSIGN_INPUT-', visible=False),
             sg.FileBrowse(font=font_text, visible=False, key='-ASSIGN_BROWSER-')],
            [sg.Button("Assign", font=font_text, key="-ASSIGN-")],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text("Calculate movements", font=font_header)],
            [sg.Button("Calculate", font=font_text, key='-CALC_MOVE-')],
            [sg.Button("Show movements", font=font_text, key='-SHOW_MOVE-', visible=False), sg.Button("Show Pairs",
            font=font_text, key='-SHOW_PAIRS-', visible=False), sg.Button("Show Pairs interactive",
            font=font_text, key='-SHOW_PAIRS_INTERACTIVE-', visible=False),
             sg.Button("Exclude Seeds", font=font_text, key='-EXCLUDE_WIN-', visible=False)],
             [sg.Button("Save results to csv", font=font_text, key='-SAVE-', visible=False)],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text("Massages Window", font=font_header)],
            [sg.Text("", font=font_header, size=(40, 7), key="-MSG-", background_color="white",
                     text_color='red')],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Exit(font=font_text)]]

        img_column = [[sg.Text('Dicom 1', font=font_header)],
                      [sg.Canvas(size=(400, 400), key='-FIXED_CANVAS-')],
                      [sg.Text('Dicom 2', font=font_header)],
                      [sg.Canvas(size=(400, 400), key='-MOVING_CANVAS-')]
                       ]
        results_column = [[sg.Text('Overlay', font=font_header)],
                      [sg.Canvas(size=(400, 400), key='-OVERLAY_CANVAS-')],
                      [sg.Text('Results', font=font_header)],
                      [sg.Image("", key='-IMAGE-')]
                       ]

        layout = [
            [
             sg.Column(data_column, vertical_alignment='top', element_justification='c'),
             sg.VSeperator(),
             sg.Column(img_column, element_justification='c'),
            sg.VSeperator(),
            sg.Column(results_column, vertical_alignment='top', element_justification='c')]]

        self.main_window = sg.Window('Demo Application - Seed Movement', layout, finalize=True,
                                     return_keyboard_events=True)
        self.case_name = None
        self.registration_plot_path = None
        self.fixed_dict = None
        self.moving_dict = None
        self.meta = None
        self.fixed_array = None
        self.moving_array = None
        self.domain = None
        self.fixed_sitk = None
        self.moving_sitk = None
        # self.warped_sitk = None
        self.fixed_viewer = None
        self.moving_viewer = None
        self.overlay_viewer = None
        self.ctrs_source = None
        self.fixed_ctrs_path = None
        self.moving_ctrs_path = None
        self.fixed_ctrs_points = None
        self.fixed_ctrs_points_down = None
        self.moving_ctrs_points = None
        self.moving_ctrs_points_down = None
        # self.warped_ctrs_points = None
        self.masked_fixed_struct = None
        self.masked_moving_struct = None
        self.moving_struct = None
        self.moving_struct_sitk = None
        self.fixed_struct = None
        self.fixed_struct_sitk = None

        self.reg_stack = []
        self.tfm_stack = []
        self.param_stack = []
        self.registration_type = None
        self.registration_params = {}
        self.euler_angles = np.zeros(3)
        self.shift = np.zeros(3)
        self.optimizer = None
        self.rmse = [None]
        self.correspondence = None
        self.local_params = {}
        # self.warped_moving = None
        self.icp_thresh = None
        self.tfm = None
        self.composite_tfm = None
        self.inv_tfm = None
        self.first_sitk = True

        self.exclude_win = None
        self.excludes = []


        self.assignment_method = None
        # self.fixed_seeds, self.fixed_orientation = None, None
        # self.moving_seeds, self.moving_orientation = None, None
        self.seeds_tips_fixed = None
        self.seeds_tips_moving = None
        # self.seeds_tips_warped = None
        self.fixed_idx, self.moving_idx = None, None
        self.assignment_dists = [None]
        self.errors = [None]

        self.df_features = ["Timestamp", "Experiment", "Registration Method", "Optimizer", "Metric", "Number Of Iterations", "Learning Rate",
                            "Accuracy Threshold", "Convergence Delta", "RMSE (mm)", "% Correspondence set","Average Movement (mm)", "Median Movement (mm)",
                            "Standard Deviation (mm)", "Maximum Movement (mm)"]
        self.init_param_dict()
        self.message = ""

    def create_exclusion_window(self):
        """
        seeds exclusion window
        :return: pysimplegui window
        """
        data_column = [[sg.Text(f"Enter outliers indexes between 1 and {self.seeds_tips_fixed.shape[-1]} seperated "
                                f"by comma", font=font_header)],
                  [sg.Input(key='-EXCLUDE_IN-', size=(30,4), enable_events=True)],
                  [sg.Button('Preview', font=font_header, key='-EXCLUDE_PREV-'),  sg.Button('Apply', font=font_header,
                                                                                            key='-EXCLUDE_APPLY-')],
                   [sg.Text('Movements', font=font_header)],
                   [sg.Image(f"./movement_output/{self.case_name}/movements.png",
                             size=(600, 600), key='-EXCLUDE_MOVES-')]]
        img_column = [[sg.Text('Seeds', font=font_header)],
                      [sg.Image(f"./movement_output/{self.case_name}/pairs.png",
                                size=(800, 800), key='-EXCLUDE_IMAGE-')]]

        layout = [
            [
                sg.Column(data_column, vertical_alignment='c', element_justification='c'),
                sg.VSeperator(),
                sg.Column(img_column, element_justification='c')]]
        return sg.Window('Exclusion window', layout, finalize=True, element_justification='c')

    def run_exclude_window(self):
        """
        run seeds exclusion window
        """
        while True:
            event, values = self.exclude_win.read()
            # if event == "-EXCLUDE_IN-":
            #     excludes = [int(i) for i in values[event].split(',')]
            #     self.inliers_idx = [i for i in range(self.seeds_tips_fixed.shape[-1]) if i not in excludes]
            if event == "-EXCLUDE_PREV-":
                try:
                    self.excludes = np.array([int(i) for i in values["-EXCLUDE_IN-"].split(',')]) - 1
                except ValueError:
                    self.update_massage("error in outliers indexes")
                    self.excludes = []
                error = np.zeros(len(self.assignment_dists))
                error += self.rmse[-1] if self.rmse[-1] is not None else 0
                error += BASE_ERROR
                # self.inliers_idx = [i for i in range(self.seeds_tips_fixed.shape[-1]) if i not in excludes]
                plot_pairs_with_outliers(self.seeds_tips_fixed, self.seeds_tips_moving, self.excludes, self.case_name)
                plot_individual_moves_outliers(self.case_name, self.assignment_dists, error, self.excludes)
                self.exclude_win['-EXCLUDE_IMAGE-'].update("./movement_output/{}/pairs_outliers.png".format(self.case_name))
                self.exclude_win['-EXCLUDE_MOVES-'].update("./movement_output/{}/movements_outliers.png".format(self.case_name))

            if event == "-EXCLUDE_APPLY-":
                inliers_bool = np.array([True]*self.seeds_tips_fixed.shape[-1])
                inliers_bool[self.excludes] = False
                self.seeds_tips_fixed = self.seeds_tips_fixed[...,inliers_bool]
                self.seeds_tips_moving = self.seeds_tips_moving[..., inliers_bool]
                self.exclude_win.close()
                # print("closing window")
                break
            if event == sg.WIN_CLOSED:  # Window close button event
                break

    @staticmethod
    def find_files_with_extention(folder, extention):
        files = os.listdir('./registration_output/{}'.format(folder))
        return [i for i in files if i.endswith(extention)]

    @staticmethod
    def analyze_distances(case, dists, seeds1, seeds2, error, base_error=0.0, save=True):
        """
        calculate errors and plot results
        :param case: case name
        :param dists: movements
        :param seeds1: (3,3,N) array of first seeds. the first axis represent 3 tips (start, middle,end). the second axis represent
                3 coordinates (x,y,z)
        :param seeds2: (3,3,N) array of second seeds.
        :return: seeds1, seeds2, dist ((N,) array), errors ((N,) array)
        """
        error = error if error is not None else np.zeros(len(dists))
        error += base_error
        plot_pairs(seeds1, seeds2, case, save)
        plot_individual_moves(case, dists, error, save)
        # display_dists(seeds1, seeds2, "%s movements and matching" % case, "%s_matchs.jpg" % case)
        return

    def run(self):
        """
        run main window
        """
        while True:
            event, values = self.main_window.read()
            # print(event, self.registration_params['iterations'])

            if event == sg.WIN_CLOSED:  # Window close button event
                self.update_massage("Bye")
                break
            if event == "-NAME-":
                self.case_name = values[event].lower()
                self.registration_plot_path = f'./registration_output/{self.case_name}/registration_res.png'
            if event == '-FIXED_FOLDER-':
                self.upload_fixed(values)
            if event == '-MOVING_FOLDER-':
                self.upload_moving(values)
            if event == "-SEEDS_INPUT_1-":
                self.get_seed_from_contour(values, event, 1)
            if event == "-SEEDS_INPUT_2-":
                self.get_seed_from_contour(values, event, 2)
            if event == "-DOMAIN-":
                if self.fixed_viewer is not None:
                    self.fixed_viewer.annotate()
                    self.update_massage("draw the desired domain on the image")
                    y0,x0,y1,x1 = self.fixed_viewer.get_rect()
                    # print(y0,x0,y1,x1)
                    self.domain = np.array([int(y0),int(x0),0]), np.array([int(y1),int(x1),self.fixed_array.shape[-1]])
            if event == "-DOMAIN_START-":
                if self.domain is not None:
                    try:
                        self.domain[0][-1] = int(values[event])
                    except ValueError:
                        pass
            if event == "-DOMAIN_END-":
                try:
                    self.domain[1][-1] = int(values[event])
                except ValueError:
                    pass
            if event == "-REG_MENU-":
                self.clear_all_params()
                self.registration_type = values[event]
                # print("reg type ", self.registration_type)
                if self.registration_type == "ICP":
                    self.show_contour_params()
                    self.show_domain_params(visible=False)
                elif self.registration_type == 'CPD':
                    self.show_probreg_params()
                    self.show_domain_params(visible=False)
                elif self.registration_type == "Bspline" or self.registration_type == 'Affine' or self.registration_type\
                == "Affine+Bspline":
                    self.show_relevant_params_input()
                    self.init_param_dict()
                    self.show_domain_params()
                elif self.registration_type == "Use saved Registration":
                    # print("should show uploader")
                    self.main_window['-TFM_INPUT-'].update(visible=True)
                    self.main_window['-TFM_UPLOADER-'].update(visible=True)
                    self.show_domain_params(visible=False)
            if event == "-TFM_INPUT-":
                tfm_path = values[event]
                self.registration_type = "saved_" + tfm_path[:-4].split('_')[-1]
                try: # TODO inverse for composite transform?
                    if self.registration_type == "saved_affine+bspline":
                        self.tfm = sitk.ReadTransform(tfm_path)
                        self.reg_stack.append(self.tfm)
                    else:
                        self.tfm = sitk.ReadTransform(tfm_path)
                        self.inv_tfm, self.disp_img = get_inverse_transform(self.tfm, self.registration_type)
                except Exception as e:
                    print(e)
                    arr = np.load(tfm_path, allow_pickle=True)
                    if self.registration_type == 'saved_ICP':
                        self.tfm = arr
                    elif self.registration_type == "saved_CPD":
                        tfm1 = arr['rigid']
                        self.reg_stack.append(tfm1)
                        self.tfm = arr['nonrigid']
                    self.reg_stack.append(self.tfm)
                self.update_massage(f"{tfm_path} uploaded successfully ")
            if event == "-CONTOURS_MENU-":
                self.ctrs_source = values[event]
                if self.ctrs_source == "csv file":
                    self.show_contours_brows()
                else:
                    self.show_contours_brows(visible=False)
            if event == "-FIXED_CONTOURS_INPUT-":
                self.fixed_ctrs_path = values[event]
            if event == "-MOVING_CONTOURS_INPUT-":
                self.moving_ctrs_path = values[event]
            if event == "-ICP_THRESH-":
                try:
                    self.icp_thresh = int(values[event])
                except ValueError:
                    self.icp_thresh = None
            if event == '-MANUAL_REGISTRATION-':
                self.clear_all_params()
                if self.moving_ctrs_points is not None:
                    self.registration_type = 'manual'
                    self.show_manual_registration_params()
                    self.message = "Generating interactive plot to analyze manual transformation.\n"\
                                    "When done, close the interactive window and fill the desired values"
                    self.update_massage(self.message)
                    overlay_contours_interactive(self.fixed_ctrs_points_down, self.moving_ctrs_points_down)
            if "ROT" in event:
                plane = event[1:3]
                try:
                    self.euler_angles[PLANE_DICT[plane]] = int(values[event])
                except ValueError:
                    pass
            if "SHIFT" in event:
                axis = event[1]
                try:
                    self.shift[SHIFT_DICT[axis]] = int(values[event])
                except ValueError:
                    pass
            if "-CPD_ITER" in event:
                if event[-2] == '2':
                    try:
                        self.registration_params['iterations 2'] = values[event]
                        # print(self.registration_params['iterations'])
                    except ValueError:
                        pass
                else:
                    try:
                        self.registration_params['iterations'] = values[event]
                        # print(self.registration_params['iterations'])
                    except ValueError:
                        pass
            if event == "-CPD_W-":
                try:
                    self.registration_params['w'] = values[event]
                except ValueError:
                    pass
            if event == "-CLEAR_REGISTRATION-":
                self.reset_data(values)
                self.message = "All registrations cleared"
                self.update_massage(self.message)
            if event == '-REGISTER-':
                self.create_case_dir()
                self.set_meta()
                if "Affine" in self.registration_type or "Bspline" in self.registration_type:
                    self.create_sitk_param_dicts(values)
                self.run_registration()
            if event == "-SAVE_REGISTRATION-":
               self.save_registration()
            if event == "-PLOT_REG-":
                try:
                    self.main_window['-IMAGE-'].update(self.registration_plot_path)
                except Exception as e:
                    self.update_massage("Nothing to show")
            if event == "-OVERLAY-":
                self.create_overlay()

            if event == "-ASSIGN_MENU-":
                self.assignment_method = values[event]

            if event == '-ASSIGN-':
                self.assign_seeds(values)

            if self.assignment_method == "Upload List":
                self.show_assignment_uploader()
            if event == "-EXCLUDE_WIN-":
                self.exclude_win = self.create_exclusion_window()
                self.run_exclude_window()
            if event == "-CALC_MOVE-":
                self.create_case_dir()
                self.calc_move()
            if event == "-SHOW_MOVE-":
                self.plot_name = "moves"
                self.main_window['-IMAGE-'].update("./movement_output/{}/movements.png".format(self.case_name))
            if event == "-SHOW_PAIRS-":
                self.plot_name = "pairs"
                self.main_window['-IMAGE-'].update("./movement_output/{}/pairs.png".format(self.case_name))
            if event == "-SHOW_PAIRS_INTERACTIVE-":
                plot_seeds_and_contour_interactive(self.seeds_tips_fixed, self.seeds_tips_moving, self.moving_ctrs_points)
            if event == "-SAVE-":
                self.save_results_to_csv()





            # if event == '-MOVING_FOLDER-':
            #     print('moving folder', values['-MOVING_FOLDER-'])
            #     self.moving_dict = get_all_dicom_files(values['-MOVING_FOLDER-'], {})
            #     self.moving_array = read_dicom(self.moving_dict['CT'])
            #     self.moving_viewer = DicomCanvas(self.moving_array, self.window['-MOVING_CANVAS-'].TKCanvas)
            if event == sg.WIN_CLOSED or event == 'Exit':
                break


    def update_massage(self, massage):
        """
        update main window message
        :param massage: str
        """
        self.main_window['-MSG-'].update(massage)

    def set_meta(self):
        """
        set relevant meta data dict
        """
        if len(self.reg_stack) == 0 or (all(np.array(self.reg_stack) == 'ICP')):
            self.meta = self.moving_dict['meta']
        else:
            self.meta = self.fixed_dict['meta']

    def get_seeds(self, dict):
        """
        read seeds
        :param dict: meta data dict
        :return: 3x3xN array of seeds. first axis represent seed tips (start,mid,end), second axis represent
        coordinates and N is the number of seed objects
        """
        seeds, orientation = get_seeds_dcm(dict['RTPLAN'])
        return get_seeds_tips(seeds, orientation)

    def create_case_dir(self):
        """
        create registration output and movement output dirs
        """
        if self.case_name not in os.listdir('./registration_output'):
            dir_name = f"./registration_output/{self.case_name}"
            os.mkdir(dir_name)
        if self.case_name not in os.listdir('./movement_output'):
            dir_name = f"./movement_output/{self.case_name}"
            os.mkdir(dir_name)

    def run_registration(self):
        """
        run registration. generally, the function will transform image, contour and seeds.
        the current version do not transform image for manual,ICP,CPD registrations. sitk registration will transform
        image to the fixed frame of reference and seeds and contour to the moving frame of reference
        """
        if self.registration_type is not None:
            if self.fixed_array is not None and self.moving_array is not None:
                if self.case_name is not None:
                    # domain = True if self.domain is not None else False
                    self.message = "Registering...\nThis may take a while"
                    self.update_massage(self.message)
                    self.main_window.refresh()
                    if self.registration_type == "Bspline" or self.registration_type == "Affine":
                        seeds_pix = None if self.registration_type == "Affine" else get_all_seeds_pixels(self.seeds_tips_fixed,
                                                                             self.fixed_dict['meta'],
                                                                             self.fixed_array.shape)
                        self.run_sitk_registration(self.registration_type, domain=self.domain, exclude=seeds_pix)
                    elif self.registration_type == "Affine+Bspline":
                        self.run_composite_sitk_registration(["Affine", "Bspline"])
                    elif self.registration_type =='ICP':
                        self.run_adaptive_icp()
                        self.tfm = self.tfm if self.tfm is not None else np.eye(4)
                        # self.moving_array = wrap_image_with_matrix(self.fixed_array, self.moving_array,
                        #                                            self.meta, np.linalg.inv(self.inv_tfm))
                        # self.warped_array = affine_transform(self.warped_array, self.tfm)
                        # self.update_arrays("icp")
                    elif self.registration_type == "CPD":
                        self.run_probreg_registration()

                    elif self.registration_type =='manual':
                        self.tfm = self.run_manual_registration()
                        # self.moving_array = wrap_image_with_matrix(self.fixed_array, self.moving_array,
                        #                                            self.tfm.as_matrix())
                        # self.moving_array = self.warped_moving
                    elif self.registration_type.split("_")[0] == "saved":
                        self.run_saved_registration()
                        # try:
                        #     self.warped_sitk = warp_image_sitk(self.fixed_sitk, self.moving_sitk, self.tfm)
                        #     self.update_arrays("sitk")
                        #     # print("**saved**")
                        #     # print("fixed ", self.fixed_array.shape, " moving ", self.moving_array.shape, "warped ",
                        #     #       self.warped_moving.shape)
                        # except Exception as e:
                        #     print("moving to icp")
                        #     print(e)
                        #     self.warped_array = wrap_image_with_matrix(self.fixed_array, self.warped_array, self.meta,
                        #                                                 self.tfm)
                        #     self.update_arrays("icp")
                    self.reg_stack.append(self.registration_type)
                    self.tfm_stack.append(self.tfm)
                    self.set_meta()
                    # print(self.tfm)
                    # print(self.moving_dict['meta'])
                else:
                    self.update_massage("Enter Experiment Name")
            else:
                self.update_massage("You must upload 2 DICOMS to perform registration")
        else:
            self.update_massage("No Registration to run")
        if self.tfm is not None:
            self.message = self.message + f"\n{self.registration_type} registration finished\n RMSE: {self.rmse[-1]}"
            self.update_massage(self.message)
            self.shift = np.array([0, 0, 0])
            self.euler_angles = np.array([0, 0, 0])
        return

    def run_probreg_registration(self):
        """
        run probreg rigid\non-rigid registration. the current version uses bayesian coherent point drift (bcpd) as non
        rigid registration and coherent point drift (cpd) for rigid registration
        """
        reg_obj = registration.ContourRegistration(f'registration_output/{self.case_name}', self.case_name)
        if self.registration_type == 'CPD':
            s = time.time()
            try:
                rigid_iter = int(self.registration_params['iterations'])
            except (ValueError, KeyError):
                rigid_iter = 50
            try:
                iter_nonrigid = int(self.registration_params['iterations 2'])
            except (ValueError, KeyError) as e:
                print(e)
                iter_nonrigid = 10
            try:
                w = float(self.registration_params['w'])
            except (ValueError, KeyError):
                w = 0.001

            print("iter ", rigid_iter, " iter nonrigid ", iter_nonrigid, " w ", w)

            self.message = self.message + ("\nRunning rigid CPD...")
            self.update_massage(self.message)
            self.tfm, self.fixed_ctrs_points_down, self.moving_ctrs_points_down = reg_obj.cpd(self.fixed_ctrs_points_down,
                                                                                    self.moving_ctrs_points_down,
                                                                                    max_iter=rigid_iter)
            self.fixed_ctrs_points = self.tfm.transform(self.fixed_ctrs_points)
            self.seeds_tips_fixed = apply_probreg_transformation_on_seeds(self.tfm, self.seeds_tips_fixed)
            self.reg_stack.append(self.registration_type + "_rigid")
            self.tfm_stack.append(self.tfm)
            criteria = log_to_dict('Logs/probreg.log')['Criteria']

            self.message = self.message + ("\nRunning non-rigid CPD...")
            self.update_massage(self.message)
            if iter_nonrigid:
                self.tfm, self.fixed_ctrs_points_down, self.moving_ctrs_points_down = reg_obj.bcpd(self.fixed_ctrs_points_down,
                                                                                        self.moving_ctrs_points_down,
                                                                                        max_iter=iter_nonrigid, w=w)
                v = self.tfm.v
                invdisttree = registration.Invdisttree(self.fixed_ctrs_points_down, v, leafsize=10, stat=1)
                interpol = invdisttree(self.fixed_ctrs_points, nnear=5, eps=0, p=1)
                self.fixed_ctrs_points += interpol
                self.seeds_tips_fixed = apply_probreg_transformation_on_seeds(self.tfm, self.seeds_tips_fixed,
                                                                              self.fixed_ctrs_points_down, type='bcpd')
                criteria_nonrigid = log_to_dict('Logs/probreg.log')['Criteria']
                plot_loss_and_contours([criteria, criteria_nonrigid], self.fixed_ctrs_points_down, self.moving_ctrs_points_down,
                                       self.registration_plot_path, num_graphs=2)
            else:
                self.tfm = np.eye(4)
                plot_loss_and_contours(criteria, self.fixed_ctrs_points_down,
                                       self.moving_ctrs_points_down,
                                       self.registration_plot_path, num_graphs=1)
            # rmse_arr = reg_obj.get_callback_obj().get_rmse()
            # criteria = log_to_dict('Logs/probreg.log')['Criteria']
            # if iter_nonrigid:
            #     loss = [criteria[:rigid_iter], criteria[rigid_iter:]]
            #     num_graphs = 2
            # else:
            #     loss = criteria
            #     num_graphs = 1
            # plot_rmse_and_contours(loss, self.fixed_ctrs_points_down, self.moving_ctrs_points_down,
            #                        self.registration_plot_path, num_graphs=num_graphs)
            # print("calculating rmse")
            self.rmse.append(calc_rmse(self.fixed_ctrs_points_down.T, self.moving_ctrs_points_down.T, 0.1))
            # print("time to transform ", time.time() - s)

    def run_saved_registration(self):
        """
        perform a saved registration
        """
        # print("running saved registration")
        # print(self.tfm)
        # print(self.fixed_sitk.GetOrigin(),self.moving_sitk.GetOrigin())

        reg_type = self.registration_type.split("_")[-1]
        if reg_type == 'Bspline' or reg_type == "Affine" or reg_type == "affine+bspline":
            self.moving_sitk = warp_image_sitk(self.fixed_sitk, self.moving_sitk, self.tfm)
            self.seeds_tips_fixed = apply_transformation_on_seeds(self.tfm,
                                                                  self.seeds_tips_fixed,
                                                                  "sitk")
            self.fixed_ctrs_points = apply_sitk_transformation_on_struct(self.tfm, self.fixed_ctrs_points)
            self.rmse.append(calc_rmse(self.fixed_ctrs_points.T, self.moving_ctrs_points.T, 0.001))

            self.update_arrays("sitk")
        # elif reg_type == "Affine+Bspline":
        #     self.seeds_tips_fixed = apply_transformation_on_seeds(self.tfm,
        #                                                           self.seeds_tips_fixed,
        #                                                           "sitk")
        #     self.fixed_ctrs_points = apply_sitk_transformation_on_struct(self.tfm, self.fixed_ctrs_points)
        elif reg_type == "ICP":
            self.seeds_tips_moving = apply_transformation_on_seeds(np.linalg.inv(self.tfm),
                                                                   self.seeds_tips_moving,
                                                                   "affine")
        elif reg_type == "CPD":
            tfm_rigid, tfm_nonrigid = self.reg_stack[-2:]
            self.fixed_ctrs_points_down = tfm_rigid.transform(self.fixed_ctrs_points_down)
            self.fixed_ctrs_points = tfm_rigid.transform(self.fixed_ctrs_points)
            self.seeds_tips_fixed = apply_probreg_transformation_on_seeds(tfm_rigid, self.seeds_tips_fixed)

            self.fixed_ctrs_points_down = tfm_nonrigid.transform(self.fixed_ctrs_points_down)
            v = tfm_nonrigid.v
            invdisttree = registration.Invdisttree(self.fixed_ctrs_points_down, v, leafsize=10, stat=1)
            interpol = invdisttree(self.fixed_ctrs_points, nnear=5, eps=0, p=1)
            self.fixed_ctrs_points += interpol
            self.seeds_tips_fixed = apply_probreg_transformation_on_seeds(tfm_nonrigid, self.seeds_tips_fixed,
                                                                          self.fixed_ctrs_points_down, type='bcpd')
            self.rmse.append(calc_rmse(self.fixed_ctrs_points_down.T, self.moving_ctrs_points_down.T, 0.1))



    def run_composite_sitk_registration(self, types):
        """
        run composite sitk registration
        :param types: types of registrations ("Affine", "Bspline", etc.)
        """
        # print("running composite registration")
        # print(self.registration_params)
        self.composite_tfm = sitk.CompositeTransform(3)
        tmp_opt, tmp_metric, tmp_smp = self.registration_params["optimizer"], self.registration_params["metric"],\
        self.registration_params['sampling_percentage']
        for i in range(len(types)):
            seeds_pix = None if i == 0 else get_all_seeds_pixels(self.seeds_tips_fixed, self.fixed_dict['meta'],
                                                                self.fixed_array.shape)
            self.run_sitk_registration(types[i], apply=False, domain=self.domain, exclude=seeds_pix)
            # print("transform type ", self.tfm.GetTransformEnum())
            if self.tfm.GetTransformEnum() == 13:
                try:
                    self.composite_tfm.AddTransform(sitk.CompositeTransform(self.tfm).GetNthTransform(0))
                except Exception as e:
                    print(e)
                    self.composite_tfm.AddTransform(self.tfm)
            else:
                self.composite_tfm.AddTransform(self.tfm)
            self.reg_stack.append(self.registration_type + f"_{i}")
            self.tfm_stack.append(self.tfm)
            self.registration_params["optimizer"] = self.registration_params["optimizer 2"]
            self.registration_params["metric"] = self.registration_params["metric 2"]
            self.registration_params["iterations"] = self.registration_params["iterations 2"]
            self.registration_params['sampling_percentage'] = self.registration_params['sampling_percentage 2']
            self.reg_stack.append("composite")
        self.reg_stack.pop()
        self.registration_params["optimizer"] = tmp_opt
        self.registration_params["metric"] = tmp_metric
        self.registration_params['sampling_percentage'] = tmp_smp
        self.seeds_tips_fixed = apply_transformation_on_seeds(self.composite_tfm,
                                                              self.seeds_tips_fixed,
                                                              "sitk")
        self.fixed_ctrs_points = apply_sitk_transformation_on_struct(self.composite_tfm, self.fixed_ctrs_points)
        self.rmse.append(calc_rmse(self.fixed_ctrs_points.T, self.moving_ctrs_points.T, 0.001))
        self.param_stack.append({"RMSE": self.rmse[-1]})

    def run_sitk_registration(self, type, apply=True, domain=None, exclude=None):
        """
        run sitk registration
        :param type: "Affine", "Bspline"
        :param apply: flag to apply on seeds
        :param domain: domain for bspline
        :param exclude: pixels to exclude in the optimization. for bspline
        """
        # print(self.registration_type)
        # print(self.registration_params['optimizer'], self.registration_params['metric'], self.registration_params['iterations'])
        if domain is not None:
            y0,x0,y1,x1 = self.fixed_viewer.get_rect()
            self.domain[0][:2] = [int(y0),int(x0)]
            self.domain[1][:2] = [int(y1),int(x1)]
            dom1, dom2 = self.domain[0], self.domain[1]
            # print("domain is ", self.domain)
        else:
            dom1, dom2 = None, None
        if len(self.reg_stack) == 0 or all(np.array(self.reg_stack) == 'ICP'):
            self.fixed_sitk, self.moving_sitk, self.tfm, self.inv_tfm, self.disp_img = register_sitk(
                self.fixed_dict['CT'], self.moving_dict['CT'], self.fixed_dict['meta'], self.registration_plot_path,
                type, self.registration_params, dom1, dom2, exclude)
        else:
            self.fixed_sitk, self.moving_sitk, self.tfm, self.inv_tfm, self.disp_img = register_sitk(
                self.fixed_sitk, self.moving_sitk, self.moving_dict['meta'], self.registration_plot_path,
                type, self.registration_params, dom1, dom2, exclude)
        if apply:
            self.seeds_tips_fixed = apply_transformation_on_seeds(self.tfm,
                                                                       self.seeds_tips_fixed,
                                                                       "sitk")
            self.fixed_ctrs_points = apply_sitk_transformation_on_struct(self.tfm, self.fixed_ctrs_points)
            self.rmse.append(calc_rmse(self.fixed_ctrs_points.T, self.moving_ctrs_points.T, 0.001))
            self.param_stack.append({"RMSE": self.rmse})
        self.update_arrays("sitk")
        # print("after registration fixed: ", self.fixed_sitk.GetOrigin(), " moving: ", self.moving_sitk.GetOrigin())
        self.param_stack.append(self.registration_params)



        # if self.moving_struct_sitk is not None:
        #     self.moving_struct_sitk = warp_image_sitk(self.fixed_sitk, self.moving_struct_sitk, self.tfm, mask=True,
        #                                               default_val=0)

    def run_manual_registration(self):
        """
        run manual registration. rotation and shift
        """

        #TODO update arrays (numpy and sitk)
        # print('shift ', self.shift)
        R = Rotation.from_euler("xyz", self.euler_angles, degrees=True)
        mean = np.mean(self.fixed_ctrs_points, axis=0)
        # print("R ", R.as_matrix())
        self.fixed_ctrs_points -= mean
        self.fixed_ctrs_points = R.apply(self.fixed_ctrs_points)
        self.fixed_ctrs_points += mean
        self.fixed_ctrs_points += self.shift[None, :]

        mean = mean[None, :, None]
        self.seeds_tips_fixed -= mean
        self.seeds_tips_fixed = apply_scipy_transformation_on_seeds(R, self.seeds_tips_fixed)
        self.seeds_tips_fixed += mean
        self.seeds_tips_fixed += self.shift[None, :, None]
        self.param_stack.append({"rotation":self.euler_angles, "shift":self.shift})
        self.rmse.append(calc_rmse(self.fixed_ctrs_points.T, self.moving_ctrs_points.T, 0.001))
        overlay_contours(self.fixed_ctrs_points, self.moving_ctrs_points, self.registration_plot_path)
        # print("scipy rotation")
        # for i, val in enumerate(self.euler_angles):
        #     if val:
        #         plane = ((i+1)%3, (i+2)%3)
        #         print(plane, val)
        #         self.moving_array = rotate(self.moving_array, -val, plane)
        # self.moving_array = shift(self.moving_array, [self.shift[1],self.shift[0],self.shift[2]])
        return R

    def run_adaptive_icp(self):
        """
        run ICP. the function will perform ICP (with fgr initialization) with different dist_thresh and will
         take the best one
        """
        reg_obj = ContourRegistration(f'registration_output/{self.case_name}', self.case_name)
        center_fixed = np.mean(self.fixed_ctrs_points, axis=0)
        center_moving = np.mean(self.moving_ctrs_points, axis=0)
        dist_thresh = np.sqrt(np.sum((center_fixed - center_moving) ** 2)) / 2
        dist_thresh = np.max((dist_thresh,0.01))
        max_dist = 10*dist_thresh
        thresh = self.icp_thresh/100 if self.icp_thresh is not None else 0.9
        min_rmse = [np.inf]
        best_reg = None
        best_init = None
        best_ratio = 0
        i = 0
        while True:
            res,init, fixed_surface, moving_surface = reg_obj.icp(self.fixed_ctrs_points_down, self.moving_ctrs_points_down,
                                                             dist_thresh)
            num_pairs = reg_obj.get_correspondence().shape[0]
            num_points = min(fixed_surface.shape[0], moving_surface.shape[0])
            rmse = res.inlier_rmse if res is not None else np.inf
            self.update_massage(f"correspondence set - {num_pairs}, num points - {num_points}, "
                                f"inlier rmse - {rmse}")
            if rmse < min_rmse[-1] and num_pairs > num_points * thresh:
                min_rmse, best_reg, best_init = reg_obj.get_params()
                best_ratio = np.min((num_pairs/num_points, 1))
            if dist_thresh >= max_dist:
                break
            dist_thresh = dist_thresh + 0.1*dist_thresh
            i += 1
        if best_reg is not None:
            self.message = f"best registration:\n inlier rmse - {min_rmse[-1][-1]}"\
                                f"\n % correspondence set - {best_ratio*100}"
            self.update_massage(self.message)
            self.tfm = best_reg.transformation.numpy()
            self.inv_tfm = np.linalg.inv(self.tfm)
            self.fixed_ctrs_points = apply_transformation(self.fixed_ctrs_points, self.tfm)
            self.fixed_ctrs_points_down = down_sample_array(self.fixed_ctrs_points)
            self.seeds_tips_fixed = apply_transformation_on_seeds(self.tfm, self.seeds_tips_fixed,
                                                                   "affine")
            # print("init ", best_init.transformation)
            # print("icp ", self.tfm)
            # self.warped_struct = get_contour_mask(self.warped_ctrs_points.T, self.meta, self.fixed_array.shape)
            # self.masked_warped_struct = np.ma.masked_where(self.warped_struct == 0, self.warped_struct)
            # self.set_moving_mask() # TODO fix
            plot_loss_and_contours(min_rmse, self.fixed_ctrs_points, self.moving_ctrs_points, self.registration_plot_path)
            self.rmse.append(calc_rmse(self.fixed_ctrs_points.T, self.moving_ctrs_points.T, 0.001))
            self.correspondence = best_ratio*100
            self.param_stack.append({"RMSE":self.rmse, "% Correspondence set":self.correspondence})
            # overlay_contours(self.fixed_ctrs_points, self.moving_ctrs_points_transformed, self.case_name, save=False)
        else:

            self.update_massage("registration failed. please try again")

    def set_moving_mask(self):
        self.moving_struct = get_contour_mask(self.moving_ctrs_points.T, self.meta,
                                              self.fixed_array.shape, flip_z=False)
        self.masked_moving_struct = np.ma.masked_where(self.moving_struct == 0, self.moving_struct)

    def set_fixed_mask(self):
        self.fixed_struct = get_contour_mask(self.fixed_ctrs_points.T, self.meta,
                                              self.fixed_array.shape, flip_z=False)
        self.masked_fixed_struct = np.ma.masked_where(self.fixed_struct == 0, self.fixed_struct)

    def update_arrays(self, source='sitk'):
        if source == "sitk":
            # print("changing numpy arrays")
            self.moving_array = np.transpose(sitk.GetArrayFromImage(self.moving_sitk), (1,2,0))
            self.fixed_array = np.transpose(sitk.GetArrayFromImage(self.fixed_sitk), (1, 2, 0))
            # print("moving array shape", self.moving_array.shape)
            if self.moving_struct_sitk is not None:
                self.moving_struct = np.transpose(sitk.GetArrayFromImage(self.moving_struct_sitk), (1,2,0))
        else:
            # print("changing sitk image")
            self.moving_sitk = sitk.Cast(sitk.GetImageFromArray(np.transpose(self.moving_array,(2,0,1))),sitk.sitkFloat32)
            if self.moving_struct is not None:
                self.moving_struct_sitk = sitk.Cast(sitk.GetImageFromArray(np.transpose(self.moving_struct,(2,0,1))),sitk.sitkFloat32)

    def upload_moving(self, values, read_im=True):
        """
        upload moving data - ct, contour, seeds
        :param values: event values
        :param read_im: flag to read ct
        """
        self.message = ""
        self.rmse = [None]
        self.masked_moving_struct = self.masked_moving_struct_orig = None
        self.moving_ctrs_points = self.moving_ctrs_points_orig = None
        self.moving_struct = None
        self.moving_dict = get_all_dicom_files(values['-MOVING_FOLDER-'], {})
        if read_im:
            self.moving_array = read_dicom(self.moving_dict['CT'], self.moving_dict['meta'])
            # self.moving_sitk = sitk.Cast(sitk.GetImageFromArray(np.transpose(self.moving_array, (2, 0, 1))),
            #                             sitk.sitkFloat32)
            self.moving_sitk = read_image(self.moving_dict['CT'])
        # try:
        #     set_meta_data_to_sitk_image(self.moving_sitk, self.moving_dict['meta'])
        # except Exception as e:
        #     print(e)
        self.moving_array_orig = self.moving_array.copy()
        # set_meta_data_to_sitk_image(self.moving_sitk, self.moving_dict['meta'])
        if "RTPLAN" in self.moving_dict.keys():
            self.seeds_tips_moving = self.get_seeds(self.moving_dict)
            self.show_seeds_uploader(2, visible=False)

            # self.seeds_tips_moving_orig = self.seeds_tips_moving.copy()
        else:
            self.show_seeds_uploader(2)
            self.message = self.message + "\ndidn't find any RTPLAN to load. if you want to load " \
                                          "seeds as contours, please load manually"
        # self.seeds_tips_warped = self.seeds_tips_moving.copy()
        if 'RTSTRUCT' in self.moving_dict.keys():
            self.moving_ctrs_points = read_structure(self.moving_dict['RTSTRUCT'])[0][1].T
            # self.moving_ctrs_points_orig = self.moving_ctrs_points.copy()
            self.moving_ctrs_points_down = down_sample_array(self.moving_ctrs_points)
            # self.warped_ctrs_points = self.moving_ctrs_points.copy()
            try:
                self.moving_struct = get_contour_mask(self.moving_ctrs_points.T, self.moving_dict['meta'], self.moving_array.shape)
                self.moving_struct_sitk = sitk.Cast(sitk.GetImageFromArray(np.transpose(self.moving_struct,(2,0,1))),sitk.sitkFloat32)
                self.masked_moving_struct = np.ma.masked_where(self.moving_struct == 0, self.moving_struct)
                # self.masked_moving_struct_orig = self.masked_moving_struct.copy()
            except Exception as e:
                print(e)
                self.message = f"error loading contours on dicom:\n {e}"
                self.update_massage(self.message)
        if read_im:
            if self.moving_viewer is not None:
                self.moving_viewer.clear()
            self.moving_viewer = DicomViewer(self.moving_array, self.main_window['-MOVING_CANVAS-'].TKCanvas, 'moving',
                                             self.masked_moving_struct)
            self.moving_viewer.show()
        self.message = self.message + "\n{} was uploaded successfully".format(self.moving_dict['meta']['ID'].value)
        self.update_massage(self.message)
        self.tfm = None

    def upload_fixed(self, values, read_im=True):
        """
        upload fixed data - ct, contour, seeds
        :param values: event values
        :param read_im: flag to read ct
        """
        self.message = ""
        self.rmse = [None]
        self.masked_fixed_struct = self.masked_fixed_struct_orig = None
        self.fixed_ctrs_points = self.fixed_ctrs_points_orig = None
        self.fixed_struct = None
        self.fixed_dict = get_all_dicom_files(values['-FIXED_FOLDER-'], {})
        if read_im:
            self.fixed_array = read_dicom(self.fixed_dict['CT'], self.fixed_dict['meta'])
            # self.fixed_sitk = sitk.Cast(sitk.GetImageFromArray(np.transpose(self.fixed_array, (2, 0, 1))),
            #                              sitk.sitkFloat32)
            self.fixed_sitk = read_image(self.fixed_dict['CT'])
        # try:
        #     set_meta_data_to_sitk_image(self.fixed_sitk, self.fixed_dict['meta'])
        # except Exception as e:
        #     print(e)
        if "RTPLAN" in self.fixed_dict.keys():
            self.seeds_tips_fixed = self.get_seeds(self.fixed_dict)
            self.show_seeds_uploader(1, visible=False)
            # self.seeds_tips_fixed_orig = self.seeds_tips_fixed.copy()
        else:
            self.show_seeds_uploader(1)
            self.message = self.message + "\ndidn't find any RTPLAN to load. if you want to load " \
                                          "seeds as contours, please load manually"
        if 'RTSTRUCT' in self.fixed_dict.keys():
            self.fixed_ctrs_points = read_structure(self.fixed_dict['RTSTRUCT'])[0][1].T
            # self.fixed_ctrs_points_orig = self.fixed_ctrs_points.copy()
            self.fixed_ctrs_points_down = down_sample_array(self.fixed_ctrs_points)
            try:
                # struct = read_structure(self.fixed_dict['RTSTRUCT'])
                self.fixed_struct = get_contour_mask(self.fixed_ctrs_points.T, self.fixed_dict['meta'], self.fixed_array.shape)
                self.masked_fixed_struct = np.ma.masked_where(self.fixed_struct == 0, self.fixed_struct)
                # self.masked_fixed_struct_orig = self.masked_fixed_struct.copy()
            except Exception as e:
                print(e)
                self.message = f"error loading contours on dicom:\n {e}"
                self.update_massage(self.message)
        if read_im:
            if self.fixed_viewer is not None:
                self.fixed_viewer.clear()
            self.fixed_viewer = DicomViewer(self.fixed_array, self.main_window['-FIXED_CANVAS-'].TKCanvas, 'fixed',
                                            self.masked_fixed_struct)
            self.fixed_viewer.show()
        self.message = self.message + "\n{} was uploaded successfully".format(self.fixed_dict['meta']['ID'].value)
        self.update_massage(self.message)
        # print("after reading ", self.fixed_sitk.GetOrigin())
        self.tfm = None

    def reset_data(self, values):
        """
        reset all data without reading ct again
        :param values: event values
        """
        self.upload_fixed(values, read_im=False)
        self.upload_moving(values, read_im=False)

        self.show_seeds_uploader(1, visible=False)
        self.show_seeds_uploader(2, visible=False)
        self.show_domain_params(visible=False)

    def show_manual_registration_params(self, visible=True):
        self.main_window['-XY_ROT_TEXT-'].update(visible=visible)
        self.main_window['-XY_ROT-'].update("", visible=visible)
        self.main_window['-XZ_ROT_TEXT-'].update(visible=visible)
        self.main_window['-XZ_ROT-'].update("", visible=visible)
        self.main_window['-YZ_ROT_TEXT-'].update(visible=visible)
        self.main_window['-YZ_ROT-'].update("", visible=visible)
        self.main_window['-X_SHIFT_TEXT-'].update(visible=visible)
        self.main_window['-X_SHIFT-'].update("", visible=visible)
        self.main_window['-Y_SHIFT_TEXT-'].update(visible=visible)
        self.main_window['-Y_SHIFT-'].update("", visible=visible)
        self.main_window['-Z_SHIFT_TEXT-'].update(visible=visible)
        self.main_window['-Z_SHIFT-'].update("", visible=visible)

    def show_domain_params(self, visible=True):
        self.main_window[f"-DOMAIN-"].update(visible=visible)
        self.main_window[f"-DOMAIN_START_TEXT-"].update(visible=visible)
        self.main_window[f"-DOMAIN_START-"].update(visible=visible)
        self.main_window[f"-DOMAIN_START-"].update('')
        self.main_window[f"-DOMAIN_END_TEXT-"].update(visible=visible)
        self.main_window[f"-DOMAIN_END-"].update(visible=visible)
        self.main_window[f"-DOMAIN_END-"].update('')

    def show_seeds_uploader(self, num, visible=True):
        self.main_window[f"-SEEDS_TEXT_{num}-"].update(visible=visible)
        self.main_window[f"-SEEDS_INPUT_{num}-"].update(visible=visible)
        self.main_window[f"-SEEDS_BROWSER_{num}-"].update(visible=visible)

    def show_probreg_params(self, visible=True):
        self.main_window['-CPD_ITER_TEXT-'].update(visible=visible)
        self.main_window['-CPD_ITER-'].update(visible=visible)
        self.main_window['-CPD_ITER_TEXT_2-'].update(visible=visible)
        self.main_window['-CPD_ITER_2-'].update(visible=visible)
        self.main_window['-CPD_W_TEXT-'].update(visible=visible)
        self.main_window['-CPD_W-'].update(visible=visible)

    def show_contour_params(self, visible=True):
        self.main_window['-GET_CONTOURS_TEXT-'].update(visible=visible)
        self.main_window['-CONTOURS_MENU-'].update(visible=visible)
        self.main_window['-ICP_THRESH_TEXT-'].update(visible=visible)
        self.main_window['-ICP_THRESH-'].update(visible=visible)

    def show_contours_brows(self, visible=True):
        self.main_window["-FIXED_CONTOURS_TEXT-"].update(visible=visible)
        self.main_window["-FIXED_CONTOURS_INPUT-"].update(visible=visible)
        self.main_window["-FIXED_CONTOURS_BROWS-"].update(visible=visible)
        self.main_window["-MOVING_CONTOURS_TEXT-"].update(visible=visible)
        self.main_window["-MOVING_CONTOURS_INPUT-"].update(visible=visible)
        self.main_window["-MOVING_CONTOURS_BROWS-"].update(visible=visible)

    def show_global_params(self, visible=True):
        self.main_window['-OPT_TEXT-'].update(visible=visible)
        self.main_window['-OPT_MENU-'].update(visible=visible)
        self.main_window['-METRIC_TEXT-'].update(visible=visible)
        self.main_window['-METRIC_MENU-'].update(visible=visible)
        for i in range(1,4):
            self.main_window[f'-GLOBAL_PARAM_TEXT_{i}-'].update(visible=visible)
            self.main_window[f'-GLOBAL_PARAM_{i}-'].update(str(GLOBAL_PARAM_INIT[i - 1]), visible=visible)
        if self.registration_type == "Affine+Bspline":
            self.main_window['-OPT_TEXT_2-'].update(visible=visible)
            self.main_window['-OPT_MENU_2-'].update(visible=visible)
            self.main_window['-METRIC_TEXT_2-'].update(visible=visible)
            self.main_window['-METRIC_MENU_2-'].update(visible=visible)
            self.main_window['-SAMPLE_TEXT_2-'].update(visible=visible)
            self.main_window['-SAMPLE_2-'].update(visible=visible)
            self.main_window['-NUM_ITER_2_TEXT-'].update(visible=visible)
            self.main_window['-NUM_ITER_2-'].update(visible=visible)
            self.main_window['-BSPLINE_RES_TEXT-'].update(visible=visible)
            self.main_window['-BSPLINE_RES-'].update(visible=visible)

    def clear_all_params(self):
        types = OPTIMIZERS
        self.show_global_params(visible=False)
        for t in types:
            self.main_window[f'-{t}_PARAM_TEXT_1-'].update(visible=False)
            self.main_window[f'-{t}_PARAM_1-'].update(visible=False)
        self.show_probreg_params(visible=False)
        self.show_contour_params(visible=False)
        self.show_contours_brows(visible=False)
        self.show_manual_registration_params(visible=False)
        self.main_window["-TFM_INPUT-"].update(visible=False)
        self.main_window["-TFM_UPLOADER-"].update(visible=False)

    def show_relevant_params_input(self):
        # print("showing params")
        if self.registration_type is not None:
            types = OPTIMIZERS
            self.clear_all_params()
            self.show_global_params()
            for t in types:
                self.main_window[f'-{t}_PARAM_TEXT_1-'].update(visible=True)
                self.main_window[f'-{t}_PARAM_1-'].update(GLOBAL_PARAM_INIT[2], visible=True)

        else:
            self.clear_all_params()

    def create_sitk_param_dicts(self, values):
        if self.registration_type is not None:
            self.registration_params['optimizer'] = values['-OPT_MENU-']
            self.registration_params['metric'] = values['-METRIC_MENU-']
            self.registration_params['sampling_percentage'] = values['-GLOBAL_PARAM_1-']
            self.registration_params['iterations'] = values['-GLOBAL_PARAM_2-']
            self.registration_params['convergence_val'] = values['-GLOBAL_PARAM_3-']
            self.registration_params['learning_rate'] = values['-GD_PARAM_1-']
            self.registration_params['accuracy'] = values['-LBFGS2_PARAM_1-']
            self.registration_params["optimizer 2"] = values['-OPT_MENU_2-']
            self.registration_params['metric 2'] = values['-METRIC_MENU_2-']
            self.registration_params['iterations 2'] = values['-NUM_ITER_2-']
            self.registration_params['sampling_percentage 2'] = values['-SAMPLE_2-']
            self.registration_params['bspline_resolution'] = values['-BSPLINE_RES-']

    def init_param_dict(self):
        self.registration_params['optimizer'] = None
        self.registration_params['metric'] = None
        self.registration_params['sampling_percentage'] = GLOBAL_PARAM_INIT[0]
        self.registration_params['iterations'] = GLOBAL_PARAM_INIT[1]
        self.registration_params['convergence_val'] = GLOBAL_PARAM_INIT[2]
        self.registration_params['learning_rate'] = GLOBAL_PARAM_INIT[2]
        self.registration_params['accuracy'] = GLOBAL_PARAM_INIT[2]

    def show_assignment_uploader(self):
        self.main_window['-ASSIGN_TEXT-'].update(visible=True)
        self.main_window['-ASSIGN_INPUT-'].update(visible=True)
        self.main_window['-ASSIGN_BROWSER-'].update(visible=True)

    def show_movement_buttons(self):
        self.main_window['-SHOW_MOVE-'].update(visible=True)
        self.main_window['-SHOW_PAIRS-'].update(visible=True)
        self.main_window['-SHOW_PAIRS_INTERACTIVE-'].update(visible=True)
        self.main_window['-EXCLUDE_WIN-'].update(visible=True)
        self.main_window['-SAVE-'].update(visible=True)

    def use_saved_transformation(self):
        """
        read transformation file. file extension should be 'tfm' (sitk), 'npy' (ICP, manual) 'npz' (CPD)
        """
        if self.case_name in os.listdir('./registration_output'):
            tfm_path = None
            tfm_files = self.find_files_with_extention(self.case_name, '.tfm')
            if len(tfm_files) > 0:
                tfm_path = "./registration_output/{}/{}".format(self.case_name, tfm_files[0])
                for file in tfm_files:
                    if "bspline" in file:
                        tfm_path = "./registration_output/{}/{}".format(self.case_name, file)
                        break
            if tfm_path is None:
                self.update_massage(NO_REGISTRATION_ERR)
            else:
                self.update_massage("Using saved transformation file found at:\n{}".format(tfm_path))
                self.tfm = sitk.ReadTransform(tfm_path)
                fixed_sitk = sitk.Cast(sitk.GetImageFromArray(self.fixed_array), sitk.sitkFloat32)
                moving_sitk = sitk.Cast(sitk.GetImageFromArray(self.moving_array), sitk.sitkFloat32)
                self.warped_moving = sitk.GetArrayFromImage(warp_image_sitk(fixed_sitk, moving_sitk, self.tfm))
        else:
            self.update_massage(NO_REGISTRATION_ERR)

    def get_ctrs_points(self, dict):
        """
        get contour points. in current version options are - read from dcm file, read from csv file.
        in future versions - model for auto segmentation
        :param dict: meta data dictionary
        :return: contour points
        """
        if self.ctrs_source == "Auto Segmentation":
            points = None
        elif self.ctrs_source == "dcm file":
            points = read_structure(dict['RTSTRUCT'])[0][1].T
        elif self.ctrs_source == 'csv file':
            points = read_structure_from_csv(self.fixed_ctrs_path)
        else:
            self.update_massage("please choose registration type first")
            points =None
        return points

    def save_registration(self):
        """
        save transformation result to file
        """
        if len(self.reg_stack) > 0:
            tfm_path = f'./registration_output/{self.case_name}/transformation_{self.registration_type}'
            if self.registration_type == "Bspline" or self.registration_type == "Affine":
                tfm_path = f'{tfm_path}.tfm'
                sitk.WriteTransform(self.tfm, tfm_path)
            elif self.registration_type == "Affine+Bspline":
                tfm_path = f'./registration_output/{self.case_name}/transformation_affine+bspline.tfm'
                # print(self.composite_tfm)
                sitk.WriteTransform(self.composite_tfm, tfm_path)
            elif self.registration_type == 'ICP':
                tfm_path = f'{tfm_path}.npy'
                np.save(tfm_path, self.tfm)
            elif self.registration_type == 'CPD':
                tfm_path = f'{tfm_path}.npz'
                tfm_1 = self.tfm_stack[-2]
                tfm_2 = self.tfm
                tfm_dict = {"rigid": tfm_1, "nonrigid": tfm_2}
                with open(tfm_path, "wb") as file:
                    pickle.dump(tfm_dict, file)

        self.message = "Registration saved"
        self.update_massage(self.message)

    def create_overlay(self):
        try:
            self.set_meta()
            self.set_moving_mask()
            self.set_fixed_mask()
        except IndexError as e:
            self.message = "Contours didn't trasnformed"
            self.update_massage(self.message)
            print(e)
        # self.masked_moving_struct = np.ma.masked_where(self.moving_struct == 0, self.moving_struct) if self.moving_struct\
        #                                                                             is not None else None
        # self.warped_moving = self.moving_array if self.warped_moving is None else self.warped_moving
        if self.overlay_viewer is not None:
            self.overlay_viewer.clear()
        # self.masked_moving_struct = self.masked_fixed_struct = None
        self.overlay_viewer = OverlayViewer(self.fixed_array, self.moving_array,
                                            self.main_window['-OVERLAY_CANVAS-'].TKCanvas,
                                            'overlay')
        self.overlay_viewer.show()
        # else:
        #     moving_ctrs_points_transformed = apply_transformation(self.moving_ctrs_points, self.tfm)
        #     overlay_contours(self.fixed_ctrs_points, moving_ctrs_points_transformed, self.case_name)
        #     self.window['-IMAGE-'].update("./registration_output/{}/icp_overlay.png".format(self.case_name))

    def calc_move(self):
        """
        caculate seeds movements
        """
        if self.fixed_idx is not None:
            if len(self.fixed_idx) > self.seeds_tips_fixed.shape[-1] or len(self.moving_idx) > \
                    self.seeds_tips_moving.shape[-1]:
                self.update_massage(f'Assignment lists length and seeds number do not match\n'
                                    f'Assignment lists lengths are: {len(self.fixed_idx)}, {len(self.moving_idx)}'
                                    f'\nSeeds number are: {self.seeds_tips_fixed.shape[-1]},'
                                    f' {self.seeds_tips_moving.shape[-1]} ')
            else:
                self.seeds_tips_fixed = self.seeds_tips_fixed[..., self.fixed_idx]
                self.seeds_tips_moving = self.seeds_tips_moving[..., self.moving_idx]
                # seeds1_assigned = self.seeds_tips_fixed[..., self.fixed_idx]
                # seeds2_assigned = self.seeds_tips_moving[..., self.moving_idx]
                self.assignment_dists = np.array(
                    [calc_dist(self.seeds_tips_fixed[..., i], self.seeds_tips_moving[..., i], calc_max=True)
                     for i in range(self.seeds_tips_moving.shape[-1])])
                error = np.ones(len(self.assignment_dists)) * self.rmse[-1] if self.rmse[-1] is not None else None
                self.analyze_distances(self.case_name, self.assignment_dists, self.seeds_tips_fixed,
                                       self.seeds_tips_moving,
                                       error, base_error=BASE_ERROR)
                self.message = 'Number of assignments - {}\n sum of distances - ' \
                               '{:.2f}\n average distance - {:.2f}' \
                    .format(len(self.assignment_dists), sum(self.assignment_dists),
                            np.mean(self.assignment_dists))
                self.update_massage(self.message)
                self.show_movement_buttons()
        else:
            self.update_massage("Please assign pair first")

    def assign_seeds(self, values):
        """
        assign seeds pairs. can be done by Munkres algorithm (default) or excplicit assign lists
        :param values: event values
        """
        if self.assignment_method is None:
            self.update_massage("Please choose assignment method first")
        else:
            if self.assignment_method == "Munkres":
                self.fixed_idx, self.moving_idx = assignment(self.seeds_tips_fixed, self.seeds_tips_moving
                                                             )
            else:
                path = values["-ASSIGN_INPUT-"]
                self.fixed_idx, self.moving_idx = parse_lists_from_file('./list')
            self.update_massage(f'assignment is:\n{self.fixed_idx}\n{self.moving_idx}')
            # self.window['-MSG-'].update(font=font_text)

        # seeds1, seeds2, dists, errors = calculate_distances(case, seeds1, seeds2, meta_fixed, meta_moving, case_idx,
        #                                                     assignment)

    def save_results_to_csv(self):
        try:
            reg_stack = None if len(self.reg_stack) == 0 else ','.join([str(i) for i in self.reg_stack])
            rmse = None if len(self.rmse) == 1 else ','.join([str(i) for i in self.rmse[1:]])
            # print(self.rmse, self.correspondence)
            df = pd.DataFrame({self.df_features[0]: pd.to_datetime('today'), self.df_features[1]: self.case_name,
                               self.df_features[2]: reg_stack,
                               self.df_features[3]: self.registration_params["optimizer"], self.df_features[4]:
                                   self.registration_params['metric'], self.df_features[5]:
                                   self.registration_params["iterations"],
                               self.df_features[6]: self.registration_params['learning_rate'],
                               self.df_features[7]: self.registration_params['accuracy'],
                               self.df_features[8]: self.registration_params['convergence_val'],
                               self.df_features[9]: rmse, self.df_features[10]: self.correspondence,
                               self.df_features[11]: np.nanmean(self.assignment_dists),
                               self.df_features[12]: np.nanmedian(self.assignment_dists),
                               self.df_features[13]: np.nanstd(self.assignment_dists),
                               self.df_features[14]: np.nanmax(self.assignment_dists)},
                              index=[0])
            df.to_csv("./results.csv", mode='a', index=False, header=False, encoding='utf-8')
            self.message = self.message + "\nExperiment results were saved to results.csv"
            self.update_massage(self.message)
        except PermissionError as e:
            self.update_massage("You have no permission to edit the file results.csv while its open."
                                " please close the file and try again")
        except TypeError as e:
            print(e)
            self.update_massage("To save your results, you must assign pairs and calculate "
                                "distances")

    def get_seed_from_contour(self, values, event, num):
        """
        read contour that represent seeds (from MIM)
        :param values: event values
        :param event: event
        :param num: 1 - fixed , 2 - moving
        """
        seeds, orientation = read_structure(values[event], seeds=True)
        if num == 1:
            self.seeds_tips_fixed = get_seeds_tips(seeds, orientation)
        elif num == 2:
            self.seeds_tips_moving = get_seeds_tips(seeds, orientation)
        self.message = f"{seeds.shape[-1]} seeds was loaded from contours"
        self.update_massage(self.message)
        self.show_seeds_uploader(num, visible=False)


if __name__ == "__main__":
    app = App()
    app.run()