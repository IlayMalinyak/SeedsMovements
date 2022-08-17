# __author:IlayK
# data:03/05/2022

import PySimpleGUI as sg
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from utils import *
from registration import register_sitk, warp_image_sitk, ContourRegistration, read_image, get_inverse_transform
import SimpleITK as sitk
from analyze import *
import pandas as pd
from visualization import *
from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform
from scipy.ndimage import rotate, shift
import time

px = 1/plt.rcParams['figure.dpi']
plt.rcParams["figure.figsize"] = (500*px,500*px)
font_header = ("Arial, 18")
font_text = ("Ariel, 12")
sg.theme("SandyBeach")

OPTIMIZERS = ['GD', 'LBFGS2']

NO_REGISTRATION_ERR = "No registration has been saved on this case yet. please choose registrarion" \
                      " type and run registrarion"

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

        data_column = [
            [sg.Text('Experiment name ', size=(15, 1), font=font_header)],
            [sg.InputText(key='-NAME-', enable_events=True)],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text('Upload files', font=font_header)],
            [sg.Text('DICOM dir 1', font=font_text), sg.In(size=(15, 1), enable_events=True, key='-FIXED_FOLDER-'),
             sg.FolderBrowse(font=font_text)],
            [sg.Text('DICOM dir 2', font=font_text), sg.In(size=(15, 1), enable_events=True, key='-MOVING_FOLDER-'),
             sg.FolderBrowse(font=font_text)],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text('Registration', font=font_header)],
            [sg.Text("Registration Type", font=font_text), sg.Combo(['Use saved Registration', 'Affine', 'Bspline', "Affine+Bspline",
                                                                     "ICP"], enable_events=True, font=font_text, key="-REG_MENU-")],
            [sg.Text('Optimizer', key='-OPT_TEXT-', visible=False),
             sg.Combo(['LBFGS2', 'Gradient Decent'], key='-OPT_MENU-', visible=False, enable_events=True),
             sg.Text('Metric', key='-METRIC_TEXT-', visible=False),
             sg.Combo(['Mean Squares', 'Mutual information'], key='-METRIC_MENU-', visible=False, enable_events=True)],
             [sg.Text('Optimizer 2', key='-OPT_TEXT_2-', visible=False),
              sg.Combo(['LBFGS2', 'Gradient Decent'], key='-OPT_MENU_2-', visible=False, enable_events=True),
              sg.Text('Metric 2', key='-METRIC_TEXT_2-', visible=False),
              sg.Combo(['Mean Squares', 'Mutual information'], key='-METRIC_MENU_2-', visible=False, enable_events=True),],
             [sg.Text('Sampling percentage (0-1)', key='-GLOBAL_PARAM_TEXT_1-', visible=False),
             sg.InputText(size=(4, 1), visible=False, key='-GLOBAL_PARAM_1-', enable_events=True),
             sg.Text('Number of Iterations', key='-GLOBAL_PARAM_TEXT_2-', visible=False),
             sg.InputText(size=(4, 1), visible=False, key='-GLOBAL_PARAM_2-', enable_events=True)
             ],
            [

             sg.Text('Sampling 2', key='-SAMPLE_TEXT_2-', visible=False),
              sg.InputText(size=(4, 1), visible=False, key='-SAMPLE_2-', enable_events=True),
             sg.Text('Number of Iterations 2', key='-NUM_ITER_2_TEXT-', visible=False),
             sg.InputText(size=(4, 1), visible=False, key='-NUM_ITER_2-', enable_events=True)
             ],
            [
             sg.Text('Convergence Tolerance', key='-GLOBAL_PARAM_TEXT_3-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-GLOBAL_PARAM_3-', enable_events=True)
             ,sg.Text('Solution accuracy', key='-LBFGS2_PARAM_TEXT_1-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-LBFGS2_PARAM_1-', enable_events=True),
             sg.Text('Learning Rate', key='-GD_PARAM_TEXT_1-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-GD_PARAM_1-', enable_events=True),
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
                                                                        font=font_text, key='-SHOW_PAIRS-', visible=False)
             ,sg.Button("Save results to csv", font=font_text, key='-SAVE-', visible=False)],
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

        self.window = sg.Window('Demo Application - Seed Movement', layout, finalize=True,
                            return_keyboard_events=True)
        self.case_name = None
        self.registration_plot_path = None
        self.fixed_dict = None
        self.moving_dict = None
        self.meta = None
        self.fixed_array = None
        self.moving_array = None
        # self.warped_array = None
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
        self.moving_ctrs_points = None
        # self.warped_ctrs_points = None
        self.masked_fixed_struct = None
        self.masked_moving_struct = None
        self.moving_struct = None
        self.moving_struct_sitk = None
        self.fixed_struct = None
        self.fixed_struct_sitk = None

        self.reg_stack = []
        self.param_stack = []
        self.registration_type = None
        self.registration_params = {}
        self.euler_angles = np.zeros(3)
        self.shift = np.zeros(3)
        self.optimizer = None
        self.rmse = []
        self.correspondence = None
        self.local_params = {}
        # self.warped_moving = None
        self.icp_thresh = None
        self.tfm = None
        self.composite_tfm = None
        self.inv_tfm = None
        self.first_sitk = True


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
                            "Accuracy Threshold", "Convergence Delta", "RMSE", "% Correspondence set","Average Movement (mm)", "Median Movement (mm)",
                            "Standard Deviation (mm)", "Maximum Movement (mm)", "Average Error (mm)"]
        self.init_param_dict()
        self.message = ""

    def update_massage(self, massage):
        self.window['-MSG-'].update(massage)

    @staticmethod
    def find_files_with_extention(folder, extention):
        files = os.listdir('./registration_output/{}'.format(folder))
        return [i for i in files if i.endswith(extention)]

    def run(self):
        while True:
            event, values = self.window.read()
            if event == "-NAME-":
                self.case_name = values[event].lower()
                self.registration_plot_path = f'./registration_output/{self.case_name}/registration_res.png'
            if event == '-FIXED_FOLDER-':
                self.upload_fixed(values)
            if event == '-MOVING_FOLDER-':
                self.upload_moving(values)
            if event == "-REG_MENU-":
                self.clear_all_params()
                self.registration_type = values[event]
                print("reg type ", self.registration_type)
                if self.registration_type == "ICP":
                    self.show_contour_params()
                elif self.registration_type == "Bspline" or self.registration_type == 'Affine' or self.registration_type\
                == "Affine+Bspline":
                    self.show_relevant_params_input()
                    self.init_param_dict()
                elif self.registration_type == "Use saved Registration":
                    print("should show uploader")
                    self.window['-TFM_INPUT-'].update(visible=True)
                    self.window['-TFM_UPLOADER-'].update(visible=True)
            if event == "-TFM_INPUT-":
                tfm_path = values[event]
                self.registration_type = "saved_" + tfm_path.removesuffix('.tfm').split('_')[-1]
                try: # TODO inverse for composite transform?
                    if self.registration_type == "saved_affine+bspline":
                        self.tfm = sitk.ReadTransform(tfm_path)
                    else:
                        self.tfm = sitk.ReadTransform(tfm_path)
                        self.inv_tfm, self.disp_img = get_inverse_transform(self.tfm, self.registration_type)
                except Exception as e:
                    print(e)
                    np.load(tfm_path)
                self.update_massage(f"{tfm_path} uploaded successfully ")
            self.create_param_dicts(values)
            if event == "-CONTOURS_MENU-":
                self.ctrs_source = values[event]
                if self.ctrs_source == "csv file":
                    self.show_contours_brows()
                else:
                    self.hide_contours_brows()
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
                    overlay_contours_interactive(self.fixed_ctrs_points, self.moving_ctrs_points)
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
            if event == "-CLEAR_REGISTRATION-":
                self.reset_data()
                self.message = "All registrations cleared"
                self.update_massage(self.message)
            if event == '-REGISTER-':
                self.create_case_dir()
                self.set_meta()
                # if self.seeds_tips_moving is None:
                #     self.seeds_tips_warped = self.seeds_tips_moving.copy()
                self.run_registration()
                if self.tfm is not None:
                    self.message = self.message + f"\n{self.registration_type} registration finished\n RMSE: {self.rmse[-1]}"
                    self.update_massage(self.message)
                    self.shift = np.array([0, 0, 0])
                    self.euler_angles = np.array([0, 0, 0])
            if event == "-SAVE_REGISTRATION-":
                if self.tfm is not None:
                    tfm_path = f'./registration_output/{self.case_name}/transformation_{self.registration_type}'
                    if self.registration_type == "Bspline" or self.registration_type == "Affine":
                        tfm_path = f'{tfm_path}.tfm'
                        sitk.WriteTransform(self.tfm, tfm_path)
                    elif self.registration_type == "Affine+Bspline":
                        tfm_path = f'./registration_output/{self.case_name}/transformation_affine+bspline.tfm'
                        print(self.composite_tfm)
                        sitk.WriteTransform(self.composite_tfm, tfm_path)
                    elif self.registration_type == 'ICP':
                        tfm_path = f'{tfm_path}.npy'
                        np.save(tfm_path, self.tfm)
                self.message = "Registration saved"
                self.update_massage(self.message)
            if event == "-PLOT_REG-":
                try:
                    self.window['-IMAGE-'].update(self.registration_plot_path)
                except Exception as e:
                    self.update_massage("Nothing to show")
            if event == "-OVERLAY-":
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
                    self.masked_moving_struct = self.masked_fixed_struct = None
                self.overlay_viewer = OverlayViewer(self.fixed_array, self.moving_array, self.window['-OVERLAY_CANVAS-'].TKCanvas,
                                                    'overlay', self.masked_fixed_struct, self.masked_moving_struct)
                self.overlay_viewer.show()
                # else:
                #     moving_ctrs_points_transformed = apply_transformation(self.moving_ctrs_points, self.tfm)
                #     overlay_contours(self.fixed_ctrs_points, moving_ctrs_points_transformed, self.case_name)
                #     self.window['-IMAGE-'].update("./registration_output/{}/icp_overlay.png".format(self.case_name))

            if event == "-ASSIGN_MENU-":
                self.assignment_method = values[event]

            if event == '-ASSIGN-':
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

            if self.assignment_method == "Upload List":
                self.show_assignment_uploader()
            if event == "-CALC_MOVE-":
                self.create_case_dir()
                if self.fixed_idx is not None:
                    if len(self.fixed_idx) > self.seeds_tips_fixed.shape[-1] or len(self.moving_idx) > \
                            self.seeds_tips_moving.shape[-1]:
                        self.update_massage(f'Assignment lists length and seeds number do not match\n'
                                            f'Assignment lists lengths are: {len(self.fixed_idx)}, {len(self.moving_idx)}'
                                            f'\nSeeds number are: {self.seeds_tips_fixed.shape[-1]},'
                                            f' {self.seeds_tips_moving.shape[-1]} ')
                    else:
                        seeds1_assigned = self.seeds_tips_fixed[..., self.fixed_idx]
                        seeds2_assigned = self.seeds_tips_moving[..., self.moving_idx]
                        self.assignment_dists = np.array(
                            [calc_dist(seeds1_assigned[..., i], seeds2_assigned[..., i], calc_max=True)
                             for i in range(seeds2_assigned.shape[-1])])
                        seeds1, seeds2, _, self.errors = analyze_distances(self.case_name, self.assignment_dists,
                                                                               seeds1_assigned, seeds2_assigned)
                        self.message = 'Number of assignments - {}\n sum of distances - '\
                                            '{:.2f}\n average distance - {:.2f}'\
                                            .format(len(self.assignment_dists), sum(self.assignment_dists),\
                                                    np.mean(self.assignment_dists))
                        self.update_massage(self.message)
                        self.show_movement_buttons()
                else:
                    self.update_massage("Please assign pair first")
            if event == "-SHOW_MOVE-":
                self.plot_name = "moves"
                self.window['-IMAGE-'].update("./movement_output/{}/movements.png".format(self.case_name))
            if event == "-SHOW_PAIRS-":
                self.plot_name = "pairs"
                self.window['-IMAGE-'].update("./movement_output/{}/pairs.png".format(self.case_name))
            if event == "-SAVE-":
                rmse = [self.param_stack[i]['RMSE'] if 'RMSE' in self.param_stack[i].keys()
                        else None for i in range(len(self.param_stack))]
                try:
                    print(self.rmse, self.correspondence)
                    df = pd.DataFrame({self.df_features[0]:pd.to_datetime('today'), self.df_features[1]:self.case_name,
                                       self.df_features[2]:self.registration_type,
                                       self.df_features[3]:self.registration_params["optimizer"],self.df_features[4]:
                                           self.registration_params['metric'], self.df_features[5]:
                                       self.registration_params["iterations"], self.df_features[6]:self.registration_params['learning_rate'],
                                      self.df_features[7]:self.registration_params['accuracy'],
                                       self.df_features[8]:self.registration_params['convergence_val'],
                                       self.df_features[9]:self.rmse, self.df_features[10]:self.correspondence,
                                       self.df_features[11]:np.nanmean(self.assignment_dists), self.df_features[12]:np.nanmedian(self.assignment_dists),
                                       self.df_features[13]:np.nanstd(self.assignment_dists),
                                       self.df_features[14]:np.nanmax(self.assignment_dists), self.df_features[15]:np.nanmean(self.errors)},
                                      index=[0])
                    df.to_csv("./results.csv", mode='a', index=False, header=False, encoding='utf-8')
                    self.message = self.message + "\nExperiment results were saved to results.csv"
                    self.update_massage(self.message)
                except PermissionError as e:
                    self.update_massage("You have no permission to edit the file results.csv while its open."
                                        " please close the file and try again")
                except TypeError as e:
                    self.update_massage("To save your results, you must register the dicoms, assign pairs and caluculate"
                                        "distances")





            # if event == '-MOVING_FOLDER-':
            #     print('moving folder', values['-MOVING_FOLDER-'])
            #     self.moving_dict = get_all_dicom_files(values['-MOVING_FOLDER-'], {})
            #     self.moving_array = read_dicom(self.moving_dict['CT'])
            #     self.moving_viewer = DicomCanvas(self.moving_array, self.window['-MOVING_CANVAS-'].TKCanvas)
            if event == sg.WIN_CLOSED or event == 'Exit':
                break

    def set_meta(self):
        if len(self.reg_stack) == 0 or (all(np.array(self.reg_stack) == 'ICP')):
            self.meta = self.moving_dict['meta']
        else:
            self.meta = self.fixed_dict['meta']

    def show_manual_registration_params(self):
        self.window['-XY_ROT_TEXT-'].update(visible=True)
        self.window['-XY_ROT-'].update("", visible=True)
        self.window['-XZ_ROT_TEXT-'].update(visible=True)
        self.window['-XZ_ROT-'].update("", visible=True)
        self.window['-YZ_ROT_TEXT-'].update(visible=True)
        self.window['-YZ_ROT-'].update("", visible=True)
        self.window['-X_SHIFT_TEXT-'].update(visible=True)
        self.window['-X_SHIFT-'].update("", visible=True)
        self.window['-Y_SHIFT_TEXT-'].update(visible=True)
        self.window['-Y_SHIFT-'].update("", visible=True)
        self.window['-Z_SHIFT_TEXT-'].update(visible=True)
        self.window['-Z_SHIFT-'].update("", visible=True)

    def hide_manual_registration_params(self):
        self.window['-XY_ROT_TEXT-'].update(visible=False)
        self.window['-XY_ROT-'].update(visible=False)
        self.window['-XZ_ROT_TEXT-'].update(visible=False)
        self.window['-XZ_ROT-'].update(visible=False)
        self.window['-YZ_ROT_TEXT-'].update(visible=False)
        self.window['-YZ_ROT-'].update(visible=False)
        self.window['-X_SHIFT_TEXT-'].update(visible=False)
        self.window['-X_SHIFT-'].update(visible=False)
        self.window['-Y_SHIFT_TEXT-'].update(visible=False)
        self.window['-Y_SHIFT-'].update(visible=False)
        self.window['-Z_SHIFT_TEXT-'].update(visible=False)
        self.window['-Z_SHIFT-'].update(visible=False)

    def get_seeds(self, dict):
        print("loading seeds")
        seeds, orientation = get_seeds_dcm(dict['RTPLAN'])
        return get_seeds_tips(seeds, orientation)
        # self.fixed_seeds, self.fixed_orientation = get_seeds_dcm(self.fixed_dict["RTPLAN"])
        # self.moving_seeds, self.moving_orientation = get_seeds_dcm(self.moving_dict["RTPLAN"])
        # self.seeds_tips_fixed = get_seeds_tips(self.fixed_seeds, self.fixed_orientation)
        # self.seeds_tips_moving = get_seeds_tips(self.moving_seeds, self.moving_orientation)
        # self.seeds_tips_moving_warped = self.seeds_tips_moving

    def create_case_dir(self):
        if self.case_name not in os.listdir('./registration_output'):
            dir_name = f"./registration_output/{self.case_name}"
            os.mkdir(dir_name)
        if self.case_name not in os.listdir('./movement_output'):
            dir_name = f"./movement_output/{self.case_name}"
            os.mkdir(dir_name)

    def run_registration(self):
        if self.registration_type is not None:
            if self.fixed_array is not None and self.moving_array is not None:
                if self.case_name is not None:
                    self.message = "Registering...\nThis may take a while"
                    self.update_massage(self.message)
                    self.window.refresh()
                    if self.registration_type == "Bspline" or self.registration_type == "Affine":
                        self.run_sitk_registration(self.registration_type)
                    elif self.registration_type == "Affine+Bspline":
                        self.run_composite_sitk_registration(["Affine", "Bspline"])
                    elif self.registration_type =='ICP':
                        self.run_adaptive_icp()
                        self.tfm = self.tfm if self.tfm is not None else np.eye(4)
                        # self.moving_array = wrap_image_with_matrix(self.fixed_array, self.moving_array,
                        #                                            self.meta, np.linalg.inv(self.tfm))
                        # # self.warped_array = affine_transform(self.warped_array, self.tfm)
                        # self.update_arrays("icp")
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
                    self.set_meta()
                    print(self.tfm)
                    print(self.moving_dict['meta'])
                else:
                    self.update_massage("Enter Experiment Name")
            else:
                self.update_massage("You must upload 2 DICOMS to perform registration")
        else:
            self.update_massage("No Registration to run")
        return

    def run_saved_registration(self):
        print("running saved registration")
        self.moving_sitk = warp_image_sitk(self.fixed_sitk, self.moving_sitk, self.tfm)
        reg_type = self.registration_type.split("_")[-1]
        if reg_type == 'Bspline' or reg_type == "Affine" or reg_type == "affine+bspline":
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
        else:
            self.seeds_tips_moving = apply_transformation_on_seeds(np.linalg.inv(self.tfm),
                                                                   self.seeds_tips_moving,
                                                                   "affine")

    def run_composite_sitk_registration(self, types):

        # fixed_path = get_all_dicom_files(r"C:\Users\ilaym\Desktop\Dart\seedsMovement\cases\brain\0", {})
        # moving_path = get_all_dicom_files(r"C:\Users\ilaym\Desktop\Dart\seedsMovement\cases\brain\20", {})
        # seeds1_center, orie1 = get_seeds_dcm(fixed_path['RTPLAN'])
        # seeds1 = get_seeds_tips(seeds1_center, orie1)
        # meta1 = fixed_path['meta']
        # meta2 = moving_path['meta']
        # seeds2_center, orie2 = get_seeds_dcm(moving_path['RTPLAN'])
        # seeds2 = get_seeds_tips(seeds2_center, orie2)
        #
        # fixed_points = read_structure(fixed_path['RTSTRUCT'])[0][1]
        # moving_points = read_structure(moving_path['RTSTRUCT'])[0][1]
        # fixed_0, warped_0, outx_0,outTx_inv, disp_img  = register_sitk(fixed_path['CT'], moving_path['CT'],
        #                                                                self.fixed_dict['meta'],'./', "Affine",
        #                                                                self.registration_params)
        # compTfm = sitk.CompositeTransform(outx_0)
        #
        # self.registration_params["optimizer"] = self.registration_params["optimizer 2"]
        # self.registration_params["metric"] = self.registration_params["metric 2"]
        # self.registration_params["iterations"] = self.registration_params["iterations 2"]
        # self.registration_params['sampling_percentage'] = self.registration_params['sampling_percentage 2']
        # fixed_0, warped_0, outx_0, outTx_inv, disp_img = register_sitk(fixed_0, warped_0, self.fixed_dict['meta'], './',
        #                                                                "Bspline",
        #                                                                self.registration_params)
        #
        # warped_contours = apply_sitk_transformation_on_struct(compTfm, fixed_points.T)
        # rmse2 = calc_rmse(warped_contours.T, moving_points.T, 0.001)
        # warped_seeds = apply_transformation_on_seeds(compTfm, seeds1)
        # compTfm.AddTransform(outx_0)
        #
        # dist = calculate_distances("", warped_seeds, seeds2, meta1, meta2, None, False)
        # print(dist)
        # print("rmse2 ", rmse2)
        print("running composite registration")
        print(self.registration_params)
        self.composite_tfm = sitk.CompositeTransform(3)
        tmp_opt, tmp_metric, tmp_smp = self.registration_params["optimizer"], self.registration_params["metric"],\
        self.registration_params['sampling_percentage']
        for i in range(len(types)):
            # self.registration_params['iterations'] = iterations[i]
            # self.registration_type = types[i]
            self.run_sitk_registration(types[i], apply=False)
            print("transform type ", self.tfm.GetTransformEnum())
            if self.tfm.GetTransformEnum() == 13:
                try:
                    self.composite_tfm.AddTransform(sitk.CompositeTransform(self.inv_tfm).GetNthTransform(0))
                except Exception as e:
                    print(e)
                    self.composite_tfm.AddTransform(self.inv_tfm)
            else:
                self.composite_tfm.AddTransform(self.inv_tfm)
            self.registration_params["optimizer"] = self.registration_params["optimizer 2"]
            self.registration_params["metric"] = self.registration_params["metric 2"]
            self.registration_params["iterations"] = self.registration_params["iterations 2"]
            self.registration_params['sampling_percentage'] = self.registration_params['sampling_percentage 2']
            self.reg_stack.append("composite")
        self.reg_stack.pop()
        self.registration_params["optimizer"] = tmp_opt
        self.registration_params["metric"] = tmp_metric
        self.registration_params['sampling_percentage'] = tmp_smp
        self.seeds_tips_moving = apply_transformation_on_seeds(self.composite_tfm,
                                                              self.seeds_tips_moving,
                                                              "sitk")
        self.moving_ctrs_points = apply_sitk_transformation_on_struct(self.composite_tfm, self.moving_ctrs_points)
        self.rmse.append(calc_rmse(self.fixed_ctrs_points.T, self.moving_ctrs_points.T, 0.001))
        self.param_stack.append({"RMSE": self.rmse[-1]})

    def run_sitk_registration(self, type, apply=True):
        print(self.registration_type)
        print(self.registration_params['optimizer'], self.registration_params['metric'], self.registration_params['iterations'])
        try:
            if len(self.reg_stack) == 0 or all(np.array(self.reg_stack) == 'ICP'):
                self.fixed_sitk, self.moving_sitk, self.tfm, self.inv_tfm, self.disp_img = register_sitk(
                    self.fixed_dict['CT'], self.moving_dict['CT'], self.fixed_dict['meta'], self.registration_plot_path,
                    type, self.registration_params)
            else:
                self.fixed_sitk, self.moving_sitk, self.tfm, self.inv_tfm, self.disp_img = register_sitk(
                    self.fixed_sitk, self.moving_sitk, self.moving_dict['meta'], self.registration_plot_path,
                    type, self.registration_params)
            if apply:
                self.seeds_tips_moving = apply_transformation_on_seeds(self.inv_tfm,
                                                                           self.seeds_tips_moving,
                                                                           "sitk")
                self.moving_ctrs_points = apply_sitk_transformation_on_struct(self.inv_tfm, self.moving_ctrs_points)
                self.rmse.append(calc_rmse(self.fixed_ctrs_points.T, self.moving_ctrs_points.T, 0.001))
                self.param_stack.append({"RMSE": self.rmse})
            self.update_arrays("sitk")
            print("after registration fixed: ", self.fixed_sitk.GetOrigin(), " moving: ", self.moving_sitk.GetOrigin())
            self.param_stack.append(self.registration_params)

        except Exception as e:
            print("sitk expection ", e)
            self.update_massage("Registration Failed")

        # if self.moving_struct_sitk is not None:
        #     self.moving_struct_sitk = warp_image_sitk(self.fixed_sitk, self.moving_struct_sitk, self.tfm, mask=True,
        #                                               default_val=0)

    def run_manual_registration(self):

        #TODO update arrays (numpy and sitk)
        print('shift ', self.shift)
        R = Rotation.from_euler("xyz", self.euler_angles, degrees=True)
        mean = np.mean(self.moving_ctrs_points, axis=0)
        # print("R ", R.as_matrix())
        self.moving_ctrs_points -= mean
        self.moving_ctrs_points = R.apply(self.moving_ctrs_points)
        self.moving_ctrs_points += mean
        self.moving_ctrs_points += self.shift[None, :]

        mean = mean[None, :, None]
        self.seeds_tips_moving -= mean
        self.seeds_tips_moving = apply_scipy_transformation_on_seeds(R, self.seeds_tips_moving)
        self.seeds_tips_moving += mean
        self.seeds_tips_moving += self.shift[None, :, None]
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
        reg_obj = ContourRegistration(f'registration_output/{self.case_name}', self.case_name)
        center_fixed = np.mean(self.fixed_ctrs_points, axis=0)
        center_moving = np.mean(self.moving_ctrs_points, axis=0)
        dist_thresh = np.sqrt(np.sum((center_fixed - center_moving) ** 2)) / 2
        dist_thresh = np.max((dist_thresh,0.01))
        max_dist = 10*dist_thresh
        thresh = self.icp_thresh/100 if self.icp_thresh is not None else 0.9
        min_rmse = [np.inf]
        best_reg = None
        best_ratio = 0
        i = 0
        while True:
            res, fixed_surface, moving_surface = reg_obj.icp(self.fixed_ctrs_points, self.moving_ctrs_points,
                                                             dist_thresh)

            num_pairs = reg_obj.get_correspondence().shape[0]
            num_points = max(fixed_surface.shape[0], moving_surface.shape[0])
            rmse = res.inlier_rmse if res is not None else np.inf
            self.update_massage(f"correspondence set - {num_pairs}, num points - {num_points}, "
                                f"inlier rmse - {rmse}")
            if rmse < min_rmse[-1] and num_pairs > num_points * thresh:
                min_rmse, best_reg = reg_obj.get_params()
                best_ratio = num_pairs/num_points
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
            self.moving_ctrs_points = apply_transformation(self.moving_ctrs_points, self.inv_tfm)
            self.seeds_tips_moving = apply_transformation_on_seeds(np.linalg.inv(self.tfm),
                                                                   self.seeds_tips_moving,
                                                                   "affine")
            # self.warped_struct = get_contour_mask(self.warped_ctrs_points.T, self.meta, self.fixed_array.shape)
            # self.masked_warped_struct = np.ma.masked_where(self.warped_struct == 0, self.warped_struct)
            # self.set_moving_mask() # TODO fix
            plot_rmse_and_contours(min_rmse, self.fixed_ctrs_points, self.moving_ctrs_points, self.registration_plot_path)
            self.rmse.append(min_rmse[-1])
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
            print("changing numpy arrays")
            self.moving_array = np.transpose(sitk.GetArrayFromImage(self.moving_sitk), (1,2,0))
            self.fixed_array = np.transpose(sitk.GetArrayFromImage(self.fixed_sitk), (1, 2, 0))
            print("moving array shape", self.moving_array.shape)
            if self.moving_struct_sitk is not None:
                self.moving_struct = np.transpose(sitk.GetArrayFromImage(self.moving_struct_sitk), (1,2,0))
        else:
            print("changing sitk image")
            self.moving_sitk = sitk.Cast(sitk.GetImageFromArray(np.transpose(self.moving_array,(2,0,1))),sitk.sitkFloat32)
            if self.moving_struct is not None:
                self.moving_struct_sitk = sitk.Cast(sitk.GetImageFromArray(np.transpose(self.moving_struct,(2,0,1))),sitk.sitkFloat32)


    def upload_moving(self, values):
        self.message = ""
        self.rmse = []
        self.masked_moving_struct = self.masked_moving_struct_orig = None
        self.moving_ctrs_points = self.moving_ctrs_points_orig = None
        self.moving_struct = None
        self.moving_dict = get_all_dicom_files(values['-MOVING_FOLDER-'], {})
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
        self.seeds_tips_moving = self.get_seeds(self.moving_dict)
        self.seeds_tips_moving_orig = self.seeds_tips_moving.copy()
        # self.seeds_tips_warped = self.seeds_tips_moving.copy()
        if 'RTSTRUCT' in self.moving_dict.keys():
            self.moving_ctrs_points = read_structure(self.moving_dict['RTSTRUCT'])[0][1].T
            self.moving_ctrs_points_orig = self.moving_ctrs_points.copy()
            # self.warped_ctrs_points = self.moving_ctrs_points.copy()
            try:
                self.moving_struct = get_contour_mask(self.moving_ctrs_points.T, self.moving_dict['meta'], self.moving_array.shape)
                self.moving_struct_sitk = sitk.Cast(sitk.GetImageFromArray(np.transpose(self.moving_struct,(2,0,1))),sitk.sitkFloat32)
                self.masked_moving_struct = np.ma.masked_where(self.moving_struct == 0, self.moving_struct)
                self.masked_moving_struct_orig = self.masked_moving_struct.copy()
            except Exception as e:
                print(e)
                self.message = f"error loading contours on dicom:\n {e}"
                self.update_massage(self.message)
        if self.moving_viewer is not None:
            self.moving_viewer.clear()
        self.moving_viewer = DicomViewer(self.moving_array, self.window['-MOVING_CANVAS-'].TKCanvas, 'moving',
                                             self.masked_moving_struct)
        self.moving_viewer.show()
        self.message = self.message + "{} was uploaded successfully".format(self.moving_dict['meta']['ID'].value)
        self.update_massage(self.message)
        self.tfm = None

    def upload_fixed(self, values):
        self.message = ""
        self.rmse = []
        self.masked_fixed_struct = self.masked_fixed_struct_orig = None
        self.fixed_ctrs_points = self.fixed_ctrs_points_orig = None
        self.fixed_struct = None
        self.fixed_dict = get_all_dicom_files(values['-FIXED_FOLDER-'], {})
        self.fixed_array = read_dicom(self.fixed_dict['CT'], self.fixed_dict['meta'])
        # self.fixed_sitk = sitk.Cast(sitk.GetImageFromArray(np.transpose(self.fixed_array, (2, 0, 1))),
        #                              sitk.sitkFloat32)
        self.fixed_sitk = read_image(self.fixed_dict['CT'])
        # try:
        #     set_meta_data_to_sitk_image(self.fixed_sitk, self.fixed_dict['meta'])
        # except Exception as e:
        #     print(e)
        self.seeds_tips_fixed = self.get_seeds(self.fixed_dict)
        self.seeds_tips_fixed_orig = self.seeds_tips_fixed.copy()
        if 'RTSTRUCT' in self.fixed_dict.keys():
            self.fixed_ctrs_points = read_structure(self.fixed_dict['RTSTRUCT'])[0][1].T
            self.fixed_ctrs_points_orig = self.fixed_ctrs_points.copy()
            try:
                # struct = read_structure(self.fixed_dict['RTSTRUCT'])
                self.fixed_struct = get_contour_mask(self.fixed_ctrs_points.T, self.fixed_dict['meta'], self.fixed_array.shape)
                self.masked_fixed_struct = np.ma.masked_where(self.fixed_struct == 0, self.fixed_struct)
                self.masked_fixed_struct_orig = self.masked_fixed_struct.copy()
            except Exception as e:
                print(e)
                self.message = f"error loading contours on dicom:\n {e}"
                self.update_massage(self.message)
        if self.fixed_viewer is not None:
            self.fixed_viewer.clear()
        self.fixed_viewer = DicomViewer(self.fixed_array, self.window['-FIXED_CANVAS-'].TKCanvas, 'fixed',
                                            self.masked_fixed_struct)
        self.fixed_viewer.show()
        self.message = self.message + "{} was uploaded successfully".format(self.fixed_dict['meta']['ID'].value)
        self.update_massage(self.message)
        print("after reading ", self.fixed_sitk.GetOrigin())
        self.tfm = None

    def reset_data(self):
        self.moving_array = self.moving_array_orig
        self.moving_sitk = read_image(self.moving_dict['CT'])
        self.fixed_sitk = read_image(self.fixed_dict['CT'])
        self.seeds_tips_moving = self.seeds_tips_moving_orig
        self.moving_ctrs_points = self.moving_ctrs_points_orig
        self.seeds_tips_fixed = self.seeds_tips_fixed_orig
        self.fixed_ctrs_points = self.fixed_ctrs_points_orig
        self.rmse = []
        # self.update_arrays("affine")
        try:
            self.masked_moving_struct = self.masked_moving_struct_orig
            self.masked_fixed_struct = self.masked_fixed_struct_orig
        except:
            pass

    def hide_contours_brows(self):
        self.window["-FIXED_CONTOURS_TEXT-"].update(visible=False)
        self.window["-FIXED_CONTOURS_INPUT-"].update(visible=False)
        self.window["-FIXED_CONTOURS_BROWS-"].update(visible=False)
        self.window["-MOVING_CONTOURS_TEXT-"].update(visible=False)
        self.window["-MOVING_CONTOURS_INPUT-"].update(visible=False)
        self.window["-MOVING_CONTOURS_BROWS-"].update(visible=False)

    def show_contours_brows(self):
        self.window["-FIXED_CONTOURS_TEXT-"].update(visible=True)
        self.window["-FIXED_CONTOURS_INPUT-"].update(visible=True)
        self.window["-FIXED_CONTOURS_BROWS-"].update(visible=True)
        self.window["-MOVING_CONTOURS_TEXT-"].update(visible=True)
        self.window["-MOVING_CONTOURS_INPUT-"].update(visible=True)
        self.window["-MOVING_CONTOURS_BROWS-"].update(visible=True)

    def show_global_params(self):
        self.window['-OPT_TEXT-'].update(visible=True)
        self.window['-OPT_MENU-'].update(visible=True)
        self.window['-METRIC_TEXT-'].update(visible=True)
        self.window['-METRIC_MENU-'].update(visible=True)
        for i in range(1,4):
            self.window[f'-GLOBAL_PARAM_TEXT_{i}-'].update(visible=True)
            self.window[f'-GLOBAL_PARAM_{i}-'].update(str(GLOBAL_PARAM_INIT[i-1]), visible=True)
        if self.registration_type == "Affine+Bspline":
            self.window['-OPT_TEXT_2-'].update(visible=True)
            self.window['-OPT_MENU_2-'].update(visible=True)
            self.window['-METRIC_TEXT_2-'].update(visible=True)
            self.window['-METRIC_MENU_2-'].update(visible=True)
            self.window['-SAMPLE_TEXT_2-'].update(visible=True)
            self.window['-SAMPLE_2-'].update(visible=True)
            self.window['-NUM_ITER_2_TEXT-'].update(visible=True)
            self.window['-NUM_ITER_2-'].update(visible=True)


    def clear_global_params(self):
        self.window['-OPT_TEXT-'].update(visible=False)
        self.window['-OPT_MENU-'].update(visible=False)
        self.window['-METRIC_TEXT-'].update(visible=False)
        self.window['-METRIC_MENU-'].update(visible=False)
        self.window['-OPT_TEXT_2-'].update(visible=False)
        self.window['-OPT_MENU_2-'].update(visible=False)
        self.window['-METRIC_TEXT_2-'].update(visible=False)
        self.window['-METRIC_MENU_2-'].update(visible=False)
        self.window['-SAMPLE_TEXT_2-'].update(visible=False)
        self.window['-SAMPLE_2-'].update(visible=False)
        self.window['-NUM_ITER_2_TEXT-'].update(visible=False)
        self.window['-NUM_ITER_2-'].update(visible=False)
        for i in range(1,4):
            self.window[f'-GLOBAL_PARAM_TEXT_{i}-'].update(visible=False)
            self.window[f'-GLOBAL_PARAM_{i}-'].update(visible=False)

    def clear_all_params(self):
        types = OPTIMIZERS
        self.clear_global_params()
        for t in types:
            self.window[f'-{t}_PARAM_TEXT_1-'].update(visible=False)
            self.window[f'-{t}_PARAM_1-'].update(visible=False)
        self.hide_contour_params()
        self.hide_contours_brows()
        self.hide_manual_registration_params()
        self.window["-TFM_INPUT-"].update(visible=False)
        self.window["-TFM_UPLOADER-"].update(visible=False)

    def show_relevant_params_input(self):
        print("showing params")
        if self.registration_type is not None:
            types = OPTIMIZERS
            self.clear_all_params()
            self.show_global_params()
            for t in types:
                self.window[f'-{t}_PARAM_TEXT_1-'].update(visible=True)
                self.window[f'-{t}_PARAM_1-'].update(GLOBAL_PARAM_INIT[2], visible=True)

        else:
            self.clear_all_params()

    def create_param_dicts(self, values):
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

    def init_param_dict(self):
        self.registration_params['optimizer'] = None
        self.registration_params['metric'] = None
        self.registration_params['sampling_percentage'] = GLOBAL_PARAM_INIT[0]
        self.registration_params['iterations'] = GLOBAL_PARAM_INIT[1]
        self.registration_params['convergence_val'] = GLOBAL_PARAM_INIT[2]
        self.registration_params['learning_rate'] = GLOBAL_PARAM_INIT[2]
        self.registration_params['accuracy'] = GLOBAL_PARAM_INIT[2]

    def show_assignment_uploader(self):
        self.window['-ASSIGN_TEXT-'].update(visible=True)
        self.window['-ASSIGN_INPUT-'].update(visible=True)
        self.window['-ASSIGN_BROWSER-'].update(visible=True)

    def show_movement_buttons(self):
        self.window['-SHOW_MOVE-'].update(visible=True)
        self.window['-SHOW_PAIRS-'].update(visible=True)
        self.window['-SAVE-'].update(visible=True)

    def use_saved_transformation(self):
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

    def show_contour_params(self):
        self.window['-GET_CONTOURS_TEXT-'].update(visible=True)
        self.window['-CONTOURS_MENU-'].update(visible=True)
        self.window['-ICP_THRESH_TEXT-'].update(visible=True)
        self.window['-ICP_THRESH-'].update(visible=True)

    def hide_contour_params(self):
        self.window['-GET_CONTOURS_TEXT-'].update(visible=False)
        self.window['-CONTOURS_MENU-'].update(visible=False)
        self.window['-ICP_THRESH_TEXT-'].update(visible=False)
        self.window['-ICP_THRESH-'].update(visible=False)

    def get_ctrs_points(self, dict):
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


if __name__ == "__main__":
    app = App()
    app.run()