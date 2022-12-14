# __author:IlayK
# data:02/05/2022
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils import *
from scipy.optimize import linear_sum_assignment
from visualization import *
import registration
from SimpleITK import AffineTransform


BASE_ERROR = 1.3


def analyze(fixed_dir, moving_dir, df, case, registration_method='bspline', assignment=None):
    """
    analyze movements
    :param fixed_dir: directory to fixed dicom 
    :param moving_dir: directory to moving dicom
    :param df: dataframe to store the results
    :param case: case name
    :param registration_method: registration method. currently either 'bspline' (default) or 'rigid'
    :param assignment: assignment list for pair assignment. if None, Munkres algorithm will create the assignment
    :return: Dataframe with movements results
    """
    print("***** %s *****" % case)

    seeds1, orientation1, seeds2, orientation2, dict_1, dict_2 = create_data(fixed_dir, moving_dir)
    if case not in df['Case'].tolist():
        df = df.append({"Case": case}, ignore_index=True)
    case_idx = np.where(df['Case'] == case)[0]

    meta_1 = dict_1['meta']
    meta_2 = dict_2['meta']
    seeds1 = get_seeds_tips(seeds1, orientation1, meta_1['pixelSpacing'][0], meta_1['pixelSpacing'][1],
                            meta_1['sliceThickness'])
    seeds2 = get_seeds_tips(seeds2, orientation2, meta_2['pixelSpacing'][0], meta_2['pixelSpacing'][1],
                            meta_2['sliceThickness'])
    if registration_method is not None:
        # fixed, moving = read_dicom(dict_1['CT']), read_dicom(dict_2['CT'])
        if case not in os.listdir('registration_output'):
            os.mkdir(f'registration_output/{case}')
        fixed, moving, warped, outTx = registration.register_sitk(dict_1['CT'], dict_2['CT'], f'registration_output/{case}',
                                                                  type=registration_method)
        overlay_images(fixed, warped)
        seeds2 = apply_transformation_on_seeds(outTx, seeds2)

    seeds1, seeds2, dists, errors = calculate_distances(case, seeds1, seeds2, meta_1, meta_2, assignment)

    df = update_df(case_idx, df, dists, errors)
    return df


def update_df(case_idx, df, dists, errors):
    """
    update Dataframe
    :param case_idx: case index
    :param df: Dataframe
    :param dists: movements
    :param errors: movements errors
    :return: Dataframe with data inserted at case_index
    """
    df.loc[case_idx, "Average Movement (mm)"] = np.average(dists)
    df.loc[case_idx, "Median Movement (mm)"] = np.median(dists)
    df.loc[case_idx, "Error (mm)"] = np.average(errors)
    df.loc[case_idx, "Standard Deviation (mm)"] = np.std(dists)
    df.loc[case_idx, "Maximum Movement (mm)"] = np.max(dists)
    return df


def apply_transformation_on_centers(transform, seeds):
    """
    apply transformation on seeds centers
    :param transform: simpleitk transformation object
    :param seeds: (3,N) array of (x,y,z) coordinated of N seeds
    :return: (3,N) array. centers transformed
    """
    new_centers = np.zeros((3, seeds.shape[-1]))
    for i in range(seeds.shape[-1]):
        new_point = transform.TransformPoint(seeds[:,i])
        new_centers[:, i] = new_point
    return new_centers


def apply_transformation(points, trans):
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    new_points = trans @ points_hom.T
    return new_points.T[:,:3]


def apply_transformation_on_seeds(transform, seeds, type='sitk'):
    """
    apply transformation on entire length seeds
    :param transform: simpleitk transformation object
    :param seeds: (3,3,N) array - (start, middle,end) tips of (x,y,z) coordinated of N seeds
    :return: (3,3,N) array. seeds transformed
    """
    new_seeds = np.zeros((3, 3, seeds.shape[-1]))
    for i in range(seeds.shape[-1]):
        new_seeds[0,:,i] = transform.TransformPoint(seeds[0,:,i]) if type == 'sitk' else\
            apply_transformation(seeds[0,:,i][None,:],transform)
        print("s ", seeds[0,:,i], new_seeds[0,:,i] - seeds[0,:,i])
        new_seeds[2,:,i] = transform.TransformPoint(seeds[2,:,i]) if type == 'sitk' else\
            apply_transformation(seeds[2,:,i][None,:],transform)
        new_seeds[1,:,i]= (new_seeds[0,:,i] + new_seeds[2,:,i])/2
    # for i in range(150):
    #     for j in range(150):
    #         for k in range(550,800):
    #             new_p = np.array(transform.TransformPoint([i,j,k]))
    #             print("manual ", new_p - [i,j,k])
    return new_seeds


def apply_scipy_transformation_on_seeds(transform, seeds):
    s = transform.apply(seeds[0, :].T).T
    e = transform.apply(seeds[2, :].T).T
    m = (s + e) / 2
    return np.vstack((s[None, ...], m[None, ...], e[None, ...]))

def apply_probreg_transformation_on_seeds(trans, seeds):
    s = trans.transform(seeds[0, :].T).T
    e = trans.transform(seeds[2, :].T).T
    m = (s + e) / 2
    return np.vstack((s[None, ...], m[None, ...], e[None, ...]))


def apply_sitk_transformation_on_struct(tfm, struct):
    if struct is not None:
        for i in range(struct.shape[0]):
            new_p = tfm.TransformPoint(struct[i, :])
            struct[i, :] = np.array(new_p)
    return struct


def calc_rmse(pt1, pt2, pct):
    step = int(1/pct)
    pt1, pt2 = pt1[:, ::step], pt2[:,::step]
    idx1, idx2 = assignment(pt1[None], pt2[None])

    pt1, pt2 = pt1[:,idx1], pt2[:, idx2]
    rmse = np.sqrt(np.mean(((pt1 - pt2)**2),axis=1))
    dist = np.mean(np.sqrt(np.sum((pt1 - pt2)**2, axis=0)))
    err = np.abs(np.mean(np.mean((pt1 - pt2),axis=1)/ dist * rmse))
    print("rmse ", rmse, "dist ", dist, "err ", err)
    return err if err is not None else 0


def calculate_distances(case, seeds1, seeds2, reg_error, assign_list=None, save=True):
    """
    calculate distance using Munkres algorithm.
    :param case: study id
    :param seeds1: (3,3,N) array of first seeds. the first axis represent 3 tips (start, middle,end). the second axis represent
            3 coordinates (x,y,z)
    :param seeds2: (3,3,N) array of second seeds.
    :param reg_error: registration error
    :param assign_list: list of indices to match seeds2 to seeds1 elements. if None, auto assignment is done
     using Munkres algorithm
    :return: seeds1, seeds2, dist ((N,) array), errors ((N,) array)
    """
    if assign_list is None:
        seed1_idx, seed2_idx = assignment(seeds1, seeds2)
    else:
        seed1_idx, seed2_idx = np.arange(seeds1.shape[1]), assign_list
    print(seed1_idx, seed2_idx)
    seeds1_assigned = seeds1[..., seed1_idx]
    seeds2_assigned = seeds2[..., seed2_idx]
    assignment_dists = np.array([calc_dist(seeds1_assigned[..., i], seeds2_assigned[..., i], calc_max=True)
                                 for i in range(seeds2_assigned.shape[-1])])
    return analyze_distances(case,assignment_dists, seeds1_assigned, seeds2_assigned, reg_error, save=save)


def calc_dist(x, y, calc_max=False):
    """
    calculate the distance between two seeds. the calculation done by the following:
    fir each pair of seeds the euclidean distance is calculated between each corresponding segments. then either
    the maximum or the average (depend on calc_max flag) among al segments distance is taken
    :param x: (3,3) array. first seed. first axis represent three tips. second axis represent coordinated
    :param y: (3,3) array. second seed
    :param meta1: meta data dictionary for first seed
    :param meta2:meta data dictionary for second seed
    :param calc_max: flag to take maximum. if False , average is taken
    :return: distance between seeds
    """

    dist = np.min(np.array([np.sqrt(np.sum((x[i,:][None,:] - y)**2, axis=1)) for i in range(x.shape[0])]), axis=1)
    return np.max(dist) if calc_max else np.mean(dist)


def assignment(seeds1, seeds2):
    """
    create assignment lists using Munkres algorithm
    :param seeds1: (3,3,N) array of first seeds. the first axis represent 3 tips (start, middle,end). the second axis represent
            3 coordinates (x,y,z)
    :param seeds2: (3,3,N) array of second seeds.
    :param meta1: meta data dictionary for first seed
    :param meta2:meta data dictionary for second seed
    :return:
    """
    C = np.zeros((seeds1.shape[-1], seeds2.shape[-1]))
    for i in range(seeds1.shape[-1]):
        C[i, :] = np.array([calc_dist(seeds1[..., i], seeds2[..., k]) for k in range(seeds2.shape[-1])])
    row_ind, col_ind = linear_sum_assignment(C)
    return row_ind, col_ind


def analyze_distances(case, dists, seeds1, seeds2, error, save=True):
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
    error += BASE_ERROR
    plot_pairs(seeds1, seeds2, case, save)
    plot_individual_moves(case, dists, error, save)
    # display_dists(seeds1, seeds2, "%s movements and matching" % case, "%s_matchs.jpg" % case)
    return


def calc_cm_shift(seeds1, seeds2):
    """
    calcualte shift between centers of seeds1 and seeds2
    :param seeds1: 3XN array of first seeds
    :param seeds2: 3XN array of second seeds
    :return: center1 (average coordinates of seeds1), center2 (average coordinates of seeds2), seeds2_shifted (seeds2 after
    subtracting diff_cm), diff_cm (difference vector between center1 and center2)
    """
    center1 = np.average(seeds1, axis=1)[..., None]
    center2 = np.average(seeds2, axis=1)[..., None]
    diff_cm = center2 - center1
    seeds2_shifted = seeds2 - diff_cm
    return center1, center2, seeds2_shifted, diff_cm


def create_data(fixed_dir, moving_dir):
    """
    get dicoms path and seeds information
    :param fixed_dir: path to directory consists dicom files of first seed
    :param moving_dir: path to directory consists dicom files of second seed
    :return: seeds1 (3,N), orientation1 (3,N), seeds2 (3,N), orientation2 (3,N), dicom_dict1 (dictionary with paths),
    dicom_dict2 (dictionary with paths)
    """
    fixed_dir = unzip_dir(fixed_dir)
    moving_dir = unzip_dir(moving_dir)

    dicom_dict_1 = get_all_dicom_files(fixed_dir, {})
    dicom_dict_2 = get_all_dicom_files(moving_dir, {})
    seeds1, orientation1 = get_seeds_dcm(dicom_dict_1["RTPLAN"])
    seeds2, orientation2 = get_seeds_dcm(dicom_dict_2["RTPLAN"])

    return seeds1, orientation1, seeds2, orientation2, dicom_dict_1, dicom_dict_2



