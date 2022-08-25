# __author:IlayK
# data:02/05/2022
import os
import zipfile
import pydicom
import operator
import numpy as np
from glob import glob
import shutil
import pandas as pd
import registration
from SimpleITK import Euler3DTransform, GetArrayFromImage, GetImageFromArray
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.spatial.transform import Rotation
import json
from matplotlib import pyplot as plt



SEED_LENGTH_MM = 10


def get_meta_data(path, cloud=False):
    """
    read chosen dicom tags and save them in a dictionary
    :param path: path to dcm file
    :return: dictionary with keys: IPP, IOP, slope, intercept, numSlices , pixelSpacing, sliceThickness
    """
    ordered_slices = slice_order(path, cloud)
    ds = pydicom.dcmread("%s/%s.dcm" % (path, ordered_slices[-1][0]))
    meta = {}
    meta['ID'] = ds[0x00100020]
    meta['name'] = ds[0x00100010]
    meta['IPP'] = ds[0x00200032]
    meta['IOP'] = ds[0x00200037]
    meta['slope'] = ds.RescaleSlope
    meta['intercept'] = ds.RescaleIntercept
    meta['numSlices'] = len(ordered_slices)
    try:
        meta['pixelSpacing'] = ds[0x00132050].value[0].PixelSpacing
        meta['sliceThickness'] = ds[0x00132050].value[0].SliceThickness
        meta['sliceSpacing'] = ds[0x00132050].value[0].SpacingBetweenSlices
    except (AttributeError, KeyError):
        try:
            meta['pixelSpacing'] = ds.PixelSpacing
            meta['sliceThickness'] = ds.SliceThickness
            meta['sliceSpacing'] = ds.SpacingBetweenSlices

        except (AttributeError, KeyError) as e:
            print(e)
    return meta


def slice_order(path, cloud=False):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    slices = []
    names = []
    for s in os.listdir(path):
        try:
            f = pydicom.read_file(path + '/' + s)
            f.pixel_array  # to ensure not to read contour file
            slices.append(f)
            names.append(s[:-4])
        except:
            continue

    slice_dict = {names[i]: slices[i].ImagePositionPatient[-1] for i in range(len(slices))} if cloud else {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
    return ordered_slices


def get_one_dcm(path, index=0):
    """
    load one slice of dcm
    :param path: path to the dcm file
    :param index: index of slice to load
    :return: data structure of dicom, number of slices in the dcm file
    """
    f = os.listdir(path)
    ds = pydicom.dcmread(path + '/' + f[index], force=True)
    return ds, len(f) - 1


def is_dicom_dir(dir):
    """
    check if directory consists dicom files directly
    :param dir: directory to check
    :return: True if it is dicom dir, False otherwise
    """
    if zipfile.is_zipfile(dir):
        return False
    for f in os.listdir(dir):
        if f.endswith("dcm"):
            return True
    return False


def get_modality(dir):
    """
    get the modality of the dicom (CT,RTSTRUCT,RTPLAN)
    :param dir: directory of dicoms
    :return: modality
    """
    ds ,_ = get_one_dcm(dir, 0)
    return ds[0x00080060].value


def get_all_dicom_files(path, dicom_dict):
    """
    get all dicom directories inside root directory
    :param path: path to root directory
    :param dicom_dict: dictionary of modalities as keys as paths as values.
    :return: dicom_dict updated
    """
    if is_dicom_dir(path):
        modality = get_modality(path)
        if modality == "CT":
            dicom_dict["CT"] = path
            dicom_dict['meta'] = get_meta_data(path)
        elif modality == "RTPLAN" or modality == 'RAW':
            dicom_dict["RTPLAN"] = path
        elif modality == "RTSTRUCT":
            dicom_dict["RTSTRUCT"] = path
    else:
        for file in os.listdir(path):
            if zipfile.is_zipfile(file):
                with zipfile.ZipFile(file, "r") as zip_ref:
                    zip_ref.extractall(path)
            new_path = path + "/" + file
            if os.path.isdir(new_path):
                get_all_dicom_files(new_path, dicom_dict)
    return dicom_dict


def get_seeds_dcm(path):
    """
    get dcm coordinated of seeds
    :param path: path to folder with dcm file of seeds information
    :return: (3,N) array of seeds positions, (3,N) array of seeds orientation. each column represent x,y,z coordinated
    of seed's center / seed's orientation
    """
    dcm , _= get_one_dcm(path)
    # print(dcm)
    seed_sequence = []
    try:
        seed_sequence = dcm[0x00132050].value[-1].ApplicationSetupSequence
    except (AttributeError, KeyError):
        try:
            seed_sequence = dcm.ApplicationSetupSequence
        except (AttributeError, KeyError) as e:
            print(e)
    seeds_position = []
    seeds_orientation = []
    for i in range(len(seed_sequence)):
        seeds_position.append(seed_sequence[i].ChannelSequence[0].BrachyControlPointSequence[0].ControlPoint3DPosition)
        seeds_orientation.append(
            (seed_sequence[i].ChannelSequence[0].BrachyControlPointSequence[0].ControlPointOrientation))
    seeds_position = np.array(seeds_position).astype(np.float64).T
    seeds_orientation = np.array(seeds_orientation).astype(np.float64).T
    # seeds_orientation[[0,1]] = seeds_orientation[[1,0]]
    # print("orientation", seeds_orientation)
    return seeds_position, seeds_orientation


def get_seeds_tips(seed_position, seeds_orientation):
    """
    get seeds tips (ends) coordinates
    :param seed_position: (3,N) nd-array of seeds center coordinated
    :param seeds_orientation: (3,N) nd-array of seeds orientations
    :param x_spacing: column spacing
    :param y_spacing: row spacing
    :param thickness: slice thickness
    :return: (3,3,N) array of seeds tips. array[0,:,:] gives the x,y,z coordinated of the first tip (start),
    array[1,:,:] gives the x,y,z coordinates of the center and array[2,:,:] gives the x,y,z coordinated
    of the end of the seeds
    """
    seeds_tips = np.zeros((3,3,seed_position.shape[1]))
    seeds_tips[2, :, :] = seed_position + (SEED_LENGTH_MM/2)*seeds_orientation
    seeds_tips[0, :, :] = seed_position - (SEED_LENGTH_MM/2)*seeds_orientation
    seeds_tips[1, :, :] = seed_position
    # seeds_tips[:, 2, :] *= -1
    return seeds_tips

def down_sample_array(arr):
    """
    uniform down-sampling coordinate
    :param arr: NX3 numpy array
    :return: MX3 numpy array where M < 5000
    """
    number_digist = int(np.floor(np.log10(arr.shape[0])))
    step = 2*10**(number_digist - 3) if number_digist > 2 else 1
    return arr[::step]

def broadcast_points(object, num_points):
    """
    broadcast number of points in an object from 3 to N = round(10 / ((x_spacing + y_spacing) / 2). this is the maximal
    number of points that a 10 mm length seeds can contain.
    :param object: usually seeds but can be any general object with shape 3x3xN. the first axis represent number of
    points in the object (two tips and center), second axis represent x,y,z and the third axis represent the number of
    objects
    :param x_spacing: pixel spacing x direction
    :param y_spacing: pixel spacing y direction
    :return: array of MX3XN. M is the new number of points
    """
    num_objects = object.shape[2]
    x = np.linspace(object[0, 0, ...], object[2, 0, ...], num_points)
    y = np.linspace(object[0, 1, ...], object[2, 1, ...], num_points)
    z = np.linspace(object[0, 2, ...], object[2, 2, ...], num_points)
    x = np.ones((num_points,num_objects)) * object[0, 0] if x is None else x
    y = np.ones((num_points,num_objects)) * object[0, 1] if y is None else y
    z = np.ones((num_points,num_objects)) * object[0, 2] if z is None else z
    return np.concatenate((x[:,None,:],y[:,None,:], z[:,None,:]), axis=1)


def read_dicom(path,meta, cloud=False):
    """
    read dicom to numpy array
    :param path: path to dicom directory
    :param cloud: flag for directory that were exported directly from the cloud
    :return: (width, height, depth) dicom array
    """
    images = []
    ordered_slices = slice_order(path, cloud)
    for k,v in ordered_slices:
        # get data from dicom.read_file
        img_arr = pydicom.read_file(path + '/' + k + '.dcm').pixel_array
        img_arr = img_arr*meta['slope'] + meta['intercept']
        images.append(img_arr.astype(np.int16))
    return np.swapaxes(np.swapaxes(np.array(images), 0,2),0,1)


def read_structure(dir, seeds=False):
    """
    read contours
    :param dir: directory to folder with dcm file of contours
    :return: list of tuples. each tuple contains (name, arr). name is contour name (string), arr is contour data
    ((3,N) nd-array) of the perimeter voxels of the contour
    """
    for f in glob("%s/*.dcm" % dir):
        pass
        # filename = f.split("/")[-1]
        ds = pydicom.dcmread(f)
        # print(ds)
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        ctrs = ds.ROIContourSequence
        meta = ds.StructureSetROISequence
        # print(ds)
        list_ctrs = []
        list_seeds = []
        list_orientations = []
        for i in range(len(ctrs)):
            data = ctrs[i].ContourSequence
            name = meta[i].ROIName
            vol = 0
            # print(name)
            arr = np.zeros((3,0))
            for j in range(len(data)):
                contour = data[j].ContourData
                np_contour = np.zeros((3, len(contour) // 3))
                for k in range(0, len(contour), 3):
                    np_contour[:, k // 3] = contour[k], contour[k + 1], contour[k + 2]
                arr = np.hstack((arr, np_contour))
            if seeds:
                cov = np.cov(arr)
                w, v = np.linalg.eig(cov)
                dxyz = v[:, np.argmax(w)]
                center = np.mean(arr, axis=1)
                list_seeds.append(center)
                list_orientations.append(dxyz)
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # ax.scatter(arr[0], arr[1], arr[2])
                # ax.scatter(center[0], center[1], center[2])
                # ax.quiver(center[0], center[1], center[2], dxyz[0],dxyz[1], dxyz[2], length=5)
                # plt.show()
            else:
                list_ctrs.append((name, arr))
        if seeds:
            return np.array(list_seeds).T, np.array(list_orientations).T
        return list_ctrs




def pixel_to_mm_transformation_mat(meta, z_factor=-1):
    """
    transformation matrix from pixel coordinates to dcm (mm) coordinates
    :param c_spacing: column spacing
    :param r_spacing: row spacing
    :param thickness: slice thickness
    :param ipp: ipp (image position patient)
    :param iop: iop (image orientation patient)
    :return: 4X4 transformation matrix for homogenous coordinates vectors
    """
    c_spacing,r_spacing,thickness = meta['pixelSpacing'][0], meta['pixelSpacing'][1],\
                                              meta['sliceThickness']
    s_spacing = abs(meta['sliceSpacing']) if "sliceSpacing" in meta.keys() else abs(thickness)
    ipp, iop = meta['IPP'].value, meta['IOP'].value
    IPP = ipp
    IOP = iop
    r_vec, c_vec = np.array(IOP[:3]), np.array(IOP[3:6])
    m1 = (r_vec * c_spacing)[:, None]
    m2 = (c_vec * r_spacing)[:, None]
    unit_z = np.cross(r_vec, c_vec)*z_factor
    m3 = (float(s_spacing) * unit_z)[:, None]
    m4 = np.array(IPP)[:, None]
    M = np.vstack((np.hstack((m1, m2, m3, m4)), np.array([0, 0, 0, 1])))
    return M

def convert_dcm_to_pixel(dcm_coords, meta):
    """
    convert dcm coordinates to pixels
    :param meta: meta data contains ipp,iop,pixel spacing and slice thickness/spacing
    :param dcm_coords: nd array of dcm coordinates
    :return: nd array with shape equal to dcm_coords.shape of pixel coordinated
    """
    M = pixel_to_mm_transformation_mat(meta)
    dicom_cord = np.vstack((dcm_coords, np.ones(dcm_coords.shape[1])))
    pixel_cord = np.abs(np.linalg.inv(M) @ dicom_cord)
    # pixel_cord[2,:] = num_slices - pixel_cord[2, :]
    return pixel_cord[:3]

def get_seeds_tips_pixels(seeds_tips_dcm, meta):
    s = convert_dcm_to_pixel(seeds_tips_dcm[0], meta)
    e = convert_dcm_to_pixel(seeds_tips_dcm[-1], meta)
    m = (s + e) // 2
    seeds_pixels = np.vstack((s[None,...], m[None,...], e[None,...])).astype(np.int16)
    # seeds_pixels[:,2,:] = meta['numSlices'] - seeds_pixels[:,2,:]
    return seeds_pixels
def get_contour_mask(arr, meta, shape, flip_z=True):
    M = np.linalg.inv(pixel_to_mm_transformation_mat(meta))
    arr = np.vstack((arr, np.ones(arr.shape[1])))
    arr = M @ arr
    arr = arr.astype(np.int16)
    if flip_z:
        arr[2,:] = abs(shape[2] - abs(arr[2,:]))
    else:
        arr[2,:] = abs(arr[2,:])
    # arr[2,:] -= shape[2]
    mask = np.zeros(shape)
    mask[arr[1],arr[0], arr[2]] = 1
    return mask


def read_structure_from_csv(path):
    """
    read structure from csv. csv is created by MIM workflow
    :param path: path to csv file
    :return: (3,N) nd array - each row in array is a voxel contained inside the contour
    """
    df = pd.read_csv(path)
    arr = df.to_numpy()
    return arr[:, :3].T

def get_spacing_array(meta):
    try:
        spacing = np.array([meta['pixelSpacing'][0],
                                                      meta['pixelSpacing'][1],meta['sliceSpacing']])
    except Exception as e:
        print("spacing array ", e)
        spacing = np.array([meta['pixelSpacing'][0],
                  meta['pixelSpacing'][1], meta['sliceThickness']])
    return spacing

def get_contour_domain(ctr1, ctr2, meta, flip_z=True):
    min_1 = np.min(ctr1, axis=1)
    max_1 = np.max(ctr1, axis=1)
    min_2 = np.min(ctr2, axis=1)
    max_2 = np.max(ctr2, axis=1)
    min_tot = np.min(np.vstack((min_1, min_2)), axis=0)
    max_tot = np.max(np.vstack((max_1, max_2)), axis=0)
    M = np.linalg.inv(pixel_to_mm_transformation_mat(meta))
    min_tot_px = M @ (np.vstack((min_tot[:,None], np.ones((1,1))))).astype(np.int16)
    max_tot_px = M @ (np.vstack((max_tot[:,None], np.ones((1,1))))).astype(np.int16)
    num_slices = meta['numSlices']
    if flip_z:
        min_tot_px[3,:] = abs(num_slices - abs(min_tot_px[3,:]))
        max_tot_px[3, :] = abs(num_slices - abs(max_tot_px[3, :]))
    else:
        min_tot_px[3,:] = abs(min_tot_px[3,:])
        max_tot_px[3,:] = abs(max_tot_px[3,:])


    # min_tot -= [10, 10, 10]
    # max_tot += [10, 10, 10]
    return np.squeeze(min_tot_px[:3],axis=-1), np.squeeze(max_tot_px[:3],axis=-1)


def get_manual_domain(origin_px, end_px, meta):
    M = pixel_to_mm_transformation_mat(meta)
    origin_hom = np.vstack((origin_px[:,None], np.ones((1, 1))))
    origin_hom[[0,1]] = origin_hom[[1,0]]
    end_hom = np.vstack((end_px[:,None], np.ones((1, 1))))
    end_hom[[0,1]] = end_hom[[1,0]]

    end_hom = np.vstack((end_px[:, None], np.ones((1, 1))))
    end_hom[[0, 1]] = end_hom[[1, 0]]
    origin = M @ origin_hom
    end = M @ end_hom
    return np.squeeze(origin[:3], -1),np.squeeze(end[:3], -1)


def get_all_seeds_pixels(seeds_tips_dcm, meta, shape=None):
    seeds_tips_px = get_seeds_tips_pixels(seeds_tips_dcm, meta)
    seeds_tips_px[:, 2, :] = meta['numSlices'] - seeds_tips_px[:, 2, :]
    seeds_pixels = np.zeros((0, 3)).astype(np.int16)
    for i in range(seeds_tips_px.shape[-1]):
        s = seeds_tips_px[:, :, i]
        num_points = np.max(
            (np.max(s[:, 0]) - np.min(s[:, 0]), np.max(s[:, 1]) - np.min(s[:, 1]), np.max(s[:, 2]) - np.min(s[:, 2])))
        s_broad = np.squeeze(broadcast_points(s[..., None], num_points), axis=-1).astype(np.int16)
        seeds_pixels = np.vstack((seeds_pixels, s_broad))
        # mask_seeds[s_broad[:, 1], s_broad[:,0], s_broad[:,2]] = 1
    if shape is not None:
        mask_seeds = np.zeros(shape)
        mask_seeds[seeds_pixels[:, 1], seeds_pixels[:, 0], seeds_pixels[:, 2]] = 1
        structure = generate_binary_structure(3, 3)
        mask_seeds = binary_dilation(mask_seeds, structure=structure, iterations=3)
        ones = np.where(mask_seeds)
        return np.array(ones)
    return seeds_pixels


def wrap_image_with_matrix(fixed_arr, moving_arr, meta, transformation_mat):
    spacing = get_spacing_array(meta)
    fixed = GetImageFromArray(np.transpose(fixed_arr,(2,0,1)))
    # set_meta_data_to_sitk_image(fixed, fixed_meta, 3)
    moving = GetImageFromArray(np.transpose(moving_arr, (2,0,1)))
    # set_meta_data_to_sitk_image(moving, moving_meta, 3)
    M = pixel_to_mm_transformation_mat(meta)
    t = transformation_mat[:3,3].flatten() / spacing
    # t = [0,0,-50]
    # t = np.hstack((t, np.ones(1)))
    # t = np.linalg.inv(M) @ t
    r = transformation_mat[:3,:3]
    rot = Rotation.from_matrix(r).as_euler('xyz')
    print(rot)
    # r = np.eye(3)
    # p_z_y = np.array([[1,0,0],[0,0,1],[0,1,0]])
    # p_x_y = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    # r_zy = p_z_y @ r @ p_z_y
    # r_xy = p_x_y @ r_zy @ p_x_y
    # # t[0],t[1], t[2] = t[2], t[0], t[1]
    tfm = Euler3DTransform()
    tfm.SetRotation(rot[0],rot[1],rot[2])
    tfm.SetTranslation(t)
    print(tfm)
    warped = registration.warp_image_sitk(fixed, moving, tfm)
    warped_arr = np.transpose(GetArrayFromImage(warped),(1,2,0))
    print("fixed arr ", fixed_arr.shape, " fixed sitk", fixed.GetSize(), " warped ", warped_arr.shape)
    return warped_arr

# def get_structure_from_segmentation(fixed_path, moving_path, name, predict=True):
#     global points_fixed, points_moving
#     pred = panc_seg.Predict()
#     fixed_seg_name = f"fixed_panc_{name}.nii"
#     moving_seg_name = f"moving_panc_{name}.nii"
#     if predict:
#         pred.predict(fixed_path['CT'], fixed_seg_name)
#         pred.predict(moving_path['CT'], moving_seg_name)
#     mask_fixed = np.array(nib.load(fixed_seg_name).dataobj)
#     mask_moving = np.array(nib.load(moving_seg_name).dataobj)
#     points_fixed = np.array(np.where(mask_fixed)).T
#     points_moving = np.array(np.where(mask_moving)).T
#     return points_fixed, points_moving


def unzip_dir(dir):
    """
    unzip directory
    :param dir: directory to unzip
    :return: path to unzipped directory
    """
    if zipfile.is_zipfile(dir):
        dir_path = dir.split('.zip')[0]
        parent = '/'.join(dir_path.split('\\')[:-1])
        child = dir_path.split('\\')[-1]
        if child not in os.listdir(parent):
            os.mkdir(dir_path)
            shutil.unpack_archive(dir, dir_path)
        dir = dir_path
    return dir


def set_meta_data_to_sitk_image(img, meta, dim=4):
    try:
        spacing = abs(meta["pixelSpacing"][0]), abs(meta["pixelSpacing"][1]), abs(meta["sliceSpacing"])
    except Exception as e:
        spacing = abs(meta["pixelSpacing"][0]), abs(meta["pixelSpacing"][1]), abs(meta["sliceThickness"])
    img.SetSpacing(spacing)
    ipp = meta['IPP'].value
    iop = meta['IOP'].value
    if len(iop) == 6:
        iop.extend([0, 0, 1])
    # iop = tuple(iop)
    img.SetOrigin(ipp)
    img.SetDirection(iop)


def log_to_dict(path):
    log_data = open(path, 'r')
    result = {}
    for line in log_data:
        # print(line)
        columns = line.split(', ')[2:]
        for c in columns:
            # print(c.split(': '))
            key = c.split(':')[0]
            if key not in result.keys():
                result[key] = []
            value = c.split(':')[1]
            result[key].append(float(value))
    return result





