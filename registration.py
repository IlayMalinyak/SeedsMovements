# __author:IlayK
# data:02/05/2022
import SimpleITK as sitk
import sys
import os
import numpy as np
import registration_gui as rgui
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import open3d as o3d
from scipy.spatial import ConvexHull
from analyze import apply_transformation
import tensorflow as tf
import DeepReg.deepreg.model.loss.image as image_loss
import DeepReg.deepreg.model.layer_util as layer_util




class ContourRegistration:

    def __init__(self, out_path, experiment_name):
        self.out_path = out_path
        self.experiment_name = experiment_name
        self.correspondence_set = None
        # self.best_correspondence_set = None
        self.registration = None
        # self.best_registration = None
        self.rmse = None
        # self.best_rmse = None

    @staticmethod
    def preprocess_point_cloud(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def prepare_dataset(self, p1,p2,voxel_size):
        # print(":: Load two point clouds and disturb initial pose.")

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(p1)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(p2)
        # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        # source.transform(trans_init)

        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def execute_fast_global_registration(self, source_down, target_down, source_fpfh,
                                         target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        # print(":: Apply fast global registration with distance threshold %.3f" \
        #         % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result

    def fast_global_registration(self, p1,p2,voxel_size):
        source,target,source_down, target_down,source_fpfh, target_fpfh = self.prepare_dataset(p1,p2,voxel_size)
        result = self.execute_fast_global_registration(source, target, source_fpfh, target_fpfh, voxel_size)
        # print("fast global - ", result)
        # print(result.transformation)
        return result

    def icp(self, p1, p2, dist_thresh=None):
        """
        run icp (iterative closest point) with fast global registration as initial transform
        :param p1: np array (N,3)
        :param p2: np array (N,3)
        :param dist_thresh: distance threshold, float
        :return: reg_p2p - open3d registration result object, fixed_surface - np array, moving surface - np array
        """
        fixed_surface, moving_surface, dist = self.prepare_points(p1, p2)
        dist_thresh = dist if dist_thresh is None else dist_thresh
        # print("distant between centers ", dist_thresh)
        # print("**** initial fast global registration *****")
        fgr = self.fast_global_registration(fixed_surface, moving_surface,dist_thresh)

        wrapped_fixed_surface = apply_transformation(fixed_surface, fgr.transformation)

        # icp(fixed_ctr.T, moving_ctr.T)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(fixed_surface[:, 0], fixed_surface[:, 1], fixed_surface[:, 2], alpha=.6, color='b',
        #            label='original source')
        # ax.scatter(wrapped_fixed_surface[:, 0], wrapped_fixed_surface[:, 1], wrapped_fixed_surface[:, 2], alpha=.6,
        #            color='g', label='wrapped source')
        # # ax.scatter(moving_ctr[:,0], moving_ctr[:,1], moving_ctr[:,2], alpha=.5, color='r')
        # ax.scatter(moving_surface[:, 0], moving_surface[:, 1], moving_surface[:, 2], alpha=.6, color='orange',
        #            label='target')
        # plt.title('fgr')
        # plt.legend()
        # plt.show()
        # self.correspondence_set = np.array(fgr.correspondence_set)
        # print("correspondence set " ,self.correspondence_set)
        trans_init = fgr.transformation
        pcd1 = o3d.t.geometry.PointCloud()
        pcd1.point['positions'] = o3d.core.Tensor(fixed_surface)
        pcd2 = o3d.t.geometry.PointCloud()
        pcd2.point['positions'] = o3d.core.Tensor(moving_surface)
        # print("***** icp ******")
        thresh_icp = max(dist_thresh/10, 0.5)
        reg_p2p = o3d.t.pipelines.registration.icp(
            pcd1, pcd2, thresh_icp, trans_init,o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000), -1, True)
        self.correspondence_set = (reg_p2p.correspondences_).numpy()
        try:
            self.registration = reg_p2p
            self.rmse = self.registration.loss_log['inlier_rmse'].numpy()
            return reg_p2p, fixed_surface, moving_surface
        except Exception:
            return None, fixed_surface, moving_surface

    # def update_params(self):
    #     self.best_correspondence_set = self.correspondence_set
    #     self.best_registration = self.registration
    #     self.best_rmse = self.best_registration.loss_log['onlier_rmse'].numpy()
    #     # self.best_rmse = self.rmse

    def get_correspondence(self):
        return self.correspondence_set

    def get_params(self):
        return self.rmse, self.registration


    @staticmethod
    def prepare_points(fixed_points_raw, moving_points_raw):
        """
        prepare points for open3d registrations. point cloud is filtered to only surface points (using convex envelope)
        :param fixed_points_raw: np array (N,3)
        :param moving_points_raw: np array (N,3)
        :return: fixed_surface = np array (N,3), moving_surface - np.array (N,3), center_dist - distance between centers
        """
        fixed_hull = ConvexHull(fixed_points_raw).simplices
        moving_hull = ConvexHull(moving_points_raw).simplices
        fixed_surface = []
        moving_surface = []
        for p in fixed_hull:
            fixed_surface.append([fixed_points_raw[p[0], 0], fixed_points_raw[p[1], 1], fixed_points_raw[p[2], 2]])
        for p in moving_hull:
            moving_surface.append([moving_points_raw[p[0], 0], moving_points_raw[p[1], 1], moving_points_raw[p[2], 2]])
        fixed_surface = np.array(fixed_surface)
        moving_surface = np.array(moving_surface)
        center_fixed = np.mean(fixed_surface, axis=0)
        center_moving = np.mean(moving_surface, axis=0)
        center_dist = np.sqrt(np.sum((center_fixed - center_moving) ** 2))
        return fixed_surface, moving_surface, center_dist

    def fgr(self, fixed_points_raw, moving_points_raw):
        fixed_surface, moving_surface, dist_thresh = self.prepare_points(fixed_points_raw, moving_points_raw)
        fgr = self.fast_global_registration(fixed_surface, moving_surface, dist_thresh)
        return fgr.transformation


def read_image(dir):
    """
    read image to ITK object
    :param dir: directory to read from
    :return: sitk.Image object
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dir)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()
    image = sitk.Cast(image, sitk.sitkFloat32)
    return image


def register_sitk(fixed_path, moving_path, meta, out_path, type="Bspline", params={}, domain_start=None, domain_end=None):
    """
    register two dicoms
    :param fixed_path: path to fixed dicom
    :param moving_path: path to moving dicom
    :param out_path: path to save transformation file
    :param type: type of registration (default is bspline)
    :param params: parameters for transformation. dictionary
    :param domain_end: coordinates of end of registration domain. array
    :param domain_start: coordinates of start of registration domain. array
    :return: parameter dict
    """
    # fixed = sitk.Cast(sitk.GetImageFromArray(fixed_path), sitk.sitkFloat32)
    # moving = sitk.Cast(sitk.GetImageFromArray(moving_path), sitk.sitkFloat32)
    try:
        print("reading from path")
        fixed = read_image(fixed_path)
        moving = read_image(moving_path)
    except:
        print("using sitk images")
        fixed = fixed_path
        moving = moving_path
    # print(fixed)
    # print("shapes itk ", fixed.GetHeight(), fixed.GetWidth(), fixed.GetDepth())

    print("----%s----" % type)
    print("origin before registration ", fixed.GetOrigin(), moving.GetOrigin())
    if type == "Bspline":
        outTx = bspline_registration(fixed, moving, out_path, params, domain_start, domain_end)
    elif type == "Affine":
        outTx = affine_registration(fixed, moving, out_path, params)
    else:
        raise ValueError("transformation of type: %s does not exist" % type)
    warped = warp_image_sitk(fixed, moving, outTx)

    outTx_inv, disp_img = get_inverse_transform(outTx, type)
    # warped_inv = warp_image_sitk(fixed, moving, outTx_inv)

    return fixed, warped, outTx, outTx_inv, disp_img


def warp_image_sitk(fixed, moving, outTx, mask=False, default_val=-1024):
    """
    warped image according to transformation
    :param fixed: fixed image (sitk object)
    :param moving: moving image (sitk object)
    :param outTx: transformation
    :return: warped image (sitk object)
    """

    print("warping ", fixed.GetOrigin(), moving.GetOrigin())
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    if mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_val)
    resampler.SetTransform(outTx)
    warped_img = resampler.Execute(moving)
    return warped_img


def get_inverse_transform(outx, type):
    if type == "Bspline":
        return inverse_bspline(outx)
    return outx.GetInverse(), None


def inverse_bspline(outX):
    print("inverting")
    df_inverter = sitk.InvertDisplacementFieldImageFilter()
    df_inverter.SetMaximumNumberOfIterations(100)
    grid_spacing = [5,5,5]
    # df_inverter.SetEnforceBoundaryCondition(True)
    outX = sitk.BSplineTransform(sitk.CompositeTransform(outX).GetNthTransform(0))
    # physical_size = outX.GetTransformDomainPhysicalDimensions()
    physical_size = outX.GetTransformDomainPhysicalDimensions()
    grid_size = [
        int(phys_sz / spc + 1)
        for phys_sz, spc in zip(physical_size, grid_spacing)
    ]
    displacement_field_image = sitk.TransformToDisplacementField(
        outX,
        outputPixelType=sitk.sitkVectorFloat64,
        size=grid_size,
        outputOrigin=outX.GetTransformDomainOrigin(),
        outputSpacing=grid_spacing,
        outputDirection=outX.GetTransformDomainDirection(),
    )
    bspline_inverse_displacement = sitk.DisplacementFieldTransform(df_inverter.Execute(displacement_field_image))
    return bspline_inverse_displacement, displacement_field_image



def bspline_registration(fixed_image, moving_image, out_path, params, domain_start, domain_end):
    """
    bspline deformable registration
    :param fixed_image: sitk.Image fixe image
    :param moving_image: sitk.Image moving image
    :param out_path: path to save transformation
    :param params: parameters dictionary
    :return: transformation
    """

    registration_method = sitk.ImageRegistrationMethod()

    origin = fixed_image.GetOrigin()
    if domain_start is not None:
        start_pixel = (np.array(domain_start - np.array(origin)) / np.array(fixed_image.GetSpacing())).astype(np.int16)
        end_pixel = (np.array(domain_end - np.array(origin)) / np.array(fixed_image.GetSpacing())).astype(np.int16)
        reduced = fixed_image[start_pixel[0]:end_pixel[0], start_pixel[1]:end_pixel[1], start_pixel[2]:end_pixel[2]]
        print("reduced size = ", reduced.GetSize())
    else:
        reduced = fixed_image

    # Determine the number of BSpline control points using the physical
    # spacing we want for the finest resolution control grid.
    grid_physical_spacing = [40, 40, 40]
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
    # The starting mesh size will be 1/4 of the original, it will be refined by
    # the multi-resolution framework.
    mesh_size = [int(sz/4 + 0.5) for sz in mesh_size]

    initial_transform = sitk.BSplineTransformInitializer(image1=reduced,
                                                         transformDomainMeshSize=mesh_size, order=3)
    # initial_transform.SetTransformDomainOrigin([-273, -208, 660])
    print("transform origin ", initial_transform.GetTransformDomainOrigin())
    # Instead of the standard SetInitialTransform we use the BSpline specific method which also
    # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with
    # the given mesh_size at the highest pyramid level then we double it in the next lower level and
    # in the full resolution image we use a mesh that is four times the original size.
    registration_method.SetInitialTransformAsBSpline(initial_transform,
                                                     inPlace=False,
                                                     scaleFactors=[1,2,4])

    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method = set_metric_and_optimizer(registration_method, params)
    registration_method.SetInterpolator(sitk.sitkBSpline)

    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    plot_metric(registration_method)

    final_transformation = registration_method.Execute(fixed_image, moving_image)
    exceute_metric_plot(registration_method, out_path, "Bspline")
    print('\nOptimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return final_transformation


def affine_registration(fixed_image, moving_image, out_path, params):
    """
    3D rigid registration
    :param fixed_image: sitk.Image fixe image
    :param moving_image: sitk.Image moving image
    :param out_path: path to save transformation
    :param params: parameters dictionary
    :return: transformation
    """


    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method = set_metric_and_optimizer(registration_method, params)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetInitialTransform(initial_transform)

    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Optimizer settings.
    # registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    plot_metric(registration_method)

    final_transform = registration_method.Execute(fixed_image, moving_image)
    exceute_metric_plot(registration_method, out_path, "Rigid")


    # Always check the reason optimization terminated.
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return final_transform


def set_metric_and_optimizer(registration_method, params):
    """
    create metric and optimizer using params dictionary
    :param registration_method: sitk.RegistrationMethod object
    :param params: parameters dictionary. should have the keys -
    :return:
    """
    if params['metric'] == 'Mean Squares':
        print("setting metric as mean squares")
        registration_method.SetMetricAsMeanSquares()
    elif params['metric'] == 'Mutual information':
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingPercentage(float(params['sampling_percentage']))
    if params['optimizer'] == 'LBFGS2':
        registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=float(params['accuracy']),
                                                 numberOfIterations=int(params['iterations']),
                                                 deltaConvergenceTolerance=float(params['convergence_val']))
    elif params['optimizer'] == 'Gradient Decent':
        registration_method.SetOptimizerAsGradientDescent(learningRate=float(params['learning_rate']), numberOfIterations=int(params['iterations']),
                                                      convergenceMinimumValue=float(params['convergence_val']),
                                                          convergenceWindowSize=10)
    return registration_method


def exceute_metric_plot(registration_method, out_path, method_name):
    """
    create metric plot
    :param registration_method: sitk.RegistrationMethod object
    :param out_path: path for saving
    :param method_name: registration type
    :return: None
    """
    handles = []
    patch = mpatches.Patch(color='b', label='Multi Resolution Event')
    # handles is a list, so append manual patch
    handles.append(patch)
    # plot the legend
    plt.legend(handles=handles, loc='upper right')
    if method_name == "demons (non rigid)":
        plt.title("%s metric final results %.2f" % (method_name, registration_method.GetMetric()))
    else:
        plt.title("%s metric final results %.2f" % (method_name, registration_method.GetMetricValue()))
    plt.xlabel("iteration number")
    plt.ylabel("mean squares value (mm)")
    plt.savefig(out_path)


def command_iteration(method):
    """
    command function to run each iteration
    :param method: sitk registration method object
    """
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")


def plot_metric(registration_method):
    """
    plot metric
    :param registration_method: sitk registration method object
    """
    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rgui.plot_values(registration_method))

@tf.function
def train_step_CT(grid, weights, optimizer, mov, fix, image_loss_name):
    """
    Train step function for backprop using gradient tape.
    GradientTape is a tensorflow API which automatically
    differentiates and facilitates the implementation of machine
    learning algorithms: https://www.tensorflow.org/guide/autodiff.

    :param grid: reference grid return from util.get_reference_grid
    :param weights: trainable affine parameters [1, 4, 3]
    :param optimizer: tf.optimizers: choice of optimizer
    :param mov: moving image, tensor shape [1, m_dim1, m_dim2, m_dim3]
    :param fix: fixed image, tensor shape[1, f_dim1, f_dim2, f_dim3]
    :return loss: image dissimilarity to minimise
    """

    # We initialise an instance of gradient tape to track operations
    with tf.GradientTape() as tape:
        pred = layer_util.resample(vol=mov, loc=layer_util.warp_grid(grid, weights))
        # Calculate the loss function between the fixed image
        # and the moving image
        loss = image_loss.dissimilarity_fn(
            y_true=fix, y_pred=pred, name=image_loss_name
        )
    gradients = tape.gradient(loss, [weights])
    # Applying the gradients
    optimizer.apply_gradients(zip(gradients, [weights]))
    return loss


# def register_self_affine(fixed_path, moving_path, learning_rate, total_iter, image_loss_name):
#     # normalisation to [0,1]
#     fixed_image = read_dicom(fixed_path)
#     moving_image = read_dicom(moving_path)
#     fixed_image = tf.cast(tf.expand_dims(fixed_image, axis=0), dtype=tf.int16)
#     fixed_image = (fixed_image - tf.reduce_min(fixed_image)) / (
#             tf.reduce_max(fixed_image) - tf.reduce_min(fixed_image).numpy()
#     )
#     moving_image = tf.cast(tf.expand_dims(moving_image, axis=0), dtype=tf.int16)
#     moving_image = (moving_image - tf.reduce_min(moving_image)) / (
#             tf.reduce_max(moving_image) - tf.reduce_min(moving_image).numpy()
#     )
#
#     # generate a radomly-affine-transformed moving image using DeepReg utils
#     fixed_image_size = fixed_image.shape
#     # The following function generates a random transform.
#     transform_random = layer_util.random_transform_generator(batch_size=1, scale=0.2)
#
#     # We create a reference grid of image size
#     grid_ref = layer_util.get_reference_grid(grid_size=fixed_image_size[1:4])
#
#     # We warp our reference grid by our random transform
#     grid_random = layer_util.warp_grid(grid_ref, transform_random)
#
#     # We create an affine transformation as a trainable weight layer
#     var_affine = tf.Variable(
#         initial_value=[
#             [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
#         ],
#         trainable=True,
#     )
#
#     # We perform an optimisation by backpropagating the loss through to our
#     # trainable weight layer.
#     optimiser = tf.optimizers.Adam(learning_rate)
#     loss_arr = []
#     # Perform an optimisation for total_iter number of steps.
#     for step in range(total_iter):
#         loss_opt = train_step_CT(grid_ref, var_affine, optimiser, moving_image, fixed_image, image_loss_name)
#         loss_arr.append(loss_opt)
#         # if (step % 50) == 0:  # print info
#         tf.print("Step", step, image_loss_name, loss_opt)
#
#     warped_moving = warp_image_sitk(moving_image, grid_ref, var_affine)
#     plt.plot(len(loss_arr), loss_arr)
#     plt.title("self affine")
#     plt.savefig(f"{save_path}/self_affine.png")
#     plt.show()
#     return fixed_image.numpy().squeeze(axis=0), moving_image.numpy().squeeze(axis=0),\
#            warped_moving.numpy().squeeze(axis=0), var_affine.numpy()
#
#
# def warp_image_tf(moving_image, grid_ref, var_affine ):
#     grid_opt = layer_util.warp_grid(grid_ref, var_affine)
#     warped_moving_image = layer_util.resample(vol=moving_image, loc=grid_opt)
#     return warped_moving_image
#
