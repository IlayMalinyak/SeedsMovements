# __author:IlayK
# data:02/05/2022
import SimpleITK as sitk
import numpy as np
import registration_gui as rgui
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import open3d as o3d
from scipy.spatial import ConvexHull, KDTree
from probreg import cpd, bcpd, callbacks


class Callback:
    """
    class of callback function
    """
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.result = None
        self.rmse = []
        self.i = 0

    def __call__(self, transformation):
        self.result = transformation.transform(self.source)
        print("iteration ", self.i)
        self.i += 1
        # self.rmse.append(calc_rmse(self.result.T, self.target.T, 1))

    def get_rmse(self):
        return self.rmse

class Invdisttree:
    """ inverse-distance-weighted interpolation using KDTree:
invdisttree = Invdisttree( X, z )  -- data points, values
interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
    interpolates z from the 3 points nearest each query point q;
    For example, interpol[ a query point q ]
    finds the 3 data points nearest q, at distances d1 d2 d3
    and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

    q may be one point, or a batch of points.
    eps: approximate nearest, dist <= (1 + eps) * true nearest
    p: use 1 / distance**p
    weights: optional multipliers for 1 / distance**p, of the same shape as q
    stat: accumulate wsum, wn for average weights

How many nearest neighbors should one take ?
a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
    |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
    I find that runtimes don't increase much at all with nnear -- ymmv.

p=1, p=2 ?
    p=2 weights nearer points more, farther points less.
    In 2d, the circles around query points have areas ~ distance**2,
    so p=2 is inverse-area weighting. For example,
        (z1/area1 + z2/area2 + z3/area3)
        / (1/area1 + 1/area2 + 1/area3)
        = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
    Similarly, in 3d, p=3 is inverse-volume weighting.

Scaling:
    if different X coordinates measure different things, Euclidean distance
    can be way off.  For example, if X0 is in the range 0 to 1
    but X1 0 to 1000, the X1 distances will swamp X0;
    rescale the data, i.e. make X0.std() ~= X1.std() .

A nice property of IDW is that it's scale-free around query points:
if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
the IDW average
    (z1/d1 + z2/d2 + z3/d3)
    / (1/d1 + 1/d2 + 1/d3)
is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
is exceedingly sensitive to distance and to h.

    """
# anykernel( dj / av dj ) is also scale-free
# error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]

class ContourRegistration:
    """
    point set registration class. implement fast global registration, ICP, CPD and BCPD registrations
    """

    def __init__(self, out_path, experiment_name):
        self.out_path = out_path
        self.experiment_name = experiment_name
        self.correspondence_set = None
        # self.best_correspondence_set = None
        self.registration = None
        self.trans_init = None
        # self.best_registration = None
        self.callback = None
        self.rmse = None
        # self.best_rmse = None

    def get_callback_obj(self):
        return self.callback

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
        """
        fgr registration. more info can be found on
         http://www.open3d.org/docs/0.8.0/tutorial/Advanced/fast_global_registration.html
        :param p1: 3XN array of points
        :param p2: 3XN array of points
        :param voxel_size: down sample resolution
        :return: fgr results registration (open3d registration result object)
        """
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
        fixed_surface, moving_surface, dist = self.prepare_points(p1, p2, surface=False)
        dist_thresh = dist if dist_thresh is None else dist_thresh
        # print("distant between centers ", dist_thresh)
        # print("**** initial fast global registration *****")
        fgr = self.fast_global_registration(fixed_surface, moving_surface,dist_thresh)

        # wrapped_fixed_surface = apply_transformation(fixed_surface, fgr.transformation)
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
            self.trans_init = fgr
            self.rmse = self.registration.loss_log['inlier_rmse'].numpy()
            return reg_p2p, fgr, fixed_surface, moving_surface
        except Exception:
            return None, None, fixed_surface, moving_surface

    def cpd(self, p1, p2, max_iter=50, type='rigid', callback=[], w=0):
        """
        coherent point drift. for more information, look at:
        "Point Set Registration: Coherent Point Drift, Andriy Myronenko and Xubo Song"
        :param p1: Nx3 numpy array of points
        :param p2: Nx3 numpy array of target points
        :param max_iter: maximum iterations
        :param type: rigid (default), nonrigid
        :param callback: callback function (implement __call__ method) that wil be executed each iteration
        :param w: outliers/noise frequency
        :return: probreg transformation object, transformed p1, p2
        """
        fixed_surface, moving_surface, dist = self.prepare_points(p1, p2, surface=False)
        self.callback = [Callback(fixed_surface, moving_surface)] if len(callback) == 0 else callback
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(fixed_surface)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(moving_surface)
        # if len(callback) == 0:
        #     callback = [callbacks.Open3dVisualizerCallback(pcd, pcd2)]
        tf_param, _, _ = cpd.registration_cpd(fixed_surface, moving_surface, tf_type_name=type, callbacks=self.callback,
                                              maxiter=max_iter,w=w, tol=1e-6)
        fixed_surface = tf_param.transform(fixed_surface)

        # p1 = tf_param.transform(p1)
        return tf_param, fixed_surface, moving_surface

    def bcpd(self, p1, p2, max_iter=50, callback=[], w=0):
        """
        coherent point drift. for more information, look at:
        "A Bayesian Formulation of Coherent Point Drift, Osamu Hirose"
        :param p1: Nx3 numpy array of points
        :param p2: Nx3 numpy array of target points
        :param max_iter: maximum iterations
        :param callback: callback function (implement __call__ method) that wil be executed each iteration
        :param w: outliers/noise frequency
        :return: probreg transformation object, transformed p1, p2
        """
        # fixed_surface, moving_surface, dist = self.prepare_points(p1, p2, surface=False)
        fixed_surface, moving_surface = p1,p2
        self.callback = [Callback(fixed_surface, moving_surface)] if len(callback) == 0 else callback
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(fixed_surface)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(moving_surface)
        # if len(callback) == 0:
        #     callback = [callbacks.Open3dVisualizerCallback(pcd, pcd2)]
        tf_param = bcpd.registration_bcpd(fixed_surface, moving_surface, w=w, maxiter=max_iter, callbacks=self.callback,
                                          )
        fixed_surface = tf_param.transform(fixed_surface)
        return tf_param, fixed_surface, moving_surface


    # def update_params(self):
    #     self.best_correspondence_set = self.correspondence_set
    #     self.best_registration = self.registration
    #     self.best_rmse = self.best_registration.loss_log['onlier_rmse'].numpy()
    #     # self.best_rmse = self.rmse

    def get_correspondence(self):
        return self.correspondence_set

    def get_params(self):
        return self.rmse, self.registration, self.trans_init


    @staticmethod
    def prepare_points(fixed_points_raw, moving_points_raw, surface=True):
        """
        prepare points for open3d registrations. point cloud is filtered to only surface points (using convex envelope)
        :param fixed_points_raw: np array (N,3)
        :param moving_points_raw: np array (N,3)
        :return: fixed_surface = np array (N,3), moving_surface - np.array (N,3), center_dist - distance between centers
        """
        if surface:
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
        else:
            fixed_surface = fixed_points_raw
            moving_surface = moving_points_raw
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


def register_sitk(fixed_path, moving_path, meta, out_path, type="Bspline", params={}, domain_start=None, domain_end=None,
                  exclude_arr=None):
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
    print("origin before crop ", fixed.GetOrigin(), moving.GetOrigin())
    print("size before crop ", fixed.GetSize())
    # f_size = fixed.GetSize()
    # domain_start = domain_start if domain_start is not None else [0,0,0]
    # domain_end = domain_end if domain_end is not None else [f_size[1], f_size[0], f)
    # fixed_reduced = fixed[domain_start[1]:domain_end[1], domain_start[0]:domain_end[0], domain_start[2]:domain_end[2]]
    print("----%s----" % type)
    # print("origin before registration ", fixed_reduced.GetOrigin(), moving.GetOrigin())
    # print("size before registration ", fixed_reduced.GetSize())
    if type == "Bspline":
        outTx = bspline_registration(fixed, moving, out_path, params, domain_start, domain_end, exclude_arr)
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
    """
    get sitk inverse transform
    :param outx: sitk transformation object
    :param type: "Bspline", "Affine"
    :return: inverse transform
    """
    if type == "Bspline":
        return inverse_bspline(outx)
    return outx.GetInverse(), None


def inverse_bspline(outX):
    """
    invert Bspline transform
    :param outX: original transform
    :return: inverse transform
    """
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



def bspline_registration(fixed_image, moving_image, out_path, params, domain_start, domain_end, exclude_arr):
    """
    bspline deformable registration
    :param fixed_image: sitk.Image fixe image
    :param moving_image: sitk.Image moving image
    :param out_path: path to save transformation
    :param params: parameters dictionary
    :return: transformation
    """

    registration_method = sitk.ImageRegistrationMethod()

    # origin = fixed_image.GetOrigin()
    # if domain_start is not None and domain_end is not None:
    #     reduced = fixed_image[domain_start[1]:domain_end[1], domain_start[0]:domain_end[0], domain_start[2]:domain_end[2]]
    #     print("reduced size = ", reduced.GetSize())
    # else:
    #     reduced = fixed_image

    # Determine the number of BSpline control points using the physical
    # spacing we want for the finest resolution control grid.
    if 'bspline_resolution' in params.keys():
        res = int(params['bspline_resolution']*10)
    else:
        res = 40
    grid_physical_spacing = [res, res, res]
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
    # The starting mesh size will be 1/4 of the original, it will be refined by
    # the multi-resolution framework.
    mesh_size = [int(sz/4 + 0.5) for sz in mesh_size]

    initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
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
        # binary_filter = sitk.BinaryThresholdImageFilter()
        # binary_filter.SetUpperThreshold(0.5)
    fixed_mask = sitk.Image(fixed_image.GetSize(), 1)
    fixed_mask.SetSpacing(fixed_image.GetSpacing())
    fixed_mask.SetOrigin(fixed_image.GetOrigin())
    fixed_mask.SetDirection(fixed_image.GetDirection())
    domain_start = [0,0,0] if domain_start is None else domain_start
    domain_end = [s - 1 for s in fixed_image.GetSize()] if domain_end is None else domain_end
    # if domain_start is not None and domain_end is not None:
    fixed_mask[domain_start[1]:domain_end[1], domain_start[0]:domain_end[0], domain_start[2]:domain_end[2]] = 1
    # else:
    #     size = fixed_image.GetSize()
    #     print("size ", size)
    #     fixed_mask[0:size[0], 0:size[1], 0:size[2]] = 1
    ones1 = len(np.where(np.transpose(sitk.GetArrayFromImage(fixed_mask), (1, 2, 0)))[0])
    if exclude_arr is not None:
        print("number of exclude points ", exclude_arr.shape[-1])
        for i in range(exclude_arr.shape[-1]):
            # if (i % 1000) == 0:
                # print(int(exclude_arr[1,i]), int(exclude_arr[0,i]), int(exclude_arr[2,i]))
            # print("excluding: ", fixed_mask[int(exclude_arr[1,i]), int(exclude_arr[0,i]), int(exclude_arr[2,i])])
            fixed_mask[int(exclude_arr[1,i]), int(exclude_arr[0,i]), int(exclude_arr[2,i])] = 0
    print("reduced on whites ", ones1 - len(np.where(np.transpose(sitk.GetArrayFromImage(fixed_mask), (1, 2, 0)))[0]))
    registration_method.SetMetricFixedMask(fixed_mask)
    # display_axial(np.transpose(sitk.GetArrayFromImage(fixed_mask), (1, 2, 0)))
        # registration_method.SetMetricMovingMask(fixed_mask)



        # fixed_mask = sitk.Cast(sitk.GetImageFromArray(np.transpose(fixed_mask),(2,0,1)),sitk.sitkFloat32)
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

