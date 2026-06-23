import sys
sys.path.append('dbaf')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import argparse
from dbaf import DBAFusion

import h5py
import pickle
import re
import math
import quaternion
import gtsam

# N6_DEBUG=1 -> trace config / IMU loading / inflation (prefix "[DBG]" for grepping).
DBG = bool(os.environ.get("N6_DEBUG"))
def dbg(*a):
    if DBG:
        print("[DBG]", *a, flush=True)

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, imagestamp, enable_h5, h5path, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    # N6 lens uses the radtan (plumb-bob) model: calib = fx fy cx cy k1 k2 p1 p2.
    # The base demo only did pinhole or KB-fisheye; feeding radtan coeffs to a fisheye
    # undistort would be the wrong model (k1=-0.44 is large barrel distortion). Undistort
    # with cv2 radtan to a fresh pinhole K (computed once), then feed THAT K to DROID --
    # the images are now distortion-free, so DROID's pinhole assumption holds.
    dist = np.array(calib[4:8]) if len(calib) >= 8 else None
    undist_maps = None      # (m1, m2), computed lazily once we know the frame size

    if not enable_h5:
        # Sort by the integer ns timestamp in the filename, NOT lexicographically.
        # N6 stamps cross the 10 s boundary (e.g. 3069301000 vs 10026000000), so they
        # differ in digit count; a string sort interleaves them ("18..." before "3...")
        # and feeds frames out of time order -- which scrambles them against the
        # time-ordered IMU and corrupts the VI fusion (and overruns the IMU buffer).
        image_list = sorted(os.listdir(imagedir),
                            key=lambda f: int(os.path.splitext(f)[0]))[::stride]
        image_stamps = np.loadtxt(imagestamp,str,delimiter=',')
        image_dict = dict(zip(image_stamps[:,1],image_stamps[:,0]))
        for t, imfile in enumerate(image_list):
            image = cv2.imread(os.path.join(imagedir, imfile))

            if dist is not None:
                if undist_maps is None:
                    h, w = image.shape[:2]
                    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)   # alpha=0: crop borders
                    undist_maps = cv2.initUndistortRectifyMap(
                        K, dist, np.eye(3), newK, (w, h), cv2.CV_32FC1)
                    fx, fy, cx, cy = newK[0,0], newK[1,1], newK[0,2], newK[1,2]   # pinhole K post-undistort
                image = cv2.remap(image, undist_maps[0], undist_maps[1],
                                  interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            tt = float(image_dict[imfile]) /1e9

            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1))
            image = image[:h1-h1%8, :w1-w1%8]
            image = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy ])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            yield tt, image[None], intrinsics
    else:
        ccount = 0
        h5_f = h5py.File(h5path,'r')
        all_keys = sorted(list(h5_f.keys()))
        for key in all_keys:
            ccount += 1
            yield pickle.loads(np.array(h5_f[key]))

if __name__ == '__main__':

    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--imagestamp", type=str, help="")
    parser.add_argument("--imupath", type=str, help="")
    parser.add_argument("--gtpath", type=str, help="")
    parser.add_argument("--enable_h5", action="store_true", help="")
    parser.add_argument("--h5path", type=str, help="")
    parser.add_argument("--resultpath", type=str, default="result.txt", help="")

    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=80)
    parser.add_argument("--image_size", default=[240, 320])

    parser.add_argument("--max_factors", type=int, default=48, help="maximum active edges (which determines the GPU memory usage)")
    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=3.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--active_window", type=int, default=8, help="maximum frames involved in DBA")
    parser.add_argument("--inac_range", type=int, default=3, help="maximum inactive frames (whose flow wouldn't be updated) involved in DBA")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--visual_only", type=int,default=0, help="wheter to disbale the IMU")
    parser.add_argument("--far_threshold", type=float, default=0.02, help="far pixels would be downweighted (unit: m^-1)")
    parser.add_argument("--translation_threshold", type=float, default=0.2, help="avoid the insertion of too close keyframes (unit: m)")
    parser.add_argument("--mask_threshold", type=float, default=-1, help="downweight too close edges (unit: m)")
    parser.add_argument("--skip_edge", type = str, default ="[]", help="whether to add 'skip' edges in the graph (for example, [-4,-5,-6] relative to the oldest active frame)")
    parser.add_argument("--save_pkl", action="store_true")
    parser.add_argument("--pklpath", default="result.pkl", help="path to saved reconstruction")
    parser.add_argument("--show_plot", action="store_true", help="plot the trajectory during running")
    parser.add_argument("--n6_config", type=str, required=True,
                        help="DBA-Fusion config JSON built by calib/to_dbaf.py from the "
                             "Kalibr calibration (Tbc extrinsic, imu_params, timeshift)")
    parser.add_argument("--apply_timeshift", action=argparse.BooleanOptionalAction, default=False,
                        help="apply the Kalibr cam-IMU timeshift (IMU.TimeShift) from the config: "
                             "all_imu_t -= timeshift_s. OFF by default: empirically it made the "
                             "working still-start case worse (2.6 m -> 12 m), and the magnitude is "
                             "unresolved (cross-corr says ~8-14 ms vs Kalibr's 3.7 ms). The sign is "
                             "correct (negative); validate the magnitude with the sync tooling before "
                             "trusting it.")

    args = parser.parse_args()
    args.skip_edge = eval(args.skip_edge)

    # N6 calibration (generated from the Kalibr calib by calib/to_dbaf.py -- not hardcoded)
    import json
    with open(args.n6_config) as f:
        N6CFG = json.load(f)

    args.stereo = False
    dbaf = None
    torch.multiprocessing.set_start_method('spawn')

    """ Load reference trajectory (for visualization) """
    all_gt ={}
    try:
        fp = open(args.gtpath,'rt')
        while True:
            line = fp.readline().strip()
            if line == '':break
            if line[0] == '#' : continue
            line = re.sub('\s\s+',' ',line)
            elem = line.split(',')
            sod = float(elem[0])/1e9
            if sod not in all_gt.keys():
                all_gt[sod] ={}
            R = quaternion.as_rotation_matrix(quaternion.from_float_array([float(elem[4]),\
                                                                           float(elem[5]),\
                                                                           float(elem[6]),\
                                                                           float(elem[7])]))
            TTT = np.eye(4,4)
            TTT[0:3,0:3] = R
            TTT[0:3,3] = np.array([ float(elem[1]), float(elem[2]), float(elem[3])])
            all_gt[sod]['T'] = TTT
        all_gt_keys =sorted(all_gt.keys())
        fp.close()
    except:
        pass

    """ Load IMU data """
    all_imu = np.loadtxt(args.imupath,delimiter=',')
    all_imu[:,0] /= 1e9
    all_imu[:,1:4] *= 180/math.pi
    dbg("IMU loaded: %d samples  t=[%.3f..%.3f]  dt~%.4f (%.1f Hz)  monotonic=%s" % (
        len(all_imu), all_imu[0,0], all_imu[-1,0],
        np.median(np.diff(all_imu[:,0])), 1.0/np.median(np.diff(all_imu[:,0])),
        bool(np.all(np.diff(all_imu[:,0]) > 0))))
    # Kalibr cam-IMU time offset (t_imu = t_cam + timeshift): put IMU stamps on the
    # camera clock via t_cam = t_imu - timeshift (so a negative timeshift_s ADVANCES the
    # IMU stamps). This is independent of, and composes with, the mid-exposure shift
    # n6_to_euroc already applied to the *camera* stamps -- no double count. OFF by default
    # (pass --apply_timeshift); the magnitude is config-driven from the Kalibr calib
    # (imu16: -5.43 ms). A sub-frame correction -- it does NOT fix the init scale
    # degeneracy (that's the extrinsic lever arm), but it's the correct calibration.
    ts = N6CFG.get('timeshift_s', 0.0)
    if args.apply_timeshift:
        all_imu[:,0] -= ts
        dbg("timeshift APPLIED: %.6f s (%.3f ms); IMU t now [%.3f..%.3f]" % (
            ts, ts * 1e3, all_imu[0,0], all_imu[-1,0]))
    else:
        dbg("timeshift NOT applied (--apply_timeshift off); config ts=%.6f s" % ts)

    tstamps = []

    """ Load images """
    clahe = cv2.createCLAHE(2.0,tileGridSize=(8, 8))
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.imagestamp, args.enable_h5,\
                                                     args.h5path, args.calib, args.stride)):
        mm = clahe.apply(image[0][0].numpy())
        image[0] = torch.tensor(mm[None].repeat(3,0))
        if args.show_plot:
            show_image(image[0])
        if dbaf is None:
            args.image_size = [image.shape[2], image.shape[3]]
            dbaf = DBAFusion(args)
            dbaf.frontend.all_imu = all_imu
            dbaf.frontend.all_gnss = []
            dbaf.frontend.all_odo = []
            dbaf.frontend.all_stamp  = np.loadtxt(args.imagestamp,str,delimiter=',')
            dbaf.frontend.all_stamp = dbaf.frontend.all_stamp[:,0].astype(np.float64)[None].transpose(1,0)/1e9
            if len(all_gt) > 0:
                dbaf.frontend.all_gt = all_gt
                dbaf.frontend.all_gt_keys = all_gt_keys
            
            # IMU-Camera extrinsic + IMU noise from the generated N6 config (built by
            # calib/to_dbaf.py from the Kalibr calibration -- NOT hand-typed). cfg['Tbc']
            # is T_imu_from_cam (== the SLAM yaml's IMU.T_b_c1), which IS Tbc (body==IMU),
            # so it's used directly with no inversion.
            dbaf.video.Ti1c = np.array(N6CFG['Tbc'])
            dbaf.video.Tbc = gtsam.Pose3(dbaf.video.Ti1c)
            # IMU noise: cfg['imu_params'] = [accel_nd, gyro_nd, accel_rw, gyro_rw]
            # are RAW LSM6DSM datasheet/Allan values (physical sensor noise). DBA-Fusion's
            # sliding-window optimizer is UNSTABLE fed raw noise -- it diverges into an
            # accelerating runaway (verified: TUM-VI room1 blows to ~15 km with raw noise,
            # stays bounded at ~5 m with the upstream inflation below; same on N6). The
            # inflation absorbs unmodeled error (vibration, scale/misalignment, time-sync
            # jitter, the linearized preintegration) -- standard VINS practice. Factors are
            # upstream's (demo_vio_tumvi); the proper N6 values want an Allan-variance run
            # (calib/allan_variance.py) + a tuned factor. Kept here, not in to_dbaf.py, so
            # the config stays the physical calibration and this stays estimator tuning.
            IMU_NOISE_INFLATION = [float(x) for x in
                os.environ.get("N6_IMU_INFLATE", "25,25,10,5000").split(",")]  # [accel_nd, gyro_nd, accel_rw, gyro_rw]
            imu_params = [p * f for p, f in zip(N6CFG['imu_params'], IMU_NOISE_INFLATION)]
            dbg("imu_params base=%s  inflate=%s  final=%s" % (
                N6CFG['imu_params'], IMU_NOISE_INFLATION, [float('%.3g' % x) for x in imu_params]))
            dbg("Tbc=\n%s" % np.round(np.array(N6CFG['Tbc']), 4))
            dbaf.video.state.set_imu_params(imu_params)
            dbaf.video.init_pose_sigma = np.array([0.1, 0.1, 0.0001, 0.0001,0.0001,0.0001])
            dbaf.video.init_bias_sigma = np.array([1.0,1.0,1.0, 1.0,1.0,1.0])
            dbaf.frontend.translation_threshold = args.translation_threshold
            dbaf.frontend.graph.mask_threshold  = args.mask_threshold

        dbaf.track(t, image, intrinsics=intrinsics)

    if args.save_pkl:
        dbaf.save_vis_easy()

    dbaf.terminate()
