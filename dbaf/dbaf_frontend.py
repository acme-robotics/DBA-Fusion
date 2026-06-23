import torch
import torchvision
import numpy as np

from lietorch import SE3, SO3
from covisible_graph import CovisibleGraph
import matplotlib.pyplot as plt

import os
import gtsam
import math
import bisect
from math import atan2, cos, sin
import geoFunc.trans as trans
from scipy.spatial.transform import Rotation

# Set N6_DEBUG=1 to trace init / IMU consumption / per-keyframe state. All lines
# are prefixed "[DBG]" so they're easy to grep out of the tqdm stream.
DBG = bool(os.environ.get("N6_DEBUG"))
def dbg(*a):
    if DBG:
        print("[DBG]", *a, flush=True)

# N6_IGNORE_LEVER=1 -> force the cam-IMU lever arm to zero in VI init (diagnostic:
# tests whether the extrinsic translation/baseline is what flips the init scale negative).
IGNORE_LEVER = bool(os.environ.get("N6_IGNORE_LEVER"))
# N6_DISABLE_SCALE=1 -> force metric scale s=1.0 at VI init commit (diagnostic: tests
# whether the committed SCALE is what corrupts the post-commit BA / breaks the attitude,
# vs the attitude/gravity solve itself). Only meaningful when DROID's visual scale ~= metric.
DISABLE_SCALE = bool(os.environ.get("N6_DISABLE_SCALE"))

# VI-init quality gate (restores VINS-Mono's LinearAlignment return-false-and-retry that
# this port had turned into a print-only no-op). Reject an ill-conditioned init -- gravity
# magnitude far from 9.81 or non-positive scale -- and retry on the next window instead of
# committing it. N6_INIT_G_TOL: |‖g‖-9.81| tolerance, m/s^2 (VINS uses 1.0). N6_INIT_MAX_DEFER:
# keyframes past vi_warmup before forcing a best-effort commit (so init can't stall forever).
N6_INIT_G_TOL = float(os.environ.get("N6_INIT_G_TOL", "1.0"))
N6_INIT_MAX_DEFER = int(os.environ.get("N6_INIT_MAX_DEFER", "40"))
# Gyro-bias robustness: the VINS-style solveGyroscopeBias reconciles VISUAL rotation with
# gyro integration over the init window. On a MOVING start (no still period) that solve
# absorbs visual-rotation noise into an implausible "bias" (rec7: 0.32 rad/s = 18 deg/s),
# corrupting the initial attitude. Fix: if the solved |bg| exceeds N6_INIT_BG_MAX (rad/s),
# fall back to a DIRECT physical estimate -- the mean gyro over the lowest-rotation IMU
# sub-window (where mean gyro ~= bias). Only trust that seed if the window is actually quiet
# (gyro std < N6_INIT_QUIET_STD rad/s); else there's no still stretch and we keep the solve.
N6_INIT_BG_MAX = float(os.environ.get("N6_INIT_BG_MAX", "0.1"))
N6_INIT_QUIET_STD = float(os.environ.get("N6_INIT_QUIET_STD", "0.1"))
# Rotation-conditioning gate (root-cause fix): DON'T commit VI init just because the
# minimum window exists. On a translation-dominated window, DROID's monocular rot-trans
# ambiguity fabricates relative rotation (tens of deg) that disagrees with the gyro; the
# IMU-active BA then trades attitude away to reconcile -> km divergence. So defer until the
# window has (a) enough REAL rotation (sum of preintegrated |dR|) so DROID's rotation is
# anchored, AND (b) low pre-BA visual-vs-IMU per-edge rotation disagreement (the gyro and the
# visual relative rotation actually agree). N6_INIT_MIN_ROT: total IMU rotation in window (deg);
# N6_INIT_MAX_ROTRES: max per-edge |R_vis^-1 R_imu| (deg).
N6_INIT_MIN_ROT = float(os.environ.get("N6_INIT_MIN_ROT", "8.0"))
N6_INIT_MAX_ROTRES = float(os.environ.get("N6_INIT_MAX_ROTRES", "5.0"))
# ...and low pre-BA velocity/position IMU residuals (the visual motion and the IMU
# actually agree before the BA, not just rotation). N6_INIT_MAX_VELRES: m/s; N6_INIT_MAX_POSRES: m.
N6_INIT_MAX_VELRES = float(os.environ.get("N6_INIT_MAX_VELRES", "0.3"))
N6_INIT_MAX_POSRES = float(os.environ.get("N6_INIT_MAX_POSRES", "0.1"))

class DBAFusionFrontend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        self.graph = CovisibleGraph(video, net.update, args=args)

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontend variables
        self.is_initialized = False
        self.count = 0

        self.warmup = args.warmup
        self.vi_warmup = 12
        if 'vi_warmup' in args: self.vi_warmup = args.vi_warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

        ### DBAFusion
        self.all_imu = None
        self.cur_imu_ii = 0
        self.is_init = False
        self.all_gnss = None
        self.all_odo = None
        self.all_gt = None
        self.all_gt_keys = None
        self.all_stamp = None
        self.cur_stamp_ii = 0
        self.visual_only = args.visual_only
        self.visual_only_init = False
        self.translation_threshold = 0.0
        self.active_window = args.active_window
        self.high_freq_output = True
        self.zupt = ('use_zupt' in args and args.use_zupt)

        if  not self.visual_only:
            self.max_age = 25
            self.iters1 = 2
            self.iters2 = 1
        else:
            self.max_age = 25
            self.iters1 = 4
            self.iters2 = 2

        # visualization/output
        self.show_plot = args.show_plot
        self.result_file = open(args.resultpath,'wt')
        self.plt_pos     = [[],[]]    # X, Y
        self.plt_pos_ref = [[],[]]    # X, Y
        self.plt_att     = [[],[],[]] # pitch, roll, yaw
        self.plt_bg      = [[],[],[]] # X, Y, Z
        self.plt_t       = []
        self.refTw       = np.eye(4,4)

        if self.show_plot:
            plt.figure('monitor',figsize=[13,4])
            plt.subplot(1,3,1); plt.gca().set_title('Trajectory')
            plt.gca().set_aspect(1)
            plt.subplot(1,3,2); plt.gca().set_title('Attitude Error/Attitude')
            plt.subplot(1,3,3); plt.gca().set_title('Gyroscope Bias')
            plt.ion()
            plt.pause(0.1)

    def get_pose_ref(self, tt:float):
        tt_found = self.all_gt_keys[bisect.bisect(self.all_gt_keys,tt)]
        return tt_found, self.all_gt[tt_found]
    
    def __rollup(self, roll):
        """ roll up window states to save memory """
        self.t1 -= roll
        self.count -= roll
        self.video.counter.value -= roll
        self.video.tstamp     = torch.roll(self.video.tstamp    ,-roll,0) 
        self.video.images     = torch.roll(self.video.images    ,-roll,0) 
        self.video.dirty      = torch.roll(self.video.dirty     ,-roll,0) 
        self.video.red        = torch.roll(self.video.red       ,-roll,0) 
        self.video.poses      = torch.roll(self.video.poses     ,-roll,0) 
        self.video.disps      = torch.roll(self.video.disps     ,-roll,0) 
        self.video.disps_sens = torch.roll(self.video.disps_sens,-roll,0) 
        self.video.disps_up   = torch.roll(self.video.disps_up  ,-roll,0) 
        self.video.intrinsics = torch.roll(self.video.intrinsics,-roll,0) 
        self.video.fmaps      = torch.roll(self.video.fmaps     ,-roll,0) 
        self.video.nets       = torch.roll(self.video.nets      ,-roll,0) 
        self.video.inps       = torch.roll(self.video.inps      ,-roll,0) 
        self.graph.ii -= roll
        self.graph.jj -= roll
        self.graph.ii_inac -= roll
        self.graph.jj_inac -= roll
        rm_inac_index = torch.logical_and(torch.greater_equal(self.graph.ii_inac,0),torch.greater_equal(self.graph.jj_inac,0))
        self.graph.ii_inac = self.graph.ii_inac[rm_inac_index]
        self.graph.jj_inac = self.graph.jj_inac[rm_inac_index]
        self.graph.target_inac = self.graph.target_inac[:,rm_inac_index,:,:,:]
        self.graph.weight_inac = self.graph.weight_inac[:,rm_inac_index,:,:,:] # need test

        self.graph.ii_bad  -= roll
        self.graph.jj_bad  -= roll

        self.video.last_t0 -= roll
        self.video.last_t1 -= roll
        # cur_ii/cur_jj are only set by the IMU bundle adjustment; in visual-only
        # mode they stay None, so don't shift them when the window rolls up.
        if self.video.cur_ii is not None: self.video.cur_ii -= roll
        if self.video.cur_jj is not None: self.video.cur_jj -= roll
        if self.video.imu_enabled:
            graph_temp = gtsam.NonlinearFactorGraph()
            for i in range(self.video.cur_graph.size()):
                f = self.video.cur_graph.at(i)
                graph_temp.push_back(f.rekey((np.array(f.keys())-roll).tolist()))
            self.video.cur_graph = graph_temp
            result_temp = gtsam.Values()
            for i in self.video.cur_result.keys():
                if gtsam.Symbol(i).chr() == ord('b'):
                    result_temp.insert(i-roll,self.video.cur_result.atConstantBias(i))
                elif gtsam.Symbol(i).chr() == ord('v'):
                    result_temp.insert(i-roll,self.video.cur_result.atVector(i))
                elif gtsam.Symbol(i).chr() == ord('x'):
                    result_temp.insert(i-roll,self.video.cur_result.atPose3(i))
                else:
                    raise Exception()
            self.video.cur_result = result_temp
            self.video.marg_factor = self.video.marg_factor.rekey((np.array(self.video.marg_factor.keys())-roll).tolist())

        self.video.state.timestamps           = self.video.state.timestamps           [roll:]
        self.video.state.wTbs                 = self.video.state.wTbs                 [roll:]
        self.video.state.vs                   = self.video.state.vs                   [roll:]
        self.video.state.bs                   = self.video.state.bs                   [roll:]
        self.video.state.preintegrations      = self.video.state.preintegrations      [roll:]
        self.video.state.preintegrations_meas = self.video.state.preintegrations_meas [roll:]
        self.video.state.gnss_valid           = self.video.state.gnss_valid           [roll:]
        self.video.state.gnss_position        = self.video.state.gnss_position        [roll:]
        self.video.state.odo_valid            = self.video.state.odo_valid            [roll:]
        self.video.state.odo_vel              = self.video.state.odo_vel              [roll:]

    def __update(self):
        """ add edges, perform update """
        self.count += 1
        self.t1 += 1

        if self.video.imu_enabled and (self.video.tstamp[self.t1-1] - self.video.vi_init_time > 5.0):
            self.video.reinit = True
            self.video.vi_init_time = 1e9

        ## new frame comes, append IMU
        cur_t = float(self.video.tstamp[self.t1-1].detach().cpu())
        self.video.logger.info('predict %f' %cur_t)

        while self.cur_imu_ii < len(self.all_imu) and self.all_imu[self.cur_imu_ii][0] < cur_t:
            ## high-frequency output
            # predict the pose of skipped frames through IMU preintegration
            if self.high_freq_output and self.video.imu_enabled: 
                if self.all_imu[self.cur_imu_ii][0] > float(self.all_stamp[self.cur_stamp_ii][0]):
                    self.video.state.append_imu_temp(float(self.all_stamp[self.cur_stamp_ii][0]),\
                                    self.all_imu[self.cur_imu_ii][4:7],\
                                    self.all_imu[self.cur_imu_ii][1:4]/180*math.pi,True)
                    if float(self.all_stamp[self.cur_stamp_ii][0]) > self.video.state.timestamps[-1] and\
                          math.fabs(cur_t - float(self.all_stamp[self.cur_stamp_ii][0]))>1e-3:
                        pose_temp = self.video.state.pose_temp
                        ppp = pose_temp.pose().translation()
                        qqq = Rotation.from_matrix(pose_temp.pose().rotation().matrix()).as_quat()
                        line = '%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f'%(float(self.all_stamp[self.cur_stamp_ii][0]),ppp[0],ppp[1],ppp[2]\
                                            ,qqq[0],qqq[1],qqq[2],qqq[3])
                        if self.video.gnss_init_t1>0:
                            p = self.video.ten0 + np.matmul(trans.Cen(self.video.ten0), ppp)
                            line += ' %.6f %.6f %.6f'% (p[0],p[1],p[2]) 
                        self.result_file.writelines(line+'\n')
                        # self.result_file.flush()
                    self.cur_stamp_ii += 1
                self.video.state.append_imu_temp(self.all_imu[self.cur_imu_ii][0],\
                                    self.all_imu[self.cur_imu_ii][4:7],\
                                    self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
                
            self.video.state.append_imu(self.all_imu[self.cur_imu_ii][0],\
                                    self.all_imu[self.cur_imu_ii][4:7],\
                                    self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
            self.cur_imu_ii += 1
        # at end-of-stream the camera can outrun the last IMU sample; clamp so the
        # final keyframe still preintegrates up to its image time (vs IndexError).
        # Only append when time actually advances -- once the IMU stream is exhausted
        # cur_t can fall at/behind state.cur_t, and append_imu raises on backward time.
        if cur_t > self.video.state.cur_t:
            imu_last = self.all_imu[min(self.cur_imu_ii, len(self.all_imu) - 1)]
            self.video.state.append_imu(cur_t, imu_last[4:7], imu_last[1:4]/180*math.pi)
        self.video.state.append_img(cur_t)
        
        ## append GNSS
        if len(self.all_gnss) > 0: gnss_found = bisect.bisect(self.all_gnss[:,0],cur_t - 1e-6)
        else: gnss_found = -1        
        if gnss_found > 0 and self.all_gnss[gnss_found,0] - cur_t < 0.01 :
            self.video.state.append_gnss(cur_t,self.all_gnss[gnss_found,1:4])

        ## append ZUPT
        if self.zupt and self.video.state.preintegrations[self.t1-3].deltaTij() > 3.0:
            if np.linalg.norm(self.video.state.vs[self.t1-2]) < 0.025:
                self.video.state.append_odo(cur_t,np.array([.0,.0,.0]))

        ## append ODO
        if len(self.all_odo) > 0: odo_found = bisect.bisect(self.all_odo[:,0],cur_t - 1e-6)
        else: odo_found = -1        
        if odo_found > 0 and self.all_odo[odo_found,0] - cur_t < 0.01 :
            self.video.state.append_odo(cur_t,self.all_odo[odo_found,1:4])

        self.video.state.append_imu(self.all_imu[self.cur_imu_ii][0],\
                        self.all_imu[self.cur_imu_ii][4:7],\
                        self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
        self.cur_imu_ii += 1

        ## predict pose (<5 ms)
        if self.video.imu_enabled:
            Twc = (self.video.state.wTbs[-1] * self.video.Tbc).matrix()
            TTT = torch.tensor(np.linalg.inv(Twc))
            q = torch.tensor(Rotation.from_matrix(TTT[:3, :3]).as_quat())
            t = TTT[:3,3]
            self.video.poses[self.t1-1] = torch.cat([t,q])
            # watch the predicted body state for the blow-out onset
            p_b = self.video.state.wTbs[-1].translation()
            v_b = self.video.state.vs[-1]
            bias = self.video.state.bs[-1]
            dbg("state t=%.3f |pos|=%.3f |vel|=%.3f ba=%s bg=%s" % (
                cur_t, float(np.linalg.norm(p_b)), float(np.linalg.norm(v_b)),
                np.round(bias.accelerometer(), 3), np.round(bias.gyroscope(), 3)))

        self.video.logger.info('manage edges')

        ## manage edges (60 ms)
        if self.graph.corr is not None:
            if self.visual_only:
                self.graph.rm_factors(torch.logical_and(self.graph.age > self.max_age,\
                torch.logical_or(self.graph.ii < self.t1-self.active_window,self.graph.jj < self.t1-self.active_window)), store=True)
            else:
                self.graph.rm_factors(torch.logical_or(self.graph.age > self.max_age,\
                torch.logical_or(self.graph.ii < self.t1-self.active_window,self.graph.jj < self.t1-self.active_window)), store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0), 
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)

        self.video.logger.info('non-keyframes %d' % self.graph.ii.shape[0])

        ## non-keyframe update
        self.video.disps[self.t1-1] = torch.where(self.video.disps_sens[self.t1-1] > 0, 
           self.video.disps_sens[self.t1-1], self.video.disps[self.t1-1])

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        self.rollup = False
        if self.t1 > 65:
            self.__rollup(30)
            print('rollup ',self.graph.ii)
            self.rollup = True

        self.video.logger.info('output')

        ## visualization/output
        poses = SE3(self.video.poses)
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)
        TTT = np.matmul(poses[self.t1-1].cpu().inv().matrix(),np.linalg.inv(self.video.Ti1c))
        if self.video.imu_enabled or (self.visual_only and self.visual_only_init):
            ppp = TTT[0:3,3]
            qqq = Rotation.from_matrix(TTT[:3, :3]).as_quat()
            line = '%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f'%(cur_t,ppp[0],ppp[1],ppp[2]\
                                        ,qqq[0],qqq[1],qqq[2],qqq[3])
            if self.video.gnss_init_t1>0:
                p = self.video.ten0 + np.matmul(trans.Cen(self.video.ten0), ppp.numpy())
                line += ' %.6f %.6f %.6f'% (p[0],p[1],p[2]) 
            self.result_file.writelines(line+'\n')
            self.result_file.flush()

        TTTref = np.matmul(self.refTw,TTT)
        ppp = TTTref[0:3,3]
        if self.show_plot:
            # if math.fabs(tt_found - cur_t) < 0.1: # for kitti and whu
            self.plt_pos[0].append(ppp[0])
            self.plt_pos[1].append(ppp[1])
            a1 = np.array(trans.m2att(TTTref[0:3,0:3])     )* 57.3
            if self.all_gt is not None:
                tt_found,dd = self.get_pose_ref(cur_t -1e-3)
                self.plt_pos_ref[0].append(dd['T'][0,3])
                self.plt_pos_ref[1].append(dd['T'][1,3])    
                a2 = np.array(trans.m2att(dd['T'][0:3,0:3]) )* 57.3    
                a1 -= a2
            self.plt_att[0].append(a1[0])
            self.plt_att[1].append(a1[1])
            self.plt_att[2].append(a1[2])
            bg = self.video.state.bs[self.t1-1].gyroscope()
            self.plt_bg[0].append(bg[0])
            self.plt_bg[1].append(bg[1])
            self.plt_bg[2].append(bg[2])
            self.plt_t.append(cur_t)
            
            if self.rollup:
                plt.subplot(1,3,1)
                plt.cla(); plt.gca().set_title('Trajectory')
                plt.plot(self.plt_pos[0],self.plt_pos[1],marker='^')
                plt.plot(self.plt_pos_ref[0],self.plt_pos_ref[1],marker='^')
                plt.subplot(1,3,2)
                plt.cla(); plt.gca().set_title('Attitude Error/Attitude')
                plt.plot(self.plt_t,self.plt_att[0],c='r')
                plt.plot(self.plt_t,self.plt_att[1],c='g')
                plt.plot(self.plt_t,self.plt_att[2],c='b')
                plt.ylim([-10,10])
                plt.subplot(1,3,3)
                plt.cla(); plt.gca().set_title('Gyroscope Bias')
                plt.plot(self.plt_t,self.plt_bg[0],c='r')
                plt.plot(self.plt_t,self.plt_bg[1],c='g')
                plt.plot(self.plt_t,self.plt_bg[2],c='b')
                plt.pause(0.1)


        ## keyframe update
        self.video.logger.info('keyframes %d' % self.graph.ii.shape[0])
        if self.t1 > 10:
            cam_translation =  torch.norm((poses[(self.t1-10):(self.t1-3)] * poses[self.t1-2].inv()[None]).translation()[:,0:3],dim=1)
        else:
            cam_translation =  torch.norm((poses[(self.t1-6):(self.t1-3)] * poses[self.t1-2].inv()[None]).translation()[:,0:3],dim=1)

        if (d.item() < self.keyframe_thresh or (self.video.imu_enabled and torch.sum(cam_translation < self.translation_threshold)>0)): # gnss
            self.video.logger.info('remove new frame!!!!!!!!!!!!1')
            self.graph.rm_keyframe(self.t1 - 2)

            # merge preintegration[self.t1-2] and preintegration[self.t1-3]
            for iii in range(len(self.video.state.preintegrations_meas[self.t1-2])):
                dd = self.video.state.preintegrations_meas[self.t1-2][iii]
                if dd[2] > 0:
                    self.video.state.preintegrations[self.t1-3].integrateMeasurement(dd[0],\
                                                                                      dd[1],\
                                                                                      dd[2])
                self.video.state.preintegrations_meas[self.t1-3].append(dd)
                
            self.video.state.preintegrations[self.t1-2] = self.video.state.preintegrations[self.t1-1]
            self.video.state.preintegrations_meas[self.t1-2] = self.video.state.preintegrations_meas[self.t1-1]
            self.video.state.preintegrations.pop()
            self.video.state.preintegrations_meas.pop()

            self.video.rm_new_gnss(self.t1-2)
            self.video.state.wTbs[self.t1-2] = self.video.state.wTbs[self.t1-1]; self.video.state.wTbs.pop()
            self.video.state.bs  [self.t1-2] = self.video.state.bs  [self.t1-1]; self.video.state.bs.pop()
            self.video.state.vs  [self.t1-2] = self.video.state.vs  [self.t1-1]; self.video.state.vs .pop()
            self.video.state.gnss_valid     [self.t1-2] = self.video.state.gnss_valid     [self.t1-1]; self.video.state.gnss_valid .pop()
            self.video.state.gnss_position  [self.t1-2] = self.video.state.gnss_position  [self.t1-1]; self.video.state.gnss_position .pop()
            self.video.state.odo_valid      [self.t1-2] = self.video.state.odo_valid      [self.t1-1]; self.video.state.odo_valid .pop()
            self.video.state.odo_vel        [self.t1-2] = self.video.state.odo_vel        [self.t1-1]; self.video.state.odo_vel .pop()            

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
        else:
            for itr in range(self.iters2):
                # print('b%d' % itr)
                self.graph.update(None, None, use_inactive=True)

        ## try initializing VI/GNSS
        dbg("update end  t1=%d vi_warmup=%d vi_init_t1=%s imu_enabled=%s cur_imu_ii=%d"
            % (self.t1, self.vi_warmup, self.video.vi_init_t1, self.video.imu_enabled, self.cur_imu_ii))
        if self.t1 > self.vi_warmup and self.video.vi_init_t1 < 0:
            self.init_VI()
            if not self.visual_only:
                for i in range(len(self.all_stamp)): # skip to next image
                    if float(self.all_stamp[i][0]) < cur_t + 1e-6: continue
                    else:
                        self.cur_stamp_ii = i
                        break
        if self.video.imu_enabled and self.video.gnss_init_time <= 0.0 and len(self.all_gnss)>0:
            self.init_GNSS()

        ## set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean() * 1.0

        self.video.dirty[self.graph.ii.min():self.t1] = True

    def init_IMU(self):
        """ initialize IMU states """
        cur_t = float(self.video.tstamp[self.t0].detach().cpu())
        for i in range(len(self.all_imu)):
            if self.all_imu[i][0] < cur_t - 1e-6: continue
            else:
                self.cur_imu_ii = i
                break

        for i in range(self.t0,self.t1):
            tt = self.video.tstamp[i]
            if i == self.t0:
                self.video.state.init_first_state(cur_t,np.zeros(3),\
                                            np.eye(3),\
                                            np.zeros(3))
                self.video.state.append_imu(self.all_imu[self.cur_imu_ii][0],\
                                        self.all_imu[self.cur_imu_ii][4:7],\
                                        self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
                self.cur_imu_ii += 1
                self.is_init = True
            else:
                cur_t = float(self.video.tstamp[i].detach().cpu())
                while self.all_imu[self.cur_imu_ii][0] < cur_t:
                    self.video.state.append_imu(self.all_imu[self.cur_imu_ii][0],\
                                            self.all_imu[self.cur_imu_ii][4:7],\
                                            self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
                    self.cur_imu_ii += 1
                self.video.state.append_imu(cur_t,\
                                            self.all_imu[self.cur_imu_ii][4:7],\
                                            self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
                self.video.state.append_img(cur_t)
                
                if len(self.all_gnss) > 0: gnss_found = bisect.bisect(self.all_gnss[:,0],cur_t - 1e-6)
                else: gnss_found = -1
                if gnss_found > 0 and self.all_gnss[gnss_found,0] - cur_t < 0.01:
                    self.video.state.append_gnss(cur_t,self.all_gnss[gnss_found,1:4])

                if len(self.all_odo) > 0: odo_found = bisect.bisect(self.all_odo[:,0],cur_t - 1e-6)
                else: odo_found = -1        
                if odo_found > 0 and self.all_odo[odo_found,0] - cur_t < 0.01 :
                    self.video.state.append_odo(cur_t,self.all_odo[odo_found,1:4])

                self.video.state.append_imu(self.all_imu[self.cur_imu_ii][0],\
                                self.all_imu[self.cur_imu_ii][4:7],\
                                self.all_imu[self.cur_imu_ii][1:4]/180*math.pi)
            
                self.cur_imu_ii += 1
            Twc = np.matmul(np.array([[1,0,0,0],\
                                     [0,1,0,0],\
                                     [0,0,1,0.02*i],\
                                     [0,0,0,1]]),self.video.Ti1c) #  perturb the camera poses, which benefits the robustness of initial BA
            TTT = torch.tensor(np.linalg.inv(Twc))
            q = torch.tensor(Rotation.from_matrix(TTT[:3, :3]).as_quat())
            t = TTT[:3,3]
            if not self.video.imu_enabled:
                self.video.poses[i] = torch.cat([t,q])

    def _quiet_gyro_bias(self, t_end, win_s=0.5):
        """Direct gyro-bias estimate = mean gyro over the lowest-rotation IMU sub-window
        ending at/<= t_end. Returns (bias_rad_s[3], gyro_std_rad_s) for the quietest window,
        or (None, None) if no IMU. The caller decides whether the window is quiet enough to
        trust (std < N6_INIT_QUIET_STD). all_imu gyro is deg/s; we return rad/s."""
        imu = self.all_imu
        if imu is None or len(imu) < 10:
            return None, None
        t_end = float(t_end)
        t = imu[:, 0]
        g = imu[:, 1:4] * (math.pi / 180.0)            # deg/s -> rad/s
        sel = t <= t_end
        t, g = t[sel], g[sel]
        if len(t) < 10:
            return None, None
        best_std, best_mean = None, None
        w0 = t[0]
        while w0 + win_s <= t[-1] + 1e-6:
            m = (t >= w0) & (t < w0 + win_s)
            if m.sum() >= 5:
                std = float(np.linalg.norm(g[m].std(axis=0)))
                if best_std is None or std < best_std:
                    best_std, best_mean = std, g[m].mean(axis=0)
            w0 += win_s / 2.0                          # 50%-overlap slide
        return best_mean, best_std

    def init_VI(self):
        """ initialize the V-I system, referring to VIN-Fusion """
        dbg("init_VI ENTER  t1=%d visual_only=%s" % (self.t1, self.visual_only))
        sum_g = np.zeros(3,dtype = np.float64)
        ccount = 0
        for i in range(self.t1 - 8 ,self.t1-1):
            dt = self.video.state.preintegrations[i].deltaTij()
            tmp_g = self.video.state.preintegrations[i].deltaVij()/dt
            sum_g += tmp_g
            ccount += 1
        aver_g = sum_g * 1.0 / ccount
        var_g = 0.0
        for i in range(self.t1 - 8 ,self.t1-1):
            dt = self.video.state.preintegrations[i].deltaTij()
            tmp_g = self.video.state.preintegrations[i].deltaVij()/dt
            var_g += np.linalg.norm(tmp_g - aver_g)**2
        var_g =math.sqrt(var_g/ccount)
        dbg("init_VI excitation var_g=%.4f (need>=0.25)  aver_g=%s" % (var_g, np.round(aver_g, 3)))
        if var_g < 0.25:
            print("IMU excitation not enough!",var_g)
        else:
            poses = SE3(self.video.poses)
            self.plt_pos = [[],[]]
            self.plt_pos_ref = [[],[]]
            for i in range(0,self.t1):
                ppp = np.matmul(poses[i].cpu().inv().matrix(),np.linalg.inv(self.video.Ti1c))[0:3,3]
                self.plt_pos[0].append(ppp[0])
                self.plt_pos[1].append(ppp[1])
                if self.all_gt is not None:
                    tt_found,dd = self.get_pose_ref(self.video.tstamp[i]-1e-3)
                    self.plt_pos_ref[0].append(dd['T'][0,3])
                    self.plt_pos_ref[1].append(dd['T'][1,3])     

            if self.show_plot:
                plt.subplot(1,3,1) 
                plt.cla(); plt.gca().set_title('Trajectory')
                plt.plot(self.plt_pos[0],self.plt_pos[1],marker='^')
                plt.plot(self.plt_pos_ref[0],self.plt_pos_ref[1],marker='^')
                plt.pause(0.1)

            if not self.visual_only:
                # Direct gyro-bias seed from the quietest IMU window so far -- only trusted
                # if that window is actually still (std < N6_INIT_QUIET_STD); else None and
                # VisualIMUAlignment keeps its own solve.
                bias_seed, quiet_std = self._quiet_gyro_bias(self.video.tstamp[self.t1-1])
                if bias_seed is not None and quiet_std is not None and quiet_std < N6_INIT_QUIET_STD:
                    dbg("quiet-window gyro-bias seed=%s (std=%.4f rad/s)" % (np.round(bias_seed,4), quiet_std))
                else:
                    dbg("no quiet-enough window for bias seed (best std=%s) -- keeping solve"
                        % (None if quiet_std is None else round(quiet_std,4)))
                    bias_seed = None
                # Gated first attempt: if the alignment is ill-conditioned (VINS gate),
                # DON'T commit -- return so init_VI re-fires on the next, better window.
                # Bounded by N6_INIT_MAX_DEFER so a persistently-poor scene still inits
                # (best-effort) rather than stalling forever.
                ok = self.VisualIMUAlignment(self.t1 - 8 ,self.t1, ignore_lever= True, gate= True, bias_prior= bias_seed, disable_scale= DISABLE_SCALE)
                force = (self.t1 - self.vi_warmup) > N6_INIT_MAX_DEFER
                if not ok and not force:
                    dbg("init_VI DEFER (VINS gate failed) -> retry next frame, t1=%d" % self.t1)
                    return
                self.graph.update(None, None, use_inactive=True)
                self.VisualIMUAlignment(self.t1 - 8 ,self.t1, ignore_lever= IGNORE_LEVER, bias_prior= bias_seed, disable_scale= DISABLE_SCALE)
                self.graph.update(None, None, use_inactive=True)
                self.VisualIMUAlignment(self.t1 - 8 ,self.t1, ignore_lever= IGNORE_LEVER, bias_prior= bias_seed, disable_scale= DISABLE_SCALE)
                if self.video.vi_init_t1 < 0:        # forced commit on a still-poor align
                    self.video.vi_init_t1 = self.t1
                    self.video.vi_init_time = self.video.tstamp[self.t1-1]
                self.video.imu_enabled = True
                dbg("init_VI DONE -> imu_enabled=True at t1=%d (forced=%s)" % (self.t1, force and not ok))
            else:
                self.VisualIMUAlignment(self.t1 - 8 ,self.t1, ignore_lever= True)
                self.graph.update(None, None, use_inactive=True)
                self.VisualIMUAlignment(self.t1 - 8 ,self.t1, ignore_lever= False)
                self.graph.update(None, None, use_inactive=True)
                self.VisualIMUAlignment(self.t1 - 8 ,self.t1, ignore_lever= False)
                self.visual_only_init = True

            self.video.set_prior(self.video.last_t0,self.t1)

            self.plt_pos = [[],[]]
            self.plt_pos_ref = [[],[]]
            for i in range(0,self.t1):
                TTT = self.video.state.wTbs[i].matrix()
                ppp = TTT[0:3,3]
                qqq = Rotation.from_matrix(TTT[:3, :3]).as_quat()
                self.result_file.writelines('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n'%(self.video.tstamp[i],ppp[0],ppp[1],ppp[2]\
                                            ,qqq[0],qqq[1],qqq[2],qqq[3]))
                
                TTTref = np.matmul(self.refTw,TTT) # for visualization
                ppp = TTTref[0:3,3]
                qqq = Rotation.from_matrix(TTTref[:3, :3]).as_quat()
                self.plt_pos[0].append(ppp[0])
                self.plt_pos[1].append(ppp[1])
                if self.all_gt is not None:
                    tt_found,dd = self.get_pose_ref(self.video.tstamp[i]-1e-3)
                    self.plt_pos_ref[0].append(dd['T'][0,3])
                    self.plt_pos_ref[1].append(dd['T'][1,3])
            if self.show_plot:
                plt.subplot(1,3,1)
                plt.cla(); plt.gca().set_title('Trajectory')
                plt.plot(self.plt_pos[0],self.plt_pos[1],marker='^')
                plt.plot(self.plt_pos_ref[0],self.plt_pos_ref[1],marker='^')
                plt.pause(0.1)

            for itr in range(1):
                self.graph.update(None, None, use_inactive=True)

    def init_GNSS(self):
        """ initialize the GNSS for geo-referencing fusion """
        ten0 = np.array([self.all_gt[self.all_gt_keys[0]]['X0'],\
                         self.all_gt[self.all_gt_keys[0]]['Y0'],\
                         self.all_gt[self.all_gt_keys[0]]['Z0']])
        self.video.ten0 = ten0
        tn0 = []; tw =[]
        for i in range(len(self.video.state.wTbs) - 10,len(self.video.state.wTbs)):
            if self.video.state.gnss_valid[i]:
                # if not is_ref_set:
                #     ten0 = self.video.sgraph.gnss_position[i]
                #     is_ref_set = True
                teg = self.video.state.gnss_position[i]
                print(self.video.ten0)
                print(self.video.state.gnss_position[i])
                tn0g = np.matmul(trans.Cen(self.video.ten0).T,(self.video.state.gnss_position[i] - self.video.ten0))
                twb = self.video.state.wTbs[i].translation()
                tn0.append(tn0g)
                tw.append(twb)
        if len(tn0) > 1:
            tn0 = np.array(tn0)
            tw = np.array(tw)
            bl = np.linalg.norm(tn0[-1] - tn0[0])
            print('GNSS Alignment Baseline: %.5f' % bl)
            if bl < 10.0:
                print('Baseline too short!!')
                return
            heading_w = math.atan2(tw[-1,1]-tw[0,1],tw[-1,0]-tw[0,0])
            heading_n0 = math.atan2(tn0[-1,1]-tn0[0,1],tn0[-1,0]-tn0[0,0])
            s_w = np.linalg.norm(tw[-1] - tw[0])
            s_n0 = np.linalg.norm(tn0[-1] - tn0[0])

            s = s_n0 / s_w
            Rn0w = trans.att2m(np.array([.0,.0,-heading_w + heading_n0]))
            tn0w = tn0  - np.matmul(Rn0w,tw.T * s).T

            poses = SE3(self.video.poses)
            wTcs = poses.inv().matrix().cpu().numpy()
            wTbs = np.matmul(wTcs,self.video.Tbc.inverse().matrix())
            wTbs[:,0:3,3] = np.matmul(Rn0w,(wTbs[:,0:3,3]*s).T).T + tn0w[0]
            wTbs[:,0:3,0:3] = np.matmul(Rn0w, (wTbs[:,0:3,0:3]).T).T
            
            self.refTw = np.eye(4,4)
            
            for i in range(0,self.t1):
                self.video.state.wTbs[i] = gtsam.Pose3(wTbs[i])
                self.video.state.vs[i] *=  s
            wTcs = np.matmul(wTbs,self.video.Tbc.matrix())
            for i in range(0,self.t1):
                TTT = np.linalg.inv(wTcs[i])
                q = torch.tensor(Rotation.from_matrix(TTT[:3, :3]).as_quat())
                t = torch.tensor(TTT[:3,3])
                self.video.poses[i] = torch.cat([t,q])
                self.video.disps[i] /= s

            self.video.gnss_init_t1 = self.t1
            self.video.gnss_init_time = self.video.tstamp[self.t1-1]
            
            self.video.set_prior(self.video.last_t0,self.t1)

            self.plt_pos = [[],[]]
            self.plt_pos_ref = [[],[]]
            for i in range(0,self.t1):
                TTT = self.video.state.wTbs[i].matrix()
                ppp = TTT[0:3,3]
                qqq = Rotation.from_matrix(TTT[:3, :3]).as_quat()
                self.result_file.writelines('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n'%(self.video.tstamp[i],ppp[0],ppp[1],ppp[2]\
                                            ,qqq[0],qqq[1],qqq[2],qqq[3]))
                
                TTTref = np.matmul(self.refTw,TTT) # for visualization
                ppp = TTTref[0:3,3]
                qqq = Rotation.from_matrix(TTTref[:3, :3]).as_quat()
                self.plt_pos[0].append(ppp[0])
                self.plt_pos[1].append(ppp[1])
                if self.all_gt is not None:
                    tt_found,dd = self.get_pose_ref(self.video.tstamp[i]-1e-3)
                    self.plt_pos_ref[0].append(dd['T'][0,3])
                    self.plt_pos_ref[1].append(dd['T'][1,3])
            if self.show_plot:
                plt.subplot(1,3,1)
                plt.cla(); plt.gca().set_title('Trajectory')
                plt.plot(self.plt_pos[0],self.plt_pos[1],marker='^')
                plt.plot(self.plt_pos_ref[0],self.plt_pos_ref[1],marker='^')
                plt.pause(0.1)

            for itr in range(1):
                self.graph.update(None, None, use_inactive=True)
            print('GNSS initialized!!!!')

    def VisualIMUAlignment(self, t0, t1, ignore_lever, disable_scale = False, gate = False, bias_prior = None):
        poses = SE3(self.video.poses)
        wTcs = poses.inv().matrix().cpu().numpy()

        if not ignore_lever:
            wTbs = np.matmul(wTcs,self.video.Tbc.inverse().matrix())
        else:
            T_tmp = self.video.Tbc.inverse().matrix()
            T_tmp[0:3,3] = 0.0
            wTbs = np.matmul(wTcs,T_tmp)
        cost = 0.0

        # solveGyroscopeBias
        A = np.zeros([3,3])
        b = np.zeros(3)
        H1 =np.zeros([15,6], order='F', dtype=np.float64)
        H2 =np.zeros([15,3], order='F', dtype=np.float64)
        H3 =np.zeros([15,6], order='F', dtype=np.float64)
        H4 =np.zeros([15,3], order='F', dtype=np.float64)
        H5 =np.zeros([15,6], order='F', dtype=np.float64) # navstate wrt. bias
        H6 =np.zeros([15,6], order='F', dtype=np.float64)
        for i in range(t0,t1-1):
            pose_i = gtsam.Pose3(wTbs[i])
            pose_j = gtsam.Pose3(wTbs[i+1])
            Rij = np.matmul(pose_i.rotation().matrix().T,pose_j.rotation().matrix())
            imu_factor = gtsam.gtsam.CombinedImuFactor(0,1,2,3,4,5,self.video.state.preintegrations[i])
            err = imu_factor.evaluateErrorCustom(pose_i,self.video.state.vs[i],\
                                                 pose_j,self.video.state.vs[i+1],\
                self.video.state.bs[i],self.video.state.bs[i+1],\
                    H1,H2,H3,H4,H5,H6)
            tmp_A = H5[0:3,3:6]
            tmp_b = err[0:3]
            cost +=  np.dot(tmp_b,tmp_b)
            A += np.matmul(tmp_A.T,tmp_A)
            b += np.matmul(tmp_A.T,tmp_b)
        bg = -np.matmul(np.linalg.inv(A),b)
        # If the visual-inertial solve produced an implausible bias (moving-start noise
        # absorption), override it with the direct quiet-window measurement, if available.
        if bias_prior is not None and np.linalg.norm(bg) > N6_INIT_BG_MAX:
            dbg("solveGyroscopeBias |bg|=%.3f > %.3f -> using quiet-window seed %s"
                % (np.linalg.norm(bg), N6_INIT_BG_MAX, np.round(bias_prior, 4)))
            bg = np.array(bias_prior, dtype=np.float64)

        for i in range(0,t1-1):
            pim = gtsam.PreintegratedCombinedMeasurements(self.video.state.params,\
                  gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),bg))
            for iii in range(len(self.video.state.preintegrations_meas[i])):
                dd = self.video.state.preintegrations_meas[i][iii]
                if dd[2] > 0: pim.integrateMeasurement(dd[0],dd[1],dd[2])
            self.video.state.preintegrations[i] = pim
            self.video.state.bs[i] = gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),bg)
        print('bg: ',bg)
        
        # linearAlignment
        all_frame_count = t1 - t0
        n_state = all_frame_count * 3 + 3 + 1
        A = np.zeros([n_state,n_state])
        b = np.zeros(n_state)
        i_count = 0
        for i in range(t0,t1-1):
            pose_i = gtsam.Pose3(wTbs[i])
            pose_j = gtsam.Pose3(wTbs[i+1])
            R_i = pose_i.rotation().matrix()
            t_i = pose_i.translation()
            R_j = pose_j.rotation().matrix()
            t_j = pose_j.translation()
            pim = self.video.state.preintegrations[i]
            tic = self.video.Tbc.translation()

            tmp_A = np.zeros([6,10])
            tmp_b = np.zeros(6)
            dt = pim.deltaTij()
            tmp_A[0:3,0:3] = -dt * np.eye(3,3)
            tmp_A[0:3,6:9] = R_i.T * dt * dt / 2
            tmp_A[0:3,9] = np.matmul(R_i.T, t_j-t_i) / 100.0
            tmp_b[0:3] = pim.deltaPij()
            tmp_A[3:6,0:3] = -np.eye(3,3)
            tmp_A[3:6,3:6] = np.matmul(R_i.T, R_j)
            tmp_A[3:6,6:9] = R_i.T * dt
            tmp_b[3:6] = pim.deltaVij()

            r_A = np.matmul(tmp_A.T,tmp_A)
            r_b = np.matmul(tmp_A.T,tmp_b)

            A[i_count*3:i_count*3+6,i_count*3:i_count*3+6] += r_A[0:6,0:6]
            b[i_count*3:i_count*3+6] += r_b[0:6]
            A[-4:,-4:] += r_A[-4:,-4:]
            b[-4:] += r_b[-4:]
            
            A[i_count*3:i_count*3+6,n_state-4:] += r_A[0:6,-4:]
            A[n_state-4:,i_count*3:i_count*3+6] += r_A[-4:,0:6]
            i_count += 1
        
        A = A * 1000.0
        b = b * 1000.0
        x = np.matmul(np.linalg.inv(A),b)
        s = x[n_state-1] / 100.0

        g = x[-4:-1]

        # RefineGravity
        g0 = g / np.linalg.norm(g) * 9.81
        lx = np.zeros(3)
        ly = np.zeros(3)
        n_state = all_frame_count * 3 + 2 + 1
        A = np.zeros([n_state,n_state])
        b = np.zeros(n_state)

        for k in range(4):
            aa = g / np.linalg.norm(g)
            tmp = np.array([.0,.0,1.0])

            bb = (tmp - np.dot(aa,tmp) * aa)
            bb /= np.linalg.norm(bb)
            cc = np.cross(aa,bb)
            bc = np.zeros([3,2])
            bc[0:3,0] = bb
            bc[0:3,1] = cc
            lxly = bc
            
            i_count = 0
            for i in range(t0,t1-1):
                pose_i = gtsam.Pose3(wTbs[i])
                pose_j = gtsam.Pose3(wTbs[i+1])
                R_i = pose_i.rotation().matrix()
                t_i = pose_i.translation()
                R_j = pose_j.rotation().matrix()
                t_j = pose_j.translation()
                tmp_A = np.zeros([6,9])
                tmp_b = np.zeros(6)
                pim = self.video.state.preintegrations[i]
                dt = pim.deltaTij()

                tmp_A[0:3,0:3] = -dt *np.eye(3,3)
                tmp_A[0:3,6:8] = np.matmul(R_i.T,lxly) * dt * dt /2 
                tmp_A[0:3,8]   = np.matmul(R_i.T,t_j - t_i) / 100.0
                tmp_b[0:3] = pim.deltaPij() - np.matmul(R_i.T,g0) * dt * dt / 2

                tmp_A[3:6,0:3] = -np.eye(3)
                tmp_A[3:6,3:6] = np.matmul(R_i.T,R_j)
                tmp_A[3:6,6:8] = np.matmul(R_i.T,lxly) * dt
                tmp_b[3:6] = pim.deltaVij() - np.matmul(R_i.T,g0) * dt

                r_A = np.matmul(tmp_A.T,tmp_A)
                r_b = np.matmul(tmp_A.T,tmp_b)

                A[i_count*3:i_count*3+6,i_count*3:i_count*3+6] += r_A[0:6,0:6]
                b[i_count*3:i_count*3+6] += r_b[0:6]
                A[-3:,-3:] += r_A[-3:,-3:]
                b[-3:] += r_b[-3:]

                A[i_count*3:i_count*3+6,n_state-3:] += r_A[0:6,-3:]
                A[n_state-3:,i_count*3:i_count*3+6] += r_A[-3:,0:6]
                i_count += 1
            
            A = A * 1000.0
            b = b * 1000.0
            x = np.matmul(np.linalg.inv(A),b)
            dg = x[-3:-1]
            g0 = g0 + np.matmul(lxly,dg)
            g0 = g0 / np.linalg.norm(g0) * 9.81
            s = x[-1] / 100.0
        print(s,g0,x)

        if disable_scale:
            s = 1.0
            
        print('g,s:',g,s)
        # VINS gate (restored). On the gated (first) init attempt, reject an ill-
        # conditioned solution -- gravity magnitude off or non-positive scale -- and
        # return BEFORE applying it to the state (matches VINS LinearAlignment returning
        # false ahead of visualInitialAlign). init_VI then retries on the next window.
        if gate:
            net_imu_rot = 0.0
            max_rotres = 0.0
            max_velres = 0.0
            max_posres = 0.0
            _Hg = [np.zeros([15,6],order='F',dtype=np.float64), np.zeros([15,3],order='F',dtype=np.float64),
                   np.zeros([15,6],order='F',dtype=np.float64), np.zeros([15,3],order='F',dtype=np.float64),
                   np.zeros([15,6],order='F',dtype=np.float64), np.zeros([15,6],order='F',dtype=np.float64)]
            bias_g = gtsam.imuBias.ConstantBias(np.zeros(3), bg)
            for i in range(t0, t1-1):
                Ri_v = gtsam.Pose3(wTbs[i]).rotation().matrix()
                Rj_v = gtsam.Pose3(wTbs[i+1]).rotation().matrix()
                Ri = self.video.state.preintegrations[i].deltaRij().matrix()
                net_imu_rot += np.degrees(np.arccos(np.clip((np.trace(Ri)-1)/2, -1, 1)))
                dR = np.matmul(np.matmul(Ri_v.T, Rj_v).T, Ri)
                max_rotres = max(max_rotres, np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1))))
                # pre-BA velocity/position residual with the just-solved velocities + scale s
                Ti = wTbs[i].copy();   Ti[0:3,3] *= s
                Tj = wTbs[i+1].copy(); Tj[0:3,3] *= s
                vi = np.matmul(Ri_v, x[(i-t0)*3:(i-t0)*3+3])
                vj = np.matmul(Rj_v, x[(i+1-t0)*3:(i+1-t0)*3+3])
                f = gtsam.gtsam.CombinedImuFactor(0,1,2,3,4,5, self.video.state.preintegrations[i])
                e = f.evaluateErrorCustom(gtsam.Pose3(Ti), vi, gtsam.Pose3(Tj), vj, bias_g, bias_g,
                                          _Hg[0],_Hg[1],_Hg[2],_Hg[3],_Hg[4],_Hg[5])
                max_posres = max(max_posres, float(np.linalg.norm(e[3:6])))
                max_velres = max(max_velres, float(np.linalg.norm(e[6:9])))
            if (net_imu_rot < N6_INIT_MIN_ROT or max_rotres > N6_INIT_MAX_ROTRES
                    or max_velres > N6_INIT_MAX_VELRES or max_posres > N6_INIT_MAX_POSRES):
                dbg("VisualIMUAlignment REJECT (cond gate): net_rot=%.1f°(min%.1f) rotres=%.1f°(max%.1f) velres=%.3f(max%.2f) posres=%.3f(max%.2f)"
                    % (net_imu_rot, N6_INIT_MIN_ROT, max_rotres, N6_INIT_MAX_ROTRES,
                       max_velres, N6_INIT_MAX_VELRES, max_posres, N6_INIT_MAX_POSRES))
                return False
            dbg("VisualIMUAlignment cond-gate PASS: net_rot=%.1f° rotres=%.1f° velres=%.3f posres=%.3f"
                % (net_imu_rot, max_rotres, max_velres, max_posres))
        if gate and not (math.fabs(np.linalg.norm(g) - 9.81) <= N6_INIT_G_TOL and s > 0):
            dbg("VisualIMUAlignment REJECT (VINS gate): |g|=%.3f s=%.4f" % (np.linalg.norm(g), s))
            return False
        if math.fabs(np.linalg.norm(g) - 9.81) < 0.5 and s > 0:
            print('V-I successfully initialized!')

        # visualInitialAlign
        wTbs[:,0:3,3] *= s # !!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(0, t1-t0):
            self.video.state.vs[i+t0] = np.matmul(wTbs[i+t0,0:3,0:3],x[i*3:i*3+3])
        
        # g2R
        ng1 = g0/ np.linalg.norm(g0)
        ng2 = np.array([0,0,1.0])
        R0 = trans.FromTwoVectors(ng1,ng2)
        yaw = trans.R2ypr(R0)[0]
        R0 = np.matmul(trans.ypr2R(np.array([-yaw,0,0])),R0)

        # align for visualization
        ppp =  np.matmul(R0,wTbs[t1-1,0:3,3])
        RRR =  np.matmul(R0,wTbs[t1-1,0:3,0:3])

        if self.all_gt is not None: # align the initial poses for visualization
            tt_found,dd = self.get_pose_ref(self.video.tstamp[t1-1]-1e-3)
            self.refTw = np.matmul(dd['T'],np.linalg.inv(wTbs[t1-1]))
            self.refTw[0:3,0:3] = trans.att2m([0,0,trans.m2att(self.refTw[0:3,0:3])[2]])

        g = np.matmul(R0,g0)
        for i in range(0,t1):
            wTbs[i,0:3,3] = np.matmul(R0,wTbs[i,0:3,3])
            wTbs[i,0:3,0:3] = np.matmul(R0,wTbs[i,0:3,0:3])
            self.video.state.vs[i] = np.matmul(R0, self.video.state.vs[i])
            self.video.state.wTbs[i] = gtsam.Pose3(wTbs[i])

        self.video.vi_init_t1 = t1
        self.video.vi_init_time = self.video.tstamp[t1-1]

        if not ignore_lever:
            wTcs = np.matmul(wTbs,self.video.Tbc.matrix())
        else:
            T_tmp = self.video.Tbc.matrix()
            T_tmp[0:3,3] = 0.0
            wTcs = np.matmul(wTbs,T_tmp)
    
        for i in range(0,t1):
            TTT = np.linalg.inv(wTcs[i])
            q = torch.tensor(Rotation.from_matrix(TTT[:3, :3]).as_quat())
            t = torch.tensor(TTT[:3,3])
            self.video.poses[i] = torch.cat([t,q])
            self.video.disps[i] /= s
        if DBG:
            # COMMIT-TIME attitude audit: estimator's gravity dir in each body frame
            # (committed, gravity-aligned world g=-z) -> compare offline to accel.
            wg = np.array([0.0, 0.0, -1.0])
            for i in range(t0, t1):
                gb = np.matmul(wTbs[i,0:3,0:3].T, wg)
                dbg("COMMIT_ATT i=%d t=%.6f s=%.4f gdir_body_est=%s" % (
                    i, float(self.video.tstamp[i]), s, np.round(gb, 4)))
        return True

    def __initialize(self):
        """ initialize the SLAM system """

        self.t0 = 0
        self.t1 = self.video.counter.value
        dbg("__initialize  t1(counter)=%d warmup=%d" % (self.t1, self.warmup))

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        self.init_IMU()

        self.graph.video.imu_enabled = False
        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.video.imu_enabled = False
        for itr in range(8):
            self.graph.update(1, use_inactive=True)
            
        # torch.concat([self.graph.ii[None],self.graph.jj[None]]).T
        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()
        
        # initialization complete
        self.is_initialized = True

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)

    def __call__(self):
        """ main update """

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()

        
