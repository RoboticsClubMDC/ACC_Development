"""This module contains QCar specific implementations of hal features"""
from pytransform3d import rotations as pr
import numpy as np
import time
import os

from pal.utilities.math import wrap_to_pi
from hal.utilities.estimation import EKF, KalmanFilter
from hal.utilities.control import PID, StanleyController

import cv2
from functools import cmp_to_key
import math
from scipy.spatial import distance
from pal.utilities.math import Signal
from os.path import splitext,basename

class QCarEKF:
    """ An EKF designed to estimate the 2D position and orientation of a QCar.

    Attributes:
        kf (KalmanFilter): Kalman filter for orientation estimation.
        ekf (EKF): Extended Kalman filter for pose estimation.
        L (float): Wheelbase of the vehicle.
        x_hat (ndarray): State estimate vector [x; y; theta].
    """

    def __init__(
            self,
            x_0,
            Q_kf=np.diagflat([0.0001, 0.001]),
            R_kf=np.diagflat([.001]),
            Q_ekf=np.diagflat([0.01, 0.01, 0.01]),
            R_ekf=np.diagflat([0.01, 0.01, 0.001])
        ):
        """Initialize QCarEKF with initial state and noise covariance matrices.

        Args:
            x_0 (ndarray): Initial state vector [x, y, theta].
            Q_kf (ndarray, optional): KF process noise covariance matrix.
            R_kf (ndarray, optional): KF measurement noise covariance matrix.
            Q_ekf (ndarray, optional): EKF process noise covariance matrix.
            R_ekf (ndarray, optional): EKF measurement noise covariance matrix.
        """

        x_0 = np.squeeze(x_0)
        self.kf = KalmanFilter(
            x_0=[x_0[2], 0],
            P0=np.eye(2),
            Q=Q_kf,
            R=R_kf,
            A=np.array([[0, -1], [0, 0]]),
            B=np.array([[1], [0]]),
            C=np.array([[1, 0]])
        )

        self.ekf = EKF(
            x_0=x_0,
            P0=np.eye(3),
            Q=Q_ekf,
            R=R_ekf,
            f=self.f,
            J_f=self.J_f,
            C=np.eye(3)
        )

        self.L = 0.2
        self.x_hat = self.ekf.x_hat

    def f(self, x, u, dt):
        """Motion model for the kinematic bicycle model.

        Args:
            x (ndarray): State vector [x, y, theta].
            u (ndarray): Control input vector [v, delta].
            dt (float): Time step in seconds.

        Returns:
            ndarray: Updated state vector after applying motion model.
        """

        return x + dt * u[0] * np.array([
            [np.cos(x[2,0])],
            [np.sin(x[2,0])],
            [np.tan(u[1]) / self.L]
        ])

    def J_f(self, x, u, dt):
        """Jacobian of the motion model for the kinematic bicycle model.

        Args:
            x (ndarray): State vector [x, y, theta].
            u (ndarray): Control input vector [v, delta].
            dt (float): Time step in seconds.

        Returns:
            ndarray: Jacobian matrix of the motion model.
        """

        return np.array([
            [1, 0, -dt*u[0]*np.sin(x[2,0])],
            [0, 1, dt*u[0]*np.cos(x[2,0])],
            [0, 0, 1]
        ])

    def update(self, u=None, dt=None, y_gps=None, y_imu=None):
        """Update the EKF state estimate using GPS and IMU measurements.

        Args:
            u (ndarray, optional): Control input vector [v, delta].
            dt (float, optional): Time step in seconds.
            y_gps (ndarray, optional): GPS measurement vector [x, y, th].
            y_imu (float, optional): IMU measurement of orientation.
        """

        if dt is not None:
            if y_imu is not None:
                self.kf.predict(y_imu, dt)
                self.kf.x_hat[0,0] = wrap_to_pi(self.kf.x_hat[0,0])
            if u is not None:
                self.ekf.predict(u, dt)
                self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        if y_gps is not None:
            y_gps = np.squeeze(y_gps)

            y_kf = (
                wrap_to_pi(y_gps[2] - self.kf.x_hat[0,0])
                + self.kf.x_hat[0,0]
            )
            self.kf.correct(y_kf, dt)
            self.kf.x_hat[0,0] = wrap_to_pi(self.kf.x_hat[0,0])

            y_ekf = np.array([
                [y_gps[0]],
                [y_gps[1]],
                [self.kf.x_hat[0,0]]
            ])
            z_ekf = y_ekf - self.ekf.C @ self.ekf.x_hat
            z_ekf[2] = wrap_to_pi(z_ekf[2])
            y_ekf = z_ekf + self.ekf.C @ self.ekf.x_hat
            self.ekf.correct(y_ekf, dt)
            self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        else:
            y_ekf = (
                wrap_to_pi(self.kf.x_hat[0,0] - self.ekf.x_hat[2,0])
                + self.ekf.x_hat[2,0]
            )
            self.ekf.correct([None, None, y_ekf], dt)
            self.ekf.x_hat[2,0] = wrap_to_pi(self.ekf.x_hat[2,0])

        self.x_hat = self.ekf.x_hat

class QCarDriveController:
    """Implements a drive controller for a QCar that handles speed and steering

    Attributes:
        speedController (PID): PI controller for speed control.
        steeringController (StanleyController): Nonlinear Stanley controller
            for steering control.
    """

    def __init__(self, waypoints, cyclic):
        """Initialize QCarDriveController

        Args:
            waypoints (list): List of waypoints for the controller to follow.
            cyclic (bool): Indicates if the waypoint path is cyclic or not.
        """

        self.speedController = PID(
            Kp=0.1,
            Ki=1,
            Kd=0,
            uLimits=(-0.3, 0.3)
        )

        self.steeringController = StanleyController(
            waypoints=waypoints,
            k=1,
            cyclic=cyclic
        )
        self.steeringController.maxSteeringAngle = np.pi/6

    def reset(self):
        """Resets the internal state of the speed and steering controllers."""

        self.speedController.reset()
        self.steeringController.wpi = 0
        self.steeringController.pathComplete = False

    def updatePath(self, waypoints, cyclic):
        """Updates the waypoint path for the steering controller.

        Args:
            waypoints (list): List of new waypoints to follow
            cyclic (bool): Indicates if the updated path is cyclic or not.
        """

        self.steeringController.updatePath(waypoints, cyclic)

    def update(self, p, th, v, v_ref, dt):
        """Updates the drive controller with the current state of the QCar.

        Args:
            p (ndarray): Position vector [x, y].
            th (float): Orientation angle in radians.
            v (float): Current speed of the QCar.
            v_ref (float): Reference speed for the QCar.
            dt (float): Time step in seconds.

        Returns:
            float: Speed control input.
            float: Steering control input.
        """

        if not self.steeringController.pathComplete:
            delta = self.steeringController.update(p, th, v)
        else:
            delta = 0
            v_ref = 0

        u = self.speedController.update(v_ref, v, dt)

        return u, delta

class ObjectDetection:
    """ An classical object detection algorithm with different options of detection
        methods, as well as visualization tools.

    """

    def __init__(self):
        ''' Initializes the object detection class.
        '''
        self.yolo=None

    def color_thresholding(self,img,lower,upper):
        """Produce a binary mask based on lower and upper thresholding values.

        Args:
            img (ndarray): RGB or HSV image.
            lower (tuple): Lower thresholds for each of the 3 RGB or HSV channels.
            upper (tuple): Upper thresholds for each of the 3 RGB or HSV channels.

        Returns:
            ndarray: Binary mask of input image.

        """
        # =================  SECTION A.1 - Thresholding Mask  =================
        mask=np.zeros(img.shape[:2],np.uint8)
        mask = cv2.inRange(img, lower, upper)
        imask = mask>0
        mask[imask]=1
        # =================        End of SECTION A.1         =================
        return mask

    def mask_img (self,img,mask):
        """Display color thresholded image.

        Args:
            img (ndarray): RGB image.
            mask (ndarray): Binary mask of the detectede object.

        """
        # =================    SECTION A.2 - Mask Image     =================
        img_thresh = cv2.bitwise_and(img,img,mask = mask)
        # =================       End of SECTION A.2        =================
        return img_thresh
    
    def obj_detect(self,img,task,mode):
        """Detect objects in an RGB image and return name, binary mask, and
           bounding box of each detected object.

        Args:
            img (ndarray): RGB image.
            task (str): 'detect' or 'classify'
            mode (str): 'shape' or 'template' or 'yolo'.
        Returns:
            tuple: Index 0 - list of binary masks of detected objects.
                   Index 1 - list of names of detected objects.
                   Index 2 - list of bounding boxes of detected objects.

        """
        assert task=='detect' or task =='classify','Task not recognized!'
        if task =='classify':
            assert mode in ['shape','template','yolo'], 'Mode not regocnized'
        
        detectedMask=[]
        detectedName=[]
        detectedBbox=[]

        if task=='detect':
        # =================    SECTION B.1 - Object Detection     =================
            yellow_bounds=[(5, 70, 70), (30, 255, 255)]
            red_bounds=[(148, 70, 70), (180, 255, 255)]
            for index,bounds in enumerate([yellow_bounds,red_bounds]):
                
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                object_mask=self.color_thresholding(hsv,bounds[0],bounds[1])
                
                CCA_ret = cv2.connectedComponentsWithStats(object_mask, 
                                                        connectivity=8, 
                                                        ltype=cv2.CV_32S)
                CCA_stats = CCA_ret[2]
                for i, stat in enumerate(CCA_stats):  
                    blob_binary=np.zeros_like(object_mask,dtype=np.uint8)
                    if i>0 and stat[4]>200:
                        idx=np.where(CCA_ret[1]==i)
                        blob_binary[idx]=1
                        detectedMask.append(blob_binary)
                        x,y,w,h = cv2.boundingRect(blob_binary)
                        detectedBbox.append((int(x),int(y),int(w),int(h)))
                        if index ==0:
                            detectedName.append('traffic')
                        else:
                            detectedName.append('stop sign')
            return detectedMask,detectedName,detectedBbox
        # =================         End of SECTION B.1          =================

        if task == 'classify':
            if mode=='shape':
        # ============   SECTION B.2 - Geometry-based Classification   ============
                yellow_bounds=[(5, 125, 70), (20, 255, 255)]
                red_bounds=[(170, 125, 70), (180, 255, 255)]
                for index,bounds in enumerate([yellow_bounds,red_bounds]):
                    
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    object_mask=self.color_thresholding(hsv,bounds[0],bounds[1])
                    
                    CCA_ret = cv2.connectedComponentsWithStats(object_mask, 
                                                            connectivity=8, 
                                                            ltype=cv2.CV_32S)
                    CCA_stats = CCA_ret[2]
                    for i, stat in enumerate(CCA_stats):  
                        blob_binary=np.zeros_like(object_mask,dtype=np.uint8)
                        if i>0 and stat[4]>2000:
                            idx=np.where(CCA_ret[1]==i)
                            blob_binary[idx]=1
                            detectedMask.append(blob_binary)
                            x,y,w,h = cv2.boundingRect(blob_binary)
                            detectedBbox.append((int(x),int(y),int(w),int(h)))

                            if index ==0:
                                detectedName.append('traffic')
                            else:
                                contours, hierarchy = cv2.findContours(blob_binary,  
                                                                    cv2.RETR_EXTERNAL,
                                                                    cv2.CHAIN_APPROX_NONE) 
                                perimeter = cv2.arcLength(contours[0], True)
                                con_area = cv2.contourArea(contours[0])
                                circularity = 4*math.pi*(con_area/(perimeter*perimeter))
                                if circularity>0.6:
                                    detectedName.append('stop sign')
                                else:
                                    detectedName.append('yeild sign')

                return detectedMask,detectedName,detectedBbox
        # ============               End of SECTION B.2                ============

            elif mode =='template':
        # ==============   SECTION B.3 - Template-based Detection   ==============
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

                stop_path = 'template/stop_sign.png'
                yield_path = 'template/yield_sign.png'
                template_paths=[stop_path,yield_path]

                for t_path in template_paths:
                    template = cv2.imread(t_path,0)
                    name = splitext(basename(t_path))[0]

                    h_template,w_template = template.shape
                    h_img,w_img = img_gray.shape
                    scale = max(math.ceil(h_template/h_img),math.ceil(w_template/w_img))
                    scales=[scale,2*scale,4*scale]
                    best_match = None
                    for s in scales:
                        resize_stop = cv2.resize(template,(w_template//s,h_template//s))
                        res_stop = cv2.matchTemplate(img_gray,resize_stop,cv2.TM_CCOEFF_NORMED)
                        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res_stop)
                        if best_match is None or maxVal > best_match[0]:
                            best_match = (maxVal, maxLoc, w_template//s,h_template//s)
                    (_, maxLoc, w, h) = best_match

                    detectedBbox.append((int(maxLoc[0]), int(maxLoc[1]),w,h))
                    detectedName.append(name)

                    mask=np.zeros(img.shape[:2],np.uint8)
                    mask = cv2.rectangle(mask, 
                                        (int(maxLoc[0]), int(maxLoc[1])), 
                                        (int(maxLoc[0])+w, int(maxLoc[1])+h),
                                        (1), -1)
                    detectedMask.append(mask)

                return detectedMask,detectedName,detectedBbox
        # =================          End of SECTION B.3          =================

            elif mode =='yolo':
        # ===============   SECTION B.4 - YOLO segmentation model   ===============
                imgProcessed = self.yolo.pre_process(img)
                results = self.yolo.predict(imgProcessed)
                if results.masks is None:
                    return detectedMask,detectedName,detectedBbox
                
                boxes=results.boxes.xywh.cpu().numpy().astype(int)
                for b in boxes:
                    w=b[2]
                    h=b[3]
                    x=b[0]-w//2
                    y=b[1]-h//2
                    detectedBbox.append((x,y,w,h))
                
                masks=results.masks.data.cpu().numpy()
                for m in masks:
                    detectedMask.append(m)
                
                ids = results.boxes.cls.cpu().numpy()
                for i in ids:
                    detectedName.append(results.names[i])

                return detectedMask,detectedName,detectedBbox
        # =================          End of SECTION B.4           =================

    def find_distance(self,depth,detectedMask):
        """Calculate distance of objects based on their binary mask and depth image.

        Args:
            depth (ndarray): Depth image.
            detectedMask (list): List of binary masks of the detected objects.

        Returns:
            list: Distance to detected objects.

        """
        detectedDist = []
        # ================   SECTION C.1 - Distance Estimation   ================
        for mask in detectedMask:
            isolated_depth = mask*depth.squeeze()
            distance = np.median(isolated_depth[isolated_depth.nonzero()])
            detectedDist.append(distance)
        return detectedDist
        # ================          End of SECTION C.1           =================

    def set_tracker(self,mode='rgb'):
        """Set trackers in CV2 window to assist in color thresholding.

        Args:
            mode (ndarray): Input image type, 'rgb' or 'hsv'.

        """

        self.mode=mode
        cv2.namedWindow('Image')

        if mode == 'hsv' or mode == 'HSV':
            # Create a trackbar for threshold value (0-255)
            cv2.createTrackbar('hue low', 'Image', 0, 180,self._nothing)
            cv2.createTrackbar('hue high', 'Image', 0, 180,self._nothing)
            cv2.createTrackbar('saturation low', 'Image', 0, 255,self._nothing)
            cv2.createTrackbar('saturation high', 'Image', 0, 255,self._nothing)
            cv2.createTrackbar('value low', 'Image', 0, 255,self._nothing)
            cv2.createTrackbar('value high', 'Image', 0, 255,self._nothing)
        else:
            # Create a trackbar for threshold value (0-255)
            cv2.createTrackbar('R low', 'Image', 0, 255,self._nothing)
            cv2.createTrackbar('R high', 'Image', 0, 255,self._nothing)
            cv2.createTrackbar('G low', 'Image', 0, 255,self._nothing)
            cv2.createTrackbar('G high', 'Image', 0, 255,self._nothing)
            cv2.createTrackbar('B low', 'Image', 0, 255,self._nothing)
            cv2.createTrackbar('B high', 'Image', 0, 255,self._nothing)

    def get_tracker(self):
        """Get tracker values to be used for creating thresholding masks.

        Return:
            tuple: Index 0 - Lower bound of the threhold values (RBG or HSV)
                   Index 1 - Upper bound of the threhold values (RBG or HSV)

        """
        if self.mode == 'hsv' or self.mode == 'HSV':
            h_l = cv2.getTrackbarPos('hue low','Image')
            s_l = cv2.getTrackbarPos('saturation low','Image')
            v_l = cv2.getTrackbarPos('value low','Image')
            h_h = cv2.getTrackbarPos('hue high','Image')
            s_h = cv2.getTrackbarPos('saturation high','Image')
            v_h = cv2.getTrackbarPos('value high','Image')
        else:
            h_l = cv2.getTrackbarPos('B low','Image')
            s_l = cv2.getTrackbarPos('G low','Image')
            v_l = cv2.getTrackbarPos('R low','Image')
            h_h = cv2.getTrackbarPos('B high','Image')
            s_h = cv2.getTrackbarPos('G high','Image')
            v_h = cv2.getTrackbarPos('R high','Image')
        return (h_l,s_l,v_l),(h_h,s_h,v_h)
    
    def annotate(sefl,img,detectedName,detectedBbox,detectedDist):
        """Add names and bounding boxes of detected objects to the input image.

        Args:
            img (ndarray): RGB image.
            detected_name (list): List of names of detected objects.
            detected_bbx (list): List of bounding boxes of detected objects.

        """
        for i,bbox in enumerate(detectedBbox):
            x,y,w,h = bbox
            id = detectedName[i]
            dist = detectedDist[i]
            if dist == 0: dist_msg= ''
            else: dist_msg = ' ('+str(round(dist,2))+' m)'
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 1)
            cv2.putText(img, id+dist_msg, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,0,255), 2)
            
    @staticmethod
    def _nothing(val):
        ''' Dummy function used in cv2.createTrackbar().
        
        '''
        pass

class LaneKeeping:
    """ Overarching class containing functions used for the lane keeping
        skills activity.

    Attributes:
        debug_out (ndarray): Visualization of lane detection algorithm.
        Kdd (float): Gain for computing look-ahead distance.
        ldMin (float): Minimum look-ahead distance.
        ldMax (float): Maximum look-ahead distance.
        ipm (InversePerspectiveMapping): Create Bird's-eye view (BEV) using camera intrisics and extrinsics.
        all_targets (TargetsTracker): Tracker for detected lane centers.
        valid_target_id (list): IDs of detected lane centers.
        bev_camera_pos (ndarray): pixel coordinates of the camera in BEV.
        bev_rear_wheel_pos (ndarray): pixel coordinates of the rear wheel in BEV.
        pp (PurePursuitController): Pure pursuit steering control.

    """
    def __init__(self,
                 Kdd,
                 ldMin,
                 ldMax,
                 maxSteer, 
                 bevShape,
                 bevWorldDims):
        """Initialize LaneKeeping.

        Args:
            Kdd (float): Gain for computing look-ahead distance.
            ldMin (float): Minimum look-ahead distance.
            ldMax (float): Maximum look-ahead distance.
            maxSteer (float): Maximum steering angle in either direction.
            bevShape (list): Shape of the BEV image [width, height].
            bevWorldDims (list): World dimensions represented by the BEV image [min x, max x, min y, max y].

        """

        self.debug_out=np.zeros((bevShape[0],bevShape[1],3), dtype=np.uint8)
        self.Kdd = Kdd
        self.ldMin = ldMin
        self.ldMax = ldMax
        self.ipm = InversePerspectiveMapping(bevShape,bevWorldDims)
        self.all_targets=TargetsTracker()
        self.valid_target_id=[10]
        self.bev_camera_pos = np.array([int(bevShape[0]/2),
                                        int(bevShape[1])])             
        self.bev_rear_wheel_pos = np.array([int(bevShape[0]/2 + 0.32/self.ipm.m_per_pix),
                                            int(bevShape[1] + 2.25/self.ipm.m_per_pix)])
        
        self.pp = PurePursuitController(self.ipm.m_per_pix,
                                        self.bev_rear_wheel_pos,
                                        maxSteer)
        
    def preprocess(self,lane_marking):
        """Merge the two edges of the same lane marking and convert all grey pixels to white.

        Args:
            lane_marking (ndarray): Lane markings IPM.
        
        Returns:
            ndarray: Binary image of preprocessed lane markings.

        """
        # ==============   SECTION C - Preprocess Lane Marking   ====================
        lane_binary = cv2.GaussianBlur(lane_marking,(5,5),0)
        ret,lane_binary = cv2.threshold(lane_marking,1,255,cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8) 
        lane_binary = cv2.dilate(lane_binary,kernel,iterations=1)
        lane_marking_processed = cv2.erode(lane_binary,kernel,iterations=1)
        return lane_marking_processed
        # ==============             END OF SECTION C            ====================

    def isolate_lane_markings(self,lane_marking):
        """Isolate seperate lane marking via connected component analysis.

        Args:
            lane_marking (ndarray): Preprocessed lane markings IPM.
        
        Returns:
            list: List of ndarrays each representing a seperate lane marking.

        """
        # ==============  SECTION D - Isolate Lane Marking ====================
        # each blob should be a binary image with the same shape as lane_binary 
        # containing only one blob
        isolated_binary_blobs=[]
        min_area=50
        CCA_ret = cv2.connectedComponentsWithStats(lane_marking, 
                                                    connectivity=8, 
                                                    ltype=cv2.CV_32S)
        CCA_stats = CCA_ret[2]
        for index, stat in enumerate(CCA_stats):
            
            lane_binary=np.zeros_like(lane_marking,dtype=np.uint8)
            if index>0 and stat[4]>min_area:
                idx=np.where(CCA_ret[1]==index)
                lane_binary[idx]=255
                isolated_binary_blobs.append(lane_binary)
        return isolated_binary_blobs
        # ==============          END OF SECTION D        ====================
    
    def find_target(self,isolated_binary_blobs,v):      
        """Compute look-ahead distance from measured velocity, then estimate 
           lane centers (targets for pure pursuit) from isolated lane markings.

        Args:
            isolated_binary_blobs (list): Preprocessed lane markings IPM.
            v (float): Measured vehicle velocity.
        
        Returns:
            dict: Dictionary with target ID as keys (float) and target coordinates as values (ndarray).

        """
        
        self.targets=[]
        self._debug_angles=[]
        self.isolated_lanes = []

        # ==============  SECTION E.1 - Calculate Look-ahead Distance ====================
        ld = np.clip(self.Kdd * v, self.ldMin, self.ldMax)
        self.ld_pix = int(ld/self.ipm.m_per_pix) #look-ahead distance in number of pixels
        # ==============          END OF SECTION E.1        ====================
        
        # create isolated lane objects, only lanes that contains an intersection is recorded
        for blob in isolated_binary_blobs:
            lane=LaneMarking(self.bev_rear_wheel_pos,blob)
            lane.find_intersection(self.ld_pix)
            if lane.intersection is not None:  #only add the lanes that has an intersection
                self.isolated_lanes.append(lane)
        # sort lanes based on the intesection points position in CCW order
        self.isolated_lanes.sort(key=cmp_to_key(self._ccw_compare))

        if len(self.isolated_lanes)==0:
            # if no valid isolated lane, update targets with empty list
            self.valid_target_id=self.all_targets.update(self.targets)
            return self.all_targets.target_dict

        if len(self.isolated_lanes) ==1:
            # only one valid isolated lane, the target should be some distnace away from
            # the lane marking, on the left or right side of the lane marking
            lane=self.isolated_lanes[0]
    
            # ==============  SECTION E.3 - Determine Side (Case I) ====================
            sign=np.sign((299.1-lane.intersection[0]))
            vec=np.array((sign*75,0)) 
            # ==============           END OF SECTION E.3           ====================

            rot_mat=self.find_rot_mat(lane)
            # re-orient the vector such that the new vector is perpendicular to the lane marking 
            new_vec = rot_mat@vec
            target_point = lane.intersection + new_vec

            self.targets.append(target_point)
            self._debug_angles.append((lane.intersection,target_point))
            self.valid_target_id=self.all_targets.update(self.targets)
            return self.all_targets.target_dict
        
        for i,lane in enumerate(self.isolated_lanes):

            if i > 0:
                delta=lane.intersection-prev_lane.intersection
                dist = np.linalg.norm(delta)

                if dist>100 and dist<300:
                    # ==============  SECTION E.5 - Find Target (Case II) ====================
                    # distance between lane markings is within the ranges of lane width
                    midpoint=(lane.intersection+prev_lane.intersection)/2
                    self.targets.append(midpoint)
                    # ==============           END OF SECTION E.5         ====================

                elif dist>300:
                    # ==============  SECTION E.6 - Find Target (Case III) ====================
                    right_vector = np.array((75,0))
                    left_vector = np.array((-75,0))

                    # find the slope at the intersection
                    rot_mat1=self.find_rot_mat(lane)
                    new_right_vec=rot_mat1@right_vector
                    target_point1 = lane.intersection+ new_right_vec
                    self.targets.append(target_point1)
                    self._debug_angles.append((lane.intersection,target_point1))

                    rot_mat2=self.find_rot_mat(prev_lane)
                    new_left_vec=rot_mat2@left_vector
                    target_point2 = prev_lane.intersection+ new_left_vec
                    self.targets.append(target_point2)
                    self._debug_angles.append((prev_lane.intersection,target_point2))
                    # ==============             END OF SECTION E.6        ====================

                
                else:
                    # isolated lane markings are too close, most likely the two edges of the 
                    # same lane marking, skip loop
                    pass
            prev_lane=lane

        self.valid_target_id=self.all_targets.update(self.targets)
        return self.all_targets.target_dict

    def find_rot_mat(self,lane):
        """Given a lane marking of interest and the its intersection point, a rotation 
           matrix, which creates a vector perpendicular to the lane marking, is computed.

        Args:
            lane (LaneMarking): LaneMarking object repesenting the lane of insterest.
        
        Returns:
            ndarray: Rotation matrix to re-orient a vector when multiplied, such that 
                     this vector will be perpendicular to the lane marking.

        """

        point = lane.intersection
        fit_square_size=40
        iteration=5
        for i in range(iteration):
            fit_square_size=fit_square_size+i*10
            start_point=(point-(fit_square_size,fit_square_size)).astype(int)
            end_point=(point+(fit_square_size,fit_square_size)).astype(int)
            fit_square=np.zeros_like(lane.binary)
            cv2.rectangle(fit_square,start_point,end_point,(255),-1)
            tangent_points=cv2.bitwise_and(lane.binary,fit_square)
            lines = cv2.HoughLinesP(
                tangent_points,
                rho=6,
                theta=np.pi / 90,
                threshold=20,
                lines=np.array([]),
                minLineLength=40,
                maxLineGap=20
                )
            slopes=[]
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        slopes.append((y2-y1)/(x2-x1))
                break
        if len(slopes)==0:
            return np.eye(2)
        slope=sum(slopes) / len(slopes)
        theta_s=math.atan(slope)
        theta=theta_s+math.pi/2
        if theta>math.pi/2:
            theta=theta-math.pi
        rot_mat = np.array([[math.cos(theta),-math.sin(theta)],
                        [math.sin(theta), math.cos(theta)]])
        return rot_mat
    
    def show_debug(self):
        """Render debug visualization and display via cv2

        """

        # add indication of ld to debug output
        cv2.circle(self.debug_out,self.bev_rear_wheel_pos,int(self.ld_pix),(255,0,0),1)
        for lane in self.isolated_lanes:
            # add lane binary blob to debug output
            lane_binary_3d=np.dstack((lane.binary,lane.binary,lane.binary))
            self.debug_out = cv2.addWeighted(self.debug_out,1,lane_binary_3d,1,0)
            # add intersection point and slope search region to debug output
            self._draw_intersect(lane.intersection)
        for (intersect,target_point) in self._debug_angles:
            self._draw_angle(intersect,target_point)
        for point in self.targets:
            cv2.circle(self.debug_out,point.astype(int),5,(0,255,0),-1)
        
        cv2.imshow('debug',self.debug_out)
        self.debug_out=np.zeros((self.ipm.bevShape[0],
                                 self.ipm.bevShape[1],
                                 3),dtype=np.uint8)

    def _draw_intersect(self,intersect):
        """ draw the intesection point on the debug output

        """

        cv2.circle(self.debug_out ,intersect,5,(0,0,255),-1)
        start_point=(intersect-(40,40)).astype(int)
        end_point=(intersect+(40,40)).astype(int)
        cv2.rectangle(self.debug_out ,start_point,end_point,(255,0,0),2)
    
    def _draw_angle(self,intersect,target_point):
        """ visualize the angle for more debug info 

        """

        vec=np.array((50,0))
        cv2.circle(self.debug_out ,(intersect+ vec).astype(int),2,(0,255,0),-1)
        cv2.circle(self.debug_out ,(intersect- vec).astype(int),2,(0,255,0),-1)
        cv2.line(self.debug_out ,(intersect+ vec).astype(int),(intersect- vec).astype(int),(0,255,0),1)
        cv2.line(self.debug_out ,target_point.astype(int),intersect,(0,255,0),1)

    def _ccw_compare(self,lane1,lane2):
        """ helper function used for sorting points in CCW order w.r.t. specified center point, 
            based on this answer https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
        
        """

        point_a=lane1.intersection
        point_b=lane2.intersection
        centerPoint=(self.debug_out.shape[0]//2-1,
                     self.debug_out.shape[1]-1) #center of the intersection circle
        if point_a[0]>=centerPoint[0] and point_b[0]<centerPoint[0]: return -1
        if point_a[0]<centerPoint[0] and point_b[0]>=centerPoint[0]: return 1
        if point_a[0]==centerPoint[0] and point_b[0]==centerPoint[0]:
            if point_a[1] - centerPoint[1] >=0 or point_b[1] - centerPoint[1] >=0: return point_b[1]-point_a[1]
            else: return point_a[1]-point_b[1]
        det = (point_a[0]-centerPoint[0])*(point_b[1]-centerPoint[1])-(point_b[0]-centerPoint[0])*(point_a[1]-centerPoint[1])
        if not det==0:
            return det
        d1=(point_a[0]-centerPoint[0])*(point_a[0]-centerPoint[0])+(point_a[1]-centerPoint[1])*(point_a[1]-centerPoint[1])
        d2=(point_b[0]-centerPoint[0])*(point_b[0]-centerPoint[0])+(point_b[1]-centerPoint[1])*(point_b[1]-centerPoint[1])
        return d1 - d2

class SpeedController:
    """A feed forward proportional-integral (PI) controller that produces a control 
       signal based on the tracking error and specified gain terms: Kff, Kp, Ki.
    
    Attributes:
        max_throttle (float): Maximum magnitude of the output throttle signal (default 0.2).
        Kff (float): Feed forward gain.
        Kp (float): Proportional gain.
        Ki (float): Integral gain.

    """
    def __init__(self, Kp, Ki ,Kff, max_throttle = 0.2):
        """Creates a PID Controller Instance

        Args:
            max_throttle (float): Maximum magnitude of the output throttle signal (default 0.2).
            Kff (float): Feed forward gain.
            Kp (float): Proportional gain.
            Ki (float): Integral gain.

        """
        self.max_throttle = max_throttle
        self.Kp = Kp
        self.Ki = Ki
        self.Kff= Kff
        self.reset()

    def reset(self):
        """This method resets numerical integrator.
        
        """
        
        self.ei = 0

    def update(self, v, vDesire, dt):
        """Update the controller output based on the current measured velocity,
           refernce velocity, and time since last update.

        Args:
            v (float): The measured velocity.
            v_ref (float): Reference velocity.
            dt (float): The time elapsed since the last update.

        Returns:
            float: The controller output.

        """
        # Calulate the error (e)
        e = vDesire - v

        # Calculate the integral of the error
        self.ei += dt * e

        u = vDesire* self.Kff + self.Kp * e + self.Ki * self.ei

        return np.clip(u,-self.max_throttle,self.max_throttle)

class InversePerspectiveMapping:
    """Inverse perspective mapping (IPM) algorithm that creates a bird's-eye view image.
    
    Attributes:
        camera_intrinsics (ndarray): Camera intrisic matrix of the IntelRealsense RGB camera.
        world_dims (list): World dimensions represented by the BEV image [min x, max x, min y, max y].
        bevShape (list): Shape of the BEV image [width, height].
        m_per_pix (float): Conversion from number of pixels to meters.

    """
    def __init__(self,bevShape,bevWorldDims):
        """Creates a IPM Instance

        Args:
            bevShape (list): Shape of the BEV image [width, height].
            bevWorldDims (list): World dimensions represented by the BEV image [min x, max x, min y, max y].

        """
        # ==============  SECTION B.1 -  Camera Intrinsics  ====================
        self.camera_intrinsics = np.array([[318.86 ,       0,    401.34 , 0],
                                           [0       ,312.14 ,    201.50 , 0],
                                           [0       ,       0,           1, 0],
                                           [0       ,       0,           0, 1]])
        # ==============         END OF SECTION B.1         ====================

        self.world_dims=bevWorldDims
        self.bevShape=bevShape
        self.m_per_pix=(bevWorldDims[1]-bevWorldDims[0])/self.bevShape[0]

        self.get_extrinsics()
        self.get_homography()

    def get_extrinsics(self):
        """This method is used to create the transformation matrices which
           bring a point from vehicle frame to camera frame.

        """
        # ==============  SECTION B.2 -  Camera Extrinsics  ====================
        phi = np.pi/2
        theta = 0
        psi = np.pi/2
        height = 1.72

        cx, sx = np.cos(phi), np.sin(phi)
        cy, sy = np.cos(theta), np.sin(theta)
        cz, sz = np.cos(psi), np.sin(psi)
        Rx=np.array([[1  ,0  ,0  ],
                     [0  ,cx ,-sx],
                     [0  ,sx ,cx ]])
        Ry=np.array([[cy ,0  ,sy ],
                     [0  ,1  ,0  ],
                     [-sy,0  ,cy ]])
        Rz=np.array([[cz ,-sz,0  ],
                     [sz ,cz ,0  ],
                     [0  ,0  ,1  ]])
        
        self.R_v2cam= Rx@Ry@Rz
        self.t_v2cam = np.array([[0,height,0]]).T
        self.T_v2cam = np.vstack((np.hstack((self.R_v2cam,self.t_v2cam)),
                                     np.array([[0,0,0,1]])))
        # ==============         END OF SECTION B.2         ====================

    def v2img(self,XYZ):
        '''Converts input locations XYZ in vehicle frame to pixel coordinates uv
           in image frame.

        Args:
            XYZ (ndarray): nx3 vector np.array([[X,Y,Z],...]) represeting 
            locations in vehicle frame.

        Returns:
            ndarray: nx2 vector np.array([[u,v],...]) representing pixel 
            coordinates of input points in image frame.

        '''
        # ==============  SECTION B.3 -  Vehicle to Image  ====================
        XYZ1 = np.hstack((XYZ,np.ones((XYZ.shape[0],1))))
        img_coords = self.camera_intrinsics@self.T_v2cam@XYZ1.T
        img_coords /= img_coords[2]
        uv = img_coords[:2,:].astype(int)
        uv = uv.T
        # ==============         END OF SECTION B.3        ====================
        return uv
    
    def get_homography (self):
        """This method creates the homography matrix (self.M) which converts. 
           RGB camera feed to BEV in desired dimensions. This is done by:
              1.Specifying four arbitrary points in vehicle frame (four corners).
              2.Converting them to image coordinates in RGB image frame.
              3.Compute the BEV coordinates of the four arbitrary points.
              4.Compute the homography matrix between the RGB corners and BEV corners.

        """
        # ==============  SECTION B.4 -  Image to Vehicle  ====================
        # four points of interest in vehicle frame
        # will be used to get homography matrix
        world_max_x=15
        world_max_y=3
        world_min_x=3
        world_min_y=-3
        corners=np.array([
                        [world_max_x,world_max_y,0],
                        [world_min_x,world_max_y,0],
                        [world_max_x,world_min_y,0],
                        [world_min_x,world_min_y,0],
                        ])    
        
        rgb_corners=self.v2img(corners) # same four points in image frame

        # convert the same four points to bev frame
        # make sure the aspect ratio of the out world coordinates 
        # are the same as the out dimesions
        bev_world_min_x=self.world_dims[0]
        bev_world_max_x=self.world_dims[1]
        bev_world_min_y=self.world_dims[2]
        bev_world_max_y=self.world_dims[3]
        bev_corners=np.array([
                    [(bev_world_max_y-world_max_y)/self.m_per_pix,(bev_world_max_x-world_max_x)/self.m_per_pix],
                    [(bev_world_max_y-world_max_y)/self.m_per_pix,self.bevShape[0]-(world_min_x-bev_world_min_x)/self.m_per_pix],
                    [self.bevShape[0]-(world_min_y-bev_world_min_y)/self.m_per_pix,(bev_world_max_x-world_max_x)/self.m_per_pix],
                    [self.bevShape[0]-(world_min_y-bev_world_min_y)/self.m_per_pix,self.bevShape[0]-(world_min_x-bev_world_min_x)/self.m_per_pix]
                    ])
        
        self.M = cv2.getPerspectiveTransform(rgb_corners.astype(np.float32),
                                             bev_corners.astype(np.float32))
        # ==============         END OF SECTION B.4        ====================

    def create_bird_eye_view(self,img):
        """Creates BEV from input camera RGB feed

        Args:
            img (ndarray): Camera RGB feed

        Returns:
            ndarray: BEV image with desired shape and world dimensions

        """
        # ==============  SECTION B.5 -  Create BEV  ====================
        dst = cv2.warpPerspective(img,
                                  self.M,
                                  self.bevShape,
                                  cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0,0,0)
                                  )
        return dst
        # ==============      END OF SECTION B.5     ====================

class PurePursuitController:
    """A pure pursuit controller that produces a steering controller based on
       specified parameters and a target point.
    
    Attributes:
        m_per_pix (float): Conversion from number of pixels to meters.
        rear_wheel_pos (ndarray): Position of the rear wheel in BEV pixel coordinate
                                  where the distance to the target point is calculated from 
        maxSteer (float): Maximum steering angle in either direction.

    """
    def __init__(self,m_per_pix,rear_wheel_pos,maxSteer):
        """Creates a pure pursuit controller instance.

        Args:
            m_per_pix (float): Conversion from number of pixels to meters.
            rear_wheel_pos (ndarray): Position of the rear wheel in BEV pixel coordinate
                                      where the distance to the target point is calculated from 
            maxSteer (float): Maximum steering angle in either direction.

        """
        self.m_per_pix=m_per_pix
        self.bev_rear_wheel=rear_wheel_pos
        self.maxSteer=maxSteer
    
    def target2steer(self,point):
        """Calculate output steering angle based on input target point.

        Args:
            point (ndarray): Target point in BEV pixel coordinates.
        
        Returns:
            float: Steering angle in radian

        """
        L=0.256*10
        P=np.linalg.norm(point-self.bev_rear_wheel)*self.m_per_pix
        psi = math.atan((self.bev_rear_wheel[0]-point[0])/(self.bev_rear_wheel[1]-point[1]))
        delta= math.atan((2*L*math.sin(psi))/P)
        return np.clip(delta,-self.maxSteer,self.maxSteer)

class LaneMarking:
    """This class stores relevant information for individual isolated lane marking.
    
    Attributes:
        m_per_pix (float): Conversion from number of pixels to meters.
        rear_wheel_pos (ndarray): Position of the rear wheel in BEV pixel coordinates
                                  where the distance to the target point is calculated from 
        maxSteer (float): Maximum steering angle in either direction.

    """
    def __init__(self,bev_rear_wheel_pos,binary):
        '''Creates a lane marking instance.
        
        Args:
            bev_rear_wheel_pos (ndarray): Position of the rear wheel in BEV pixel coordinates.
            binary (ndarray): Binary blob representing an isolated lane marking, should be the 
                              same shape as the BEV.
        '''
        self.bev_rear_wheel_pos=bev_rear_wheel_pos
        self.binary=binary

    def find_intersection(self,ld):
        '''Find the intersection point, and stores it in self.intersection
        
        Args:
            ld (int): Look-ahead distance starting from the rear wheel in 
                          pixel values.

        '''
        # ==============  SECTION E.2 -  Find Intersection  ====================
        circle_bi=cv2.circle(np.zeros(self.binary.shape,dtype=np.uint8),
                             self.bev_rear_wheel_pos,ld,(255),1)
        intersection =cv2.bitwise_and(circle_bi,self.binary)
        CCA_ret = cv2.connectedComponentsWithStats(intersection, 
                                                    connectivity=8, 
                                                    ltype=cv2.CV_32S)
        CCA_stats = CCA_ret[2]
        CCA_centroid = CCA_ret[3]
        self.intersection = None
        temp=[]
        for index, stat in enumerate(CCA_stats):
            if index>0:
                temp.append(CCA_centroid[index].astype(int))
        if len(temp)==0:
            return
        self.intersection = temp[0]
        # ==============          END OF SECTION E.2        ====================
      
class TargetsTracker:
    ''' Basic object tracker for target points. Consecutive IDs are assigned
        to each target point such that the target point to the left has larger ID value.

    Attributes:
        starting_id (int): Default ID for the rightmost target point.
        target_dict (dict): Dictionary that stores IDs (key) of tracked target points (value).
        disappeared_dict (dict): Dictionary that stores the number of frames of 
                                 disappearance (value) for associated ID (key).
        max_frame (int): When the disappearance frame exceeds max_frame, the tracked 
                         ID and associated point is removed from both dictionaries.

    '''
    def __init__(self,max_frame=2):
        '''Creates a target point tracker instance.

        Args:
            max_frame (int): When the disappearance frame exceeds max_frame, the tracked 
                             ID and associated point is removed from both dictionaries.

        '''
        self.starting_id = 10
        self.target_dict={}
        self.disappeared_dict={}
        self.max_frame=max_frame

    def update(self,new_targets):
        """Assigned IDs to new target points by comparing their distances to
           registered target points.

        Args:
            new_targets (list): list of coordinates of detected target points.
        
        Returns:
            list: IDs of registered target pionts.

        """
        if not new_targets:
            for id in list(self.disappeared_dict.keys()): 
                # num of disappeared frame increase by one for all
                self.disappeared_dict[id]+=1
                if  self.disappeared_dict[id] > self.max_frame:
                    # remove this point if it disappeared for too long
                    self.deregister(id)
            # terminate this loop early if no target is found
            return list(self.target_dict.keys())
        
        if len(list(self.target_dict.keys()))==0: 
            # first iteration or all points expired, register new points 
            for i, point in enumerate(new_targets): 
                # assigning index to each point, starting from starting index
                # bigger index for the points to the left of starting point
                self.register(point,self.starting_id+i)
            return list(self.target_dict.keys())
        else:
            # print('matching new targets')
            # get all registered points and ids
            registered_targets=list(self.target_dict.values())
            registered_ids=list(self.target_dict.keys())

            # compute distances between each pair of registered targets and new targets
            # scipy.spatial.distance is utilized
            D = distance.cdist(np.array(registered_targets),
                               np.array(new_targets)) 

            # each row in D represent the distances of one registered target to all new targets
            # for each of the registeed target, find the smallest distance to the new target
            row_min = D.min(axis=1)
            # sort the rows based on the smallest distnaces (row_min) such that the row with 
            # smllaer distance is at the front of the list (returning the sorted index)
            sorted_row = row_min.argsort()
            # find the index (col) of the smallest distance in each row
            min_dist_index=D.argmin(axis=1)
            # rearranging col based on the associated sorted row
            sorted_col = min_dist_index[sorted_row]

            usedRows=set()
            usedCols=set()

            for (row,col) in zip(sorted_row,sorted_col):
                if row in usedRows or col in usedCols:
                    continue

                #check distance to the registered first?
                # update the registered target and disappear log
                target_id = registered_ids[row]
                self.target_dict[target_id] = new_targets[col]
                self.disappeared_dict[target_id] = 0

                #log this update so this point doesnt get examined again
                usedCols.add(col)
                usedRows.add(row)

            unusedCols = set(range(0,D.shape[1])).difference(usedCols)
            unusedRows = set(range(0,D.shape[0])).difference(usedRows)

            if D.shape[0]>=D.shape[1]: 
                # if number of registered targets >= new targets
                # check if any of the registered targets disppeared
                for row in unusedRows:
                    target_id = registered_ids[row]
                    self.disappeared_dict[target_id] += 1
                    if  self.disappeared_dict[target_id] > self.max_frame:
                        # remove this point if it disappeared for too long
                        self.deregister(target_id)
            else:
                # if there are more new targets than registered target
                # assign the new point a id based on position and register
                allTargets = [] 
                for i in self.target_dict:
                    allTargets.append([i,self.target_dict[i]])
                for col in unusedCols:
                    allTargets.append([-1,new_targets[col]])
                allTargets.sort(key=lambda x:x[1][0])
                num_unused=len(unusedCols)
                self._assign_id(allTargets,num_unused)
            return list(self.target_dict.keys())

    def register(self,point,id):
        """Update self.target_dict with an ID-target pair.

        Args:
            point (ndarray): Coordinates of the target point
            ID (int): Pre-determined ID of the target point

        """
        self.target_dict[id]=point
        self.disappeared_dict[id]=0
    
    def deregister(self,id):
        """Remove ID-target pair in dictionaries if they disappear for too long.

        Args:
            id (int): ID of the target point

        """
        del self.target_dict[id]
        del self.disappeared_dict[id]

    def _assign_id(self,all_targets,num_unused):
        ''' Assign consecutive IDs to each target point such that the target 
            point to the left has larger ID value.

        Args:
            all_targets (list): Sorted list of tuples with index 0 being the 
                                ID (-1 for un-registered) and index 1 being 
                                the coordinate.This list is sorted such that 
                                the tuple to the front of the list contains
                                the leftmost target point.
            num_unused (int): Number of un-registered target points.

        '''
        if num_unused==0:
            return
        l_p=0
        if all_targets[l_p][0] ==-1:
            while all_targets[l_p+1][0] ==-1:
                l_p+=1
            r_p=l_p
        else:
            l_p=-1
            r_p=0
        r_p=l_p
        for l in range(l_p,-1,-1):
            all_targets[l][0] = all_targets[l+1][0]+1
            self.register(all_targets[l][1],all_targets[l][0])
        for r in range(r_p,len(all_targets)-1):
            if all_targets[r+1][0] == -1:
                all_targets[r+1][0] = all_targets[r][0]-1
                self.register(all_targets[r+1][1],all_targets[r+1][0])

class LaneSelection:
    '''Facilitate the usage of gamepad left and right buttons for selecting
       detected lane centers.

    Attributes:
        target_select (int): ID of selected target point.
        left_edge_detect (generator): Signal edge detector for left button.
        right_edge_detect (generator): Signal edge detector for right button.
        clip (int): clipping function to keep selected ID within the range of valid IDs.

    '''
    def __init__(self):
        '''Create a Lane Selector instance and initialize signal edge detector.

        '''
        self.target_select=0
        self.left_edge_detect=Signal.edge_detector()
        next(self.left_edge_detect)
        self.right_edge_detect=Signal.edge_detector()
        next(self.right_edge_detect)
        self.clip = lambda x, l, u: l if x < l else u if x > u else x

    def select(self,myLaneKeeping,kb):
        '''Use gamepad input to select from valid ID of detected target points.

        Args:
            myLaneKeeping (LaneKeeping): Lane Keeping task instance which 
                                         contains detected target points and IDs.
            kb (PygameKeyboard): Keyboard instance that contains key input signals.

        Returns:
            ndarray: Selected target point.

        '''
        _,left_pressed=self.left_edge_detect.send(kb.k_left)
        _,right_pressed=self.right_edge_detect.send(kb.k_right)
        if left_pressed: self.target_select+=1
        if right_pressed: self.target_select-=1
        self.target_select=self.clip(self.target_select,
                                     min(myLaneKeeping.valid_target_id),
                                     max(myLaneKeeping.valid_target_id))
        try:
            selected = myLaneKeeping.all_targets.target_dict[self.target_select]
        except KeyError:
            print('Lost tracked target, selecting rightmost target')
            self.target_select=min(myLaneKeeping.valid_target_id)
            selected = myLaneKeeping.all_targets.target_dict[self.target_select]
        return  selected