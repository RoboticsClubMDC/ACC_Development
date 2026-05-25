import sys
import platform
import numpy as np
import time
import cv2
from pathlib import Path

REPO_PYTHON_LIBS = Path(__file__).resolve().parents[5] / "MDC_libraries" / "python"
QUANSER_PYTHON_LIBS = Path.home() / "Documents" / "Quanser" / "0_libraries" / "python"

for python_libs in (REPO_PYTHON_LIBS, QUANSER_PYTHON_LIBS):
    python_libs_str = str(python_libs)
    while python_libs_str in sys.path:
        sys.path.remove(python_libs_str)

if QUANSER_PYTHON_LIBS.exists():
    sys.path.insert(0, str(QUANSER_PYTHON_LIBS))
if REPO_PYTHON_LIBS.exists():
    sys.path.insert(0, str(REPO_PYTHON_LIBS))

from pit.YOLO.nets import YOLOv8
from pit.YOLO.utils import QCar2DepthAligned

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class DirectD435Aligned:
    def __init__(self, image_width=640, image_height=480, fps=30):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed.")

        self.rgb = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        self.depth = np.zeros((image_height, image_width), dtype=np.float32)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, fps)

        profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return False

        self.rgb[:, :, :] = np.asanyarray(color_frame.get_data()).astype(np.uint8)
        self.depth[:, :] = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        return True

    def terminate(self):
        self.pipeline.stop()


def create_camera(image_width, image_height, fps):
    if platform.system() == "Windows":
        try:
            print("Using direct USB D435 capture.")
            return DirectD435Aligned(image_width=image_width, image_height=image_height, fps=fps)
        except Exception as exc:
            print(f"Direct USB D435 capture failed: {exc}")
            print("Falling back to Quanser depth-aligned stream.")

    return QCar2DepthAligned()

## Timing Parameters and methods 
def elapsed_time():
    return time.time() - startTime

sampleRate     = 30.0
sampleTime     = 1/sampleRate
simulationTime = 30.0
useHalfPrecision = False
print('Sample Time: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Additional parameters
imageWidth  = 640
imageHeight = 480
modelPath = (
    Path(__file__).resolve().parents[5]
    / "ros2"
    / "src"
    / "qcar2_autonomy"
    / "models"
    / "quanser_yolov8s-seg-cone.pt"
)
classesToDetect = [2, 9, 11, 80]

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
# Initialize YOLOv8 segmentation model
myYolo  = YOLOv8(
                 modelPath = str(modelPath),
                 imageHeight= imageHeight,
                 imageWidth = imageWidth,
                )

# Initialize Depth/RGB alignment RT model
QCarImg = create_camera(imageWidth, imageHeight, int(sampleRate))

try:
    startTime = time.time()
    while elapsed_time()<simulationTime:
        start = time.time()

        # Get aligned RGB and Depth images
        QCarImg.read()
            
        rgbProcessed = myYolo.pre_process(QCarImg.rgb)
        predecion = myYolo.predict(inputImg = rgbProcessed,
                                   classes = classesToDetect,
                                   confidence = 0.3,
                                   half = useHalfPrecision,
                                   verbose = False
                                   )
        
        processedResults=myYolo.post_processing(alignedDepth = QCarImg.depth,
                                                clippingDistance = 5)
        for object in processedResults:
            print(object.__dict__)
        print('---------------------------')

        # annotatedImg=myYolo.render(showFPS = True)
        annotatedImg=myYolo.post_process_render(showFPS = True)
        cv2.imshow('Object Segmentation', annotatedImg)

        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - ( computationTime % sampleTime )

        # Pause/sleep for sleepTime in milliseconds
        msSleepTime = int(1000*sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1
        key = cv2.waitKey(msSleepTime) & 0xFF
        if key in (27, ord('q')):
            break

except KeyboardInterrupt:
    print("User interrupted!")
    
finally:
    QCarImg.terminate()

