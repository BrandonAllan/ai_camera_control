import cv2
from traitlets import HasTraits, Int

class CameraConfiguration(HasTraits):
    width = Int(640).tag(config=True)
    height = Int(480).tag(config=True)
    capture_width = Int(640).tag(config=True)
    capture_height = Int(480).tag(config=True)

class Camera:
    def __init__(self):
        self.cap = None
        self.configuration = CameraConfiguration()
        self.width = self.configuration.width
        self.height = self.configuration.height
        self.capture_width = self.configuration.capture_width
        self.capture_height = self.configuration.capture_height

    def start_capture(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

    def stop_capture(self):
        if self.cap is not None:
            self.cap.release()

    def capture_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return "Camera not initialized"



    def set_frame_size(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

