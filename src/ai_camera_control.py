#!/usr/bin/env python3

import os
import rclpy
import cv2
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from camera import Camera

class CameraAINode(Node):
    def __init__(self):
        super().__init__('camera_ai_node')
        self.camera = Camera()
        self.camera.start_capture()
        self.camera.set_frame_size()

        self.model_path = os.path.join('.', 'src', 'runs', 'detect', 'train', 'weights', 'last.pt')
        self.model = YOLO(self.model_path)
        self.threshold = 0.5

        self.wheel_publisher = self.create_publisher(Float32MultiArray, 'wheel_coordinates', 10)
        self.image_publisher = self.create_publisher(Image, 'captured_image', 10)
        self.bridge = CvBridge()

        self.timer = self.create_timer(0.1, self.capture_and_publish)

    def capture_and_publish(self):
        frame = self.camera.capture_frame()
        if frame is not None and frame != "Camera not initialized":
            results = self.model(frame)[0]
            self.draw_rectangles(frame, results)
            cv2.imshow('Object Detection', frame)  # Display the frame
            cv2.waitKey(1)
            wheel_coordinates = self.extract_wheel_coordinates(results)
            self.publish_wheel_coordinates(wheel_coordinates)
            self.publish_image(frame)

    def draw_rectangles(self, frame, results):
        class_name_dict = {0: 'wheel', 1: 'ball'}  # Update class names and IDs accordingly

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > self.threshold and class_id in class_name_dict:
                class_name = class_name_dict[class_id]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, class_name.upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    def extract_wheel_coordinates(self, results):
        wheel_coordinates = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.threshold and class_id == 0:  # Only consider class 'wheel'
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                wheel_coordinates.append((center_x, center_y))
        return wheel_coordinates

    def publish_wheel_coordinates(self, wheel_coordinates):
        if len(wheel_coordinates) > 0:
            wheel_msg = Float32MultiArray()
            wheel_msg.data = [coord for point in wheel_coordinates for coord in point]
            self.wheel_publisher.publish(wheel_msg)

    def publish_image(self, frame):
        image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.image_publisher.publish(image_msg)

    def on_shutdown(self):
        self.camera.stop_capture()

def main(args=None):
    rclpy.init(args=args)
    camera_ai_node = CameraAINode()
    rclpy.spin(camera_ai_node)
    camera_ai_node.on_shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
