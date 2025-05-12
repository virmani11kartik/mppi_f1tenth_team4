import cv2
import numpy as np
import pyrealsense2 as rs
import pytesseract
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("Running visual command interpreter. Press 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        output = frame.copy()
        command = "None"

        # Threshold red (two ranges in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circle_detected = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 < circularity < 1.2 and area > 3000:
                circle_detected = True
                cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)
                break

        if circle_detected:
            command = "Stop"
            print("Red circle detected — stopping for 5 seconds")
            cv2.putText(output, "Stopping...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Visual Command", output)
            cv2.waitKey(1)
            time.sleep(5)
            continue

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 500:
                continue
            digit_roi = red_mask[y:y+h, x:x+w]
            text = pytesseract.image_to_string(digit_roi, config='--psm 10 digits').strip()

            if text.isdigit():
                command = text
                print(f"Digit detected: {text}")
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(output, f"Digit: {text}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

        cv2.putText(output, f"Command: {command}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Visual Command", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
