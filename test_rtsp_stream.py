import cv2
import yaml

def play_rtsp_stream(rtsp_url, width, height):
    """
    Open and play an RTSP stream using OpenCV.
    """
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from RTSP stream.")
            break

        # Resize the frame to match the expected height and width
        frame = cv2.resize(frame, (width, height))

        # Display the frame
        cv2.imshow("RTSP Stream", frame)

        # Break the loop if ESC is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for ESC
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Read the config file.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    RTSP_URL = config["rtsp_url"]
    HEIGHT = config["height"]
    WIDTH = config["width"]

    play_rtsp_stream(rtsp_url=RTSP_URL, width=WIDTH, height=HEIGHT)
