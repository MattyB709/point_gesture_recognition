import cv2
import mediapipe as mp

def main():
    # --- 1. Setup MediaPipe Hands ---
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize the Hands model
    # min_detection_confidence: Threshold to consider a hand detected (0.0 to 1.0)
    # min_tracking_confidence: Threshold to consider tracking successful
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # --- 2. Setup Webcam ---
    # '0' is usually the default ID for the built-in webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # --- 3. Image Processing ---
        # MediaPipe works with RGB images, but OpenCV reads images in BGR.
        # We must convert the image to RGB before processing.
        image.flags.writeable = False # Performance optimization
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform the hand detection
        results = hands.process(image_rgb)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convert back to BGR for rendering

        # --- 4. Drawing Landmarks ---
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the connections (bones) and landmarks (joints)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # --- 5. Display ---
        # Flip the image horizontally for a selfie-view display
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        # Break loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- 6. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()