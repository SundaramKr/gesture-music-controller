import cv2
import pygame
import os
import math
from collections import deque
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GestureMusicPlayer:
    def __init__(self, music_folder):
        # mediapipe hand landmarker
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            import urllib.request
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(model_url, model_path)
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise

        # create hand landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # for music
        pygame.mixer.init()

        self.music_folder = music_folder
        self.playlist = self.load_playlist()
        self.current_track_index = 0
        self.is_playing = False
        self.is_paused = False

        # for gestures
        self.pinch_threshold = 0.08  # change this if the pinch is detected too early or late
        self.pinch_history = deque(maxlen=10)
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5
        self.consecutive_pinches = 0
        self.pinch_window = 1.0
        self.first_pinch_time = 0

        # volume control
        self.volume = 0.7  # starts at 70% volume
        pygame.mixer.music.set_volume(self.volume)
        self.is_controlling_volume = False
        self.pinch_start_y = None
        self.volume_start = self.volume
        self.volume_sensitivity = 0.006  # volume change per pixel movement in the y axis

        # camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            raise RuntimeError("Camera not available")
        print("Camera opened")

    def load_playlist(self):
        supported_formats = ['.mp3', '.wav', '.ogg']
        playlist = []

        if os.path.exists(self.music_folder):
            for file in os.listdir(self.music_folder):
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    playlist.append(os.path.join(self.music_folder, file))

        if not playlist:
            print(f"No music files found in {self.music_folder}. Only supports mp3, wav and ogg for now.")

        return playlist

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    # landmark 4 = thumb tip, 8 = index finger tip, 12 = middle finger tip
    def is_pinching(self, hand_landmarks):
        # thumb n index
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]

        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < self.pinch_threshold

    def is_middle_pinching(self, hand_landmarks):
        # thumb n middle finger
        thumb_tip = hand_landmarks[4]
        middle_tip = hand_landmarks[12]

        distance = self.calculate_distance(thumb_tip, middle_tip)
        return distance < self.pinch_threshold

    def detect_gesture(self, is_pinch):
        # single, double n triple pinch for play, next n prev
        current_time = time.time()

        self.pinch_history.append(is_pinch)

        # not pinching -> pinching
        if len(self.pinch_history) >= 2:
            if not self.pinch_history[-2] and self.pinch_history[-1]:
                # new pinch
                if current_time - self.first_pinch_time > self.pinch_window:
                    self.consecutive_pinches = 1
                    self.first_pinch_time = current_time
                else:
                    self.consecutive_pinches += 1

        # gesture complete
        if self.consecutive_pinches > 0 and current_time - self.first_pinch_time > self.pinch_window:
            if current_time - self.last_gesture_time > self.gesture_cooldown:
                gesture_count = self.consecutive_pinches
                self.consecutive_pinches = 0
                self.last_gesture_time = current_time
                return gesture_count

        return 0

    def update_volume(self, current_y):
        if self.pinch_start_y is None:
            self.pinch_start_y = current_y
            self.volume_start = self.volume
            return

        # distance moved in y axis
        delta_y = current_y - self.pinch_start_y

        new_volume = self.volume_start - (delta_y * self.volume_sensitivity)
        self.volume = max(0.0, min(1.0, new_volume))
        pygame.mixer.music.set_volume(self.volume)

    def play_pause(self):
        if not self.playlist:
            print("No music files in playlist")
            return

        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
            self.is_paused = True
            print("Paused")
        else:
            if self.is_paused:
                # resume from where it was paused so that it doesn't start from 0:00 of a track
                pygame.mixer.music.unpause()
                self.is_paused = False
            else:
                # load and play from beginning
                pygame.mixer.music.load(self.playlist[self.current_track_index])
                pygame.mixer.music.play()
            self.is_playing = True
            print(f"Playing: {os.path.basename(self.playlist[self.current_track_index])}")

    def next_track(self):
        if not self.playlist:
            return

        self.current_track_index = (self.current_track_index + 1) % len(self.playlist)
        pygame.mixer.music.load(self.playlist[self.current_track_index])
        pygame.mixer.music.play()
        self.is_playing = True
        self.is_paused = False
        print(f"Next: {os.path.basename(self.playlist[self.current_track_index])}")

    def previous_track(self):
        if not self.playlist:
            return

        self.current_track_index = (self.current_track_index - 1) % len(self.playlist)
        pygame.mixer.music.load(self.playlist[self.current_track_index])
        pygame.mixer.music.play()
        self.is_playing = True
        self.is_paused = False
        print(f"Previous: {os.path.basename(self.playlist[self.current_track_index])}")

    def draw_hand_landmarks(self, frame, hand_landmarks, h, w):
        # 4, 8, 12, 16 n 20 -> tips of thumb, index, middle, ring n pinky
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # index
            (0, 9), (9, 10), (10, 11), (11, 12),  # middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]

        # drawing connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]

            start_pos = (int(start_point.x * w), int(start_point.y * h))
            end_pos = (int(end_point.x * w), int(end_point.y * h))

            cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)

        # drawing landmarks
        for landmark in hand_landmarks:
            pos = (int(landmark.x * w), int(landmark.y * h))
            cv2.circle(frame, pos, 5, (0, 0, 255), -1)

    def run(self):
        print("Gesture Music Player Started!")
        print("\nControls:")
        print(" > Single Pinch (thumb + index): Play/Pause")
        print(" > Double Pinch (thumb + index): Next Track")
        print(" > Triple Pinch (thumb + index): Previous Track")
        print(" > Pinch & Move (thumb + middle): Volume Control")
        print(" > Press 'q' to quit\n")

        if self.playlist:
            print(f"Playlist ({len(self.playlist)} songs):")
            for i, track in enumerate(self.playlist):
                print(f"  {i + 1}. {os.path.basename(track)}")
            print()
        else:
            print("Add music files to the ./music folder to enable playback")

        frame_count = 0

        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to read frame")
                break

            frame_count += 1

            # mirror view by flipping horizontally
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # detect hands
            try:
                detection_result = self.detector.detect(mp_image)
            except Exception as e:
                print(f"Detection error: {e}")
                continue

            is_pinch = False
            is_middle_pinch = False
            pinch_midpoint_y = None

            # hand landmarks
            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]

                # draw hand landmarks
                self.draw_hand_landmarks(frame, hand_landmarks, h, w)

                # check for both types of pinches
                is_pinch = self.is_pinching(hand_landmarks)  # Index + Thumb
                is_middle_pinch = self.is_middle_pinching(hand_landmarks)  # Middle + Thumb

                # index pinch indicator (for play/pause/next)
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                middle_tip = hand_landmarks[12]

                thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))

                # index+thumb pinch (play/pause/next)
                index_color = (0, 255, 0) if is_pinch else (0, 0, 255)
                cv2.line(frame, thumb_pos, index_pos, index_color, 3)
                cv2.circle(frame, thumb_pos, 12, index_color, -1)
                cv2.circle(frame, index_pos, 12, index_color, -1)

                # middle+thumb pinch (volume control)
                middle_color = (255, 255, 0) if is_middle_pinch else (100, 100, 100)
                cv2.line(frame, thumb_pos, middle_pos, middle_color, 2)
                cv2.circle(frame, middle_pos, 10, middle_color, -1)

                # calculate midpoint for volume control (middle finger)
                pinch_midpoint_y = (thumb_pos[1] + middle_pos[1]) // 2

                # volume control with middle finger pinch
                if is_middle_pinch:
                    # check if pinch is held for volume control
                    current_time = time.time()
                    if not self.is_controlling_volume:
                        self.is_controlling_volume = True
                        self.pinch_start_y = pinch_midpoint_y
                        self.volume_start = self.volume

                    # update volume based on y axis movement
                    self.update_volume(pinch_midpoint_y)

                    cv2.putText(frame, f"VOLUME: {int(self.volume * 100)}%", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                elif is_pinch:
                    cv2.putText(frame, "PINCH DETECTED!", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # reset volume control when hand is not detected
                if self.is_controlling_volume:
                    self.is_controlling_volume = False
                    self.pinch_start_y = None

            # reset volume control when middle pinch is released
            if self.is_controlling_volume and not is_middle_pinch:
                self.is_controlling_volume = False
                self.pinch_start_y = None

            # detect gesture with index pinch
            gesture_count = 0
            if not self.is_controlling_volume and not is_middle_pinch:
                gesture_count = self.detect_gesture(is_pinch)

            if gesture_count == 1:
                self.play_pause()
            elif gesture_count == 2:
                self.next_track()
            elif gesture_count >= 3:
                self.previous_track()

            # status display
            status = "Playing" if self.is_playing else "Paused"

            if self.playlist:
                current_song = os.path.basename(self.playlist[self.current_track_index])
                next_idx = (self.current_track_index + 1) % len(self.playlist)
                next_song = os.path.basename(self.playlist[next_idx])
            else:
                current_song = "No songs"
                next_song = "-"

            # left side - Status
            cv2.rectangle(frame, (5, 5), (200, 95), (0, 0, 0), -1)
            cv2.putText(frame, f"Status: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Volume: {int(self.volume * 100)}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Pinches: {self.consecutive_pinches}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # bottom - Current n Next track
            cv2.rectangle(frame, (5, h - 50), (w - 5, h - 5), (0, 0, 0), -1)
            cv2.putText(frame, "Now Playing:", (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(frame, current_song[:60], (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            (text_w, _), _ = cv2.getTextSize(next_song[:60], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            right_x = w - text_w - 10
            cv2.putText(frame, "Next:", (right_x, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(frame, next_song[:60], (right_x, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Gesture Music Player', frame)

            # quit on 'q' press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Cleanup
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        self.detector.close()
        print("Music player stopped!")


if __name__ == "__main__":
    music_folder = "./music"

    # create music folder if it doesn't exist
    if not os.path.exists(music_folder):
        os.makedirs(music_folder)
        print(f"Created music folder at: {os.path.abspath(music_folder)}")

    try:
        player = GestureMusicPlayer(music_folder)
        player.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()