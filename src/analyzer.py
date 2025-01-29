from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import cv2
import base64
import time
from datetime import datetime
import os
import mediapipe as mp
import numpy as np

class OutfitAnalyzerChain:
    def __init__(self, api_key):

        self.cap = cv2.VideoCapture(0)
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4-turbo",
            max_tokens=500,
            temperature=0.7
        )
        
        self.system_prompt = SystemMessage(
            content="""You are a professional fashion consultant. Analyze outfits and provide detailed feedback.
            Focus on:
            1. what I am wearing
            2. What you suggest
            4. Color harmony analysis"""
        )
        
        self.analysis_prompt = PromptTemplate(
            input_variables=["image_b64"],
            template="""Please analyze this outfit in detail:
            {image_b64}
            
            Provide your analysis in the following structure:
            1. Current Style:
            2. Suggested Improvements:
            3. Recommended Accessories:
            4. Color Combination Analysis:"""
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.analysis_prompt,
            verbose=True
        )
        
        # Gesture detection state
        self.last_gesture_time = 0
        self.gesture_cooldown = 2  # Cooldown in seconds

    def detect_peace_sign(self, hand_landmarks):
        """Detect peace sign gesture"""
        if hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            
            # Check if index and middle fingers are up, others are down
            if (index_tip.y < hand_landmarks.landmark[6].y and  # Index up
                middle_tip.y < hand_landmarks.landmark[10].y and  # Middle up
                ring_tip.y > hand_landmarks.landmark[14].y and  # Ring down
                pinky_tip.y > hand_landmarks.landmark[18].y):  # Pinky down
                return True
        return False

    def capture_frame(self):
        """Capture a frame and process hand gestures"""
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture frame")
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hand landmarks
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks on frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Check for peace sign gesture
                current_time = time.time()
                if (self.detect_peace_sign(hand_landmarks) and 
                    current_time - self.last_gesture_time > self.gesture_cooldown):
                    self.last_gesture_time = current_time
                    return frame, True
        
        return frame, False

    def process_frame(self, frame):
        """Process the frame for vision API"""
        # Resize if needed
        max_dim = 2048
        height, width = frame.shape[:2]
        if height > max_dim or width > max_dim:
            scale = max_dim / max(height, width)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        
        # Convert to jpg format
        _, buffer = cv2.imencode('.jpg', frame)
        # Convert to base64
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        return image_b64

    def analyze_outfit(self, image_b64):
        """Analyze outfit using LangChain"""
        try:
            # Create vision message
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": self.analysis_prompt.format(image_b64="")
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            )
            
            # Get response using LangChain
            response = self.llm.invoke([self.system_prompt, message])
            return response.content
            
        except Exception as e:
            print(f"Error in analysis chain: {e}")
            return None

    def save_analysis(self, frame, analysis, timestamp):
        """Save the captured image and analysis"""
        # Create results directory if it doesn't exist
        os.makedirs("outfit_results", exist_ok=True)
        
        # Save image
        image_path = f"outfit_results/outfit_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        
        # Save analysis
        analysis_path = f"outfit_results/analysis_{timestamp}.txt"
        with open(analysis_path, "w") as f:
            f.write(analysis)
        
        return image_path, analysis_path

    def run(self):
        """Main loop for the outfit analyzer"""
        try:
            print("Starting Outfit Analyzer...")
            print("Make a peace sign ✌️ to capture and analyze outfit")
            print("Press Q to quit")
            
            while True:
                # Show live feed with gesture detection
                frame, gesture_detected = self.capture_frame()
                
                # Display frame
                cv2.imshow('Outfit Analyzer - Make peace sign ✌️ to capture, Q to quit', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif gesture_detected:
                    print("\nPeace sign detected! Capturing and analyzing outfit...")
                    
                    # Process frame and get analysis
                    image_b64 = self.process_frame(frame)
                    analysis = self.analyze_outfit(image_b64)
                    
                    if analysis:
                        # Save results
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path, analysis_path = self.save_analysis(frame, analysis, timestamp)
                        
                        print("\nOutfit Analysis:")
                        print(analysis)
                        print(f"\nResults saved to:")
                        print(f"Image: {image_path}")
                        print(f"Analysis: {analysis_path}")
                        print("\nMake peace sign ✌️ to analyze again, Q to quit")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()