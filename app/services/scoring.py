import time

class ScoringService:
    def __init__(self):
        pass

    def calculate_attention(self, yaw, pitch):
        """
        Calculate Attention Score (0-100) based on head pose.
        Center = 100
        Extreme Side/Up/Down = 0
        """
        # Yaw penalty: 0.6 is threshold for "Side". 
        # Map 0 -> 1.0 (score 100), 0.8 -> 0.0 (score 0)
        yaw_score = max(0, 100 - (abs(yaw) / 0.8 * 100))
        
        # Pitch penalty: 
        # Normal 0.2-0.8. 
        # > 1.0 (Down) -> penalty. < 0.2 (Up) -> penalty.
        pitch_score = 100
        if pitch > 0.8:
            pitch_score = max(0, 100 - ((pitch - 0.8) / 0.5 * 100)) # Looking down
        elif pitch < 0.3:
            pitch_score = max(0, 100 - ((0.3 - pitch) / 0.3 * 100)) # Looking up
            
        return (yaw_score * 0.7 + pitch_score * 0.3)

    def calculate_posture(self, keypoints):
        """
        Calculate Posture Score (0-100).
        Based on shoulder levelness and spine straightness (nose to mid-shoulder).
        """
        if len(keypoints) < 7:
            return 0
            
        # 5: L-Shoulder, 6: R-Shoulder
        l_sh = keypoints[5]
        r_sh = keypoints[6]
        
        # Shoulder levelness (y-diff should be small)
        if l_sh[0] == 0 or r_sh[0] == 0:
            return 50 # Unknown
            
        y_diff = abs(l_sh[1] - r_sh[1])
        width = abs(l_sh[0] - r_sh[0])
        
        if width == 0: return 50
        
        tilt_ratio = y_diff / width
        # Ratio 0 -> 100, Ratio 0.5 -> 0
        posture_score = max(0, 100 - (tilt_ratio * 200))
        
        return posture_score

    def calculate_confidence(self, keypoints, emotion):
        """
        Calculate Confidence Score (0-100).
        Open posture + Positive/Neutral emotion.
        """
        score = 70 # Base
        
        # Emotion modifier
        if emotion in ['happy', 'neutral']:
            score += 20
        elif emotion in ['sad', 'fear']:
            score -= 20
            
        return min(100, max(0, score))

    def calculate_engagement(self, attention_score, emotion):
        """
        Engagement combines attention and emotional state.
        """
        score = attention_score * 0.7
        
        if emotion in ['happy', 'surprise']:
            score += 30
        elif emotion == 'neutral':
            score += 10
        elif emotion in ['bored', 'sad']: # 'bored' not standard deepface, but logic holds
            score -= 20
            
        return min(100, max(0, score))
