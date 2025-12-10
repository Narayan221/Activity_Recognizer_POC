import numpy as np

class ScoringService:
    def __init__(self):
        pass

    def calculate_attention(self, yaw, pitch):
        """
        Calculate Attention Score (0-100) based on head pose.
        Enhanced with smoother scoring curve.
        
        Args:
            yaw: Head yaw angle ratio
            pitch: Head pitch angle ratio
            
        Returns:
            Attention score (0-100)
        """
        # Yaw scoring with smoother curve
        # abs(yaw) < 0.3 = excellent (90-100)
        # abs(yaw) 0.3-0.6 = good (60-90)
        # abs(yaw) > 0.6 = poor (0-60)
        yaw_abs = abs(yaw)
        if yaw_abs < 0.3:
            yaw_score = 100 - (yaw_abs / 0.3 * 10)  # 90-100
        elif yaw_abs < 0.6:
            yaw_score = 90 - ((yaw_abs - 0.3) / 0.3 * 30)  # 60-90
        else:
            yaw_score = max(0, 60 - ((yaw_abs - 0.6) / 0.4 * 60))  # 0-60
        
        # Pitch scoring with optimal range
        # pitch 0.3-0.8 = good (looking forward)
        # pitch > 0.8 = looking down (penalty)
        # pitch < 0.3 = looking up (penalty)
        if 0.3 <= pitch <= 0.8:
            pitch_score = 100
        elif pitch > 0.8:
            pitch_score = max(0, 100 - ((pitch - 0.8) / 0.5 * 100))
        else:  # pitch < 0.3
            pitch_score = max(0, 100 - ((0.3 - pitch) / 0.3 * 100))
            
        # Weighted combination (yaw is more important)
        return yaw_score * 0.7 + pitch_score * 0.3

    def calculate_posture(self, keypoints):
        """
        Enhanced Posture Score (0-100) using multiple keypoints.
        Considers shoulder levelness, spine alignment, and body symmetry.
        
        Args:
            keypoints: Array of pose keypoints
            
        Returns:
            Posture score (0-100)
        """
        if len(keypoints) < 12:
            return 50  # Not enough keypoints
            
        # Keypoint indices: 0=Nose, 5=L-Shoulder, 6=R-Shoulder, 11=L-Hip, 12=R-Hip
        nose = keypoints[0]
        l_sh = keypoints[5]
        r_sh = keypoints[6]
        l_hip = keypoints[11] if len(keypoints) > 11 else None
        r_hip = keypoints[12] if len(keypoints) > 12 else None
        
        scores = []
        
        # 1. Shoulder levelness (most important)
        if l_sh[0] > 0 and r_sh[0] > 0:
            y_diff = abs(l_sh[1] - r_sh[1])
            width = abs(l_sh[0] - r_sh[0])
            
            if width > 0:
                tilt_ratio = y_diff / width
                # Ratio < 0.1 = excellent, 0.1-0.3 = good, > 0.3 = poor
                if tilt_ratio < 0.1:
                    shoulder_score = 100
                elif tilt_ratio < 0.3:
                    shoulder_score = 100 - ((tilt_ratio - 0.1) / 0.2 * 30)  # 70-100
                else:
                    shoulder_score = max(0, 70 - ((tilt_ratio - 0.3) / 0.2 * 70))  # 0-70
                scores.append(shoulder_score)
        
        # 2. Spine alignment (nose to mid-shoulder line)
        if nose[0] > 0 and l_sh[0] > 0 and r_sh[0] > 0:
            mid_shoulder_x = (l_sh[0] + r_sh[0]) / 2
            mid_shoulder_y = (l_sh[1] + r_sh[1]) / 2
            
            # Check if nose is roughly centered above shoulders
            horizontal_offset = abs(nose[0] - mid_shoulder_x)
            shoulder_width = abs(l_sh[0] - r_sh[0])
            
            if shoulder_width > 0:
                offset_ratio = horizontal_offset / shoulder_width
                # Offset < 0.2 = centered, > 0.5 = leaning
                spine_score = max(0, 100 - (offset_ratio / 0.5 * 100))
                scores.append(spine_score)
        
        # 3. Hip levelness (if available)
        if l_hip is not None and r_hip is not None and l_hip[0] > 0 and r_hip[0] > 0:
            hip_y_diff = abs(l_hip[1] - r_hip[1])
            hip_width = abs(l_hip[0] - r_hip[0])
            
            if hip_width > 0:
                hip_tilt = hip_y_diff / hip_width
                hip_score = max(0, 100 - (hip_tilt / 0.3 * 100))
                scores.append(hip_score * 0.5)  # Lower weight for hips
        
        # Return weighted average
        return sum(scores) / len(scores) if scores else 50

    def calculate_confidence(self, keypoints, emotion):
        """
        Enhanced Confidence Score (0-100).
        Based on open posture, body language, and emotional state.
        
        Args:
            keypoints: Array of pose keypoints
            emotion: Detected emotion string
            
        Returns:
            Confidence score (0-100)
        """
        base_score = 60
        
        # Emotion modifier (stronger impact)
        emotion_bonus = 0
        if emotion in ['happy', 'neutral']:
            emotion_bonus = 25
        elif emotion == 'surprise':
            emotion_bonus = 15
        elif emotion in ['sad', 'fear', 'angry']:
            emotion_bonus = -30
        
        # Posture openness (shoulders)
        if len(keypoints) > 6:
            l_sh = keypoints[5]
            r_sh = keypoints[6]
            
            if l_sh[0] > 0 and r_sh[0] > 0:
                shoulder_width = abs(l_sh[0] - r_sh[0])
                # Wider shoulders = more open posture = more confidence
                # This is relative, but we can use it as a bonus
                if shoulder_width > 50:  # Arbitrary threshold
                    base_score += 10
        
        final_score = base_score + emotion_bonus
        return min(100, max(0, final_score))

    def calculate_engagement(self, attention_score, emotion):
        """
        Enhanced Engagement Score (0-100).
        Combines attention with emotional engagement indicators.
        
        Args:
            attention_score: Pre-calculated attention score
            emotion: Detected emotion string
            
        Returns:
            Engagement score (0-100)
        """
        # Base from attention (primary factor)
        score = attention_score * 0.6
        
        # Emotion modifier (engagement-specific)
        if emotion in ['happy', 'surprise']:
            # Highly engaged emotions
            score += 35
        elif emotion == 'neutral':
            # Moderately engaged
            score += 20
        elif emotion == 'angry':
            # Engaged but negatively
            score += 10
        elif emotion in ['sad', 'fear']:
            # Disengaged emotions
            score -= 15
        
        return min(100, max(0, score))
    
    def calculate_temporal_consistency(self, scores_timeline):
        """
        Calculate consistency score based on temporal variance.
        Lower variance = more consistent behavior.
        
        Args:
            scores_timeline: List of scores over time
            
        Returns:
            Consistency score (0-10)
        """
        if not scores_timeline or len(scores_timeline) < 2:
            return 5.0
        
        variance = np.var(scores_timeline)
        # Map variance to consistency score (inverse relationship)
        # Low variance (0-100) -> high consistency (10-8)
        # High variance (>500) -> low consistency (<5)
        consistency = max(0, min(10, 10 - (variance / 100)))
        return round(consistency, 2)
