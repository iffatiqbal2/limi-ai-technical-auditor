"""
NLP Classifier for Limi Installation Reports
Classifies condition as Optimal, Needs Maintenance, or Critical
"""

import re
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class LimiConditionClassifier:
    """
    NLP-based classifier for installation condition
    Uses keyword matching with weighted scoring
    """
    
    def __init__(self):
        # Define condition keywords with weights
        self.keywords = {
            'optimal': {
                'positive': ['optimal', 'excellent', 'good', 'stable', 'normal', 
                            'functioning', 'within parameters', 'peak', 'efficient',
                            'nominal', 'satisfactory', 'perfect', 'strong', 'success'],
                'weight': 1.0
            },
            'maintenance': {
                'positive': ['degraded', 'unstable', 'fluctuating', 'noise', 'check',
                            'maintenance', 'service', 'attention', 'warning', 'caution',
                            'degrading', 'intermittent', 'unreliable', 'repair'],
                'weight': 1.5  # Higher weight for maintenance indicators
            },
            'critical': {
                'positive': ['failed', 'offline', 'critical', 'emergency', 'down',
                            'error', 'fault', 'alarm', 'immediate', 'urgent',
                            'non-functional', 'broken', 'damaged', 'overheating', 'stop'],
                'weight': 2.0  # Highest weight for critical issues
            }
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep words
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def classify(self, text: str, signal_strength: str = None) -> Dict:
        """
        Classify based on weighted keyword matching
        Returns: Dictionary with condition and confidence
        """
        logger.info("Classifying report condition...")
        
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        # Initialize scores
        scores = {'optimal': 0, 'maintenance': 0, 'critical': 0}
        matched_keywords = []
        
        # Check signal strength if provided
        if signal_strength:
            try:
                signal_val = int(re.search(r'\d+', signal_strength).group())
                if signal_val < 50:
                    scores['critical'] += 10
                    matched_keywords.append(f"signal:critical({signal_val}%)")
                elif signal_val < 80:
                    scores['maintenance'] += 5
                    matched_keywords.append(f"signal:maintenance({signal_val}%)")
                else:
                    scores['optimal'] += 3
                    matched_keywords.append(f"signal:optimal({signal_val}%)")
            except:
                pass
        
        # Keyword matching
        for condition, data in self.keywords.items():
            for keyword in data['positive']:
                if keyword in processed_text:
                    scores[condition] += data['weight']
                    matched_keywords.append(f"{condition}:{keyword}")
        
        # Determine winner
        max_score = max(scores.values())
        if max_score == 0:
            condition = "Unknown"
            confidence = 0.0
        else:
            # Get condition with highest score
            condition = max(scores, key=scores.get).capitalize()
            
            # Calculate confidence (normalized)
            total = sum(scores.values())
            confidence = scores[condition.lower()] / total if total > 0 else 0
        
        # Capitalize properly
        if condition.lower() == 'maintenance':
            condition = "Needs Maintenance"
        
        result = {
            'condition': condition,
            'confidence': round(confidence, 2),
            'signal_strength': signal_strength,
            'scores': scores,
            'keywords_matched': matched_keywords[:10]  # Return first 10
        }
        
        logger.info(f"Classification result: {condition} (confidence: {confidence:.2f})")
        return result


# Integration with OCR
class LimiAIPipeline:
    """Complete AI pipeline combining OCR and NLP"""
    
    def __init__(self):
        from ocr_pipeline import LimiOCRExtractor
        self.ocr = LimiOCRExtractor()
        self.classifier = LimiConditionClassifier()
    
    def process_report(self, file_path: str) -> Dict:
        """
        Complete pipeline: OCR + NLP classification
        """
        logger.info(f"Processing report: {file_path}")
        
        # Step 1: OCR extraction
        text = self.ocr.extract_text(file_path)
        
        # Step 2: Extract structured data
        extracted = self.ocr.extract_structured_data(text)
        
        # Step 3: Classify condition
        classification = self.classifier.classify(
            text, 
            signal_strength=extracted.get('signal_strength')
        )
        
        # Combine results
        result = {
            **extracted,
            'condition': classification['condition'],
            'confidence': classification['confidence'],
            'timestamp': datetime.now().isoformat()
        }
        
        return result


# Demo
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 LIMI AI NLP CLASSIFIER")
    print("="*60)
    
    # Test the classifier
    classifier = LimiConditionClassifier()
    
    # Test with different texts
    test_cases = [
        ("Module operating optimally with excellent signal strength", "98%"),
        ("Signal degradation detected, maintenance recommended", "75%"),
        ("Critical failure - module offline, immediate attention required", "45%"),
    ]
    
    for text, signal in test_cases:
        print(f"\n📝 Text: {text[:50]}...")
        result = classifier.classify(text, signal)
        print(f"✅ Condition: {result['condition']}")
        print(f"📊 Confidence: {result['confidence']}")
        print(f"🔑 Keywords: {', '.join(result['keywords_matched'][:3])}")
    
    # Try with the sample report
    print("\n" + "="*60)
    print("📄 Analyzing Sample Report")
    print("="*60)
    
    try:
        with open("sample_reports/report1.txt", 'r') as f:
            sample_text = f.read()
        
        result = classifier.classify(sample_text, "98%")
        print(f"Condition: {result['condition']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Signal: {result['signal_strength']}")
        
        # Save results
        with open("classification_result.json", 'w') as f:
            json.dump(result, f, indent=2)
        print("\n📁 Results saved to: classification_result.json")
        
    except FileNotFoundError:
        print("❌ Sample report not found")