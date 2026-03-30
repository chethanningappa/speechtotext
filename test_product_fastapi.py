import os
import json
import asyncio
import base64
import pyaudio
import wave
import audioop
import re
import uuid
from datetime import datetime
from typing import Optional, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import AsyncOpenAI
import aiohttp
import io
import threading
import queue
import time

load_dotenv()

# Initialize DeepSeek LLM
deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Sarvam API
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_API_URL = "https://api.sarvam.ai"

# ============== Conversation Manager ==============

class ConversationSession:
    """Manages a single conversation session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.stage = "language_selection"  # language_selection, greeting, asking_name, etc.
        self.language = None
        self.lead_data = {
            "name": None,
            "phone": None,
            "email": None,
            "city": None,
            "interested_products": [],
            "preferred_delivery_time": None,
            "heard_about": None,
            "timestamp": None
        }
        self.created_at = datetime.now().isoformat()
        self.is_active = True
        
        # Audio streaming
        self.audio_buffer = bytearray()
        self.silence_counter = 0
        self.is_speaking = False
        
        # Audio settings
        self.sample_rate = 16000
        self.silence_threshold = 500
        self.silence_duration = 1.5  # seconds
        
        # Response queue
        self.response_queue = asyncio.Queue()
    
    def detect_silence(self, audio_chunk):
        """Detect if audio chunk is silence"""
        try:
            rms = audioop.rms(audio_chunk, 2)
            return rms < self.silence_threshold
        except:
            return True
    
    async def process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio chunk"""
        self.audio_buffer.extend(audio_data)
        
        # Check for silence
        is_silent = self.detect_silence(audio_data)
        
        if self.is_speaking:
            if is_silent:
                self.silence_counter += len(audio_data) / (self.sample_rate * 2)
                if self.silence_counter >= self.silence_duration:
                    # Speech ended - process the audio
                    await self.process_speech()
            else:
                self.silence_counter = 0
        else:
            if not is_silent:
                self.is_speaking = True
                self.silence_counter = 0
    
    async def process_speech(self):
        """Process complete speech segment"""
        if len(self.audio_buffer) < 16000:  # Too short
            self.audio_buffer = bytearray()
            self.is_speaking = False
            return
        
        # Save audio to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            wf = wave.open(tmp.name, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.audio_buffer)
            wf.close()
            
            # Transcribe
            transcript = await self.transcribe_audio(tmp.name)
            
            # Clean up
            os.unlink(tmp.name)
        
        # Clear buffer
        self.audio_buffer = bytearray()
        self.is_speaking = False
        
        if transcript:
            # Process the transcript and get response
            response = await self.process_input(transcript)
            await self.response_queue.put(response)
    
    async def transcribe_audio(self, audio_path):
        """Transcribe using Sarvam API"""
        try:
            url = f"{SARVAM_API_URL}/speech-to-text"
            headers = {"api-subscription-key": SARVAM_API_KEY}
            
            with open(audio_path, 'rb') as f:
                files = {'file': (os.path.basename(audio_path), f, 'audio/wav')}
                data = {'model': 'saaras:v3', 'language_code': 'auto'}
                
                import requests
                response = requests.post(url, headers=headers, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('transcript', '').strip()
        except Exception as e:
            print(f"STT Error: {e}")
        return ""
    
    async def text_to_speech(self, text):
        """Convert text to speech"""
        try:
            url = f"{SARVAM_API_URL}/text-to-speech/stream"
            headers = {
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "text": text,
                "target_language_code": self.get_language_code(),
                "speaker": self.get_speaker(),
                "model": "bulbul:v3",
                "output_audio_codec": "mp3",
                "output_audio_bitrate": "128k",
                "pace": 1.0,
                "enable_preprocessing": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    return await response.read()
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    
    def get_language_code(self):
        """Get Sarvam language code"""
        lang_map = {
            "english": "en-IN",
            "kannada": "kn-IN",
            "hindi": "hi-IN",
            "telugu": "te-IN",
            "tamil": "ta-IN"
        }
        return lang_map.get(self.language, "en-IN")
    
    def get_speaker(self):
        """Get speaker voice"""
        speaker_map = {
            "english": "aditya",
            "kannada": "aditya",
            "hindi": "shubh",
            "telugu": "aditya",
            "tamil": "aditya"
        }
        return speaker_map.get(self.language, "aditya")
    
    async def process_input(self, user_input: str) -> Dict:
        """Process user input and generate response"""
        
        user_input_lower = user_input.lower()
        
        # Language Selection Stage
        if self.stage == "language_selection":
            return await self.handle_language_selection(user_input)
        
        # Lead Capture Stages
        elif self.stage == "greeting":
            return await self.handle_greeting(user_input)
        elif self.stage == "asking_name":
            return await self.handle_name(user_input)
        elif self.stage == "asking_phone":
            return await self.handle_phone(user_input)
        elif self.stage == "asking_email":
            return await self.handle_email(user_input)
        elif self.stage == "asking_city":
            return await self.handle_city(user_input)
        elif self.stage == "asking_products":
            return await self.handle_products(user_input)
        elif self.stage == "asking_delivery_time":
            return await self.handle_delivery_time(user_input)
        elif self.stage == "asking_referral":
            return await self.handle_referral(user_input)
        
        return {"response": "I didn't understand. Please try again.", "stage": self.stage}
    
    async def handle_language_selection(self, user_input):
        """Handle language selection"""
        number_map = {
            "1": "english", "one": "english",
            "2": "kannada", "two": "kannada",
            "3": "hindi", "three": "hindi",
            "4": "telugu", "four": "telugu",
            "5": "tamil", "five": "tamil"
        }
        
        user_input_lower = user_input.lower()
        language = number_map.get(user_input_lower)
        
        # Try to extract number
        if not language:
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                language = number_map.get(numbers[0])
        
        if language:
            self.language = language
            self.stage = "greeting"
            
            # Greetings in selected language
            greetings = {
                "english": "Hello! Welcome to Akshayakalpa Organic. We're India's premium organic dairy brand. Our cows are grass-fed and hormone-free. Would you like to learn more about our products and get started with home delivery?",
                "kannada": "ನಮಸ್ಕಾರ! ಅಕ್ಷಯಕಲ್ಪ ಆರ್ಗಾನಿಕ್‍ಗೆ ಸುಸ್ವಾಗತ. ನಾವು ಭಾರತದ ಪ್ರೀಮಿಯಂ ಸಾವಯವ ಡೈರಿ ಬ್ರಾಂಡ್. ನಮ್ಮ ಹಸುಗಳು ಹುಲ್ಲು ತಿನ್ನುತ್ತವೆ ಮತ್ತು ಹಾರ್ಮೋನ್ ಮುಕ್ತವಾಗಿವೆ. ನಮ್ಮ ಉತ್ಪನ್ನಗಳ ಬಗ್ಗೆ ಇನ್ನಷ್ಟು ತಿಳಿದುಕೊಳ್ಳಲು ಮತ್ತು ಹೋಮ್ ಡೆಲಿವರಿ ಪ್ರಾರಂಭಿಸಲು ನೀವು ಬಯಸುವಿರಾ?",
                "hindi": "नमस्ते! अक्षयकल्प ऑर्गेनिक में आपका स्वागत है। हम भारत का प्रीमियम ऑर्गेनिक डेयरी ब्रांड हैं। हमारी गायें घास खाती हैं और हार्मोन-मुक्त हैं। क्या आप हमारे उत्पादों के बारे में अधिक जानना चाहते हैं और होम डिलीवरी शुरू करना चाहते हैं?",
                "telugu": "నమస్కారం! అక్షయకల్ప ఆర్గానిక్‌కు స్వాగతం. మేము భారతదేశంలోని ప్రీమియం ఆర్గానిక్ డైరీ బ్రాండ్. మా ఆవులు గడ్డి తింటాయి మరియు హార్మోన్ రహితంగా ఉంటాయి. మా ఉత్పత్తుల గురించి మరింత తెలుసుకోవడానికి మరియు హోమ్ డెలివరీ ప్రారంభించడానికి మీరు కోరుకుంటున్నారా?",
                "tamil": "வணக்கம்! அக்ஷயகல்பா ஆர்கானிக்கிற்கு வருக. நாங்கள் இந்தியாவின் பிரீமியம் ஆர்கானிக் பால் பிராண்ட். எங்கள் மாடுகள் புல் மேய்கின்றன மற்றும் ஹார்மோன் இல்லாதவை. எங்கள் தயாரிப்புகளைப் பற்றி மேலும் அறியவும், வீட்டு டெலிவரியைத் தொடங்கவும் விரும்புகிறீர்களா?"
            }
            
            return {
                "response": greetings[language],
                "stage": self.stage,
                "language": self.language
            }
        else:
            return {
                "response": "Please say 1 for English, 2 for Kannada, 3 for Hindi, 4 for Telugu, or 5 for Tamil.",
                "stage": "language_selection"
            }
    
    async def handle_greeting(self, user_input):
        """Handle initial greeting response"""
        if any(word in user_input.lower() for word in ["yes", "yeah", "sure", "interested", "tell me", "know more"]):
            self.stage = "asking_name"
            prompts = {
                "english": "Wonderful! Could you please tell me your name?",
                "kannada": "ಅದ್ಭುತ! ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹೆಸರನ್ನು ಹೇಳಬಹುದೇ?",
                "hindi": "बहुत अच्छा! कृपया अपना नाम बता सकते हैं?",
                "telugu": "అద్భుతం! దయచేసి మీ పేరు చెప్పగలరా?",
                "tamil": "அருமை! தயவுசெய்து உங்கள் பெயரை சொல்ல முடியுமா?"
            }
            return {
                "response": prompts[self.language],
                "stage": self.stage
            }
        else:
            return {
                "response": "No problem! Visit our website to learn more. Thank you!",
                "stage": "complete",
                "is_complete": True
            }
    
    async def handle_name(self, user_input):
        """Handle name input"""
        self.lead_data["name"] = user_input.title()
        self.stage = "asking_phone"
        
        prompts = {
            "english": f"Thank you {self.lead_data['name']}! Could you please share your 10-digit mobile number? Or say skip if you prefer not to share.",
            "kannada": f"ಧನ್ಯವಾದಗಳು {self.lead_data['name']}! ದಯವಿಟ್ಟು ನಿಮ್ಮ 10-ಅಂಕೆಯ ಮೊಬೈಲ್ ಸಂಖ್ಯೆಯನ್ನು ಹಂಚಿಕೊಳ್ಳಬಹುದೇ? ಅಥವಾ ಹಂಚಿಕೊಳ್ಳಲು ಇಷ್ಟವಿಲ್ಲದಿದ್ದರೆ skip ಎಂದು ಹೇಳಿ.",
            "hindi": f"धन्यवाद {self.lead_data['name']}! कृपया अपना 10-अंकीय मोबाइल नंबर साझा करें? या skip कहें यदि आप साझा नहीं करना चाहते।",
            "telugu": f"ధన్యవాదాలు {self.lead_data['name']}! దయచేసి మీ 10-అంకెల మొబైల్ నంబర్‌ను షేర్ చేయగలరా? లేదా షేర్ చేయకూడదనుకుంటే skip చెప్పండి.",
            "tamil": f"நன்றி {self.lead_data['name']}! தயவுசெய்து உங்கள் 10 இலக்க மொபைல் எண்ணைப் பகிர முடியுமா? அல்லது பகிர விரும்பவில்லை என்றால் skip என்று சொல்லுங்கள்."
        }
        
        return {
            "response": prompts[self.language],
            "stage": self.stage,
            "lead_data": self.lead_data
        }
    
    def extract_phone(self, text):
        """Extract phone number"""
        cleaned = re.sub(r'[\s\-\(\)]', '', text)
        pattern = r'[6-9]\d{9}'
        match = re.search(pattern, cleaned)
        return match.group() if match else None
    
    def should_skip(self, text):
        """Check if user wants to skip"""
        return any(word in text.lower() for word in ["skip", "dont want", "not comfortable", "later"])
    
    async def handle_phone(self, user_input):
        """Handle phone input"""
        if self.should_skip(user_input):
            self.lead_data["phone"] = "Not provided"
            self.stage = "asking_email"
        else:
            phone = self.extract_phone(user_input)
            if phone and len(phone) == 10:
                self.lead_data["phone"] = phone
            else:
                self.lead_data["phone"] = user_input
            self.stage = "asking_email"
        
        prompts = {
            "english": "Great! What's your email address? Or say skip to continue.",
            "kannada": "ಉತ್ತಮ! ನಿಮ್ಮ ಇಮೇಲ್ ವಿಳಾಸ ಯಾವುದು? ಅಥವಾ skip ಎಂದು ಹೇಳಿ.",
            "hindi": "बढ़िया! आपका ईमेल पता क्या है? या skip कहें।",
            "telugu": "గ్రేట్! మీ ఇమెయిల్ చిరునామా ఏమిటి? లేదా skip చెప్పండి.",
            "tamil": "அருமை! உங்கள் மின்னஞ்சல் முகவரி என்ன? அல்லது skip என்று சொல்லுங்கள்."
        }
        
        return {
            "response": prompts[self.language],
            "stage": self.stage,
            "lead_data": self.lead_data
        }
    
    def extract_email(self, text):
        """Extract email"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(pattern, text)
        return match.group() if match else None
    
    async def handle_email(self, user_input):
        """Handle email input"""
        if self.should_skip(user_input):
            self.lead_data["email"] = "Not provided"
        else:
            email = self.extract_email(user_input)
            self.lead_data["email"] = email if email else user_input
        self.stage = "asking_city"
        
        prompts = {
            "english": "Thanks! Which city do you live in?",
            "kannada": "ಧನ್ಯವಾದಗಳು! ನೀವು ಯಾವ ನಗರದಲ್ಲಿ ವಾಸಿಸುತ್ತೀರಿ?",
            "hindi": "धन्यवाद! आप किस शहर में रहते हैं?",
            "telugu": "ధన్యవాదాలు! మీరు ఏ నగరంలో నివసిస్తున్నారు?",
            "tamil": "நன்றி! நீங்கள் எந்த நகரத்தில் வசிக்கிறீர்கள்?"
        }
        
        return {
            "response": prompts[self.language],
            "stage": self.stage,
            "lead_data": self.lead_data
        }
    
    def extract_city(self, text):
        """Extract city"""
        cities = ["bangalore", "bengaluru", "mumbai", "delhi", "chennai", "hyderabad", "pune"]
        text_lower = text.lower()
        for city in cities:
            if city in text_lower:
                return city.title()
        return None
    
    async def handle_city(self, user_input):
        """Handle city input"""
        city = self.extract_city(user_input)
        self.lead_data["city"] = city if city else user_input
        self.stage = "asking_products"
        
        prompts = {
            "english": "Perfect! We have these products: A2 Milk (₹53/litre), A1 Milk (₹45/litre), Curd (₹40/500g), Ghee (₹450/500ml), Paneer (₹80/200g). Which products are you interested in?",
            "kannada": "ಪರಿಪೂರ್ಣ! ನಮ್ಮಲ್ಲಿ ಈ ಉತ್ಪನ್ನಗಳಿವೆ: ಎ2 ಹಾಲು (₹53/ಲೀಟರ್), ಎ1 ಹಾಲು (₹45/ಲೀಟರ್), ಮೊಸರು (₹40/500g), ತುಪ್ಪ (₹450/500ml), ಪನೀರ್ (₹80/200g). ನೀವು ಯಾವ ಉತ್ಪನ್ನಗಳಲ್ಲಿ ಆಸಕ್ತಿ ಹೊಂದಿದ್ದೀರಿ?",
            "hindi": "बिल्कुल सही! हमारे पास ये उत्पाद हैं: ए2 दूध (₹53/लीटर), ए1 दूध (₹45/लीटर), दही (₹40/500g), घी (₹450/500ml), पनीर (₹80/200g). आप किन उत्पादों में रुचि रखते हैं?",
            "telugu": "పర్ఫెక్ట్! మాకు ఈ ఉత్పత్తులు ఉన్నాయి: ఎ2 పాలు (₹53/లీటర్), ఎ1 పాలు (₹45/లీటర్), పెరుగు (₹40/500g), నెయ్యి (₹450/500ml), పనీర్ (₹80/200g). మీకు ఏ ఉత్పత్తులపై ఆసక్తి ఉంది?",
            "tamil": "சரியானது! எங்களிடம் இந்த தயாரிப்புகள் உள்ளன: ஏ2 பால் (₹53/லிட்டர்), ஏ1 பால் (₹45/லிட்டர்), தயிர் (₹40/500g), நெய் (₹450/500ml), பன்னீர் (₹80/200g). நீங்கள் எந்த தயாரிப்புகளில் ஆர்வமாக உள்ளீர்கள்?"
        }
        
        return {
            "response": prompts[self.language],
            "stage": self.stage,
            "lead_data": self.lead_data
        }
    
    async def handle_products(self, user_input):
        """Handle product selection"""
        products = ["A2 Milk", "A1 Milk", "Curd", "Ghee", "Paneer"]
        interested = [p for p in products if p.lower() in user_input.lower()]
        
        if interested:
            self.lead_data["interested_products"] = interested
        else:
            self.lead_data["interested_products"] = [user_input]
        
        self.stage = "asking_delivery_time"
        
        prompts = {
            "english": "Great choice! When would you prefer delivery? Morning (6-9 AM) or Evening (4-7 PM)?",
            "kannada": "ಉತ್ತಮ ಆಯ್ಕೆ! ನೀವು ಯಾವಾಗ ಡೆಲಿವರಿ ಬಯಸುತ್ತೀರಿ? ಬೆಳಿಗ್ಗೆ (6-9 AM) ಅಥವಾ ಸಂಜೆ (4-7 PM)?",
            "hindi": "बढ़िया विकल्प! आप डिलीवरी कब पसंद करेंगे? सुबह (6-9 AM) या शाम (4-7 PM)?",
            "telugu": "గ్రేట్ ఛాయిస్! మీరు ఎప్పుడు డెలివరీ కోరుకుంటారు? ఉదయం (6-9 AM) లేదా సాయంత్రం (4-7 PM)?",
            "tamil": "அருமையான தேர்வு! நீங்கள் எப்போது டெலிவரி விரும்புகிறீர்கள்? காலை (6-9 AM) அல்லது மாலை (4-7 PM)?"
        }
        
        return {
            "response": prompts[self.language],
            "stage": self.stage,
            "lead_data": self.lead_data
        }
    
    async def handle_delivery_time(self, user_input):
        """Handle delivery time"""
        if "morning" in user_input.lower():
            self.lead_data["preferred_delivery_time"] = "Morning (6-9 AM)"
        elif "evening" in user_input.lower():
            self.lead_data["preferred_delivery_time"] = "Evening (4-7 PM)"
        else:
            self.lead_data["preferred_delivery_time"] = user_input
        
        self.stage = "asking_referral"
        
        prompts = {
            "english": "One last question - how did you hear about Akshayakalpa?",
            "kannada": "ಕೊನೆಯ ಪ್ರಶ್ನೆ - ಅಕ್ಷಯಕಲ್ಪ ಬಗ್ಗೆ ನಿಮಗೆ ಹೇಗೆ ತಿಳಿಯಿತು?",
            "hindi": "आखिरी सवाल - आपको अक्षयकल्प के बारे में कैसे पता चला?",
            "telugu": "చివరి ప్రశ్న - అక్షయకల్ప గురించి మీకు ఎలా తెలిసింది?",
            "tamil": "கடைசி கேள்வி - அக்ஷயகல்பா பற்றி உங்களுக்கு எப்படி தெரியும்?"
        }
        
        return {
            "response": prompts[self.language],
            "stage": self.stage,
            "lead_data": self.lead_data
        }
    
    async def handle_referral(self, user_input):
        """Handle referral and save lead"""
        self.lead_data["heard_about"] = user_input
        self.lead_data["timestamp"] = datetime.now().isoformat()
        self.lead_data["language"] = self.language
        
        # Save lead
        await self.save_lead(self.lead_data)
        
        products_summary = ", ".join(self.lead_data["interested_products"]) if self.lead_data["interested_products"] else "our products"
        
        prompts = {
            "english": f"Thank you so much {self.lead_data['name']}! I've noted your interest in {products_summary}. Our team will contact you soon. Welcome to Akshayakalpa Organic family!",
            "kannada": f"ತುಂಬಾ ಧನ್ಯವಾದಗಳು {self.lead_data['name']}! {products_summary} ನಲ್ಲಿ ನಿಮ್ಮ ಆಸಕ್ತಿಯನ್ನು ನಾನು ಗಮನಿಸಿದ್ದೇನೆ. ನಮ್ಮ ತಂಡವು ಶೀಘ್ರದಲ್ಲೇ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತದೆ. ಅಕ್ಷಯಕಲ್ಪ ಆರ್ಗಾನಿಕ್ ಕುಟುಂಬಕ್ಕೆ ಸುಸ್ವಾಗತ!",
            "hindi": f"बहुत-बहुत धन्यवाद {self.lead_data['name']}! मैंने {products_summary} में आपकी रुचि नोट कर ली है। हमारी टीम जल्द ही आपसे संपर्क करेगी। अक्षयकल्प ऑर्गेनिक परिवार में आपका स्वागत है!",
            "telugu": f"చాలా ధన్యవాదాలు {self.lead_data['name']}! {products_summary} పై మీ ఆసక్తిని నేను గమనించాను. మా బృందం త్వరలో మిమ్మల్ని సంప్రదిస్తుంది. అక్షయకల్ప ఆర్గానిక్ కుటుంబానికి స్వాగతం!",
            "tamil": f"மிக்க நன்றி {self.lead_data['name']}! {products_summary} இல் உங்கள் ஆர்வத்தை நான் கவனித்தேன். எங்கள் குழு விரைவில் உங்களைத் தொடர்பு கொள்ளும். அக்ஷயகல்பா ஆர்கானிக் குடும்பத்திற்கு வரவேற்கிறோம்!"
        }
        
        self.stage = "complete"
        
        return {
            "response": prompts[self.language],
            "stage": "complete",
            "lead_data": self.lead_data,
            "is_complete": True
        }
    
    async def save_lead(self, lead_data):
        """Save lead to JSON file"""
        leads_file = "leads.json"
        try:
            if os.path.exists(leads_file):
                with open(leads_file, 'r', encoding='utf-8') as f:
                    leads = json.load(f)
            else:
                leads = []
            
            lead_data["lead_id"] = f"LEAD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            leads.append(lead_data)
            
            with open(leads_file, 'w', encoding='utf-8') as f:
                json.dump(leads, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Save Error: {e}")

# ============== FastAPI App with WebSocket ==============

app = FastAPI(title="Akshayakalpa Voice Agent - Real-time Conversation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
sessions: Dict[str, ConversationSession] = {}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time voice conversation"""
    await websocket.accept()
    print(f"✅ Client connected: {session_id}")
    
    # Create or get session
    if session_id not in sessions:
        sessions[session_id] = ConversationSession(session_id)
    session = sessions[session_id]
    
    try:
        # Send initial welcome message
        welcome = {
            "type": "welcome",
            "message": "Please select your language by saying: 1 for English, 2 for Kannada, 3 for Hindi, 4 for Telugu, 5 for Tamil",
            "session_id": session_id
        }
        await websocket.send_json(welcome)
        
        # Audio streaming variables
        audio_buffer = bytearray()
        silence_counter = 0
        is_speaking = False
        sample_rate = 16000
        silence_threshold = 500
        silence_duration_samples = int(sample_rate * 1.5)  # 1.5 seconds
        
        # Main loop for receiving audio
        while True:
            # Receive message from client
            data = await websocket.receive_bytes()
            
            # Process as audio chunk (16-bit PCM)
            audio_buffer.extend(data)
            
            # Calculate RMS for silence detection
            try:
                rms = audioop.rms(data, 2)
            except:
                rms = 0
            
            if rms < silence_threshold:
                # Silence detected
                if is_speaking:
                    silence_counter += len(data) / (sample_rate * 2)
                    if silence_counter >= 1.5:  # 1.5 seconds of silence
                        # Speech ended - process the audio
                        if len(audio_buffer) > sample_rate * 0.5:  # At least 0.5 seconds
                            # Save audio to temp file
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                                wf = wave.open(tmp.name, 'wb')
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(sample_rate)
                                wf.writeframes(audio_buffer)
                                wf.close()
                                
                                # Transcribe
                                transcript = await session.transcribe_audio(tmp.name)
                                os.unlink(tmp.name)
                            
                            # Process transcript
                            if transcript:
                                print(f"📝 Transcript: {transcript}")
                                
                                # Process input
                                result = await session.process_input(transcript)
                                
                                # Generate TTS response
                                audio_response = await session.text_to_speech(result["response"])
                                
                                if audio_response:
                                    # Send audio response
                                    await websocket.send_bytes(audio_response)
                                    
                                    # Send text response as JSON
                                    await websocket.send_json({
                                        "type": "text",
                                        "text": result["response"],
                                        "stage": result["stage"],
                                        "is_complete": result.get("is_complete", False)
                                    })
                                    
                                    # If conversation complete, close session
                                    if result.get("is_complete"):
                                        await websocket.send_json({
                                            "type": "complete",
                                            "message": "Thank you for your interest! Goodbye!"
                                        })
                                        break
                            
                            # Reset buffers
                            audio_buffer = bytearray()
                            is_speaking = False
                            silence_counter = 0
            else:
                # Voice detected
                if not is_speaking:
                    is_speaking = True
                    silence_counter = 0
                    
    except WebSocketDisconnect:
        print(f"❌ Client disconnected: {session_id}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up session if complete
        if session_id in sessions and sessions[session_id].stage == "complete":
            del sessions[session_id]
        print(f"Session {session_id} closed")

@app.get("/")
async def root():
    return {
        "name": "Akshayakalpa Voice Agent",
        "version": "1.0.0",
        "websocket_endpoint": "/ws/{session_id}",
        "instructions": "Connect to WebSocket and stream 16-bit PCM audio"
    }

@app.get("/api/sessions")
async def get_sessions():
    """Get all active sessions"""
    return {
        "active_sessions": list(sessions.keys()),
        "count": len(sessions)
    }

@app.get("/api/leads")
async def get_leads():
    """Get all leads"""
    leads_file = "leads.json"
    if os.path.exists(leads_file):
        with open(leads_file, 'r', encoding='utf-8') as f:
            leads = json.load(f)
        return {"leads": leads, "count": len(leads)}
    return {"leads": [], "count": 0}

# ============== HTML Test Client ==============

@app.get("/test")
async def test_client():
    """Simple HTML test client for WebSocket"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Akshayakalpa Voice Agent Test</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }
            #status { padding: 10px; margin: 10px 0; background: #f0f0f0; border-radius: 5px; }
            #response { padding: 10px; margin: 10px 0; background: #e0f0e0; border-radius: 5px; white-space: pre-wrap; }
            .recording { background-color: red; color: white; }
        </style>
    </head>
    <body>
        <h1>🎤 Akshayakalpa Voice Agent</h1>
        
        <div>
            <button id="connectBtn">Connect</button>
            <button id="startBtn" disabled>Start Recording</button>
            <button id="stopBtn" disabled>Stop Recording</button>
            <button id="disconnectBtn" disabled>Disconnect</button>
        </div>
        
        <div id="status">Status: Disconnected</div>
        <div id="response">Responses will appear here...</div>
        
        <script>
            let ws = null;
            let mediaRecorder = null;
            let audioChunks = [];
            let sessionId = 'session_' + Date.now();
            
            const connectBtn = document.getElementById('connectBtn');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const disconnectBtn = document.getElementById('disconnectBtn');
            const statusDiv = document.getElementById('status');
            const responseDiv = document.getElementById('response');
            
            connectBtn.onclick = async () => {
                ws = new WebSocket(`ws://localhost:8000/ws/${sessionId}`);
                
                ws.onopen = () => {
                    statusDiv.innerHTML = 'Status: Connected ✅';
                    connectBtn.disabled = true;
                    startBtn.disabled = false;
                    disconnectBtn.disabled = false;
                    addResponse('Connected to agent');
                };
                
                ws.onmessage = (event) => {
                    if (typeof event.data === 'string') {
                        const data = JSON.parse(event.data);
                        if (data.type === 'welcome') {
                            addResponse(data.message);
                        } else if (data.type === 'text') {
                            addResponse(`🤖 Agent: ${data.text}`);
                            if (data.is_complete) {
                                startBtn.disabled = true;
                                stopBtn.disabled = true;
                            }
                        }
                    } else {
                        // Audio response - play it
                        const audioBlob = new Blob([event.data], { type: 'audio/mpeg' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const audio = new Audio(audioUrl);
                        audio.play();
                    }
                };
                
                ws.onclose = () => {
                    statusDiv.innerHTML = 'Status: Disconnected ❌';
                    connectBtn.disabled = false;
                    startBtn.disabled = true;
                    stopBtn.disabled = true;
                    disconnectBtn.disabled = true;
                };
            };
            
            startBtn.onclick = async () => {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                        // Send audio chunk immediately
                        event.data.arrayBuffer().then(buffer => {
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                ws.send(buffer);
                            }
                        });
                    }
                };
                
                mediaRecorder.start(100); // Send chunks every 100ms
                startBtn.disabled = true;
                stopBtn.disabled = false;
                startBtn.classList.add('recording');
                addResponse('🎙️ Recording... Speak now');
            };
            
            stopBtn.onclick = () => {
                if (mediaRecorder) {
                    mediaRecorder.stop();
                    mediaRecorder.stream.getTracks().forEach(track => track.stop());
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    startBtn.classList.remove('recording');
                    addResponse('⏹️ Recording stopped');
                }
            };
            
            disconnectBtn.onclick = () => {
                if (ws) ws.close();
            };
            
            function addResponse(text) {
                responseDiv.innerHTML += `<div>${new Date().toLocaleTimeString()} - ${text}</div>`;
                responseDiv.scrollTop = responseDiv.scrollHeight;
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)