# akshayakalpa_sales_agent_final.py
import os
import json
import asyncio
import base64
import pyaudio
import wave
import tempfile
import audioop
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
import pygame
import io
import requests
import time

load_dotenv()

# Initialize DeepSeek LLM
deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Sarvam API key
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_API_URL = "https://api.sarvam.ai"

# Initialize pygame mixer for audio playback
pygame.mixer.init(frequency=22050)

class AkshayakalpaSalesAgent:
    def __init__(self):
        # Audio settings
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.p = pyaudio.PyAudio()
        
        # Voice activity detection
        self.silence_threshold = 1908
        self.silence_duration = 2.0
        self.max_duration = 20
        
        # Conversation state
        self.current_language = None
        self.call_active = True
        self.current_lang_code = "en-IN"
        
        # Lead capture state
        self.capturing_lead = False
        self.lead_data = {
            "name": None,
            "phone": None,
            "email": None,
            "city": None,
            "interested_products": [],
            "preferred_delivery_time": None,
            "heard_about": None,
            "timestamp": None,
            "language": None
        }
        self.lead_stage = "greeting"
        
        # Language mapping
        self.language_map = {
            "english": "en-IN",
            "tamil": "ta-IN",
            "hindi": "hi-IN",
            "kannada": "kn-IN",
            "telugu": "te-IN",
            "malayalam": "ml-IN"
        }
        
        # Number to language mapping
        self.number_to_language = {
            "1": "english",
            "2": "kannada",
            "3": "hindi",
            "4": "telugu",
            "5": "tamil"
        }
        
        # Word to number mapping
        self.word_to_number = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5"
        }
        
        # Speaker mapping for TTS
        self.speaker_map = {
            "english": "aditya",
            "tamil": "aditya",
            "hindi": "shubh",
            "kannada": "aditya",
            "telugu": "aditya",
            "malayalam": "aditya"
        }
        
        # Language selection menu with numbers only
        self.language_selection_prompt = """
        Please select your language by saying the number:
        
        1 - English
        2 - Kannada
        3 - Hindi
        4 - Telugu
        5 - Tamil
        
        Say 1, 2, 3, 4, or 5
        """
        
        self.greetings = {
            "kannada": "ನಮಸ್ಕಾರ! ಅಕ್ಷಯಕಲ್ಪ ಆರ್ಗಾನಿಕ್‍ಗೆ ಸುಸ್ವಾಗತ. ನಾವು ಭಾರತದ ಪ್ರೀಮಿಯಂ ಸಾವಯವ ಡೈರಿ ಬ್ರಾಂಡ್. ನಮ್ಮ ಹಸುಗಳು ಹುಲ್ಲು ತಿನ್ನುತ್ತವೆ ಮತ್ತು ಹಾರ್ಮೋನ್ ಮುಕ್ತವಾಗಿವೆ. ನಮ್ಮ ಉತ್ಪನ್ನಗಳ ಬಗ್ಗೆ ಇನ್ನಷ್ಟು ತಿಳಿದುಕೊಳ್ಳಲು ಮತ್ತು ಹೋಮ್ ಡೆಲಿವರಿ ಪ್ರಾರಂಭಿಸಲು ನೀವು ಬಯಸುವಿರಾ?",
            "telugu": "నమస్కారం! అక్షయకల్ప ఆర్గానిక్‌కు స్వాగతం. మేము భారతదేశంలోని ప్రీమియం ఆర్గానిక్ డైరీ బ్రాండ్. మా ఆవులు గడ్డి తింటాయి మరియు హార్మోన్ రహితంగా ఉంటాయి. మా ఉత్పత్తుల గురించి మరింత తెలుసుకోవడానికి మరియు హోమ్ డెలివరీ ప్రారంభించడానికి మీరు కోరుకుంటున్నారా?",
            "tamil": "வணக்கம்! அக்ஷயகல்பா ஆர்கானிக்கிற்கு வருக. நாங்கள் இந்தியாவின் பிரீமியம் ஆர்கானிக் பால் பிராண்ட். எங்கள் மாடுகள் புல் மேய்கின்றன மற்றும் ஹார்மோன் இல்லாதவை. எங்கள் தயாரிப்புகளைப் பற்றி மேலும் அறியவும், வீட்டு டெலிவரியைத் தொடங்கவும் விரும்புகிறீர்களா?",
            "english": "Hello! Welcome to Akshayakalpa Organic. We're India's premium organic dairy brand. Our cows are grass-fed and hormone-free. Would you like to learn more about our products and get started with home delivery?",
            "hindi": "नमस्ते! अक्षयकल्प ऑर्गेनिक में आपका स्वागत है। हम भारत का प्रीमियम ऑर्गेनिक डेयरी ब्रांड हैं। हमारी गायें घास खाती हैं और हार्मोन-मुक्त हैं। क्या आप हमारे उत्पादों के बारे में अधिक जानना चाहते हैं और होम डिलीवरी शुरू करना चाहते हैं?"
        }
        
        self.prompts = {
            "ask_name": {
                "kannada": "ಅದ್ಭುತ! ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹೆಸರನ್ನು ಹೇಳಬಹುದೇ?",
                "telugu": "అద్భుతం! దయచేసి మీ పేరు చెప్పగలరా?",
                "tamil": "அருமை! தயவுசெய்து உங்கள் பெயரை சொல்ல முடியுமா?",
                "english": "Wonderful! Could you please tell me your name?",
                "hindi": "बहुत अच्छा! कृपया अपना नाम बता सकते हैं?"
            },
            "ask_phone": {
                "kannada": "ಧನ್ಯವಾದಗಳು {name}! ದಯವಿಟ್ಟು ನಿಮ್ಮ 10-ಅಂಕೆಯ ಮೊಬೈಲ್ ಸಂಖ್ಯೆಯನ್ನು ಹಂಚಿಕೊಳ್ಳಬಹುದೇ? ಅಥವಾ ಹಂಚಿಕೊಳ್ಳಲು ಇಷ್ಟವಿಲ್ಲದಿದ್ದರೆ 'skip' ಎಂದು ಹೇಳಿ.",
                "telugu": "ధన్యవాదాలు {name}! దయచేసి మీ 10-అంకెల మొబైల్ నంబర్‌ను షేర్ చేయగలరా? లేదా షేర్ చేయకూడదనుకుంటే 'skip' చెప్పండి.",
                "tamil": "நன்றி {name}! தயவுசெய்து உங்கள் 10 இலக்க மொபைல் எண்ணைப் பகிர முடியுமா? அல்லது பகிர விரும்பவில்லை என்றால் 'skip' என்று சொல்லுங்கள்.",
                "english": "Thank you {name}! Could you please share your 10-digit mobile number? Or say 'skip' if you prefer not to share.",
                "hindi": "धन्यवाद {name}! कृपया अपना 10-अंकीय मोबाइल नंबर साझा करें? या 'skip' कहें यदि आप साझा नहीं करना चाहते।"
            },
            "ask_email": {
                "kannada": "ಉತ್ತಮ! ನಿಮ್ಮ ಇಮೇಲ್ ವಿಳಾಸ ಯಾವುದು? ಅಥವಾ 'skip' ಎಂದು ಹೇಳಿ.",
                "telugu": "గ్రేట్! మీ ఇమెయిల్ చిరునామా ఏమిటి? లేదా 'skip' చెప్పండి.",
                "tamil": "அருமை! உங்கள் மின்னஞ்சல் முகவரி என்ன? அல்லது 'skip' என்று சொல்லுங்கள்.",
                "english": "Great! What's your email address? Or say 'skip' to continue.",
                "hindi": "बढ़िया! आपका ईमेल पता क्या है? या 'skip' कहें।"
            },
            "ask_city": {
                "kannada": "ಧನ್ಯವಾದಗಳು! ನೀವು ಯಾವ ನಗರದಲ್ಲಿ ವಾಸಿಸುತ್ತೀರಿ?",
                "telugu": "ధన్యవాదాలు! మీరు ఏ నగరంలో నివసిస్తున్నారు?",
                "tamil": "நன்றி! நீங்கள் எந்த நகரத்தில் வசிக்கிறீர்கள்?",
                "english": "Thanks! Which city do you live in?",
                "hindi": "धन्यवाद! आप किस शहर में रहते हैं?"
            },
            "ask_products": {
                "kannada": "ಪರಿಪೂರ್ಣ! ನಮ್ಮಲ್ಲಿ ಈ ಉತ್ಪನ್ನಗಳಿವೆ: ಎ2 ಹಾಲು (₹53/ಲೀಟರ್), ಎ1 ಹಾಲು (₹45/ಲೀಟರ್), ಮೊಸರು (₹40/500g), ತುಪ್ಪ (₹450/500ml), ಪನೀರ್ (₹80/200g). ನೀವು ಯಾವ ಉತ್ಪನ್ನಗಳಲ್ಲಿ ಆಸಕ್ತಿ ಹೊಂದಿದ್ದೀರಿ?",
                "telugu": "పర్ఫెక్ట్! మాకు ఈ ఉత్పత్తులు ఉన్నాయి: ఎ2 పాలు (₹53/లీటర్), ఎ1 పాలు (₹45/లీటర్), పెరుగు (₹40/500g), నెయ్యి (₹450/500ml), పనీర్ (₹80/200g). మీకు ఏ ఉత్పత్తులపై ఆసక్తి ఉంది?",
                "tamil": "சரியானது! எங்களிடம் இந்த தயாரிப்புகள் உள்ளன: ஏ2 பால் (₹53/லிட்டர்), ஏ1 பால் (₹45/லிட்டர்), தயிர் (₹40/500g), நெய் (₹450/500ml), பன்னீர் (₹80/200g). நீங்கள் எந்த தயாரிப்புகளில் ஆர்வமாக உள்ளீர்கள்?",
                "english": "Perfect! We have these products: A2 Milk (₹53/litre), A1 Milk (₹45/litre), Curd (₹40/500g), Ghee (₹450/500ml), Paneer (₹80/200g). Which products are you interested in?",
                "hindi": "बिल्कुल सही! हमारे पास ये उत्पाद हैं: ए2 दूध (₹53/लीटर), ए1 दूध (₹45/लीटर), दही (₹40/500g), घी (₹450/500ml), पनीर (₹80/200g). आप किन उत्पादों में रुचि रखते हैं?"
            },
            "ask_delivery": {
                "kannada": "ಉತ್ತಮ ಆಯ್ಕೆ! ನೀವು ಯಾವಾಗ ಡೆಲಿವರಿ ಬಯಸುತ್ತೀರಿ? ಬೆಳಿಗ್ಗೆ (6-9 AM) ಅಥವಾ ಸಂಜೆ (4-7 PM)?",
                "telugu": "గ్రేట్ ఛాయిస్! మీరు ఎప్పుడు డెలివరీ కోరుకుంటారు? ఉదయం (6-9 AM) లేదా సాయంత్రం (4-7 PM)?",
                "tamil": "அருமையான தேர்வு! நீங்கள் எப்போது டெலிவரி விரும்புகிறீர்கள்? காலை (6-9 AM) அல்லது மாலை (4-7 PM)?",
                "english": "Great choice! When would you prefer delivery? Morning (6-9 AM) or Evening (4-7 PM)?",
                "hindi": "बढ़िया विकल्प! आप डिलीवरी कब पसंद करेंगे? सुबह (6-9 AM) या शाम (4-7 PM)?"
            },
            "ask_referral": {
                "kannada": "ಕೊನೆಯ ಪ್ರಶ್ನೆ - ಅಕ್ಷಯಕಲ್ಪ ಬಗ್ಗೆ ನಿಮಗೆ ಹೇಗೆ ತಿಳಿಯಿತು? (ಸ್ನೇಹಿತ/ಗೂಗಲ್/ಸೋಶಿಯಲ್ ಮೀಡಿಯಾ/ಇತರೆ)",
                "telugu": "చివరి ప్రశ్న - అక్షయకల్ప గురించి మీకు ఎలా తెలిసింది? (స్నేహితుడు/గూగుల్/సోషల్ మీడియా/ఇతర)",
                "tamil": "கடைசி கேள்வி - அக்ஷயகல்பா பற்றி உங்களுக்கு எப்படி தெரியும்? (நண்பர்/கூகுள்/சோஷியல் மீடியா/மற்றவை)",
                "english": "One last question - how did you hear about Akshayakalpa? (Friend/Google/Social Media/Other)",
                "hindi": "आखिरी सवाल - आपको अक्षयकल्प के बारे में कैसे पता चला? (दोस्त/गूगल/सोशल मीडिया/अन्य)"
            },
            "thankyou": {
                "kannada": "ತುಂಬಾ ಧನ್ಯವಾದಗಳು {name}! {products} ನಲ್ಲಿ ನಿಮ್ಮ ಆಸಕ್ತಿಯನ್ನು ನಾನು ಗಮನಿಸಿದ್ದೇನೆ. ನಮ್ಮ ತಂಡವು ಶೀಘ್ರದಲ್ಲೇ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತದೆ. ಅಕ್ಷಯಕಲ್ಪ ಆರ್ಗಾನಿಕ್ ಕುಟುಂಬಕ್ಕೆ ಸುಸ್ವಾಗತ! 🌱",
                "telugu": "చాలా ధన్యవాదాలు {name}! {products} పై మీ ఆసక్తిని నేను గమనించాను. మా బృందం త్వరలో మిమ్మల్ని సంప్రదిస్తుంది. అక్షయకల్ప ఆర్గానిక్ కుటుంబానికి స్వాగతం! 🌱",
                "tamil": "மிக்க நன்றி {name}! {products} இல் உங்கள் ஆர்வத்தை நான் கவனித்தேன். எங்கள் குழு விரைவில் உங்களைத் தொடர்பு கொள்ளும். அக்ஷயகல்பா ஆர்கானிக் குடும்பத்திற்கு வரவேற்கிறோம்! 🌱",
                "english": "Thank you so much {name}! I've noted your interest in {products}. Our team will contact you soon. Welcome to Akshayakalpa Organic family! 🌱",
                "hindi": "बहुत-बहुत धन्यवाद {name}! मैंने {products} में आपकी रुचि नोट कर ली है। हमारी टीम जल्द ही आपसे संपर्क करेगी। अक्षयकल्प ऑर्गेनिक परिवार में आपका स्वागत है! 🌱"
            }
        }
        
        self.products = {
            "A2 Milk": {"price": "₹53 per litre", "kannada": "ಎ2 ಹಾಲು", "telugu": "ఎ2 పాలు", "tamil": "ஏ2 பால்", "hindi": "ए2 दूध"},
            "A1 Milk": {"price": "₹45 per litre", "kannada": "ಎ1 ಹಾಲು", "telugu": "ఎ1 పాలు", "tamil": "ஏ1 பால்", "hindi": "ए1 दूध"},
            "Curd": {"price": "₹40 for 500g, ₹75 for 1kg", "kannada": "ಮೊಸರು", "telugu": "పెరుగు", "tamil": "தயிர்", "hindi": "दही"},
            "Ghee": {"price": "₹450 for 500ml", "kannada": "ತುಪ್ಪ", "telugu": "నెయ్యి", "tamil": "நெய்", "hindi": "घी"},
            "Paneer": {"price": "₹80 for 200g", "kannada": "ಪನೀರ್", "telugu": "పనీర్", "tamil": "பன்னீர்", "hindi": "पनीर"}
        }
        
        self.leads_file = "customer_leads.json"
        
        print("=" * 60)
        print("🐄 Akshayakalpa Organic - Sales Agent")
        print("=" * 60)
        print("🌐 Choose your language by saying the number:")
        print("   1 - English")
        print("   2 - Kannada")
        print("   3 - Hindi")
        print("   4 - Telugu")
        print("   5 - Tamil")
        print("=" * 60)
    
    async def speak(self, text, language):
        """Text to speech using Sarvam HTTP API"""
        print(f"\n🤖 Agent ({language}): {text}")
        
        try:
            lang_code = self.language_map.get(language, "en-IN")
            speaker = self.speaker_map.get(language, "aditya")
            
            url = f"{SARVAM_API_URL}/text-to-speech/stream"
            headers = {
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "text": text,
                "target_language_code": lang_code,
                "speaker": speaker,
                "model": "bulbul:v3",
                "output_audio_codec": "mp3",
                "output_audio_bitrate": "128k",
                "pace": 1.0,
                "enable_preprocessing": True
            }
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    audio_data = await response.read()
            
            audio_buffer = io.BytesIO(audio_data)
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            
            pygame.mixer.music.unload()
            
        except Exception as e:
            print(f"TTS Error: {e}")
    
    async def transcribe_audio(self, audio_file_path):
        """Transcribe audio using Sarvam REST API - WORKING VERSION"""
        try:
            url = f"{SARVAM_API_URL}/speech-to-text"
            headers = {
                "api-subscription-key": SARVAM_API_KEY
            }
            
            with open(audio_file_path, 'rb') as f:
                files = {
                    'file': (os.path.basename(audio_file_path), f, 'audio/wav')
                }
                data = {
                    'model': 'saaras:v3',
                    'language_code': self.current_lang_code or 'en-IN'
                }
                
                response = requests.post(url, headers=headers, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    transcript = result.get('transcript', '')
                    return transcript.strip()
                else:
                    print(f"STT API Error: {response.status_code} - {response.text}")
                    return ""
                    
        except Exception as e:
            print(f"STT Error: {e}")
            return ""
    
    async def record_and_transcribe(self):
        """Record audio and transcribe"""
        
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("\n🎤 Listening... (speak now)")
        
        frames = []
        silent_chunks = 0
        started_speaking = False
        silence_chunks_needed = int(self.silence_duration * self.rate / self.chunk)
        
        for i in range(int(self.max_duration * self.rate / self.chunk)):
            try:
                data = stream.read(self.chunk)
                frames.append(data)
                
                rms = audioop.rms(data, 2)
                
                if rms < self.silence_threshold:
                    if started_speaking:
                        silent_chunks += 1
                        if silent_chunks > silence_chunks_needed:
                            print("🔇 Silence detected")
                            break
                else:
                    if not started_speaking:
                        started_speaking = True
                        print("🎙️ Recording started...")
                    silent_chunks = 0
                    
            except Exception as e:
                print(f"Audio error: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        
        if not started_speaking:
            print("⚠️ No speech detected")
            return ""
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        wf = wave.open(temp_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        await asyncio.sleep(0.1)
        
        transcript = await self.transcribe_audio(temp_path)
        
        try:
            os.unlink(temp_path)
        except:
            pass
        
        if transcript:
            print(f"📝 Transcribed: {transcript}")
        else:
            print("⚠️ No transcription received")
        
        return transcript
    
    def save_lead(self, lead_data):
        """Save lead to JSON"""
        try:
            if os.path.exists(self.leads_file):
                with open(self.leads_file, 'r', encoding='utf-8') as f:
                    leads = json.load(f)
            else:
                leads = []
            
            lead_data["timestamp"] = datetime.now().isoformat()
            lead_data["lead_id"] = f"LEAD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            leads.append(lead_data)
            
            with open(self.leads_file, 'w', encoding='utf-8') as f:
                json.dump(leads, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Lead saved: {lead_data['lead_id']}")
            return lead_data["lead_id"]
            
        except Exception as e:
            print(f"Save Error: {e}")
            return None
    
    def extract_phone_number(self, text):
        """Extract phone number"""
        cleaned = re.sub(r'[\s\-\(\)]', '', text)
        pattern = r'[6-9]\d{9}'
        match = re.search(pattern, cleaned)
        if match:
            return match.group()
        return None
    
    def should_skip(self, text):
        """Check if user wants to skip"""
        skip_words = ["skip", "dont want", "not comfortable", "later", "no thanks"]
        return any(word in text.lower() for word in skip_words)
    
    def extract_email(self, text):
        """Extract email"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(pattern, text)
        return match.group() if match else None
    
    def extract_city(self, text):
        """Extract city"""
        cities = ["bangalore", "bengaluru", "mumbai", "delhi", "chennai", "hyderabad", "pune", "coimbatore", "mysore"]
        text_lower = text.lower()
        for city in cities:
            if city in text_lower:
                return city.title()
        return None
    
    async def process_with_llm_intent(self, user_input):
        """Use LLM for intent detection"""
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an intent classifier. Determine if user is interested in learning about Akshayakalpa products. Respond with ONLY 'yes' or 'no'."},
                    {"role": "user", "content": user_input}
                ],
                temperature=0,
                max_tokens=10
            )
            intent = response.choices[0].message.content.strip().lower()
            return intent == "yes"
        except:
            return any(word in user_input.lower() for word in ["yes", "yeah", "sure", "interested", "tell me", "know more"])
    
    async def select_language(self):
        """Language selection using numbered menu - FIXED NUMBER DETECTION"""
        
        await self.speak(self.language_selection_prompt, "english")
        
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            user_text = await self.record_and_transcribe()
            
            if not user_text:
                attempts += 1
                await self.speak(f"Please say a number 1, 2, 3, 4, or 5. Attempt {attempts} of {max_attempts}.", "english")
                continue
            
            user_text_lower = user_text.lower()
            print(f"📝 Language selection: '{user_text}'")
            
            # Try to extract number from the text
            selected_number = None
            
            # Check if it's a digit
            if user_text in ["1", "2", "3", "4", "5"]:
                selected_number = user_text
            # Check if it's a word like "one", "two", etc.
            elif user_text_lower in self.word_to_number:
                selected_number = self.word_to_number[user_text_lower]
            # Try to find any number in the text
            else:
                numbers = re.findall(r'\d+', user_text)
                if numbers:
                    selected_number = numbers[0]
            
            if selected_number and selected_number in self.number_to_language:
                self.current_language = self.number_to_language[selected_number]
                break
            else:
                attempts += 1
                await self.speak(f"Invalid choice. Please say 1 for English, 2 for Kannada, 3 for Hindi, 4 for Telugu, or 5 for Tamil. Attempt {attempts} of {max_attempts}.", "english")
        
        if not self.current_language:
            print("⚠️ Could not detect language, defaulting to English")
            self.current_language = "english"
        
        self.current_lang_code = self.language_map[self.current_language]
        print(f"\n🔒 Language locked: {self.current_language}")
        
        await self.speak(self.greetings[self.current_language], self.current_language)
    
    async def process_lead_capture(self, user_input):
        """Process lead capture"""
        
        user_input_lower = user_input.lower()
        
        if self.lead_stage == "greeting":
            is_interested = await self.process_with_llm_intent(user_input)
            
            if is_interested:
                self.lead_stage = "asking_name"
                self.lead_data["language"] = self.current_language
                return self.prompts["ask_name"][self.current_language]
            else:
                return None
        
        elif self.lead_stage == "asking_name":
            self.lead_data["name"] = user_input.title()
            self.lead_stage = "asking_phone"
            prompt = self.prompts["ask_phone"][self.current_language]
            return prompt.format(name=self.lead_data["name"])
        
        elif self.lead_stage == "asking_phone":
            if self.should_skip(user_input):
                self.lead_data["phone"] = "Not provided"
                self.lead_stage = "asking_email"
                return self.prompts["ask_email"][self.current_language]
            
            phone = self.extract_phone_number(user_input)
            if phone and len(phone) == 10:
                self.lead_data["phone"] = phone
                self.lead_stage = "asking_email"
                return self.prompts["ask_email"][self.current_language]
            else:
                self.lead_data["phone"] = user_input if user_input else "Not provided"
                self.lead_stage = "asking_email"
                return self.prompts["ask_email"][self.current_language]
        
        elif self.lead_stage == "asking_email":
            if self.should_skip(user_input):
                self.lead_data["email"] = "Not provided"
                self.lead_stage = "asking_city"
                return self.prompts["ask_city"][self.current_language]
            
            email = self.extract_email(user_input)
            if email:
                self.lead_data["email"] = email
                self.lead_stage = "asking_city"
                return self.prompts["ask_city"][self.current_language]
            else:
                self.lead_data["email"] = user_input if user_input else "Not provided"
                self.lead_stage = "asking_city"
                return self.prompts["ask_city"][self.current_language]
        
        elif self.lead_stage == "asking_city":
            city = self.extract_city(user_input)
            if city:
                self.lead_data["city"] = city
                self.lead_stage = "asking_products"
                return self.prompts["ask_products"][self.current_language]
            else:
                self.lead_data["city"] = user_input if user_input else "Not specified"
                self.lead_stage = "asking_products"
                return self.prompts["ask_products"][self.current_language]
        
        elif self.lead_stage == "asking_products":
            interested = []
            for product in self.products.keys():
                product_local = self.products[product].get(self.current_language, product)
                if product.lower() in user_input_lower or product_local.lower() in user_input_lower:
                    interested.append(product)
            
            if interested:
                self.lead_data["interested_products"] = interested
                self.lead_stage = "asking_delivery_time"
                return self.prompts["ask_delivery"][self.current_language]
            else:
                self.lead_data["interested_products"] = [user_input] if user_input else []
                self.lead_stage = "asking_delivery_time"
                return self.prompts["ask_delivery"][self.current_language]
        
        elif self.lead_stage == "asking_delivery_time":
            if "morning" in user_input_lower:
                self.lead_data["preferred_delivery_time"] = "Morning (6-9 AM)"
            elif "evening" in user_input_lower:
                self.lead_data["preferred_delivery_time"] = "Evening (4-7 PM)"
            else:
                self.lead_data["preferred_delivery_time"] = user_input if user_input else "Not specified"
            
            self.lead_stage = "asking_referral"
            return self.prompts["ask_referral"][self.current_language]
        
        elif self.lead_stage == "asking_referral":
            self.lead_data["heard_about"] = user_input if user_input else "Not specified"
            
            self.save_lead(self.lead_data)
            
            self.lead_stage = "greeting"
            self.capturing_lead = False
            
            products_summary = ", ".join(self.lead_data["interested_products"]) if self.lead_data["interested_products"] else "our products"
            
            thankyou = self.prompts["thankyou"][self.current_language]
            return thankyou.format(name=self.lead_data["name"], products=products_summary)
        
        return None
    
    async def run(self):
        """Main loop"""
        
        print("\n" + "=" * 60)
        print("🐄 Welcome to Akshayakalpa Organic")
        print("=" * 60)
        print("🌐 Choose your language by saying the number:")
        print("   1 - English")
        print("   2 - Kannada")
        print("   3 - Hindi")
        print("   4 - Telugu")
        print("   5 - Tamil")
        print("=" * 60)
        
        # Select language using numbered menu
        await self.select_language()
        
        waiting_for_response = True
        
        while self.call_active:
            try:
                user_text = await self.record_and_transcribe()
                
                if not user_text:
                    await self.speak("I didn't hear you. Please say yes or no.", self.current_language)
                    continue
                
                print(f"\n📝 User: {user_text}")
                
                if user_text.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                    farewell = "Thank you for your time! Visit akshayakalpa.com to learn more. Have a great day!"
                    await self.speak(farewell, self.current_language)
                    break
                
                if waiting_for_response:
                    is_interested = await self.process_with_llm_intent(user_text)
                    
                    if is_interested:
                        waiting_for_response = False
                        self.capturing_lead = True
                        self.lead_stage = "asking_name"
                        
                        response = self.prompts["ask_name"][self.current_language]
                        await self.speak(response, self.current_language)
                        continue
                    else:
                        response = "No problem! Visit our website to learn more. Thank you!"
                        await self.speak(response, self.current_language)
                        break
                
                if self.capturing_lead:
                    response = await self.process_lead_capture(user_text)
                    
                    if response:
                        await self.speak(response, self.current_language)
                        
                        if not self.capturing_lead:
                            await asyncio.sleep(1)
                            break
                
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue

async def main():
    """Entry point"""
    if not os.getenv("SARVAM_API_KEY"):
        print("❌ SARVAM_API_KEY not found in .env file")
        return
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ DEEPSEEK_API_KEY not found in .env file")
        return
    
    print("✅ API keys found!")
    
    agent = AkshayakalpaSalesAgent()
    await agent.run()

if __name__ == "__main__":
    try:
        import aiohttp
    except ImportError:
        print("Installing aiohttp...")
        os.system("pip install aiohttp")
        import aiohttp
    
    asyncio.run(main())