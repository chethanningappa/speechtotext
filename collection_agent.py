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
from sarvamai import AsyncSarvamAI
import pygame
import io

load_dotenv()

# Initialize Async clients
sarvam_client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Initialize pygame mixer
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
        self.conversation_history = []
        self.current_language = "kannada"  
        self.call_active = True
        self.current_lang_code = "kn-IN" 
        
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
        
        # Speaker mapping
        self.speaker_map = {
            "english": "aditya",
            "tamil": "aditya",
            "hindi": "shubh",
            "kannada": "aditya",
            "telugu": "aditya",
            "malayalam": "aditya"
        }
        
        # Multilingual greetings and prompts
        self.greetings = {
            "kannada": "ನಮಸ್ಕಾರ! ಅಕ್ಷಯಕಲ್ಪ ಆರ್ಗಾನಿಕ್‍ಗೆ ಸುಸ್ವಾಗತ. ನಾವು ಭಾರತದ ಪ್ರೀಮಿಯಂ ಸಾವಯವ ಡೈರಿ ಬ್ರಾಂಡ್. ನಮ್ಮ ಹಸುಗಳು ಹುಲ್ಲು ತಿನ್ನುತ್ತವೆ ಮತ್ತು ಹಾರ್ಮೋನ್ ಮುಕ್ತವಾಗಿವೆ. ನಮ್ಮ ಉತ್ಪನ್ನಗಳ ಬಗ್ಗೆ ಇನ್ನಷ್ಟು ತಿಳಿದುಕೊಳ್ಳಲು ಮತ್ತು ಹೋಮ್ ಡೆಲಿವರಿ ಪ್ರಾರಂಭಿಸಲು ನೀವು ಬಯಸುವಿರಾ?",
            "telugu": "నమస్కారం! అక్షయకల్ప ఆర్గానిక్‌కు స్వాగతం. మేము భారతదేశంలోని ప్రీమియం ఆర్గానిక్ డైరీ బ్రాండ్. మా ఆవులు గడ్డి తింటాయి మరియు హార్మోన్ రహితంగా ఉంటాయి. మా ఉత్పత్తుల గురించి మరింత తెలుసుకోవడానికి మరియు హోమ్ డెలివరీ ప్రారంభించడానికి మీరు కోరుకుంటున్నారా?",
            "tamil": "வணக்கம்! அக்ஷயகல்பா ஆர்கானிக்கிற்கு வருக. நாங்கள் இந்தியாவின் பிரீமியம் ஆர்கானிக் பால் பிராண்ட். எங்கள் மாடுகள் புல் மேய்கின்றன மற்றும் ஹார்மோன் இல்லாதவை. எங்கள் தயாரிப்புகளைப் பற்றி மேலும் அறியவும், வீட்டு டெலிவரியைத் தொடங்கவும் விரும்புகிறீர்களா?",
            "english": "Hello! Welcome to Akshayakalpa Organic. We're India's premium organic dairy brand. Our cows are grass-fed and hormone-free. Would you like to learn more about our products and get started with home delivery?"
        }
        
        self.ask_name_prompts = {
            "kannada": "ಅದ್ಭುತ! ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹೆಸರನ್ನು ಹೇಳಬಹುದೇ?",
            "telugu": "అద్భుతం! దయచేసి మీ పేరు చెప్పగలరా?",
            "tamil": "அருமை! தயவுசெய்து உங்கள் பெயரை சொல்ல முடியுமா?",
            "english": "Wonderful! Could you please tell me your name?"
        }
        
        self.ask_phone_prompts = {
            "kannada": "ಧನ್ಯವಾದಗಳು {name}! ದಯವಿಟ್ಟು ನಿಮ್ಮ 10-ಅಂಕೆಯ ಮೊಬೈಲ್ ಸಂಖ್ಯೆಯನ್ನು ಹಂಚಿಕೊಳ್ಳಬಹುದೇ? ಅಥವಾ ಹಂಚಿಕೊಳ್ಳಲು ಇಷ್ಟವಿಲ್ಲದಿದ್ದರೆ 'skip' ಎಂದು ಹೇಳಿ.",
            "telugu": "ధన్యవాదాలు {name}! దయచేసి మీ 10-అంకెల మొబైల్ నంబర్‌ను షేర్ చేయగలరా? లేదా షేర్ చేయకూడదనుకుంటే 'skip' చెప్పండి.",
            "tamil": "நன்றி {name}! தயவுசெய்து உங்கள் 10 இலக்க மொபைல் எண்ணைப் பகிர முடியுமா? அல்லது பகிர விரும்பவில்லை என்றால் 'skip' என்று சொல்லுங்கள்.",
            "english": "Thank you {name}! Could you please share your 10-digit mobile number? Or say 'skip' if you prefer not to share."
        }
        
        self.ask_email_prompts = {
            "kannada": "ಉತ್ತಮ! ನಿಮ್ಮ ಇಮೇಲ್ ವಿಳಾಸ ಯಾವುದು? ಅಥವಾ 'skip' ಎಂದು ಹೇಳಿ.",
            "telugu": "గ్రేట్! మీ ఇమెయిల్ చిరునామా ఏమిటి? లేదా 'skip' చెప్పండి.",
            "tamil": "அருமை! உங்கள் மின்னஞ்சல் முகவரி என்ன? அல்லது 'skip' என்று சொல்லுங்கள்.",
            "english": "Great! What's your email address? Or say 'skip' to continue."
        }
        
        self.ask_city_prompts = {
            "kannada": "ಧನ್ಯವಾದಗಳು! ನೀವು ಯಾವ ನಗರದಲ್ಲಿ ವಾಸಿಸುತ್ತೀರಿ?",
            "telugu": "ధన్యవాదాలు! మీరు ఏ నగరంలో నివసిస్తున్నారు?",
            "tamil": "நன்றி! நீங்கள் எந்த நகரத்தில் வசிக்கிறீர்கள்?",
            "english": "Thanks! Which city do you live in?"
        }
        
        self.ask_products_prompts = {
            "kannada": "ಪರಿಪೂರ್ಣ! ನಮ್ಮಲ್ಲಿ ಈ ಉತ್ಪನ್ನಗಳಿವೆ: A2 ಹಾಲು (₹53/ಲೀಟರ್), A1 ಹಾಲು (₹45/ಲೀಟರ್), ಮೊಸರು (₹40/500g), ತುಪ್ಪ (₹450/500ml), ಪನೀರ್ (₹80/200g). ನೀವು ಯಾವ ಉತ್ಪನ್ನಗಳಲ್ಲಿ ಆಸಕ್ತಿ ಹೊಂದಿದ್ದೀರಿ?",
            "telugu": "పర్ఫెక్ట్! మాకు ఈ ఉత్పత్తులు ఉన్నాయి: A2 పాలు (₹53/లీటర్), A1 పాలు (₹45/లీటర్), పెరుగు (₹40/500g), నెయ్యి (₹450/500ml), పనీర్ (₹80/200g). మీకు ఏ ఉత్పత్తులపై ఆసక్తి ఉంది?",
            "tamil": "சரியானது! எங்களிடம் இந்த தயாரிப்புகள் உள்ளன: A2 பால் (₹53/லிட்டர்), A1 பால் (₹45/லிட்டர்), தயிர் (₹40/500g), நெய் (₹450/500ml), பன்னீர் (₹80/200g). நீங்கள் எந்த தயாரிப்புகளில் ஆர்வமாக உள்ளீர்கள்?",
            "english": "Perfect! We have these products: A2 Milk (₹53/litre), A1 Milk (₹45/litre), Curd (₹40/500g), Ghee (₹450/500ml), Paneer (₹80/200g). Which products are you interested in?"
        }
        
        self.ask_delivery_prompts = {
            "kannada": "ಉತ್ತಮ ಆಯ್ಕೆ! ನೀವು ಯಾವಾಗ ಡೆಲಿವರಿ ಬಯಸುತ್ತೀರಿ? ಬೆಳಿಗ್ಗೆ (6-9 AM) ಅಥವಾ ಸಂಜೆ (4-7 PM)?",
            "telugu": "గ్రేట్ ఛాయిస్! మీరు ఎప్పుడు డెలివరీ కోరుకుంటారు? ఉదయం (6-9 AM) లేదా సాయంత్రం (4-7 PM)?",
            "tamil": "அருமையான தேர்வு! நீங்கள் எப்போது டெலிவரி விரும்புகிறீர்கள்? காலை (6-9 AM) அல்லது மாலை (4-7 PM)?",
            "english": "Great choice! When would you prefer delivery? Morning (6-9 AM) or Evening (4-7 PM)?"
        }
        
        self.ask_referral_prompts = {
            "kannada": "ಕೊನೆಯ ಪ್ರಶ್ನೆ - ಅಕ್ಷಯಕಲ್ಪ ಬಗ್ಗೆ ನಿಮಗೆ ಹೇಗೆ ತಿಳಿಯಿತು? (ಸ್ನೇಹಿತ/ಗೂಗಲ್/ಸೋಶಿಯಲ್ ಮೀಡಿಯಾ/ಇತರೆ)",
            "telugu": "చివరి ప్రశ్న - అక్షయకల్ప గురించి మీకు ఎలా తెలిసింది? (స్నేహితుడు/గూగుల్/సోషల్ మీడియా/ఇతర)",
            "tamil": "கடைசி கேள்வி - அக்ஷயகல்பா பற்றி உங்களுக்கு எப்படி தெரியும்? (நண்பர்/கூகுள்/சோஷியல் மீடியா/மற்றவை)",
            "english": "One last question - how did you hear about Akshayakalpa? (Friend/Google/Social Media/Other)"
        }
        
        self.thankyou_prompts = {
            "kannada": "ತುಂಬಾ ಧನ್ಯವಾದಗಳು {name}! {products} ನಲ್ಲಿ ನಿಮ್ಮ ಆಸಕ್ತಿಯನ್ನು ನಾನು ಗಮನಿಸಿದ್ದೇನೆ. ನಮ್ಮ ತಂಡವು ಶೀಘ್ರದಲ್ಲೇ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತದೆ. ಅಕ್ಷಯಕಲ್ಪ ಆರ್ಗಾನಿಕ್ ಕುಟುಂಬಕ್ಕೆ ಸುಸ್ವಾಗತ! 🌱",
            "telugu": "చాలా ధన్యవాదాలు {name}! {products} పై మీ ఆసక్తిని నేను గమనించాను. మా బృందం త్వరలో మిమ్మల్ని సంప్రదిస్తుంది. అక్షయకల్ప ఆర్గానిక్ కుటుంబానికి స్వాగతం! 🌱",
            "tamil": "மிக்க நன்றி {name}! {products} இல் உங்கள் ஆர்வத்தை நான் கவனித்தேன். எங்கள் குழு விரைவில் உங்களைத் தொடர்பு கொள்ளும். அக்ஷயகல்பா ஆர்கானிக் குடும்பத்திற்கு வரவேற்கிறோம்! 🌱",
            "english": "Thank you so much {name}! I've noted your interest in {products}. Our team will contact you soon. Welcome to Akshayakalpa Organic family! 🌱"
        }
        
        self.products = {
            "A2 Milk": {
                "price": "₹53 per litre",
                "kannada": "ಎ2 ಹಾಲು",
                "telugu": "ఎ2 పాలు",
                "tamil": "ஏ2 பால்",
                "description": "Premium organic milk from grass-fed cows"
            },
            "A1 Milk": {
                "price": "₹45 per litre",
                "kannada": "ಎ1 ಹಾಲು",
                "telugu": "ఎ1 పాలు",
                "tamil": "ஏ1 பால்",
                "description": "Fresh organic milk from happy cows"
            },
            "Curd": {
                "price": "₹40 for 500g, ₹75 for 1kg",
                "kannada": "ಮೊಸರು",
                "telugu": "పెరుగు",
                "tamil": "தயிர்",
                "description": "Probiotic-rich, traditional culture curd"
            },
            "Ghee": {
                "price": "₹450 for 500ml",
                "kannada": "ತುಪ್ಪ",
                "telugu": "నెయ్యి",
                "tamil": "நெய்",
                "description": "Pure cow ghee, slow-cooked"
            },
            "Paneer": {
                "price": "₹80 for 200g",
                "kannada": "ಪನೀರ್",
                "telugu": "పనీర్",
                "tamil": "பன்னீர்",
                "description": "Fresh, soft, and protein-rich paneer"
            }
        }
        
        self.leads_file = "customer_leads.json"
        
        print("=" * 60)
        print("🐄 Akshayakalpa Organic - Multilingual Sales Agent")
        print("=" * 60)
        print("🎤 Languages: ಕನ್ನಡ (Kannada) | తెలుగు (Telugu) | தமிழ் (Tamil) | English")
        print("🎤 Powered by: Sarvam Streaming ASR + DeepSeek LLM")
        print("🔊 Powered by: Sarvam HTTP Streaming TTS")
        print("=" * 60)
    
    async def speak(self, text, language="kannada"):
        """Text to speech in specified language"""
        print(f"\n🤖 Agent ({language}): {text}")
        
        try:
            lang_code = self.language_map.get(language, "kn-IN")
            speaker = self.speaker_map.get(language, "aditya")
            
            audio_chunks = []
            
            async for chunk in sarvam_client.text_to_speech.convert_stream(
                text=text,
                target_language_code=lang_code,
                speaker=speaker,
                model="bulbul:v3",
                output_audio_codec="mp3",
                output_audio_bitrate="128k",
                pace=1.0,
                enable_preprocessing=True
            ):
                audio_chunks.append(chunk)
            
            audio_data = b"".join(audio_chunks)
            
            # Play audio
            audio_buffer = io.BytesIO(audio_data)
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            
            pygame.mixer.music.unload()
            
        except Exception as e:
            print(f"TTS Error: {e}")
    
    async def listen_streaming(self):
        """Stream audio and transcribe"""
        
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("\n🎤 Listening... (speak now)")
        
        audio_buffer = []
        silent_chunks = 0
        started_speaking = False
        silence_chunks_needed = int(self.silence_duration * self.rate / self.chunk)
        
        transcription_result = ""
        
        try:
            async with sarvam_client.speech_to_text_streaming.connect(
                model="saaras:v3",
                mode="transcribe",
                language_code=self.current_lang_code,
                high_vad_sensitivity=True,
                vad_signals=False
            ) as ws:
                
                async def send_audio():
                    nonlocal silent_chunks, started_speaking, audio_buffer
                    
                    for i in range(int(self.max_duration * self.rate / self.chunk)):
                        try:
                            data = stream.read(self.chunk)
                            audio_buffer.append(data)
                            
                            rms = audioop.rms(data, 2)
                            
                            if rms < self.silence_threshold:
                                if started_speaking:
                                    silent_chunks += 1
                                    if silent_chunks > silence_chunks_needed:
                                        print("🔇 Silence detected")
                                        await ws.flush()
                                        break
                            else:
                                if not started_speaking:
                                    started_speaking = True
                                    print("🎙️ Recording started...")
                                silent_chunks = 0
                                
                                if len(audio_buffer) >= 10:
                                    combined_audio = b''.join(audio_buffer)
                                    audio_base64 = base64.b64encode(combined_audio).decode('utf-8')
                                    
                                    await ws.transcribe(
                                        audio=audio_base64,
                                        encoding="audio/wav",
                                        sample_rate=self.rate
                                    )
                                    audio_buffer = []
                                    
                        except Exception as e:
                            print(f"Send error: {e}")
                            break
                
                send_task = asyncio.create_task(send_audio())
                
                # Wait for response
                try:
                    async for message in ws:
                        if hasattr(message, 'type') and message.type == 'data':
                            if hasattr(message, 'data'):
                                data_obj = message.data
                                if hasattr(data_obj, 'transcript'):
                                    transcription_result = data_obj.transcript
                                    print(f"📝 Transcribed: {transcription_result}")
                                    break
                        elif hasattr(message, 'transcript'):
                            transcription_result = message.transcript
                            print(f"📝 Transcribed: {transcription_result}")
                            break
                        elif isinstance(message, dict):
                            if 'transcript' in message:
                                transcription_result = message['transcript']
                                print(f"📝 Transcribed: {transcription_result}")
                                break
                        
                except Exception as e:
                    print(f"Receive error: {e}")
                
                send_task.cancel()
                
        except Exception as e:
            print(f"WebSocket error: {e}")
        
        finally:
            stream.stop_stream()
            stream.close()
        
        if not started_speaking:
            print("⚠️ No speech detected")
            return ""
        
        if not transcription_result:
            print("⚠️ No transcription received")
            return ""
        
        return transcription_result
    
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
    
    def detect_language_from_text(self, text):
        """Detect language from user input"""
        kannada_indicators = ["ನಾನು", "ನಿಮ್ಮ", "ಹೆಸರು", "ನಮಸ್ಕಾರ", "ಹಾಲು", "ಮೊಸರು", "ಬೆಲೆ"]
        telugu_indicators = ["నేను", "మీ", "పేరు", "నమస్కారం", "పాలు", "పెరుగు", "ధర"]
        tamil_indicators = ["நான்", "உங்கள்", "பெயர்", "வணக்கம்", "பால்", "தயிர்", "விலை"]
        
        for word in kannada_indicators:
            if word in text:
                return "kannada"
        for word in telugu_indicators:
            if word in text:
                return "telugu"
        for word in tamil_indicators:
            if word in text:
                return "tamil"
        
        return "english"
    
    def extract_phone_number(self, text):
        """Extract phone number"""
        cleaned = re.sub(r'[\s\-\(\)]', '', text)
        pattern = r'[6-9]\d{9}'
        match = re.search(pattern, cleaned)
        if match:
            return match.group()
        pattern = r'\+91[6-9]\d{9}'
        match = re.search(pattern, cleaned)
        if match:
            return match.group()[3:]
        return None
    
    def should_skip(self, text):
        """Check if user wants to skip"""
        skip_words = ["skip", "dont want", "not comfortable", "later", "no thanks", "prefer not", "ವರ್ಜಿಸಿ", "దాటవేయి", "தவிர்"]
        return any(word in text.lower() for word in skip_words)
    
    def extract_email(self, text):
        """Extract email"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(pattern, text)
        return match.group() if match else None
    
    def extract_city(self, text):
        """Extract city"""
        cities = ["bangalore", "bengaluru", "mumbai", "delhi", "chennai", "hyderabad", "pune", "coimbatore", "mysore", "mysuru"]
        text_lower = text.lower()
        for city in cities:
            if city in text_lower:
                return city.title()
        return None
    
    async def process_lead_capture(self, user_input, language):
        """Process lead capture with multilingual support"""
        
        user_input_lower = user_input.lower()
        
        if self.lead_stage == "greeting":
            if any(word in user_input_lower for word in ["yes", "yeah", "sure", "interested", "want", "like", "know more", "ಹೌದು", "చెప్పండి", "ஆம்"]):
                self.lead_stage = "asking_name"
                self.lead_data["language"] = language
                return self.ask_name_prompts.get(language, self.ask_name_prompts["english"])
            else:
                return None
        
        elif self.lead_stage == "asking_name":
            self.lead_data["name"] = user_input.title()
            self.lead_stage = "asking_phone"
            prompt = self.ask_phone_prompts.get(language, self.ask_phone_prompts["english"])
            return prompt.format(name=self.lead_data["name"])
        
        elif self.lead_stage == "asking_phone":
            if self.should_skip(user_input):
                self.lead_data["phone"] = "Not provided"
                self.lead_stage = "asking_email"
                return self.ask_email_prompts.get(language, self.ask_email_prompts["english"])
            
            phone = self.extract_phone_number(user_input)
            if phone and len(phone) == 10:
                self.lead_data["phone"] = phone
                self.lead_stage = "asking_email"
                return self.ask_email_prompts.get(language, self.ask_email_prompts["english"])
            else:
                self.lead_data["phone"] = user_input if user_input else "Not provided"
                self.lead_stage = "asking_email"
                return self.ask_email_prompts.get(language, self.ask_email_prompts["english"])
        
        elif self.lead_stage == "asking_email":
            if self.should_skip(user_input):
                self.lead_data["email"] = "Not provided"
                self.lead_stage = "asking_city"
                return self.ask_city_prompts.get(language, self.ask_city_prompts["english"])
            
            email = self.extract_email(user_input)
            if email:
                self.lead_data["email"] = email
                self.lead_stage = "asking_city"
                return self.ask_city_prompts.get(language, self.ask_city_prompts["english"])
            else:
                self.lead_data["email"] = user_input if user_input else "Not provided"
                self.lead_stage = "asking_city"
                return self.ask_city_prompts.get(language, self.ask_city_prompts["english"])
        
        elif self.lead_stage == "asking_city":
            city = self.extract_city(user_input)
            if city:
                self.lead_data["city"] = city
                self.lead_stage = "asking_products"
                return self.ask_products_prompts.get(language, self.ask_products_prompts["english"])
            else:
                self.lead_data["city"] = user_input if user_input else "Not specified"
                self.lead_stage = "asking_products"
                return self.ask_products_prompts.get(language, self.ask_products_prompts["english"])
        
        elif self.lead_stage == "asking_products":
            interested = []
            for product in self.products.keys():
                product_local = self.products[product].get(language, product)
                if product.lower() in user_input_lower or product_local.lower() in user_input_lower:
                    interested.append(product)
            
            if interested:
                self.lead_data["interested_products"] = interested
                self.lead_stage = "asking_delivery_time"
                return self.ask_delivery_prompts.get(language, self.ask_delivery_prompts["english"])
            else:
                self.lead_data["interested_products"] = [user_input] if user_input else []
                self.lead_stage = "asking_delivery_time"
                return self.ask_delivery_prompts.get(language, self.ask_delivery_prompts["english"])
        
        elif self.lead_stage == "asking_delivery_time":
            if "morning" in user_input_lower or "ಬೆಳಿಗ್ಗೆ" in user_input_lower or "ఉదయం" in user_input_lower or "காலை" in user_input_lower:
                self.lead_data["preferred_delivery_time"] = "Morning (6-9 AM)"
            elif "evening" in user_input_lower or "ಸಂಜೆ" in user_input_lower or "సాయంత్రం" in user_input_lower or "மாலை" in user_input_lower:
                self.lead_data["preferred_delivery_time"] = "Evening (4-7 PM)"
            else:
                self.lead_data["preferred_delivery_time"] = user_input if user_input else "Not specified"
            
            self.lead_stage = "asking_referral"
            return self.ask_referral_prompts.get(language, self.ask_referral_prompts["english"])
        
        elif self.lead_stage == "asking_referral":
            self.lead_data["heard_about"] = user_input if user_input else "Not specified"
            
            self.save_lead(self.lead_data)
            
            self.lead_stage = "greeting"
            self.capturing_lead = False
            
            products_summary = ", ".join(self.lead_data["interested_products"]) if self.lead_data["interested_products"] else "our products"
            
            thankyou = self.thankyou_prompts.get(language, self.thankyou_prompts["english"])
            return thankyou.format(name=self.lead_data["name"], products=products_summary)
        
        return None
    
    async def run(self):
        """Main loop"""
        print("\n" + "=" * 60)
        print("🐄 Welcome to Akshayakalpa Organic")
        print("=" * 60)
        print("🌐 We speak: ಕನ್ನಡ | తెలుగు | தமிழ் | English")
        print("💬 Please respond in your preferred language")
        print("=" * 60)
        
        # Initial greeting - Start with Kannada, then offer other languages
        intro_kannada = self.greetings["kannada"]
        await self.speak(intro_kannada, "kannada")
        
        # Also offer Telugu option
        await asyncio.sleep(1)
        intro_telugu = "తెలుగులో మాట్లాడాలనుకుంటే, దయచేసి 'తెలుగు' అని చెప్పండి. మీరు కన్నడ, తమిళం లేదా ఇంగ్లీష్ లో కూడా మాట్లాడవచ్చు."
        await self.speak(intro_telugu, "telugu")
        
        waiting_for_response = True
        
        while self.call_active:
            try:
                # Listen for user response
                user_text = await self.listen_streaming()
                
                if not user_text:
                    await self.speak("I didn't hear you. Please say yes or no. ದಯವಿಟ್ಟು ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಹೇಳಿ.", "kannada")
                    continue
                
                # Detect language from user input
                detected_lang = self.detect_language_from_text(user_text)
                self.current_lang_code = self.language_map.get(detected_lang, "kn-IN")
                self.current_language = detected_lang
                
                print(f"\n📝 User ({detected_lang}): {user_text}")
                
                # Exit check
                if user_text.lower() in ['exit', 'quit', 'goodbye', 'bye', 'no', 'ಬೈ', 'నమస్కారం', 'பிரியாவிடை']:
                    farewell = "Thank you for your time! Visit akshayakalpa.com to learn more. Have a great day! ಧನ್ಯವಾದಗಳು!"
                    await self.speak(farewell, "kannada")
                    break
                
                # Language selection check
                if "telugu" in user_text.lower() or "తెలుగు" in user_text:
                    self.current_language = "telugu"
                    self.current_lang_code = "te-IN"
                    await self.speak("మీరు తెలుగులో మాట్లాడవచ్చు. దయచేసి చెప్పండి.", "telugu")
                    continue
                elif "tamil" in user_text.lower() or "தமிழ்" in user_text:
                    self.current_language = "tamil"
                    self.current_lang_code = "ta-IN"
                    await self.speak("நீங்கள் தமிழில் பேசலாம். தயவுசெய்து சொல்லுங்கள்.", "tamil")
                    continue
                elif "kannada" in user_text.lower() or "ಕನ್ನಡ" in user_text:
                    self.current_language = "kannada"
                    self.current_lang_code = "kn-IN"
                    await self.speak("ನೀವು ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡಬಹುದು. ದಯವಿಟ್ಟು ಹೇಳಿ.", "kannada")
                    continue
                
                # Start lead capture if interested
                if waiting_for_response:
                    if any(word in user_text.lower() for word in ["yes", "yeah", "sure", "interested", "tell me", "know more", "ಹೌದು", "చెప్పండి", "ஆம்"]):
                        waiting_for_response = False
                        self.capturing_lead = True
                        self.lead_stage = "asking_name"
                        self.lead_data["language"] = self.current_language
                        
                        response = self.ask_name_prompts.get(self.current_language, self.ask_name_prompts["english"])
                        await self.speak(response, self.current_language)
                        continue
                    else:
                        response = "No problem! Visit our website to learn more. Thank you! ಧನ್ಯವಾದಗಳು!"
                        await self.speak(response, "kannada")
                        break
                
                # Process lead capture
                if self.capturing_lead:
                    response = await self.process_lead_capture(user_text, self.current_language)
                    
                    if response:
                        await self.speak(response, self.current_language)
                        
                        # If lead capture complete, end call
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
    asyncio.run(main())