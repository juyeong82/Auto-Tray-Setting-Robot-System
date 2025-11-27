#!/usr/bin/env python3
import os
import time
import numpy as np
import pyaudio
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
import openwakeword
from openwakeword.model import Model
from scipy.signal import resample

# LangChain 관련
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# 모델 파일 이름 (같은 폴더에 있어야 함)
WAKEWORD_MODEL_FILE = "hello_rokey_8332_32.tflite"

# --- [내장된 WakeupWord 클래스] ---
class WakeupWordDetector:
    def __init__(self, model_file, buffer_size=1024):
        # 모델 파일 경로 확인
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_file)
        
        if not os.path.exists(model_path):
            # 없으면 다운로드 시도 (기본 모델인 경우)
            print(f"모델 파일 '{model_file}'을 찾을 수 없습니다. 다운로드를 시도합니다...")
            openwakeword.utils.download_models()
            model_path = model_file # 다운로드 되면 현재 폴더에 생김

        self.model = Model(wakeword_models=[model_path])
        self.buffer_size = buffer_size
        self.stream = None

    def set_stream(self, stream):
        self.stream = stream

    def is_wakeup(self):
        if self.stream is None:
            print("Error: 마이크 스트림이 연결되지 않았습니다.")
            return False
            
        # 마이크에서 소리 데이터 읽기
        audio_data = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16
        )

        # 16000Hz로 리샘플링 (openwakeword 요구사항)
        # 마이크가 16000Hz라면 리샘플링 없이 써도 됨 (여기서는 안전하게 변환)
        # audio_16k = resample(audio_data, int(len(audio_data) * 16000 / 44100)) 
        
        # 만약 마이크 설정을 16000으로 했다면 그대로 사용
        audio_16k = audio_data 

        # 모델 예측
        prediction = self.model.predict(audio_16k)
        
        # 감지 확인 (점수가 0.5 이상이면 감지된 것으로 간주)
        for mdl in self.model.prediction_buffer.keys():
            scores = self.model.prediction_buffer[mdl]
            if scores[-1] > 0.5:
                return True
        return False

# --- [LLM 설정] ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=OPENAI_API_KEY)
prompt = PromptTemplate(
    input_variables=["order"],
    template="""
    당신은 친절한 로봇 웨이터 '로키'입니다.
    손님의 주문: "{order}"
    
    위 주문에 대해 친절하게 대답하고, 주문 내용을 확인하는 멘트를 작성해주세요.
    단, 50자 이내로 짧게 대답하세요.
    """
)
chain = prompt | llm | StrOutputParser()

# --- [기능 함수] ---
def speak(text):
    print(f"[Robot]: {text}")
    try:
        tts = gTTS(text=text, lang='ko')
        filename = "voice.mp3"
        tts.save(filename)
        os.system(f"mpg123 -q {filename}")
        os.remove(filename)
    except Exception as e:
        print(f"TTS Error: {e}")

def listen_order():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 주문을 말씀해주세요...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            text = r.recognize_google(audio, language='ko-KR') 
            print(f"[User]: {text}")
            return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            print(f"STT Error: {e}")
            return None

# --- [메인 실행] ---
def main():
    # 1. 마이크 스트림 열기 (16000Hz 설정)
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000, # openwakeword는 16kHz를 권장함
        input=True,
        frames_per_buffer=1280 # 1280 샘플 = 0.08초 (openwakeword 기본 청크)
    )

    # 2. 감지기 생성 및 연결
    detector = WakeupWordDetector(model_file=WAKEWORD_MODEL_FILE, buffer_size=1280)
    detector.set_stream(stream)
    
    print(f"🤖 로봇이 대기 중입니다. '{WAKEWORD_MODEL_FILE}' 모델 사용 중.")

    while True:
        # 호출어 감지 (블로킹 없이 계속 체크)
        if detector.is_wakeup():
            print("\n✨ 호출어 감지됨! ✨")
            speak("네, 부르셨나요?")
            
            # 잠시 마이크 스트림을 멈추거나 비워주는 것이 좋음 (생략 가능)
            
            order_text = listen_order()
            
            if order_text:
                answer = chain.invoke({"order": order_text})
                speak(answer)
            else:
                speak("잘 못 들었습니다. 다시 불러주세요.")
            
            print("--- 대기 모드로 복귀 ---")
            # 다시 루프를 돌며 대기

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")