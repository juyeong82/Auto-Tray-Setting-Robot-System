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
    # 손님의 주문 금액 : "{cost}"
    # 포장 주문 : "{eat_out}"
    # 매장에서 식사 : "{eat_in}"
    # 음식 준비 중 : "{ready}"
    # 음식 준비 완료 : "{done}"

    # 메뉴
    # 불고기버거 : "{buger1}", 
    # 치즈버거 : "{buger2}",
    # 새우버거 : "{buger3}",
    # 감자튀김 : "{fries}",
    # 치킨너겟 : "{nugget}",
    # 콜라 : "{coke}",
    # 사이다 : "{soda}",

    
    template="""
    <입력 변수>
    손님의 주문 : "{order}"

    ----------------------------------------------------------------------
    <메뉴 정보>
    - 버거: 불고기버거(3000), 치즈버거(3500), 새우버거(3200)
    - 사이드: 감자튀김(1500), 치킨너겟(2000)
    - 음료: 콜라(1800), 사이다(1800)
    - 세트메뉴 (+ 3000)(감자튀김 기본, 치킨너겟으로 변경 시 + 500원 / 콜라 기본, 사이다로 변경 시 + 0원)
    ----------------------------------------------------------------------
    <행동 지침>
    1. 손님의 말에서 '포장' 또는 '가져갈게'라는 의도가 보이면 답변 맨 앞에 "[eat_out]"을 붙이세요.
    2. 손님의 말에서 '매장', '먹고 갈게'라는 의도가 보이면 답변 맨 앞에 "[eat_in]"을 붙이세요.
    3. 아직 주문 중이거나 식사 장소가 불분명하면 태그를 붙이지 마세요.
    4. 세트메뉴를 시키면 사이드메뉴 변경 여부와 음료 종류를 꼭 확인하세요.
    5. 단품버거 또는 사이드메뉴만 시키면 추가 메뉴 주문 여부를 꼭 확인하세요.
    6. 주문이 끝나면 총 금액을 알려주고 포스기가 앞에 있으니 계산하라고 안내하세요.
    7. 손님에게는 태그를 제외한 친절한 멘트로 대답하세요.
    ----------------------------------------------------------------------
    <답변 예시>
    - 손님: "불고기버거 포장해줘" -> [eat_out] 네, 알겠습니다. 불고기버거를 꼼꼼히 포장해 드릴게요. 잠시만 기다려주세요.
    - 손님: "여기서 먹고 갈게" -> [eat_in] 네, 알겠습니다. 식판 트레이에 준비해 드릴게요. 편한 자리에 앉아 계세요.
    - 손님: "치즈버거 하나 줘" -> 치즈버거 하나 주문하셨습니다. 포장이신가요, 아니면 매장에서 드시고 가시나요?
    ----------------------------------------------------------------------
    위 규칙을 지켜서 한국어로 짧고 친절하고 자연스럽게 대답해주세요.
    1. 어떤 버거를 시키는지
    2. 사이드는 무엇으로 할지
    3. 음료는 무엇으로 할지
    4. 포장인지 매장 식사인지
    5. 총 금액 안내 및 계산 안내
    순으로 하나씩 듣고 이야기 하세요.
    """
)

# LCEL(LangChain Expression Language) 문법을 사용한 체인 생성
# 데이터 흐름: 프롬프트 입력 -> LLM 모델 처리 -> 문자열 파서로 텍스트 추출
chain = prompt | llm | StrOutputParser()

# ==========================================
# 2. 기능 함수 정의 (말하기, 듣기)
# ==========================================
def speak(text):                    # TTS로 음성 출력
    print(f"[Robot]: {text}")       # 로봇이 하는말 출력 
    try:        # gTTS를 이용해 텍스트를 한국어 음성 mp3 파일로 저장
        tts = gTTS(text=text, lang='ko')
        filename = "voice.mp3"
        tts.save(filename)                      # 텍스트를 한국어 음성 mp3파일로 저장
        os.system(f"mpg123 -q {filename}")      # mpg123으로 음성 재생
        os.remove(filename)                     # 재생 후 파일 삭제
    except Exception as e:
        print(f"TTS Error: {e}")                # 에러 발생 시 메시지 출력

def listen_order():               # 마이크를 통해 사용자의 목소리를 듣고 텍스트로 변환하는 함수 (STT)
    r = sr.Recognizer()           # 음성 인식기 생성
    with sr.Microphone() as source:     # 마이크를 음성 소스(입력)로 사용
        print("🎤 주문중...")
        try:
            # listen: 사용자가 말을 할 때까지 기다렸다가 녹음
            # timeout: 30초 동안 아무 말도 없으면 에러 발생
            # phrase_time_limit: 한 번 말할 때 최대 10초까지만 듣음
            audio = r.listen(source, timeout=30, phrase_time_limit=10)
            # 구글의 무료 STT 서버를 사용하여 음성을 텍스트로 변환 (한국어 설정)
            text = r.recognize_google(audio, language='ko-KR') 
            print(f"[User]: {text}")        # 사용자가 한 말을 출력
            return text
        except sr.WaitTimeoutError:
            print("음성이 감지되지 않았습니다.") # 5초간 침묵 시            
            return None
        except sr.UnknownValueError:
            print("무슨 말인지 못 알아들었어요.") # 말은 했으나 인식이 안 될 때
            return None
        except Exception as e:
            print(f"STT Error: {e}") # 그 외 에러 (마이크 연결 끊김 등)
            return None

# ==========================================
# 3. 메인 실행 루프 (프로그램의 본체)
# ==========================================
def main():
    # [중요] 1. PyAudio를 사용하여 마이크 하드웨어 스트림 열기
    # 이 스트림은 '호출어 감지(openwakeword)'를 위해 사용됩니다.
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000, # openwakeword는 16kHz를 권장함
        input=True,
        frames_per_buffer=1280 # 1280 샘플 = 0.08초 (openwakeword 기본 청크)
    )

    # 2. 감지기 생성 및 연결 - 호출어 감지기 객체 생성 및 스트림 연결
    # 이 부분이 없으면 'NoneType has no attribute read' 에러가 발생합니다.
    detector = WakeupWordDetector(model_file=WAKEWORD_MODEL_FILE, buffer_size=1280)
    detector.set_stream(stream)     # 열어둔 마이크 스트림을 감지기에 전달
    
    print(f"🤖 로봇이 대기 중입니다. 메뉴를 정하셨으면 'Hello 로키' 라고 불러주세요.")

    # 무한 루프: 프로그램을 끄기 전까지 계속 반복
    while True:
        if detector.is_wakeup():        # 1. 호출어("Hello Rokey")가 들리는지 0.08초마다 검사
            print("\n✨ 호출어 감지됨! ✨")
            speak("""안녕하세요! 로키버거입니다. 메뉴를 말씀해주세요.""")     # 2. 호출어가 들리면 대답
            
            # 잠시 마이크 스트림을 멈추거나 비워주는 것이 좋음 (생략 가능)
            
            order_text = listen_order()        # 3. 사용자의 주문 듣기 (STT)
            
            if order_text:            # 주문이 정상적으로 인식되었다면
                answer = chain.invoke({"order": order_text})    # 4. LLM(AI)에게 주문 내용을 보내고 적절한 대답 생성
                speak(answer)         # 5. AI가 만든 대답을 말하기 (TTS)
                break
            else:                     # 주문을 못 들었을 경우
                speak("잘 못 들었습니다. 다시 불러주세요.")
            
            print("--- 대기 모드로 복귀 ---")
            # 다시 루프를 돌며 대기
# 이 파일이 직접 실행될 때만 main() 함수를 호출
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")