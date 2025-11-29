#!/usr/bin/env python3
# v1.6 - 대화 끝나고 대기모드로 복귀 찐막
import os
import time
import numpy as np
import pyaudio
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
import openwakeword
from openwakeword.model import Model

# LangChain 관련막
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. 환경 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WAKEWORD_MODEL_FILE = "hello_rokey_8332_32.tflite"

# --- [WakeupWord 클래스] ---
class WakeupWordDetector:
    def __init__(self, model_file, buffer_size=1280):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_file)
        
        if not os.path.exists(model_path):
            print(f"모델 파일 '{model_file}'을 찾을 수 없습니다. 다운로드를 시도합니다...")
            openwakeword.utils.download_models()
            model_path = model_file

        self.model = Model(wakeword_models=[model_path])
        self.buffer_size = buffer_size
        self.stream = None

    def set_stream(self, stream):
        self.stream = stream

    def is_wakeup(self):
        if self.stream is None: return False
        audio_data = np.frombuffer(self.stream.read(self.buffer_size, exception_on_overflow=False), dtype=np.int16)
        prediction = self.model.predict(audio_data)
        for mdl in self.model.prediction_buffer.keys():
            if self.model.prediction_buffer[mdl][-1] > 0.5: return True
        return False

# --- [LLM 설정] ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=OPENAI_API_KEY)

template = """
당신은 '로키버거'의 친절한 로봇 점원입니다.

<메뉴 정보>
- 버거: 불고기버거(buger1), 치즈버거(buger2), 새우버거(buger3)
- 사이드: 감자튀김(fries), 치킨너겟(nugget)
- 음료: 콜라(coke), 사이다(soda)
<메뉴 가격>
- 버거: 불고기버거(3000), 치즈버거(3500), 새우버거(3200)
- 사이드: 감자튀김(1500), 치킨너겟(2000)
- 음료: 콜라(1800), 사이다(1800)
- 세트메뉴 (+3000): (기본: 감자튀김+콜라)

<대화 규칙>
1. 손님에게 필요한 정보(버거, 사이드, 음료, 포장여부)를 자연스럽게 하나씩 물어보세요.
2. 손님이 "계산할게", "이상이야" 등 주문을 마치는 말을 하면 아래 <데이터 추출 규칙>을 따르세요.
3. 아직 주문 중이거나 식사 장소가 불분명하면 태그를 붙이지 마세요.
4. 세트메뉴를 시키면 사이드메뉴와 음료 변경 여부를 꼭 확인하세요.
5. 단품버거 또는 사이드메뉴만 시키면 추가 메뉴 주문 여부를 꼭 확인하세요.
6. 한번에 메뉴 이야기하고 계산해달라고 하면 바로 총 금액 알려주고 계산하라고 하세요.
7. 주문이 끝나면 총 금액을 알려주고 포스기가 앞에 있으니 계산하라고 안내하세요.
8. 손님에게는 태그를 제외한 친절한 멘트로 대답하세요.

<데이터 추출 규칙>
- 주문이 최종 확정되었을 때만, 답변 맨 마지막에 [ORDER: 메뉴코드1, 수량1, 메뉴코드2, 수량2, ...] 형식을 추가하세요.
- 메뉴 이름 대신 위 <메뉴 정보>에 있는 **영어 코드**를 사용하세요.
- 세트 메뉴인 경우 구성품을 풀어서 각각의 코드로 적으세요.
- 예시 1: "불고기버거 하나 주세요" -> [ORDER: buger1, 1]
- 예시 2: "치즈버거 2개랑 콜라 1개" -> [ORDER: buger2, 2, coke, 1]
- 예시 3: "불고기버거 세트 하나" -> [ORDER: buger1, 1, fries, 1, coke, 1]

<이전 대화 내역>
{history}

손님: {input}
로봇: 
"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
chain = prompt | llm | StrOutputParser()

# 대화 기억 저장소
memory_store = {}

def get_session_history(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- [기능 함수] ---
def speak(text):
    print(f"🔊 [Robot]: {text}")
    try:
        tts = gTTS(text=text, lang='ko')
        filename = "voice.mp3"
        tts.save(filename)
        os.system(f"mpg123 -q {filename}")
        os.remove(filename)
    except Exception as e: print(f"TTS Error: {e}")

def listen_order():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 말씀해주세요...")
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=10)
            text = r.recognize_google(audio, language='ko-KR')
            print(f"👤 [User]: {text}")
            return text
        except: return None

# [추가] 리스트 생성 및 전송 함수
def process_order_data(order_data_str):
    try:
        items = [item.strip() for item in order_data_str.split(",")]
        
        O_list = [] 
        N_list = [] 
        
        for i in range(0, len(items), 2):
            menu_code = items[i]
            quantity = int(items[i+1])
            O_list.append(menu_code)
            N_list.append(quantity)
            
        print("\n📊 [System] 주문 데이터 생성 완료")
        print(f"   >>> O_list (주문): {O_list}")
        print(f"   >>> N_list (수량): {N_list}")
        
        # send_to_ros2(O_list, N_list) # ROS2 전송
        return True
    except Exception as e:
        print(f"❌ 데이터 파싱 오류: {e}")
        return False

# --- [메인 실행] ---
def main():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)
    
    print(f"🤖 로봇이 대기 중입니다. 'Hello 로키'라고 불러주세요.")

    while True:
        # [수정 포인트 1] 루프 시작할 때마다 감지기 상태를 깨끗하게 유지
        # (여기서는 매번 새로 생성하는 것이 가장 확실합니다)
        detector = WakeupWordDetector(model_file=WAKEWORD_MODEL_FILE, buffer_size=1280)
        detector.set_stream(stream)
        
        # 1. 호출어 감지 대기 (여기서 계속 멈춰 있음)
        # (주의: detector 내부에서 stream을 읽어오므로, 이전 대화 내용이 stream 버퍼에 남아있다면
        #  이를 비워주는 작업이 필요할 수 있습니다.)
        
        # [수정 포인트 2] 스트림 버퍼 비우기 (대화 중 쌓인 소리 제거)
        if stream.get_read_available() > 0:
            _ = stream.read(stream.get_read_available(), exception_on_overflow=False)
            
        is_detected = False
        while not is_detected:
            if detector.is_wakeup():
                is_detected = True
                print("\n✨ 호출어 감지됨! 손님 응대 시작 ✨")
                session_id = str(time.time())
                speak("어서오세요 로키버거입니다. 주문하시겠어요?")
                
                # [2] 안쪽 루프: 손님과 대화 (주문)
                while True:
                    order_text = listen_order()
                    
                    if order_text:
                        response = chain_with_history.invoke(
                            {"input": order_text},
                            config={"configurable": {"session_id": session_id}}
                        )
                        
                        if "[ORDER:" in response:
                            parts = response.split("[ORDER:")
                            robot_ment = parts[0].strip()
                            order_data = parts[1].replace("]", "").strip()
                            
                            speak(robot_ment)
                            
                            if process_order_data(order_data):
                                print("⏳ 결제 대기 중... (10초)")
                                time.sleep(10) 
                                
                                speak("결제가 완료되었습니다. 주문하신 음식은 곧 준비해 드릴게요.")
                                print("--- 주문 처리 완료: 대기 모드로 복귀합니다 ---")
                                print(f"🤖 로봇이 대기 중입니다. 'Hello 로키'라고 불러주세요.")

                                # [수정 포인트 3] 대화 종료 후 바깥 루프(호출어 대기)로 돌아감
                                break 
                        else:
                            speak(response)
                    else:
                        speak("잘 못 들었습니다. 다시 말씀해 주세요.")
                
                # 안쪽 while문이 break로 끝나면 여기로 옴 -> 다시 바깥쪽 while문의 처음으로

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")