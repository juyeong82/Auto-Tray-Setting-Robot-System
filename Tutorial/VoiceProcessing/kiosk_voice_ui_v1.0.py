#!/usr/bin/env python3
# kiosk_voice_ui_v1.0.py - 통합 ROS 2 + 음성 키오스크 노드
import os
import time
import sys
import numpy as np
import pyaudio
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
import openwakeword
from openwakeword.model import Model

# ROS 2 관련 라이브러리
import rclpy
from rclpy.node import Node
from ff_robot_interfaces.srv import OrderService

# LangChain 관련 라이브러리
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. 환경 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# 모델 파일 경로 (실행 위치에 파일이 있어야 함)
WAKEWORD_MODEL_FILE = "hello_rokey_8332_32.tflite"

# --- [WakeupWord 클래스 (rokey_buger.py 그대로 사용)] ---
class WakeupWordDetector:
    def __init__(self, model_file, buffer_size=1280):
        # 현재 파일 위치 기준으로 모델 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_file)
        
        # 모델 파일 없으면 다운로드 시도 (혹시 모르니 유지)
        if not os.path.exists(model_path):
            print(f"모델 파일 '{model_file}'을 찾을 수 없습니다. 현재 폴더를 확인해주세요.")
            # openwakeword.utils.download_models() # 필요시 주석 해제
            model_path = model_file

        self.model = Model(wakeword_models=[model_path])
        self.buffer_size = buffer_size
        self.stream = None

    def set_stream(self, stream):
        self.stream = stream

    def is_wakeup(self):
        if self.stream is None: return False
        try:
            audio_data = np.frombuffer(self.stream.read(self.buffer_size, exception_on_overflow=False), dtype=np.int16)
            prediction = self.model.predict(audio_data)
            for mdl in self.model.prediction_buffer.keys():
                if self.model.prediction_buffer[mdl][-1] > 0.5: return True
        except Exception:
            pass
        return False

# --- [통합 ROS 2 노드 클래스] ---
class VoiceKioskNode(Node):
    def __init__(self):
        super().__init__('voice_kiosk_node')
        
        # 1. ROS 2 서비스 클라이언트 설정
        self.cli = self.create_client(OrderService, '/dsr01/order_service')
        
        # 서비스 서버 연결 대기
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('메인 로봇 컨트롤러(Service) 연결 대기 중...')
            
        self.get_logger().info('✅ ROS 2 서비스 연결 완료.')
        
        # 2. LLM 설정
        self.setup_llm()
        self.memory_store = {}
        
        self.get_logger().info('✅ 음성 키오스크 시스템 준비 완료.')

    def setup_llm(self):
        """LangChain 및 프롬프트 설정"""
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=OPENAI_API_KEY)

        template = """
        당신은 '로키버거'의 친절한 로봇 점원입니다.

        <메뉴 정보>
        - 버거: 불고기버거(burger1), 치즈버거(burger2), 새우버거(burger3)
        - 사이드: 감자튀김(fries), 치킨너겟(nugget)
        - 음료: 콜라(coke), 사이다(cider)
        <메뉴 가격>
        - 버거: 불고기버거(3000), 치즈버거(3500), 새우버거(3200)
        - 사이드: 감자튀김(1500), 치킨너겟(2000)
        - 음료: 콜라(1800), 사이다(1800)

        <대화 규칙>
        1. 손님에게 필요한 정보(버거, 사이드, 음료)를 자연스럽게 물어보세요.
        2. 손님이 "계산할게", "이상이야" 등 주문을 마치는 말을 하면 아래 <데이터 추출 규칙>을 따르세요.
        3. 주문이 끝나면 총 금액을 알려주고 결제를 안내하세요.
        4. 손님에게는 태그를 제외한 친절한 멘트로 대답하세요.

        <데이터 추출 규칙>
        - 주문이 최종 확정되었을 때만, 답변 맨 마지막에 [ORDER: 메뉴코드1, 수량1, 메뉴코드2, 수량2, ...] 형식을 추가하세요.
        - 메뉴 이름 대신 위 <메뉴 정보>에 있는 **영어 코드**를 사용하세요.
        - 예시: "불고기버거 하나랑 콜라 두개" -> [ORDER: burger1, 1, coke, 2]

        <이전 대화 내역>
        {history}

        손님: {input}
        로봇: 
        """
        
        prompt = PromptTemplate(input_variables=["history", "input"], template=template)
        self.chain = prompt | self.llm | StrOutputParser()
        
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def get_session_history(self, session_id: str):
        if session_id not in self.memory_store:
            self.memory_store[session_id] = ChatMessageHistory()
        return self.memory_store[session_id]

    # --- [음성 입출력 기능] ---
    def speak(self, text):
        """TTS 출력"""
        print(f"🔊 [Robot]: {text}")
        try:
            tts = gTTS(text=text, lang='ko')
            filename = "voice_temp.mp3"
            tts.save(filename)
            os.system(f"mpg123 -q {filename}")
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            self.get_logger().error(f"TTS Error: {e}")

    def listen_order(self):
        """STT 입력"""
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("🎤 듣고 있습니다...")
            try:
                # 주변 소음 적응
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=5, phrase_time_limit=8)
                text = r.recognize_google(audio, language='ko-KR')
                print(f"👤 [User]: {text}")
                return text
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                return None
            except Exception as e:
                print(f"STT Error: {e}")
                return None

    # --- [ROS 2 통신 기능] ---
    def send_order_to_robot(self, item_names, item_quantities):
        """서비스 요청 전송 (동기식 처리)"""
        req = OrderService.Request()
        req.item_names = item_names
        req.item_quantities = item_quantities
        
        self.get_logger().info(f"🚀 주문 전송 시도: {item_names} / {item_quantities}")
        
        # 비동기 호출 후 대기 (spin_until_future_complete 사용)
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        return future.result()

    def process_order_string(self, order_str):
        """문자열 파싱 후 ROS 서비스 호출"""
        try:
            # 예: "burger1, 1, coke, 2" -> 리스트로 변환
            parts = [item.strip() for item in order_str.split(",")]
            
            o_list = [] # 메뉴 이름 리스트
            n_list = [] # 수량 리스트
            
            for i in range(0, len(parts), 2):
                o_list.append(parts[i])
                n_list.append(int(parts[i+1]))
            
            # ROS 2 서비스 호출
            result = self.send_order_to_robot(o_list, n_list)
            
            if result.assigned_order_id:
                return True, result.message
            else:
                return False, result.message
                
        except Exception as e:
            self.get_logger().error(f"데이터 파싱 오류: {e}")
            return False, "주문 데이터 처리 중 오류가 발생했습니다."

    # --- [메인 실행 로직] ---
    def run_kiosk(self):
        """전체 키오스크 실행 루프"""
        pa = pyaudio.PyAudio()
        # 마이크 스트림 열기
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)
        
        print("\n" + "="*40)
        print(f"🤖 로봇이 대기 중입니다. 'Hello 로키'라고 불러주세요.")
        print("="*40)

        # ROS가 살아있는 동안 반복
        while rclpy.ok():
            # 호출어 감지기 초기화
            detector = WakeupWordDetector(model_file=WAKEWORD_MODEL_FILE)
            detector.set_stream(stream)
            
            # 버퍼 비우기 (이전 소리 제거)
            if stream.get_read_available() > 0:
                _ = stream.read(stream.get_read_available(), exception_on_overflow=False)
            
            # 1. 호출어 감지 루프
            is_detected = False
            while rclpy.ok() and not is_detected:
                if detector.is_wakeup():
                    is_detected = True
                    print("\n✨ 호출어 감지됨! ✨")
                    self.speak("어서오세요 로키버거입니다. 주문하시겠어요?")
                else:
                    # CPU 점유율 방지
                    time.sleep(0.01)
            
            # 2. 대화 및 주문 루프
            session_id = str(time.time())
            while rclpy.ok() and is_detected:
                user_text = self.listen_order()
                
                if user_text:
                    # LLM 응답 생성
                    response = self.chain_with_history.invoke(
                        {"input": user_text},
                        config={"configurable": {"session_id": session_id}}
                    )
                    
                    # 주문 확정 태그 확인
                    if "[ORDER:" in response:
                        parts = response.split("[ORDER:")
                        robot_ment = parts[0].strip()
                        order_data = parts[1].replace("]", "").strip()
                        
                        # 멘트 먼저 출력
                        self.speak(robot_ment)
                        
                        # ROS로 주문 전송
                        success, msg = self.process_order_string(order_data)
                        
                        if success:
                            self.speak("결제가 완료되었습니다. 로봇이 곧 음식을 준비합니다.")
                            print(f"✅ 서버 응답: {msg}")
                        else:
                            self.speak(f"죄송합니다. 주문 처리에 실패했습니다. {msg}")
                        
                        print("\n--- 대기 모드로 복귀합니다 ---")
                        # 대화 루프 탈출 -> 다시 호출어 감지로
                        break 
                    else:
                        # 일반 대화 응답
                        self.speak(response)
                else:
                    self.speak("잘 못 들었습니다. 다시 말씀해 주세요.")

        # 종료 처리
        stream.stop_stream()
        stream.close()
        pa.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = VoiceKioskNode()
    
    try:
        node.run_kiosk()
    except KeyboardInterrupt:
        node.get_logger().info("키오스크를 종료합니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()