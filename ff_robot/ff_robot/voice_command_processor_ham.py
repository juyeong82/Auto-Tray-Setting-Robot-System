import os
import tempfile
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import warnings

from dotenv import load_dotenv
from openai import OpenAI

# LangChain 관련 import (최신 버전 기준)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(current_dir, ".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")


class InputHandler:
    """
    사용자 입력을 처리하는 클래스 (음성 또는 텍스트)
    """
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.duration = 5       # 녹음 시간 (초)
        self.samplerate = 16000 # Whisper 권장 샘플링 레이트

    def get_input(self):
        # 입력 모드 선택 (음성/텍스트)
        print("\n--- 입력 모드 선택 ---")
        print("1. 음성 입력 (마이크)")
        print("2. 텍스트 입력 (키보드)")
        mode = input("선택 > ").strip()

        if mode == "1":
            return self._speech2text()
        else:
            return self._text_input()

    def _speech2text(self):
        print("\n[음성 모드] 5초 동안 말해주세요...")
        
        # 녹음 수행
        audio = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        print("녹음 완료. 텍스트 변환 중...")

        # 임시파일 생성 및 Whisper API 전송
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)
            
            with open(temp_wav.name, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=f
                )

        print(f"STT 인식 결과: {transcript.text}")
        return transcript.text

    def _text_input(self):
        print("\n[텍스트 모드] 명령을 입력하세요.")
        user_text = input("입력 > ")
        return user_text


class ExtractKeyword:
    """
    자연어 문장에서 주문 메뉴(Objects)와 배치 위치(Tray Destination)를 추출하는 클래스
    """
    def __init__(self, api_key):
        # ChatOpenAI 인스턴스 생성
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.0, # 정확한 추출을 위해 temperature 낮춤
            api_key=api_key
        )

        prompt_content = """
            당신은 햄버거 가게의 주문 접수 AI입니다. 
            사용자의 자연어 주문에서 '메뉴'를 추출하고, 각 메뉴에 대해 자동으로 '배치 위치(트레이)'를 지정해야 합니다.

            <목표>
            1. 사용자의 말에서 아래 <메뉴 리스트>에 있는 항목을 찾아 시스템 ID로 변환하세요.
            2. 모든 정상 메뉴의 배치 위치는 사용자가 말하지 않아도 무조건 'tray_A'로 지정하세요.
            3. 단, '쓰레기', '불량품' 등의 언급이 있을 때만 'reject_bin'으로 지정하세요.

            <메뉴 리스트 (로봇 시스템 ID)>
            - burger_bulgogi, burger_cheese, fries, coke, sprite, coffee

            <출력 형식>
            - 반드시 다음 형식을 지키세요: [메뉴1 메뉴2 ... / 위치1 위치2 ...]
            - 메뉴와 위치는 1:1로 매칭되어야 하며, 개수가 정확히 일치해야 합니다.
            - 구분자 '/' 앞에는 메뉴 나열, 뒤에는 위치 나열 (공백 구분)

            <변환 규칙>
            - "불고기 버거" -> burger_bulgogi
            - "치즈 버거" -> burger_cheese
            - "감자튀김", "감튀" -> fries
            - "콜라" -> coke
            - "사이다" -> sprite
            - "커피" -> coffee

            <예시>
            - 입력: "치즈버거 하나 주세요"
            출력: burger_cheese / tray_A

            - 입력: "불고기 버거랑 콜라 하나씩 줘"
            출력: burger_bulgogi coke / tray_A tray_A

            - 입력: "감자튀김 두 개랑 사이다 하나"
            출력: fries fries sprite / tray_A tray_A tray_A

            - 입력: "이거 불량품이야 버려줘"
            출력: / reject_bin

            <사용자 입력>
            "{user_input}"
        """
        
        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], 
            template=prompt_content
        )

        # LCEL 체인 구성
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def run(self, text_input):
        print(f"\n[AI 분석 중] '{text_input}' 해석 시작...")
        
        # LLM 호출
        response = self.chain.invoke({"user_input": text_input})
        
        # 결과 파싱 (기존 로직 유지)
        result = response.strip().split("/")
        
        if len(result) != 2:
            warnings.warn("AI 응답 형식이 올바르지 않음.")
            return [], []

        object_part, destination_part = result[0], result[1]
        
        object_list = object_part.split()
        destination_list = destination_part.split()

        return object_list, destination_list


if __name__ == "__main__":
    # 1. 인스턴스 초기화
    if not openai_api_key:
        print("Error: .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
        exit()

    input_handler = InputHandler(openai_api_key)
    keyword_extractor = ExtractKeyword(openai_api_key)

    # 2. 사용자 입력 받기 (음성 or 텍스트)
    user_command = input_handler.get_input()

    # 3. 키워드 추출 수행
    if user_command:
        objects, destinations = keyword_extractor.run(user_command)

        # 4. 최종 결과 출력
        print("\n" + "="*30)
        print("      최종 명령 데이터      ")
        print("="*30)
        print(f"Target Objects : {objects}")
        print(f"Destinations   : {destinations}")
        print("="*30)