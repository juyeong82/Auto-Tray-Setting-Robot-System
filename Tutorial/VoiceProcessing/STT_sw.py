# # from langchain.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI
# import openai
# import sounddevice as sd
# import scipy.io.wavfile as wav
# import numpy as np
# import tempfile
# import os

# # from ament_index_python.packages import get_package_share_directory
# from dotenv import load_dotenv


# load_dotenv(dotenv_path=os.path.join(".env"))
# openai_api_key = os.getenv("OPENAI_API_KEY")


# class STT:
#     def __init__(self, openai_api_key):
#         self.openai_api_key = openai_api_key
#         self.duration = 5  # seconds
#         self.samplerate = 16000  # Whisper는 16kHz를 선호

#     def speech2text(self):
#         # 녹음 설정
#         print("음성 녹음을 시작합니다. \n 5초 동안 말해주세요...")
#         audio = sd.rec(
#             int(self.duration * self.samplerate),
#             samplerate=self.samplerate,
#             channels=1,
#             dtype="int16",
#         )
#         sd.wait()
#         print("녹음 완료. Whisper에 전송 중...")

#         # 임시 WAV 파일 저장
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
#             wav.write(temp_wav.name, self.samplerate, audio)

#             # Whisper API 호출
#             with open(temp_wav.name, "rb") as f:
#                 transcript = openai.Audio.transcribe(
#                     model="whisper-1", file=f, api_key=self.openai_api_key
#                 )

#         print("STT 결과: ", transcript["text"])
#         return transcript["text"]


# if __name__ == "__main__":
#     stt = STT(openai_api_key)
#     output_message = stt.speech2text()


from langchain_openai import ChatOpenAI
from openai import OpenAI  # [수정 1] OpenAI 클라이언트 클래스 import
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import tempfile
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(dotenv_path=os.path.join(".env"))
openai_api_key = os.getenv("OPENAI_API_KEY")

class STT:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        # [수정 2] 클라이언트 인스턴스 생성
        self.client = OpenAI(api_key=self.openai_api_key)
        self.duration = 5  # seconds
        self.samplerate = 16000  # Whisper는 16kHz를 선호

    def speech2text(self):
        # 녹음 설정
        print("음성 녹음을 시작합니다. \n 5초 동안 말해주세요...")
        audio = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        print("녹음 완료. Whisper에 전송 중...")

        # 임시 WAV 파일 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)
            
            # 파일 포인터를 다시 처음으로 돌릴 필요 없이, 이름을 이용해 다시 엽니다.
            # (Windows 등 일부 환경 호환성을 위해 파일을 닫았다가 다시 여는 것이 안전합니다)
            
        # [수정 3] 최신 문법으로 API 호출 (client 사용)
        with open(temp_wav.name, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )

        # [수정 4] 결과 접근 방식 변경 (딕셔너리['text'] -> 객체.text)
        print("STT 결과: ", transcript.text)
        
        # 임시 파일 삭제 (선택 사항)
        os.remove(temp_wav.name)
        
        return transcript.text

if __name__ == "__main__":
    stt = STT(openai_api_key)
    output_message = stt.speech2text()