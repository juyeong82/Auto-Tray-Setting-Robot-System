import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import warnings
from langchain_core.prompts import PromptTemplate
from STT import STT

# [수정 1] 전역에서 .env 로드 (가장 먼저 실행)
load_dotenv(dotenv_path=".env")
openai_api_key = os.getenv("OPENAI_API_KEY")

class ExtractKeyword:
    def __init__(self):
        # 클래스 내부에서 또 로드할 필요 없음 (위에서 했으므로)
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.0, openai_api_key=openai_api_key
        )
        # [수정 2] 프롬프트 강화 (한글 -> 영어 매핑 명시)
        prompt_content = """
            당신은 사용자의 발화(STT 결과)에서 로봇 제어 명령을 추출하는 에이전트입니다.
            
            <매핑 규칙>
            - 입력 "해머", "망치" -> 출력 "hammer"
            - 입력 "포즈1", "포스1", "1번" -> 출력 "pos1"
            - 입력 "포즈2", "포스2", "2번" -> 출력 "pos2"
            
            <도구 리스트>
            - hammer, screwdriver, wrench

            <위치 리스트>
            - pos1, pos2, pos3

            <출력 형식>
            - 반드시 "도구 / 목적지" 형식을 지킬 것.
            - 구분자 "/"가 반드시 존재해야 함.
            
            <예시>
            - 입력: "해머를 포즈1에 갖다 넣어" -> 출력: hammer / pos1
            - 입력: "드라이버 줘" -> 출력: screwdriver /

            <사용자 입력>
            "{user_input}"
        """
        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], template=prompt_content
        )
        self.lang_chain = self.prompt_template | self.llm

    def extract_keyword(self, output_message):
        response = self.lang_chain.invoke({"user_input": output_message})
        raw_content = response.content.strip()
        
        # [수정 3] 디버깅용 출력 (LLM이 뭐라고 했는지 확인 필수!)
        print(f"DEBUG: LLM Raw Output -> '{raw_content}'")

        if "/" not in raw_content:
            # LLM이 형식을 어겼을 경우 강제 처리
            print("WARNING: '/' 구분자가 없습니다. 형식을 맞춥니다.")
            raw_content += " /"

        result = raw_content.split("/")
        
        if len(result) != 2:
            warnings.warn(f"Format Error. Split result: {result}")
            return None, None

        object_part, destination_part = result[0].strip(), result[1].strip()
        objects = object_part.split()
        destinations = destination_part.split()

        print(f"Extracted Objects: {objects}")
        print(f"Extracted Destinations: {destinations}")

        return objects, destinations


if __name__ == "__main__":
    # [수정 4] 전역 변수 사용
    if openai_api_key is None:
        print("ERROR: .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
    else:
        stt = STT(openai_api_key)
        output_message = stt.speech2text()
        print(f"STT Result: {output_message}")
        
        if output_message:
            extract_keyword = ExtractKeyword()
            tools, locs = extract_keyword.extract_keyword(output_message)
            
            # 로봇에게 보낼 데이터 확인
            print(f"Final Result -> Tools: {tools}, Locs: {locs}")
