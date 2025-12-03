#!/usr/bin/env python3
# kiosk_continuous.py
#
# [최종 완성]
# - 연속 대화 기능 (Multi-turn): 주문 완료까지 대화 루프 유지
# - 대화 기억 (Memory): LangChain History 적용
# - 메뉴 사진 추가
# - ROS2 서비스 서버로 메뉴 및 수량 리스트 전송
# - UI/TTS/로봇 통신 통합

import os
import sys
import threading
import time
from collections import defaultdict
import numpy as np
import pyaudio

from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from ff_robot_interfaces.srv import OrderService

import speech_recognition as sr
import openwakeword
from openwakeword.model import Model

# TTS
from gtts import gTTS

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QGridLayout,
    QFrame, QListWidget, QListWidgetItem, QMessageBox,
    QSizePolicy, QScrollArea
)
from PyQt5.QtGui import QFont, QPixmap

# LangChain (기억 기능 추가)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ==============================
#  1. 설정 및 데이터
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")
WAKEWORD_MODEL_FILE = "hello_rokey_8332_32.tflite"
MENU_BOARD_IMAGE = "menu.jpg"

MENU_ITEMS = {
    "burger1": {"name": "불고기버거", "price": 3000, "category": "burger", "image": "burger_bulgogi.jpg"},
    "burger2": {"name": "치즈버거", "price": 3500, "category": "burger", "image": "burger_cheese.jpg"},
    "burger3": {"name": "새우버거", "price": 3200, "category": "burger", "image": "burger_shrimp.jpg"},
    "fries": {"name": "감자튀김", "price": 1500, "category": "side", "image": "side_fries.jpg"},
    "nugget": {"name": "치킨너겟", "price": 2000, "category": "side", "image": "side_nugget.jpg"},
    "coke": {"name": "콜라", "price": 1800, "category": "drink", "image": "drink_coke.jpg"},
    "cider": {"name": "사이다", "price": 1800, "category": "drink", "image": "drink_soda.jpg"},
}

SORTED_MENU_KEYS = [
    "burger1", "burger2", "burger3", 
    "fries", "nugget",
    "coke", "cider"
]

SET_EXTRA_PRICE = 3000
DEFAULT_SET_COMPONENTS = ["fries", "coke"] 

BASE_BG = "#FFF5E1"
PRIMARY_RED = "#D9251D"
PRIMARY_YELLOW = "#FFC72C"
DARK_BROWN = "#3E2723"
CARD_BG = "#FFFFFF"
SET_CARD_BG = "#FFF0C2"

# ==============================
#  2. 유틸리티 (TTS, Wakeup, LLM)
# ==============================

def speak(text):
    """TTS: 텍스트를 음성으로 변환하여 재생"""
    print(f"🔊 [Robot]: {text}")
    try:
        tts = gTTS(text=text, lang='ko')
        filename = os.path.join(BASE_DIR, "voice_temp.mp3")
        tts.save(filename)
        os.system(f"mpg123 -q {filename}")
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"TTS Error: {e}")

class WakeupWordDetector:
    def __init__(self, model_file, buffer_size=1280):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_file)
        if not os.path.exists(model_path):
            print(f"모델 파일 다운로드 중...")
            openwakeword.utils.download_models()
            model_path = model_file
        self.model = Model(wakeword_models=[model_path])
        self.buffer_size = buffer_size
        self.stream = None

    def set_stream(self, stream):
        self.stream = stream

    def is_wakeup(self):
        if self.stream is None: return False
        try:
            data = self.stream.read(self.buffer_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            prediction = self.model.predict(audio_data)
            for mdl in self.model.prediction_buffer.keys():
                if self.model.prediction_buffer[mdl][-1] > 0.5: return True
        except Exception:
            pass
        return False

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# [수정] 대화 내역(history)을 포함하는 프롬프트
VOICE_PROMPT_TEMPLATE = """
당신은 '로키버거'의 친절한 로봇 점원입니다.

<메뉴 정보>
- 버거: 불고기버거(burger1), 치즈버거(burger2), 새우버거(burger3)
- 사이드: 감자튀김(fries), 치킨너겟(nugget)
- 음료: 콜라(coke), 사이다(cider)
- 세트메뉴: 각 버거 코드 뒤에 '_set'을 붙임 (예: burger1_set)

<대화 규칙>
1. 손님의 말을 듣고 필요한 정보(버거 종류, 사이드/음료 변경 여부)를 차근차근 물어보세요.
2. **한 번에 하나씩만 질문하세요.** (예: "어떤 버거로 드릴까요?", "음료는 콜라 괜찮으신가요?")
3. 손님이 "주문 끝", "그게 다야", "결제해줘" 등 주문을 마치는 표현을 하면, 
   최종 주문 내역을 정리해주고 마지막 줄에 <데이터 추출 규칙>을 적용하세요.
4. 아직 주문이 완료되지 않았다면 [ORDER: ...] 태그를 절대 붙이지 마세요.
5. 손님에게는 태그를 제외한 친절한 한국어 멘트로만 대답하세요.

<데이터 추출 규칙>
- 주문이 최종 확정되었을 때만, 답변 맨 마지막에 [ORDER: 메뉴코드1, 수량1, 메뉴코드2, 수량2, ...] 형식을 추가하고, 
  마지막에 총 금액을 알려주세요.
- 세트 메뉴는 '_set'을 붙이세요.
- 예시: "주문하신 불고기버거 세트 하나 맞으신가요? 결제 도와드릴게요. [ORDER: burger1_set, 1]"

<이전 대화 내역>
{history}

손님: {input}
로봇:
"""

_llm_chain = None
_memory_store = {}

def get_session_history(session_id: str):
    if session_id not in _memory_store:
        _memory_store[session_id] = ChatMessageHistory()
    return _memory_store[session_id]

def init_llm():
    global _llm_chain
    if _llm_chain is not None: return
    if not OPENAI_API_KEY: raise RuntimeError("OPENAI_API_KEY 환경변수 확인 필요")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(input_variables=["history", "input"], template=VOICE_PROMPT_TEMPLATE)
    
    # RunnableWithMessageHistory로 감싸서 대화 기억 능력 추가
    chain = prompt | llm | StrOutputParser()
    _llm_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

def parse_order_data(order_data_str):
    items = [item.strip() for item in order_data_str.split(",") if item.strip()]
    if len(items) % 2 != 0: raise ValueError("잘못된 ORDER 데이터 형식")
    
    item_names = []
    item_quantities = []
    
    for i in range(0, len(items), 2):
        code = items[i]
        qty = int(items[i+1])
        
        if code.endswith("_set"):
            base_burger = code.replace("_set", "")
            item_names.append(base_burger)
            item_quantities.append(qty)
            for side in DEFAULT_SET_COMPONENTS:
                item_names.append(side)
                item_quantities.append(qty)
        else:
            item_names.append(code)
            item_quantities.append(qty)
    
    return item_names, item_quantities

def get_display_name(code):
    return MENU_ITEMS.get(code, {}).get("name", code)

def load_menu_pixmap(info, size=(140, 140)):
    image_name = info.get("image")
    if not image_name: return None
    path = os.path.join(BASE_DIR, image_name)
    if not os.path.exists(path):
         path = os.path.join(ASSET_DIR, image_name)
         if not os.path.exists(path): return None
    pixmap = QPixmap(path)
    if pixmap.isNull(): return None
    return pixmap.scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)

# ==============================
#  3. ROS2 서비스 클라이언트
# ==============================
class OrderWorker(QThread):
    finished = pyqtSignal(bool, str, str)

    def __init__(self, node, item_names, item_quantities):
        super().__init__()
        self.node = node
        self.item_names = item_names
        self.item_quantities = item_quantities

    def run(self):
        cli = None
        try:
            cli = self.node.create_client(OrderService, "/dsr01/order_service")
            if not cli.wait_for_service(timeout_sec=2.0):
                self.finished.emit(False, "Order service를 찾을 수 없습니다.", "")
                return

            req = OrderService.Request()
            req.item_names = self.item_names
            req.item_quantities = [int(q) for q in self.item_quantities]
            
            print(f"\n🚀 [OrderWorker] 전송: {req.item_names}, {list(req.item_quantities)}") 

            future = cli.call_async(req)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)

            if future.done():
                res = future.result()
                if res:
                    self.finished.emit(bool(res.assigned_order_id), res.message, res.assigned_order_id)
                else:
                    self.finished.emit(False, "응답 없음", "")
            else:
                self.finished.emit(False, "시간 초과", "")

        except Exception as e:
            self.finished.emit(False, f"오류: {e}", "")
        finally:
            if cli: self.node.destroy_client(cli)

class OrderClient(QObject):
    orderFinished = pyqtSignal(bool, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        if not rclpy.ok(): rclpy.init(args=None)
        self.node = rclpy.create_node("kiosk_order_client_node")
        self.worker = None

    def send_order(self, item_names, item_quantities):
        if self.worker is not None:
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait()
            self.worker.deleteLater()
        
        self.worker = OrderWorker(self.node, item_names, item_quantities)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_finished(self, success, msg, oid):
        self.orderFinished.emit(success, msg, oid)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def shutdown(self):
        if self.worker:
            self.worker.quit()
            self.worker.wait()
        if self.node:
            self.node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


# ==============================
#  4. 음성 주문 스레드 (연속 대화형)
# ==============================
class VoiceOrderThread(QThread):
    statusUpdate = pyqtSignal(str)
    recognizedText = pyqtSignal(str)
    resultReady = pyqtSignal(str, list, list)
    error = pyqtSignal(str)

    def run(self):
        pa = None
        stream = None
        try:
            # 1. 호출어 대기
            self.statusUpdate.emit("🤖 'Hello 로키'라고 불러주세요!")
            pa = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)
            detector = WakeupWordDetector(model_file=WAKEWORD_MODEL_FILE, buffer_size=1280)
            detector.set_stream(stream)
            
            while True:
                if self.isInterruptionRequested(): raise InterruptedError
                if detector.is_wakeup():
                    self.statusUpdate.emit("✨ 네! 말씀해주세요.")
                    speak("네! 말씀해주세요.")
                    break
            
            # 호출어 감지 후 마이크 스트림 닫기 (STT를 위해)
            stream.stop_stream()
            stream.close()
            pa.terminate()

            # 2. 연속 대화 루프 시작
            init_llm()
            session_id = str(time.time()) # 대화 세션 ID 생성
            r = sr.Recognizer()

            while True: # 주문 완료될 때까지 반복
                if self.isInterruptionRequested(): raise InterruptedError

                # STT 듣기
                self.statusUpdate.emit("👂 듣고 있습니다...")
                try:
                    with sr.Microphone() as source:
                        r.adjust_for_ambient_noise(source, duration=0.5)
                        audio = r.listen(source, timeout=5, phrase_time_limit=8)
                    text = r.recognize_google(audio, language='ko-KR')
                    self.recognizedText.emit(f"🗣️ 손님: {text}")
                except sr.WaitTimeoutError:
                    # 말이 없으면 다시 듣기 시도 (혹은 안내 멘트 후 재시도)
                    continue 
                except Exception:
                    continue

                # LLM 응답 생성
                self.statusUpdate.emit("🧠 생각 중...")
                response = _llm_chain.invoke(
                    {"input": text},
                    config={"configurable": {"session_id": session_id}}
                )
                
                # 주문 완료([ORDER: ...]) 확인
                if "[ORDER:" in response:
                    parts = response.split("[ORDER:")
                    robot_ment = parts[0].strip()
                    order_data = parts[1].replace("]", "").strip()
                    
                    # 마지막 멘트 말하기
                    speak(robot_ment)
                    
                    # 데이터 파싱 후 GUI로 전송 및 종료
                    item_names, item_quantities = parse_order_data(order_data)
                    self.resultReady.emit(robot_ment, item_names, item_quantities)
                    break # 대화 루프 탈출
                else:
                    # 주문 미완료: 로봇 대답 말하고 다시 듣기 루프로
                    self.recognizedText.emit(f"🤖 로키: {response}")
                    speak(response)
                    # 다시 while 처음으로 돌아가서 듣기

        except InterruptedError:
            pass
        except Exception as e:
            self.error.emit(f"오류 발생: {e}")
        finally:
            if stream and stream.is_active(): stream.stop_stream(); stream.close()
            if pa: pa.terminate()

    def stop(self):
        self.requestInterruption()
        self.wait()


# ==============================
#  5. 메인 윈도우 (GUI)
# ==============================
class KioskWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.single_counts = defaultdict(int)
        self.set_counts = defaultdict(int)
        self.qty_labels_single = {}
        self.qty_labels_set = {}
        self.pending_voice_order_names = None
        self.pending_voice_order_quantities = None
        self.order_in_progress = False
        self.last_order_id = ""
        self.order_client = OrderClient()
        self.order_client.orderFinished.connect(self.on_order_finished)
        self.voice_thread = None 
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("ROKEY BURGER 스마트 키오스크")
        self.setStyleSheet(f"QWidget {{ background-color: {BASE_BG}; font-family: Noto Sans KR; }} QPushButton {{ background-color: {PRIMARY_YELLOW}; border-radius: 15px; padding: 10px; font-size: 16px; color: {DARK_BROWN}; border: 2px solid {PRIMARY_RED}; }} QPushButton#PrimaryCTA {{ background-color: {PRIMARY_RED}; color: #FFFFFF; border: none; }}")
        main_layout = QVBoxLayout(self)
        
        # 상단바
        top_layout = QHBoxLayout()
        logo_path = os.path.join(ASSET_DIR, "logo.png")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            pix = QPixmap(logo_path).scaledToHeight(60, Qt.SmoothTransformation)
            logo_label.setPixmap(pix)
            top_layout.addWidget(logo_label)
        
        title = QLabel("ROKEY BURGER")
        title.setFont(QFont("Noto Sans KR", 24, QFont.Bold))
        title.setStyleSheet(f"color: {DARK_BROWN};")
        top_layout.addWidget(title)
        top_layout.addStretch()
        main_layout.addLayout(top_layout)

        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        self.home_index = self.stack.addWidget(self.build_home_page())
        self.touch_index = self.stack.addWidget(self.build_touch_order_page())
        self.voice_index = self.stack.addWidget(self.build_voice_order_page())
        self.complete_index = self.stack.addWidget(self.build_complete_page())

        self.stack.setCurrentIndex(self.home_index)
        self.resize(1024, 650)

    def reset_and_go_home(self):
        self.clear_touch_order()
        self.pending_voice_order_names = None
        self.pending_voice_order_quantities = None
        if self.voice_thread and self.voice_thread.isRunning():
            self.voice_thread.stop()
        self.lbl_voice_status.setText("준비 중...")
        self.lbl_voice_robot.setText("")
        self.btn_voice_confirm.setEnabled(False)
        self.btn_start_voice.setEnabled(True)
        self.order_in_progress = False
        self.stack.setCurrentIndex(self.home_index)

    def build_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()
        btn_voice = QPushButton("🎙️ 음성으로 주문")
        btn_voice.setMinimumHeight(120)
        btn_voice.setObjectName("PrimaryCTA")
        btn_voice.clicked.connect(self.on_select_voice_mode)
        btn_touch = QPushButton("👆 터치로 주문")
        btn_touch.setMinimumHeight(120)
        btn_touch.setObjectName("PrimaryCTA")
        btn_touch.clicked.connect(self.on_select_touch_mode)
        layout.addWidget(btn_voice)
        layout.addSpacing(20)
        layout.addWidget(btn_touch)
        layout.addStretch()
        return page

    def build_touch_order_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        nav_layout = QHBoxLayout()
        btn_back = QPushButton("← 처음으로")
        btn_back.setFixedSize(120, 50)
        btn_back.clicked.connect(self.reset_and_go_home)
        nav_layout.addWidget(btn_back)
        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        body_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        cat_layout = QHBoxLayout()
        self.btn_cat_all = QPushButton("전체")
        self.btn_cat_set = QPushButton("버거(세트)")
        self.btn_cat_single = QPushButton("버거(단품)")
        self.btn_cat_side = QPushButton("사이드")
        self.btn_cat_drink = QPushButton("음료")

        self.category_buttons = [self.btn_cat_all, self.btn_cat_set, self.btn_cat_single, self.btn_cat_side, self.btn_cat_drink]
        for btn in self.category_buttons:
            btn.setCheckable(True)
            cat_layout.addWidget(btn)

        self.btn_cat_all.clicked.connect(lambda: self.populate_menu_category("all"))
        self.btn_cat_set.clicked.connect(lambda: self.populate_menu_category("set"))
        self.btn_cat_single.clicked.connect(lambda: self.populate_menu_category("single"))
        self.btn_cat_side.clicked.connect(lambda: self.populate_menu_category("side"))
        self.btn_cat_drink.clicked.connect(lambda: self.populate_menu_category("drink"))

        left_panel.addLayout(cat_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.menu_grid_container = QWidget()
        self.menu_grid_layout = QGridLayout(self.menu_grid_container)
        self.menu_grid_layout.setSpacing(15)
        scroll.setWidget(self.menu_grid_container)
        left_panel.addWidget(scroll)
        body_layout.addLayout(left_panel, stretch=3)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("🛒 주문 내역"))
        self.order_list_widget = QListWidget()
        right_panel.addWidget(self.order_list_widget)
        self.lbl_total_price = QLabel("총 금액: 0원")
        self.lbl_total_price.setFont(QFont("Noto Sans KR", 18, QFont.Bold))
        self.lbl_total_price.setAlignment(Qt.AlignRight)
        right_panel.addWidget(self.lbl_total_price)
        
        btn_clear = QPushButton("전체 취소")
        btn_clear.clicked.connect(self.clear_touch_order)
        btn_order = QPushButton("주문 완료")
        btn_order.setObjectName("PrimaryCTA")
        btn_order.clicked.connect(self.on_complete_touch_order)
        self.btn_complete_order = btn_order 
        self.btn_clear_order = btn_clear
        right_panel.addWidget(btn_clear)
        right_panel.addWidget(btn_order)
        body_layout.addLayout(right_panel, stretch=1)
        layout.addLayout(body_layout)
        self.btn_cat_all.setChecked(True)
        self.populate_menu_category("all")
        return page

    def populate_menu_category(self, mode):
        self.btn_cat_all.setChecked(mode == "all")
        self.btn_cat_set.setChecked(mode == "set")
        self.btn_cat_single.setChecked(mode == "single")
        self.btn_cat_side.setChecked(mode == "side")
        self.btn_cat_drink.setChecked(mode == "drink")
        while self.menu_grid_layout.count():
            item = self.menu_grid_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.qty_labels_single.clear()
        self.qty_labels_set.clear()
        cards = []
        for code in SORTED_MENU_KEYS:
            info = MENU_ITEMS[code]
            cat = info["category"]
            if cat == "burger":
                if mode in ["all", "set"]: cards.append(self.create_burger_set_card(code, info))
                if mode in ["all", "single"]: cards.append(self.create_burger_single_card(code, info))
            elif cat == "side":
                if mode in ["all", "side"]: cards.append(self.create_single_item_card(code, info))
            elif cat == "drink":
                if mode in ["all", "drink"]: cards.append(self.create_single_item_card(code, info))
        col_max = 3
        row, col = 0, 0
        for card in cards:
            self.menu_grid_layout.addWidget(card, row, col)
            col += 1
            if col >= col_max: col = 0; row += 1
        if row == 0 and col == 0: self.menu_grid_layout.addWidget(QLabel("표시할 메뉴가 없습니다."), 0, 0)

    def _make_card(self, code, name, price, count_dict, label_dict, callback, is_set=False):
        frame = QFrame()
        frame.setStyleSheet(f"background-color: {SET_CARD_BG if is_set else CARD_BG}; border-radius: 10px; border: 1px solid #DDD;")
        frame.setFixedSize(200, 240)
        layout = QVBoxLayout(frame)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)
        info = MENU_ITEMS[code]
        pix = load_menu_pixmap(info, (120, 100))
        if pix:
            img = QLabel(); img.setPixmap(pix); img.setAlignment(Qt.AlignCenter); layout.addWidget(img)
        lbl_name = QLabel(name); lbl_name.setFont(QFont("Noto Sans KR", 12, QFont.Bold)); lbl_name.setAlignment(Qt.AlignCenter); lbl_name.setWordWrap(True); layout.addWidget(lbl_name)
        lbl_price = QLabel(f"{price:,}원"); lbl_price.setFont(QFont("Noto Sans KR", 11)); lbl_price.setAlignment(Qt.AlignCenter); layout.addWidget(lbl_price)
        btn_box = QHBoxLayout()
        btn_minus = QPushButton("-"); btn_plus = QPushButton("+")
        btn_minus.setFixedSize(30, 30); btn_plus.setFixedSize(30, 30)
        current_qty = count_dict[code]
        lbl_qty = QLabel(str(current_qty)); lbl_qty.setAlignment(Qt.AlignCenter)
        label_dict[code] = lbl_qty 
        btn_minus.clicked.connect(lambda: callback(code, -1)); btn_plus.clicked.connect(lambda: callback(code, 1))
        btn_box.addWidget(btn_minus); btn_box.addWidget(lbl_qty); btn_box.addWidget(btn_plus); layout.addLayout(btn_box)
        return frame

    def create_burger_single_card(self, code, info): return self._make_card(code, info["name"]+"(단품)", info["price"], self.single_counts, self.qty_labels_single, self.add_single_item)
    def create_burger_set_card(self, code, info): return self._make_card(code, info["name"]+" 세트", info["price"] + SET_EXTRA_PRICE, self.set_counts, self.qty_labels_set, self.add_set_item, is_set=True)
    def create_single_item_card(self, code, info): return self._make_card(code, info["name"], info["price"], self.single_counts, self.qty_labels_single, self.add_single_item)
    def add_single_item(self, code, delta):
        self.single_counts[code] = max(0, self.single_counts[code] + delta)
        if code in self.qty_labels_single: self.qty_labels_single[code].setText(str(self.single_counts[code]))
        self.update_order_summary()
    def add_set_item(self, code, delta):
        self.set_counts[code] = max(0, self.set_counts[code] + delta)
        if code in self.qty_labels_set: self.qty_labels_set[code].setText(str(self.set_counts[code]))
        self.update_order_summary()
    def clear_touch_order(self):
        self.single_counts.clear(); self.set_counts.clear()
        if self.btn_cat_all.isChecked(): self.populate_menu_category("all")
        elif self.btn_cat_set.isChecked(): self.populate_menu_category("set")
        elif self.btn_cat_single.isChecked(): self.populate_menu_category("single")
        elif self.btn_cat_side.isChecked(): self.populate_menu_category("side")
        elif self.btn_cat_drink.isChecked(): self.populate_menu_category("drink")
        self.update_order_summary()
    def update_order_summary(self):
        self.order_list_widget.clear(); total = 0
        for code, qty in self.single_counts.items():
            if qty > 0: price = MENU_ITEMS[code]["price"] * qty; total += price; self.order_list_widget.addItem(f"{MENU_ITEMS[code]['name']} x{qty} : {price:,}원")
        for code, qty in self.set_counts.items():
            if qty > 0: price = (MENU_ITEMS[code]["price"] + SET_EXTRA_PRICE) * qty; total += price; self.order_list_widget.addItem(f"{MENU_ITEMS[code]['name']} 세트 x{qty} : {price:,}원")
        self.lbl_total_price.setText(f"총 금액: {total:,}원")
    def on_complete_touch_order(self):
        if self.order_in_progress: return
        names, qtys = [], []
        for c, q in self.single_counts.items():
            if q > 0: names.append(c); qtys.append(q)
        for c, q in self.set_counts.items():
            if q > 0: names.extend([c, "fries", "coke"]); qtys.extend([q, q, q])
        if not names: QMessageBox.warning(self, "경고", "메뉴를 선택해주세요."); return
        self.order_in_progress = True
        self.order_client.send_order(names, qtys)

    def build_voice_order_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        btn_back = QPushButton("← 처음으로")
        btn_back.setFixedSize(150, 50)
        btn_back.clicked.connect(self.reset_and_go_home)
        layout.addWidget(btn_back, alignment=Qt.AlignLeft)
        menu_board_path = os.path.join(BASE_DIR, MENU_BOARD_IMAGE)
        if not os.path.exists(menu_board_path): menu_board_path = os.path.join(ASSET_DIR, MENU_BOARD_IMAGE)
        if os.path.exists(menu_board_path):
            lbl_menu_board = QLabel()
            pix = QPixmap(menu_board_path)
            if not pix.isNull(): pix = pix.scaledToHeight(360, Qt.SmoothTransformation); lbl_menu_board.setPixmap(pix); lbl_menu_board.setAlignment(Qt.AlignCenter); layout.addWidget(lbl_menu_board)
        self.lbl_voice_status = QLabel("준비 중..."); self.lbl_voice_status.setAlignment(Qt.AlignCenter); self.lbl_voice_status.setFont(QFont("Noto Sans KR", 20)); layout.addWidget(self.lbl_voice_status)
        self.lbl_voice_robot = QLabel(""); self.lbl_voice_robot.setAlignment(Qt.AlignCenter); layout.addWidget(self.lbl_voice_robot)
        self.btn_start_voice = QPushButton("🎙️ 음성 입력 시작"); self.btn_start_voice.hide()
        self.btn_voice_confirm = QPushButton("이대로 주문하기"); self.btn_voice_confirm.setEnabled(False); self.btn_voice_confirm.clicked.connect(self.on_confirm_voice_order); layout.addWidget(self.btn_voice_confirm)
        self.btn_voice_retry = QPushButton("다시 말하기"); self.btn_voice_retry.clicked.connect(self.start_voice_recognition); layout.addWidget(self.btn_voice_retry)
        return page
    def on_select_touch_mode(self): self.stack.setCurrentIndex(self.touch_index)
    def on_select_voice_mode(self): self.stack.setCurrentIndex(self.voice_index); self.start_voice_recognition()
    def start_voice_recognition(self):
        if self.voice_thread and self.voice_thread.isRunning(): self.voice_thread.stop()
        self.lbl_voice_status.setText("로봇을 불러주세요...")
        self.btn_voice_confirm.setEnabled(False)
        self.voice_thread = VoiceOrderThread()
        self.voice_thread.statusUpdate.connect(self.lbl_voice_status.setText)
        self.voice_thread.recognizedText.connect(self.lbl_voice_robot.setText)
        self.voice_thread.resultReady.connect(self.on_voice_result)
        self.voice_thread.error.connect(self.on_voice_error)
        self.voice_thread.start()
    def on_back_to_home(self):
        if self.voice_thread and self.voice_thread.isRunning(): self.voice_thread.stop()
        self.stack.setCurrentIndex(self.home_index)
    def on_voice_result(self, ment, names, qtys):
        self.lbl_voice_status.setText("주문 확인")
        self.lbl_voice_robot.setText(f"로봇: {ment}")
        self.pending_voice_order_names = names
        self.pending_voice_order_quantities = qtys
        self.btn_voice_confirm.setEnabled(True)
    def on_voice_error(self, msg): self.lbl_voice_status.setText("오류"); self.lbl_voice_robot.setText(msg)
    def on_confirm_voice_order(self):
        if self.pending_voice_order_names:
            self.order_in_progress = True
            self.order_client.send_order(self.pending_voice_order_names, self.pending_voice_order_quantities)

    def build_complete_page(self):
        page = QWidget(); layout = QVBoxLayout(page)
        layout.addWidget(QLabel("주문 완료! 카운터에서 번호를 확인하세요.", alignment=Qt.AlignCenter))
        self.lbl_order_num = QLabel("-"); self.lbl_order_num.setFont(QFont("Arial", 30)); self.lbl_order_num.setAlignment(Qt.AlignCenter); layout.addWidget(self.lbl_order_num)
        btn_home = QPushButton("처음으로"); btn_home.clicked.connect(self.reset_and_go_home); layout.addWidget(btn_home)
        return page
    def on_order_finished(self, success, msg, oid):
        self.order_in_progress = False
        if success: self.lbl_order_num.setText(f"주문번호: {oid}"); self.stack.setCurrentIndex(self.complete_index)
        else: QMessageBox.warning(self, "실패", msg)
    def on_next_customer(self): self.reset_and_go_home()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = KioskWindow()
    win.show()
    sys.exit(app.exec_())