#!/usr/bin/env python3
# kiosk_node.py (PyQt5 GUI Ver.)
#
# 기능: 음성(AI) 및 터치(버튼) 주문이 가능한 통합 키오스크 v1.6 버전 통합
# 특징: PyQt5 기반 GUI, ROS2 Service 통신, OpenAI 연동합

import sys
import os
import tempfile
import time
import threading

# ROS2
import rclpy
from rclpy.node import Node
from ff_robot_interfaces.srv import OrderService

# Audio / AI v1.6
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

# PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget, 
                             QGridLayout, QMessageBox, QListWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# ==============================================================================
# [1] 백엔드 로직: AI 처리 (STT & LLM) - GUI와 무관한 로직
# ==============================================================================
class AILogic:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(os.path.join(current_dir, ".env"))
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=self.api_key)
            self._init_chain()

    def _init_chain(self):
        prompt = """
        당신은 햄버거 가게 주문 AI입니다. 사용자의 말에서 아래 메뉴를 찾아 시스템 ID로 변환하세요.
        <메뉴 리스트>
        - burger_bulgogi, burger_cheese, fries, coke, sprite, coffee
        
        <규칙>
        - 메뉴 ID만 공백으로 구분하여 나열 (예: "버거 2개" -> "burger_bulgogi burger_bulgogi")
        - 메뉴 외 잡담은 무시. 인식 불가시 빈 문자열 반환.
        
        <입력>: "{user_input}"
        """
        self.chain = PromptTemplate(input_variables=["user_input"], template=prompt) | self.llm | StrOutputParser()

    def speech_to_text(self, duration=4, samplerate=16000):
        if not self.api_key: return None
        try:
            # 녹음
            audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
            sd.wait()
            
            # 파일 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav.write(tmp.name, samplerate, audio)
                tmp_path = tmp.name
            
            # Whisper API
            with open(tmp_path, "rb") as f:
                transcript = self.client.audio.transcriptions.create(model="whisper-1", file=f)
            
            os.remove(tmp_path)
            return transcript.text
        except Exception as e:
            print(f"STT Error: {e}")
            return None

    def analyze_order(self, text):
        if not self.api_key or not text: return []
        try:
            response = self.chain.invoke({"user_input": text})
            return response.strip().split()
        except Exception as e:
            print(f"LLM Error: {e}")
            return []

# ==============================================================================
# [2] 작업 스레드: GUI 멈춤 방지용 (녹음, 로봇 통신)
# ==============================================================================
class VoiceThread(QThread):
    finished_signal = pyqtSignal(str, list) # (인식된 텍스트, 메뉴 리스트)

    def __init__(self, ai_logic):
        super().__init__()
        self.ai = ai_logic

    def run(self):
        text = self.ai.speech_to_text()
        if text:
            items = self.ai.analyze_order(text)
            self.finished_signal.emit(text, items)
        else:
            self.finished_signal.emit("", [])

class RobotClient(Node):
    def __init__(self):
        super().__init__('kiosk_gui_node')
        self.cli = self.create_client(OrderService, '/dsr01/order_service')
        # 주의: 생성자에서 wait_for_service를 하면 GUI가 뜰때까지 멈출 수 있으므로 체크만 함

    def send_order_sync(self, names, quantities):
        if not self.cli.service_is_ready():
            return False, "로봇 연결 안됨"
        
        req = OrderService.Request()
        req.item_names = names
        req.item_quantities = quantities
        
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.done():
            res = future.result()
            if res.assigned_order_id:
                return True, f"주문 완료! (번호: {res.assigned_order_id})"
            else:
                return False, f"주문 거절: {res.message}"
        else:
            return False, "응답 시간 초과"

# ==============================================================================
# [3] 메인 GUI 클래스
# ==============================================================================
class KioskApp(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.node = ros_node
        self.ai = AILogic()
        self.setWindowTitle("🤖 AI Robot Kiosk")
        self.setGeometry(100, 100, 800, 600)
        
        # 스타일 설정
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QPushButton { 
                background-color: #007bff; color: white; border-radius: 10px; 
                font-size: 18px; padding: 15px; 
            }
            QPushButton:hover { background-color: #0056b3; }
            QLabel { font-size: 16px; color: #333; }
        """)

        # 메인 위젯 및 스택(페이지) 설정
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.pages = QStackedWidget()
        self.layout.addWidget(self.pages)

        # 페이지들 생성
        self.init_home_page()
        self.init_voice_page()
        self.init_touch_page()
        
        # 첫 페이지 설정
        self.pages.setCurrentIndex(0)

    # --- 1. 홈 화면 ---
    def init_home_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("🍔 햄버거 로봇 키오스크 🍟")
        title.setFont(QFont("Arial", 30, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        btn_voice = QPushButton("🎤 음성으로 주문하기")
        btn_voice.clicked.connect(lambda: self.pages.setCurrentIndex(1))
        
        btn_touch = QPushButton("👆 터치로 주문하기")
        btn_touch.clicked.connect(lambda: self.pages.setCurrentIndex(2))

        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(50)
        layout.addWidget(btn_voice)
        layout.addWidget(btn_touch)
        layout.addStretch()
        
        page.setLayout(layout)
        self.pages.addWidget(page)

    # --- 2. 음성 주문 화면 ---
    def init_voice_page(self):
        page = QWidget()
        layout = QVBoxLayout()

        lbl_status = QLabel("버튼을 누르고 말씀해주세요.")
        lbl_status.setAlignment(Qt.AlignCenter)
        lbl_status.setFont(QFont("Arial", 20))
        self.lbl_voice_status = lbl_status

        btn_record = QPushButton("🔴 말하기 시작 (5초)")
        btn_record.clicked.connect(self.start_voice_recording)
        
        btn_back = QPushButton("🏠 처음으로")
        btn_back.setStyleSheet("background-color: #6c757d;")
        btn_back.clicked.connect(lambda: self.pages.setCurrentIndex(0))

        layout.addStretch()
        layout.addWidget(lbl_status)
        layout.addSpacing(30)
        layout.addWidget(btn_record)
        layout.addSpacing(10)
        layout.addWidget(btn_back)
        layout.addStretch()

        page.setLayout(layout)
        self.pages.addWidget(page)

    def start_voice_recording(self):
        if not self.ai.api_key:
            QMessageBox.critical(self, "오류", "API KEY가 없습니다.")
            return

        self.lbl_voice_status.setText("👂 듣고 있습니다... (녹음중)")
        self.voice_thread = VoiceThread(self.ai)
        self.voice_thread.finished_signal.connect(self.on_voice_finished)
        self.voice_thread.start()
        
        # 버튼 비활성화 (중복 방지)
        self.sender().setEnabled(False)
        self.voice_record_btn = self.sender()

    def on_voice_finished(self, text, items):
        self.voice_record_btn.setEnabled(True)
        if not text:
            self.lbl_voice_status.setText("❌ 아무 소리도 들리지 않았습니다.")
            return

        self.lbl_voice_status.setText(f"인식됨: \"{text}\"")
        
        if not items:
            QMessageBox.warning(self, "인식 실패", "주문 메뉴를 찾지 못했습니다.\n다시 말씀해 주세요.")
            return

        # 주문 확인 팝업
        msg = f"주문하시겠습니까?\n\n메뉴: {items}"
        reply = QMessageBox.question(self, "주문 확인", msg, QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.process_order(items)

    # --- 3. 터치 주문 화면 ---
    def init_touch_page(self):
        page = QWidget()
        layout = QHBoxLayout() # 좌: 메뉴판, 우: 장바구니

        # 메뉴 그리드
        menu_widget = QWidget()
        grid = QGridLayout()
        
        menus = [
            ("치즈버거", "burger_cheese"),
            ("불고기버거", "burger_bulgogi"),
            ("감자튀김", "fries"),
            ("콜라", "coke"),
            ("사이다", "sprite"),
            ("커피", "coffee")
        ]

        row, col = 0, 0
        for name, code in menus:
            btn = QPushButton(name)
            btn.setFixedSize(150, 100)
            btn.clicked.connect(lambda _, c=code, n=name: self.add_to_cart(c, n))
            grid.addWidget(btn, row, col)
            col += 1
            if col > 1:
                col = 0; row += 1
        
        menu_widget.setLayout(grid)

        # 장바구니 영역
        cart_layout = QVBoxLayout()
        self.cart_list = QListWidget()
        self.cart_data = {} # {code: qty}

        btn_order = QPushButton("✅ 주문 전송")
        btn_order.setStyleSheet("background-color: #28a745;")
        btn_order.clicked.connect(self.send_touch_order)

        btn_clear = QPushButton("🗑️ 비우기")
        btn_clear.setStyleSheet("background-color: #dc3545;")
        btn_clear.clicked.connect(self.clear_cart)
        
        btn_back = QPushButton("🏠 뒤로")
        btn_back.setStyleSheet("background-color: #6c757d;")
        btn_back.clicked.connect(lambda: self.pages.setCurrentIndex(0))

        cart_layout.addWidget(QLabel("🛒 장바구니"))
        cart_layout.addWidget(self.cart_list)
        cart_layout.addWidget(btn_order)
        cart_layout.addWidget(btn_clear)
        cart_layout.addWidget(btn_back)

        layout.addWidget(menu_widget, 2)
        layout.addLayout(cart_layout, 1)

        page.setLayout(layout)
        self.pages.addWidget(page)

    def add_to_cart(self, code, name):
        if code in self.cart_data:
            self.cart_data[code] += 1
        else:
            self.cart_data[code] = 1
        self.update_cart_ui()

    def clear_cart(self):
        self.cart_data.clear()
        self.update_cart_ui()

    def update_cart_ui(self):
        self.cart_list.clear()
        for code, qty in self.cart_data.items():
            self.cart_list.addItem(f"{code} x {qty}")

    def send_touch_order(self):
        if not self.cart_data:
            QMessageBox.warning(self, "경고", "장바구니가 비어있습니다.")
            return
        
        # 딕셔너리를 리스트로 변환 (burger_cheese가 2개면 -> ['burger_cheese', 'burger_cheese'])
        # 또는 서비스가 (이름, 수량) 리스트를 받으므로 그대로 전달
        item_names = list(self.cart_data.keys())
        item_quantities = list(self.cart_data.values())
        
        # 로봇 전송
        self.send_to_robot(item_names, item_quantities)
        self.clear_cart()

    # --- 공통 주문 전송 로직 ---
    def process_order(self, items_list):
        """음성 주문 결과(리스트)를 수량 집계하여 전송"""
        unique_items = list(set(items_list))
        quantities = [items_list.count(i) for i in unique_items]
        self.send_to_robot(unique_items, quantities)

    def send_to_robot(self, names, quantities):
        # 로봇 통신은 메인 스레드 멈춤 방지를 위해 짧게 처리하거나 스레드 사용 권장.
        # 여기서는 간단히 처리하되, GUI가 약간 멈출 수 있음 (Thread로 빼면 더 좋음)
        success, msg = self.node.send_order_sync(names, quantities)
        
        if success:
            QMessageBox.information(self, "주문 성공", msg)
            self.pages.setCurrentIndex(0) # 홈으로
        else:
            QMessageBox.critical(self, "주문 실패", msg)

# ==============================================================================
# [4] 메인 실행부
# ==============================================================================
def main():
    # 1. ROS2 초기화
    rclpy.init()
    ros_node = RobotClient()

    # 2. PyQt 앱 실행
    app = QApplication(sys.argv)
    kiosk = KioskApp(ros_node)
    kiosk.show()

    # 3. 종료 처리
    try:
        sys.exit(app.exec_())
    except:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()