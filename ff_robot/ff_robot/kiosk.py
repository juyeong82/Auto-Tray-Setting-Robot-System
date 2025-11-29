#!/usr/bin/env python3
# customer_kiosk_gui.py
#
# 손님용 로키버거 키오스크 GUI
# - ROS2 OrderService로 주문 전송
# - 음성 주문 / 터치 주문 선택
# - 세트 메뉴는 개별 구성(버거 + 감자튀김 + 콜라)으로 풀어서 전송

import os
import sys
import threading
from collections import defaultdict

from dotenv import load_dotenv

import rclpy
from rclpy.node import Node
from ff_robot_interfaces.srv import OrderService

import speech_recognition as sr

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QStackedWidget, QGridLayout,
    QFrame, QListWidget, QListWidgetItem, QMessageBox,
    QSizePolicy,
)
from PyQt5.QtGui import QFont, QPixmap

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==============================
#  공통 설정 / 메뉴 정의
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")  # <- 여기에 이미지 넣을 거야

# 메뉴 정의 (kvu.py에서 쓰던 메뉴 정보와 맞춤)
# image 필드는 assets/ 안에 넣을 파일 이름
MENU_ITEMS = {
    "burger1": {
        "name": "불고기버거",
        "price": 3000,
        "category": "burger",
        "image": "burger_bulgogi.jpeg",
    },
    "burger2": {
        "name": "치즈버거",
        "price": 3500,
        "category": "burger",
        "image": "burger_cheese.jpeg",
    },
    "burger3": {
        "name": "새우버거",
        "price": 3200,
        "category": "burger",
        "image": "burger_shrimp.jpeg",
    },
    "fries": {
        "name": "감자튀김",
        "price": 1500,
        "category": "side",
        "image": "side_fries.jpeg",
    },
    "nugget": {
        "name": "치킨너겟",
        "price": 2000,
        "category": "side",
        "image": "side_nugget.jpeg",
    },
    "coke": {
        "name": "콜라",
        "price": 1800,
        "category": "drink",
        "image": "drink_coke.jpeg",
    },
    "cider": {
        "name": "사이다",
        "price": 1800,
        "category": "drink",
        "image": "drink_soda.jpeg",
    },
}

SET_EXTRA_PRICE = 3000
DEFAULT_SET_SIDE = "fries"
DEFAULT_SET_DRINK = "coke"

# ---- 색상 테마 (맥도날드/버거킹 느낌) ----
BASE_BG = "#FFF5E1"        # 크림색 배경
PRIMARY_RED = "#D9251D"    # 메인 레드
PRIMARY_YELLOW = "#FFC72C" # 포인트 옐로
DARK_BROWN = "#3E2723"     # 텍스트/테두리용
CARD_BG = "#FFFFFF"
SET_CARD_BG = "#FFF0C2"

# ==============================
#  LLM (음성 주문용) 설정
# ==============================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VOICE_PROMPT_TEMPLATE = """
당신은 '로키버거'의 친절한 로봇 점원입니다.

<메뉴 정보>
- 버거: 불고기버거(burger1), 치즈버거(burger2), 새우버거(burger3)
- 사이드: 감자튀김(fries), 치킨너겟(nugget)
- 음료: 콜라(coke), 사이다(cider)

<메뉴 가격>
- 버거: 불고기버거(3000), 치즈버거(3500), 새우버거(3200)
- 사이드: 감자튀김(1500), 치킨너겟(2000)
- 음료: 콜라(1800), 사이다(1800)
- 세트메뉴 (+3000): (기본 구성: 감자튀김 + 콜라)

<대화 규칙>
1. 손님에게 필요한 정보(버거, 사이드, 음료, 포장/매장)를 자연스럽게 물어보세요.
2. 손님이 주문 관련 발화를 했다면, 그 주문이 중간이든 최종이든 항상 아래 <데이터 추출 규칙>에 따라
   현재까지의 주문 내역을 [ORDER: ...] 태그로 구조화하여 답변 맨 마지막 줄에 추가하세요.
3. 주문이 더 이어질 것 같으면, 멘트에서 "불고기버거 하나, 치즈버거 하나 주문해 두었습니다. 더 필요하신 건 없으신가요?"
   처럼 부드럽게 물어보되, 그래도 지금까지 인식한 내용을 [ORDER: ...]에 포함하세요.
4. 세트메뉴를 시키면 사이드와 음료 변경 여부를 확인하고, 최종 구성에 맞게 [ORDER]를 만드세요.
5. 손님에게는 태그를 제외한 자연스러운 한국어 멘트로 응답하세요.
6. 주문이 끝난 것 같으면 대략적인 총 금액도 말해 주세요.

<데이터 추출 규칙>
- 손님이 주문 관련으로 말한 내용이 있다면, 항상 답변 맨 마지막에 [ORDER: 메뉴코드1, 수량1, 메뉴코드2, 수량2, ...] 형식을 추가하세요.
- 메뉴 이름 대신 위 <메뉴 정보>에 있는 영어 코드(burger1, fries, coke 등)를 사용하세요.
- 세트 메뉴인 경우 구성품을 풀어서 각각의 코드로 적으세요.
- 예시 1: "불고기버거 하나 주세요" -> [ORDER: burger1, 1]
- 예시 2: "치즈버거 2개랑 콜라 1개" -> [ORDER: burger2, 2, coke, 1]
- 예시 3: "불고기버거 세트 하나" -> [ORDER: burger1, 1, fries, 1, coke, 1]

손님: {input}
로봇:
"""

_llm = None
_llm_chain = None


def init_llm():
    """지연 초기화: 처음 음성 주문할 때만 LLM 설정"""
    global _llm, _llm_chain
    if _llm_chain is not None:
        return

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

    _llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        api_key=OPENAI_API_KEY,
    )
    prompt = PromptTemplate(
        input_variables=["input"],
        template=VOICE_PROMPT_TEMPLATE,
    )
    _llm_chain = prompt | _llm | StrOutputParser()


def parse_order_data(order_data_str):
    """[ORDER: ...] 안의 문자열 -> (item_names, item_quantities)"""
    items = [item.strip() for item in order_data_str.split(",") if item.strip()]
    if len(items) % 2 != 0:
        raise ValueError("잘못된 ORDER 데이터 형식입니다.")

    item_names = []
    item_quantities = []
    for i in range(0, len(items), 2):
        name = items[i]
        qty = int(items[i + 1])
        item_names.append(name)
        item_quantities.append(qty)
    return item_names, item_quantities


def get_display_name(code: str) -> str:
    """메뉴 코드(burger1 등)를 손님에게 보여줄 이름으로 변환"""
    info = MENU_ITEMS.get(code)
    if info:
        return info["name"]
    return code


def load_menu_pixmap(info, size=(140, 140)):
    """
    메뉴 dict에서 image 필드를 읽어 QPixmap 반환.
    assets/ 경로에 파일이 없으면 None 반환해서 그냥 텍스트 카드만 쓰게 함.
    """
    image_name = info.get("image")
    if not image_name:
        return None
    path = os.path.join(ASSET_DIR, image_name)
    if not os.path.exists(path):
        return None
    pixmap = QPixmap(path)
    if pixmap.isNull():
        return None
    return pixmap.scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ==============================
#  ROS2 서비스 클라이언트
# ==============================

class OrderClient(QObject):
    """
    ROS2 OrderService 클라이언트.
    - 다른 스레드에서 서비스 호출하고
    - Qt 시그널로 결과를 GUI에 전달.
    """
    orderFinished = pyqtSignal(bool, str, str)  # success, message, order_id

    def __init__(self, parent=None):
        super().__init__(parent)
        rclpy.init(args=None)
        self.node: Node = rclpy.create_node("kiosk_order_client")
        self.cli = self.node.create_client(OrderService, "/dsr01/order_service")

    def send_order(self, item_names, item_quantities):
        """
        주문 데이터를 ROS2 서비스로 전송 (스레드에서 실행)
        """
        def _worker():
            try:
                while not self.cli.wait_for_service(timeout_sec=1.0):
                    self.node.get_logger().warn("Order service 대기 중...")

                req = OrderService.Request()
                req.item_names = item_names
                req.item_quantities = item_quantities

                future = self.cli.call_async(req)
                rclpy.spin_until_future_complete(self.node, future)
                res = future.result()

                if res is None:
                    self.orderFinished.emit(False, "서비스 응답이 없습니다.", "")
                    return

                success = bool(res.assigned_order_id)
                self.orderFinished.emit(success, res.message, res.assigned_order_id)
            except Exception as e:
                self.orderFinished.emit(False, f"서비스 호출 오류: {e}", "")

        threading.Thread(target=_worker, daemon=True).start()

    def shutdown(self):
        if self.node is not None:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ==============================
#  음성 주문 스레드
# ==============================

class VoiceOrderThread(QThread):
    recognizedText = pyqtSignal(str)                # STT 결과 텍스트
    resultReady = pyqtSignal(str, list, list)       # robot_message, item_names, item_quantities
    error = pyqtSignal(str)

    def run(self):
        # 1) 음성 인식
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                self.recognizedText.emit("마이크를 준비하는 중입니다...")
                r.adjust_for_ambient_noise(source, duration=0.5)
                self.recognizedText.emit("지금 주문을 말씀해 주세요.")
                audio = r.listen(source, timeout=5, phrase_time_limit=8)

            text = r.recognize_google(audio, language='ko-KR')
            self.recognizedText.emit(f"손님: {text}")
        except Exception:
            self.error.emit("음성을 인식하지 못했습니다. 다시 시도해 주세요.")
            return

        # 2) LLM으로 주문 파싱
        try:
            init_llm()
            global _llm_chain
            response = _llm_chain.invoke({"input": text})
        except Exception as e:
            self.error.emit(f"주문 분석 중 오류가 발생했습니다: {e}")
            return

        if "[ORDER:" not in response:
            self.error.emit("주문 정보를 인식하지 못했습니다. 다시 말씀해 주세요.")
            return

        parts = response.split("[ORDER:")
        robot_ment = parts[0].strip()
        order_data = parts[1].replace("]", "").strip()

        try:
            item_names, item_quantities = parse_order_data(order_data)
        except Exception:
            self.error.emit("주문 데이터 파싱에 실패했습니다. 다시 말씀해 주세요.")
            return

        self.resultReady.emit(robot_ment, item_names, item_quantities)


# ==============================
#  메인 키오스크 GUI
# ==============================

class KioskWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 주문 상태
        self.single_counts = defaultdict(int)  # 단품/사이드/음료 수량
        self.set_counts = defaultdict(int)     # 버거 세트 수량
        self.qty_labels_single = {}            # code -> QLabel
        self.qty_labels_set = {}               # burger_code -> QLabel

        self.pending_voice_order_names = None
        self.pending_voice_order_quantities = None

        self.order_in_progress = False
        self.last_order_id = ""

        # ROS 주문 클라이언트
        self.order_client = OrderClient()
        self.order_client.orderFinished.connect(self.on_order_finished)

        self.init_ui()

    # ------------- UI 구성 -------------

    def init_ui(self):
        self.setWindowTitle("ROKEY BURGER 셀프 키오스크")
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {BASE_BG};
                font-family: Noto Sans CJK KR, Noto Sans KR, Pretendard, Arial;
            }}
            QPushButton {{
                border-radius: 16px;
                padding: 14px 18px;
                font-size: 18px;
                background-color: {PRIMARY_YELLOW};
                color: {DARK_BROWN};
                border: 2px solid {PRIMARY_RED};
            }}
            QPushButton:hover {{
                background-color: #FFE48A;
            }}
            QPushButton:disabled {{
                background-color: #E0E0E0;
                color: #888888;
                border: none;
            }}
            QPushButton#PrimaryCTA {{
                background-color: {PRIMARY_RED};
                color: #FFFFFF;
                border: none;
            }}
            QPushButton#PrimaryCTA:hover {{
                background-color: #F44336;
            }}
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)

        # 상단 로고 (assets/logo.png 있으면 표시)
        logo_path = os.path.join(ASSET_DIR, "logo.png")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            pix = QPixmap(logo_path)
            if not pix.isNull():
                pix = pix.scaledToHeight(80, Qt.SmoothTransformation)
                logo_label.setPixmap(pix)
                logo_label.setAlignment(Qt.AlignCenter)
                main_layout.addWidget(logo_label)

        # 상단 타이틀
        title = QLabel("ROKEY BURGER 셀프 키오스크")
        title.setFont(QFont("Noto Sans KR", 26, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {DARK_BROWN};")
        main_layout.addWidget(title)

        # 중앙 스택 (화면 전환)
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, stretch=1)

        # 각 페이지 생성
        self.home_index = self.stack.addWidget(self.build_home_page())
        self.touch_index = self.stack.addWidget(self.build_touch_order_page())
        self.voice_index = self.stack.addWidget(self.build_voice_order_page())
        self.complete_index = self.stack.addWidget(self.build_complete_page())

        # 처음 화면
        self.stack.setCurrentIndex(self.home_index)

        self.resize(1024, 600)

    # ------------- 페이지: 첫 화면 -------------

    def build_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)

        label = QLabel("어서오세요!\n주문 방식을 선택해 주세요.")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Noto Sans KR", 26, QFont.Bold))
        label.setStyleSheet(f"color: {DARK_BROWN};")

        sub_label = QLabel("음성으로 말해서 주문하거나, 화면을 터치해서 직접 메뉴를 선택할 수 있습니다.")
        sub_label.setAlignment(Qt.AlignCenter)
        sub_label.setFont(QFont("Noto Sans KR", 14))
        sub_label.setStyleSheet(f"color: {DARK_BROWN};")

        layout.addStretch(1)
        layout.addWidget(label)
        layout.addWidget(sub_label)
        layout.addStretch(1)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(40)

        btn_voice = QPushButton("🎙️ 음성으로 주문")
        btn_voice.setFont(QFont("Noto Sans KR", 20, QFont.Bold))
        btn_voice.setMinimumHeight(120)
        btn_voice.setObjectName("PrimaryCTA")
        btn_voice.clicked.connect(self.on_select_voice_mode)

        btn_touch = QPushButton("👆 화면 터치로 주문")
        btn_touch.setFont(QFont("Noto Sans KR", 20, QFont.Bold))
        btn_touch.setMinimumHeight(120)
        btn_touch.setObjectName("PrimaryCTA")
        btn_touch.clicked.connect(self.on_select_touch_mode)

        btn_layout.addWidget(btn_voice)
        btn_layout.addWidget(btn_touch)

        layout.addLayout(btn_layout)
        layout.addStretch(2)

        return page

    # ------------- 페이지: 터치 주문 -------------

    def build_touch_order_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # 상단 바
        top_bar = QHBoxLayout()
        btn_back = QPushButton("← 처음으로")
        btn_back.setFont(QFont("Noto Sans KR", 14))
        btn_back.clicked.connect(self.on_back_to_home)

        label_mode = QLabel("👆 화면 터치로 주문")
        label_mode.setFont(QFont("Noto Sans KR", 18, QFont.Bold))
        label_mode.setAlignment(Qt.AlignCenter)
        label_mode.setStyleSheet(f"color: {DARK_BROWN};")

        top_bar.addWidget(btn_back, alignment=Qt.AlignLeft)
        top_bar.addStretch(1)
        top_bar.addWidget(label_mode, alignment=Qt.AlignCenter)
        top_bar.addStretch(1)

        layout.addLayout(top_bar)

        # 본문: 좌측 메뉴 / 우측 주문 요약
        body_layout = QHBoxLayout()
        body_layout.setSpacing(20)

        # --- 좌측: 메뉴 선택 ---
        left = QVBoxLayout()
        left.setSpacing(10)

        # 카테고리 버튼
        cat_layout = QHBoxLayout()
        self.btn_cat_burger = QPushButton("버거")
        self.btn_cat_side = QPushButton("사이드")
        self.btn_cat_drink = QPushButton("음료")

        for b in [self.btn_cat_burger, self.btn_cat_side, self.btn_cat_drink]:
            b.setFont(QFont("Noto Sans KR", 14))
            b.setCheckable(True)

        self.btn_cat_burger.clicked.connect(lambda: self.populate_menu_category("burger"))
        self.btn_cat_side.clicked.connect(lambda: self.populate_menu_category("side"))
        self.btn_cat_drink.clicked.connect(lambda: self.populate_menu_category("drink"))

        cat_layout.addWidget(self.btn_cat_burger)
        cat_layout.addWidget(self.btn_cat_side)
        cat_layout.addWidget(self.btn_cat_drink)
        left.addLayout(cat_layout)

        # 메뉴 카드 영역
        self.menu_grid_container = QWidget()
        self.menu_grid_layout = QGridLayout(self.menu_grid_container)
        self.menu_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.menu_grid_layout.setSpacing(16)

        left.addWidget(self.menu_grid_container, stretch=1)
        body_layout.addLayout(left, stretch=3)

        # --- 우측: 주문 요약 ---
        right = QVBoxLayout()
        right.setSpacing(10)

        order_title = QLabel("주문 내역")
        order_title.setFont(QFont("Noto Sans KR", 16, QFont.Bold))
        order_title.setStyleSheet(f"color: {DARK_BROWN};")
        right.addWidget(order_title)

        self.order_list_widget = QListWidget()
        right.addWidget(self.order_list_widget, stretch=1)

        self.lbl_total_price = QLabel("총 금액: 0원")
        self.lbl_total_price.setFont(QFont("Noto Sans KR", 16, QFont.Bold))
        self.lbl_total_price.setStyleSheet(f"color: {DARK_BROWN};")
        right.addWidget(self.lbl_total_price)

        btn_row = QHBoxLayout()
        self.btn_clear_order = QPushButton("전체 취소")
        self.btn_clear_order.setFont(QFont("Noto Sans KR", 14))
        self.btn_clear_order.clicked.connect(self.clear_touch_order)

        self.btn_complete_order = QPushButton("주문 완료")
        self.btn_complete_order.setFont(QFont("Noto Sans KR", 14))
        self.btn_complete_order.setObjectName("PrimaryCTA")
        self.btn_complete_order.clicked.connect(self.on_complete_touch_order)

        btn_row.addWidget(self.btn_clear_order)
        btn_row.addWidget(self.btn_complete_order)

        right.addLayout(btn_row)

        guide_label = QLabel("결제는 주문 완료 후 오른쪽 POS에서 진행해 주세요.")
        guide_label.setFont(QFont("Noto Sans KR", 12))
        guide_label.setStyleSheet(f"color: {DARK_BROWN};")
        right.addWidget(guide_label)

        body_layout.addLayout(right, stretch=2)

        layout.addLayout(body_layout)

        # 기본 카테고리: 버거
        self.btn_cat_burger.setChecked(True)
        self.populate_menu_category("burger")

        return page

    def populate_menu_category(self, category):
        # 카테고리 버튼 토글
        self.btn_cat_burger.setChecked(category == "burger")
        self.btn_cat_side.setChecked(category == "side")
        self.btn_cat_drink.setChecked(category == "drink")

        # 기존 카드 제거
        while self.menu_grid_layout.count():
            item = self.menu_grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # 🔴 죽은 QLabel 참조도 같이 버리기
        self.qty_labels_single.clear()
        self.qty_labels_set.clear()


        # ---- 여기부터: 가로 버전 레이아웃 ----
        # 버거면 3열(가로로 넓게), 나머지는 2열 유지
        col_max = 3 if category == "burger" else 2
        row, col = 0, 0

        # 먼저 카드들을 리스트로 만든 뒤, 공통 로직으로 배치
        cards = []

        if category == "burger":
            # 버거: 단품 + 세트 카드 둘 다 넣기
            for code, info in MENU_ITEMS.items():
                if info["category"] != "burger":
                    continue
                cards.append(self.create_burger_single_card(code, info))
                cards.append(self.create_burger_set_card(code, info))
        else:
            # 사이드/음료: 단일 카드만
            for code, info in MENU_ITEMS.items():
                if info["category"] != category:
                    continue
                cards.append(self.create_single_item_card(code, info))

        # cards 리스트를 col_max 기준으로 그리드에 쭉 배치
        for card in cards:
            self.menu_grid_layout.addWidget(card, row, col)
            col += 1
            if col >= col_max:
                col = 0
                row += 1


    def create_burger_single_card(self, code, info):
        CARD_HEIGHT = 260   # 필요하면 220~280 사이로 취향껏 조절   
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ background-color: {CARD_BG}; border-radius: 18px; "
            f"border: 2px solid {PRIMARY_YELLOW}; }}"
        )
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        frame.setFixedHeight(CARD_HEIGHT)

        layout = QVBoxLayout(frame)
        layout.setSpacing(6)

        pix = load_menu_pixmap(info)
        if pix:
            img_label = QLabel()
            img_label.setPixmap(pix)
            img_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(img_label)

        name = QLabel(f"{info['name']} (단품)")
        name.setFont(QFont("Noto Sans KR", 14, QFont.Bold))
        name.setStyleSheet(f"color: {DARK_BROWN};")
        price = QLabel(f"{info['price']:,}원")
        price.setFont(QFont("Noto Sans KR", 12))
        price.setStyleSheet(f"color: {DARK_BROWN};")

        layout.addWidget(name, alignment=Qt.AlignCenter)
        layout.addWidget(price, alignment=Qt.AlignCenter)

        btn_row = QHBoxLayout()
        btn_minus = QPushButton("-")
        btn_minus.setFixedWidth(36)
        btn_plus = QPushButton("+")
        btn_plus.setFixedWidth(36)

        qty_label = QLabel("0")
        qty_label.setFont(QFont("Noto Sans KR", 14))
        qty_label.setAlignment(Qt.AlignCenter)
        qty_label.setFixedWidth(40)

        self.qty_labels_single[code] = qty_label

        btn_minus.clicked.connect(lambda _, c=code: self.add_single_item(c, -1))
        btn_plus.clicked.connect(lambda _, c=code: self.add_single_item(c, +1))

        btn_row.addWidget(btn_minus)
        btn_row.addWidget(qty_label)
        btn_row.addWidget(btn_plus)
        layout.addLayout(btn_row)

        return frame

    def create_burger_set_card(self, code, info):
        CARD_HEIGHT = 260   # 필요하면 220~280 사이로 취향껏 조절
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ background-color: {SET_CARD_BG}; border-radius: 18px; "
            f"border: 2px solid {PRIMARY_RED}; }}"
        )
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        frame.setFixedHeight(CARD_HEIGHT)
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)

        pix = load_menu_pixmap(info)
        if pix:
            img_label = QLabel()
            img_label.setPixmap(pix)
            img_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(img_label)

        name = QLabel(f"{info['name']} 세트")
        name.setFont(QFont("Noto Sans KR", 14, QFont.Bold))
        name.setStyleSheet(f"color: {DARK_BROWN};")
        desc = QLabel("(감자튀김 + 콜라 포함)")
        desc.setFont(QFont("Noto Sans KR", 11))
        desc.setStyleSheet(f"color: {DARK_BROWN};")
        price_value = info["price"] + SET_EXTRA_PRICE
        price = QLabel(f"{price_value:,}원")
        price.setFont(QFont("Noto Sans KR", 12))
        price.setStyleSheet(f"color: {DARK_BROWN};")

        layout.addWidget(name, alignment=Qt.AlignCenter)
        # layout.addWidget(desc, alignment=Qt.AlignCenter)
        layout.addWidget(price, alignment=Qt.AlignCenter)

        btn_row = QHBoxLayout()
        btn_minus = QPushButton("-")
        btn_minus.setFixedWidth(36)
        btn_plus = QPushButton("+")
        btn_plus.setFixedWidth(36)

        qty_label = QLabel("0")
        qty_label.setFont(QFont("Noto Sans KR", 14))
        qty_label.setAlignment(Qt.AlignCenter)
        qty_label.setFixedWidth(40)

        self.qty_labels_set[code] = qty_label

        btn_minus.clicked.connect(lambda _, c=code: self.add_set_item(c, -1))
        btn_plus.clicked.connect(lambda _, c=code: self.add_set_item(c, +1))

        btn_row.addWidget(btn_minus)
        btn_row.addWidget(qty_label)
        btn_row.addWidget(btn_plus)
        layout.addLayout(btn_row)

        return frame

    def create_single_item_card(self, code, info):
        CARD_HEIGHT = 260   # 필요하면 220~280 사이로 취향껏 조절
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ background-color: {CARD_BG}; border-radius: 18px; "
            f"border: 2px solid #E0E0E0; }}"
        )
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        frame.setFixedHeight(CARD_HEIGHT)
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)

        pix = load_menu_pixmap(info)
        if pix:
            img_label = QLabel()
            img_label.setPixmap(pix)
            img_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(img_label)

        name = QLabel(info["name"])
        name.setFont(QFont("Noto Sans KR", 14, QFont.Bold))
        name.setStyleSheet(f"color: {DARK_BROWN};")
        price = QLabel(f"{info['price']:,}원")
        price.setFont(QFont("Noto Sans KR", 12))
        price.setStyleSheet(f"color: {DARK_BROWN};")

        layout.addWidget(name, alignment=Qt.AlignCenter)
        layout.addWidget(price, alignment=Qt.AlignCenter)

        btn_row = QHBoxLayout()
        btn_minus = QPushButton("-")
        btn_minus.setFixedWidth(36)
        btn_plus = QPushButton("+")
        btn_plus.setFixedWidth(36)

        qty_label = QLabel("0")
        qty_label.setFont(QFont("Noto Sans KR", 14))
        qty_label.setAlignment(Qt.AlignCenter)
        qty_label.setFixedWidth(40)

        # 사이드/음료도 single_counts 사용
        self.qty_labels_single[code] = qty_label

        btn_minus.clicked.connect(lambda _, c=code: self.add_single_item(c, -1))
        btn_plus.clicked.connect(lambda _, c=code: self.add_single_item(c, +1))

        btn_row.addWidget(btn_minus)
        btn_row.addWidget(qty_label)
        btn_row.addWidget(btn_plus)
        layout.addLayout(btn_row)

        return frame

    def add_single_item(self, code, delta):
        new_value = max(0, self.single_counts[code] + delta)
        self.single_counts[code] = new_value
        if code in self.qty_labels_single:
            self.qty_labels_single[code].setText(str(new_value))
        self.update_order_summary()

    def add_set_item(self, burger_code, delta):
        new_value = max(0, self.set_counts[burger_code] + delta)
        self.set_counts[burger_code] = new_value
        if burger_code in self.qty_labels_set:
            self.qty_labels_set[burger_code].setText(str(new_value))
        self.update_order_summary()

    def clear_touch_order(self):
        self.single_counts.clear()
        self.set_counts.clear()
        for lbl in self.qty_labels_single.values():
            lbl.setText("0")
        for lbl in self.qty_labels_set.values():
            lbl.setText("0")
        self.update_order_summary()

    def update_order_summary(self):
        self.order_list_widget.clear()
        total_price = 0

        # 1) 버거 단품 + 사이드 + 음료
        for code, qty in self.single_counts.items():
            if qty <= 0:
                continue
            info = MENU_ITEMS.get(code)
            if not info:
                continue
            line_price = info["price"] * qty
            total_price += line_price
            item = QListWidgetItem(f"{info['name']} x{qty}  ({line_price:,}원)")
            self.order_list_widget.addItem(item)

        # 2) 버거 세트
        for bcode, qty in self.set_counts.items():
            if qty <= 0:
                continue
            info = MENU_ITEMS.get(bcode)
            if not info:
                continue
            set_price_per = info["price"] + SET_EXTRA_PRICE
            line_price = set_price_per * qty
            total_price += line_price
            text = f"{info['name']} 세트 x{qty}  ({line_price:,}원)\n  - 감자튀김 + 콜라 포함"
            self.order_list_widget.addItem(QListWidgetItem(text))

        self.lbl_total_price.setText(f"총 금액: {total_price:,}원")

        has_items = any(v > 0 for v in self.single_counts.values()) or any(v > 0 for v in self.set_counts.values())
        self.btn_complete_order.setEnabled(has_items)
        self.btn_clear_order.setEnabled(has_items)

    def build_order_request_from_touch(self):
        """
        터치로 선택된 내용을 OrderService에 맞는 item_names, item_quantities로 변환.
        - 세트: burger + fries + coke 를 각각 qty만큼 추가.
        """
        item_counts = defaultdict(int)

        # 단품 / 사이드 / 음료
        for code, qty in self.single_counts.items():
            if qty > 0:
                item_counts[code] += qty

        # 세트 -> burger + fries + coke
        for bcode, qty in self.set_counts.items():
            if qty > 0:
                item_counts[bcode] += qty
                item_counts[DEFAULT_SET_SIDE] += qty
                item_counts[DEFAULT_SET_DRINK] += qty

        item_names = list(item_counts.keys())
        item_quantities = [item_counts[c] for c in item_names]
        return item_names, item_quantities

    def on_complete_touch_order(self):
        if self.order_in_progress:
            return

        item_names, item_quantities = self.build_order_request_from_touch()
        if not item_names:
            QMessageBox.information(self, "주문 없음", "선택된 메뉴가 없습니다.")
            return

        reply = QMessageBox.question(
            self,
            "주문 확인",
            "선택하신 메뉴로 주문을 진행할까요?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.order_in_progress = True
        self.btn_complete_order.setEnabled(False)
        self.btn_clear_order.setEnabled(False)

        self.order_client.send_order(item_names, item_quantities)

    # ------------- 페이지: 음성 주문 -------------

    def build_voice_order_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        top_bar = QHBoxLayout()
        btn_back = QPushButton("← 처음으로")
        btn_back.setFont(QFont("Noto Sans KR", 14))
        btn_back.clicked.connect(self.on_back_to_home)

        label_mode = QLabel("🎙️ 음성으로 주문")
        label_mode.setFont(QFont("Noto Sans KR", 18, QFont.Bold))
        label_mode.setAlignment(Qt.AlignCenter)
        label_mode.setStyleSheet(f"color: {DARK_BROWN};")

        top_bar.addWidget(btn_back, alignment=Qt.AlignLeft)
        top_bar.addStretch(1)
        top_bar.addWidget(label_mode, alignment=Qt.AlignCenter)
        top_bar.addStretch(1)

        layout.addLayout(top_bar)

        info_label = QLabel("버거 세트, 단품, 사이드, 음료를 한 번에 말씀해 주세요.\n"
                            "예: '불고기버거 세트 하나랑 치즈버거 하나, 콜라 하나 주세요. 계산할게요.'")
        info_label.setFont(QFont("Noto Sans KR", 13))
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet(f"color: {DARK_BROWN};")
        layout.addWidget(info_label)

        self.lbl_voice_status = QLabel("버튼을 눌러 음성 주문을 시작해 주세요.")
        self.lbl_voice_status.setFont(QFont("Noto Sans KR", 14))
        self.lbl_voice_status.setAlignment(Qt.AlignCenter)
        self.lbl_voice_status.setStyleSheet(f"color: {DARK_BROWN};")
        layout.addWidget(self.lbl_voice_status)

        self.lbl_voice_robot = QLabel("")
        self.lbl_voice_robot.setFont(QFont("Noto Sans KR", 13))
        self.lbl_voice_robot.setAlignment(Qt.AlignCenter)
        self.lbl_voice_robot.setWordWrap(True)
        self.lbl_voice_robot.setStyleSheet(f"color: {DARK_BROWN};")
        layout.addWidget(self.lbl_voice_robot)

        self.lbl_voice_order = QLabel("")
        self.lbl_voice_order.setFont(QFont("Noto Sans KR", 13))
        self.lbl_voice_order.setAlignment(Qt.AlignCenter)
        self.lbl_voice_order.setWordWrap(True)
        self.lbl_voice_order.setStyleSheet(f"color: {DARK_BROWN};")
        layout.addWidget(self.lbl_voice_order)

        btn_layout = QHBoxLayout()
        self.btn_start_voice = QPushButton("🎙️ 음성 입력 시작")
        self.btn_start_voice.setFont(QFont("Noto Sans KR", 16, QFont.Bold))
        self.btn_start_voice.setObjectName("PrimaryCTA")
        self.btn_start_voice.clicked.connect(self.on_start_voice_order)

        self.btn_voice_confirm = QPushButton("이대로 주문하기")
        self.btn_voice_confirm.setFont(QFont("Noto Sans KR", 16))
        self.btn_voice_confirm.setObjectName("PrimaryCTA")
        self.btn_voice_confirm.setEnabled(False)
        self.btn_voice_confirm.clicked.connect(self.on_confirm_voice_order)

        btn_layout.addWidget(self.btn_start_voice)
        btn_layout.addWidget(self.btn_voice_confirm)

        layout.addLayout(btn_layout)

        guide_label = QLabel("음성이 잘 인식되지 않으면 다시 시도하거나, 화면 터치 주문을 이용해 주세요.")
        guide_label.setFont(QFont("Noto Sans KR", 11))
        guide_label.setAlignment(Qt.AlignCenter)
        guide_label.setStyleSheet(f"color: {DARK_BROWN};")
        layout.addWidget(guide_label)

        layout.addStretch(1)

        return page

    def on_start_voice_order(self):
        if self.order_in_progress:
            return
        self.lbl_voice_status.setText("음성 인식을 시작합니다...")
        self.lbl_voice_robot.setText("")
        self.lbl_voice_order.setText("")
        self.btn_start_voice.setEnabled(False)
        self.btn_voice_confirm.setEnabled(False)
        self.pending_voice_order_names = None
        self.pending_voice_order_quantities = None

        self.voice_thread = VoiceOrderThread()
        self.voice_thread.recognizedText.connect(self.on_voice_recognized_text)
        self.voice_thread.resultReady.connect(self.on_voice_result)
        self.voice_thread.error.connect(self.on_voice_error)
        self.voice_thread.finished.connect(lambda: self.btn_start_voice.setEnabled(True))
        self.voice_thread.start()

    def on_voice_recognized_text(self, text):
        self.lbl_voice_status.setText(text)

    def on_voice_result(self, robot_message, item_names, item_quantities):
        self.lbl_voice_robot.setText(f"로봇 응답:\n{robot_message}")

        preview_lines = []
        for code, qty in zip(item_names, item_quantities):
            display_name = get_display_name(code)
            preview_lines.append(f"- {display_name} x{qty}")

        preview = "\n".join(preview_lines)
        self.lbl_voice_order.setText(f"[주문 데이터 미리보기]\n{preview}")

        self.pending_voice_order_names = item_names
        self.pending_voice_order_quantities = item_quantities
        self.btn_voice_confirm.setEnabled(True)

    def on_voice_error(self, msg):
        self.lbl_voice_status.setText(msg)
        self.lbl_voice_robot.setText("")
        self.lbl_voice_order.setText("")
        self.btn_voice_confirm.setEnabled(False)

    def on_confirm_voice_order(self):
        if self.order_in_progress:
            return
        if not self.pending_voice_order_names:
            QMessageBox.information(self, "주문 없음", "인식된 주문 데이터가 없습니다.")
            return

        reply = QMessageBox.question(
            self,
            "주문 확인",
            "인식된 내용으로 주문을 진행할까요?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.order_in_progress = True
        self.btn_voice_confirm.setEnabled(False)
        self.btn_start_voice.setEnabled(False)
        self.lbl_voice_status.setText("주문을 전송하는 중입니다...")

        self.order_client.send_order(
            self.pending_voice_order_names,
            self.pending_voice_order_quantities
        )

    # ------------- 페이지: 주문 완료 -------------

    def build_complete_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        label = QLabel("주문이 완료되었습니다!")
        label.setFont(QFont("Noto Sans KR", 24, QFont.Bold))
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"color: {DARK_BROWN};")
        layout.addWidget(label)

        self.lbl_complete_msg = QLabel("")
        self.lbl_complete_msg.setFont(QFont("Noto Sans KR", 16))
        self.lbl_complete_msg.setAlignment(Qt.AlignCenter)
        self.lbl_complete_msg.setStyleSheet(f"color: {DARK_BROWN};")
        layout.addWidget(self.lbl_complete_msg)

        self.lbl_order_number = QLabel("주문 번호: -")
        self.lbl_order_number.setFont(QFont("Noto Sans KR", 26, QFont.Bold))
        self.lbl_order_number.setAlignment(Qt.AlignCenter)
        self.lbl_order_number.setStyleSheet(f"color: {PRIMARY_RED};")
        layout.addWidget(self.lbl_order_number)

        info = QLabel("잠시 후 번호가 호출되면 카운터에서 음식을 받아 주세요.\n"
                      "다음 손님을 위해 '다음 손님' 버튼을 눌러 초기 화면으로 돌아갈 수 있습니다.")
        info.setFont(QFont("Noto Sans KR", 13))
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet(f"color: {DARK_BROWN};")
        layout.addWidget(info)

        btn_next = QPushButton("다음 손님")
        btn_next.setFont(QFont("Noto Sans KR", 18, QFont.Bold))
        btn_next.setObjectName("PrimaryCTA")
        btn_next.clicked.connect(self.on_next_customer)
        layout.addWidget(btn_next, alignment=Qt.AlignCenter)

        layout.addStretch(1)
        return page

    # ------------- 공통 이벤트 -------------

    def on_select_touch_mode(self):
        self.stack.setCurrentIndex(self.touch_index)

    def on_select_voice_mode(self):
        self.stack.setCurrentIndex(self.voice_index)

    def on_back_to_home(self):
        if self.order_in_progress:
            QMessageBox.warning(self, "진행 중", "현재 주문 처리 중입니다. 잠시만 기다려 주세요.")
            return
        self.stack.setCurrentIndex(self.home_index)

    def on_next_customer(self):
        # 다음 손님 준비: 상태/선택 초기화
        self.clear_touch_order()
        self.pending_voice_order_names = None
        self.pending_voice_order_quantities = None
        self.lbl_voice_status.setText("버튼을 눌러 음성 주문을 시작해 주세요.")
        self.lbl_voice_robot.setText("")
        self.lbl_voice_order.setText("")
        self.btn_start_voice.setEnabled(True)
        self.btn_voice_confirm.setEnabled(False)

        self.stack.setCurrentIndex(self.home_index)

    def on_order_finished(self, success, message, order_id):
        self.order_in_progress = False

        if success:
            self.last_order_id = order_id
            self.lbl_complete_msg.setText(message or "주문이 접수되었습니다.")
            if order_id:
                self.lbl_order_number.setText(f"주문 번호: {order_id}")
            else:
                self.lbl_order_number.setText("주문 번호: -")

            self.clear_touch_order()
            self.pending_voice_order_names = None
            self.pending_voice_order_quantities = None

            self.stack.setCurrentIndex(self.complete_index)
        else:
            QMessageBox.warning(self, "주문 실패", message or "주문 처리에 실패했습니다.")
            self.btn_complete_order.setEnabled(True)
            self.btn_clear_order.setEnabled(True)
            self.btn_start_voice.setEnabled(True)
            self.btn_voice_confirm.setEnabled(bool(self.pending_voice_order_names))

    def closeEvent(self, event):
        self.order_client.shutdown()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = KioskWindow()
    # win.showMaximized()  # 키오스크 느낌
    win.resize(1000, 650)
    win.show()
    ret = app.exec_()
    sys.exit(ret)


if __name__ == "__main__":
    main()
