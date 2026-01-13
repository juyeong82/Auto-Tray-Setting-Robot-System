# 🍔 Auto Tray Setting Robot System (패스트푸드 무인화 솔루션)

![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?style=flat&logo=ros&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?style=flat&logo=ubuntu&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv11](https://img.shields.io/badge/AI-YOLOv11_OBB-00FFFF?style=flat&logo=ultralytics&logoColor=black)
![Doosan](https://img.shields.io/badge/Robot-Doosan_M0609-blue?style=flat)

---

## 🎬 프로젝트 시연 영상 (Demo)

> *로봇이 키오스크 주문을 인식하고, 햄버거와 음료를 트레이에 세팅하는 시연 영상입니다.*

[![Demo Video](http://img.youtube.com/vi/7-8vbY-XjlU/0.jpg)](https://youtu.be/7-8vbY-XjlU)
---

## 1. 프로젝트 개요 (Project Overview)

### 🎯 개발 목표
패스트푸드 매장의 점심/저녁 피크타임에 발생하는 **병목 현상을 해결**하고, 단순 반복 업무(트레이 세팅)를 자동화하여 **인력난 해소 및 위생 관리**를 강화하기 위한 협동 로봇 솔루션입니다.

### 🛠️ 핵심 기능
* **키오스크 연동:** 사용자의 주문 정보를 받아 로봇에게 작업 명령 전달.
* **AI 객체 인식:** YOLO 기반의 Vision 시스템을 통해 비정형 물체(햄버거, 음료 등)의 위치와 각도 파악.
* **자동 트레이 세팅:** 인식된 좌표로 로봇팔(M0609)을 제어하여 트레이에 메뉴 정렬.
* **이중 안전 장치:** 작업자와 로봇 보호를 위한 다중 안전 로직 적용.

---

## 2. 시스템 아키텍처 (System Architecture)

### 🔄 Logic Flowchart
키오스크 주문부터 객체 인식, 로봇 제어, 안전 점검까지의 전체 프로세스 흐름도입니다.

```mermaid
graph LR
    User((User)) -->|Order| Kiosk[Kiosk UI]
    Kiosk -->|Pub: Order List| TaskMgr[Task Manager Node]
    
    subgraph Vision System
        Cam[RealSense D435] -->|RGB-D Data| YOLO[YOLOv11 OBB]
        YOLO -->|Class, Box, Angle| Coord[Coord Converter]
        Coord -->|TF: Pixel -> RobotBase| TargetPos[Target Pose]
    end
    
    TaskMgr -->|Trigger| Vision System
    TargetPos -->|Return Pose| TaskMgr
    
    subgraph Robot Control
        TaskMgr -->|Action Goal| MoveIt[MoveIt2 Planner]
        MoveIt -->|Trajectory| Controller[M0609 Controller]
        Controller -->|Execute| Gripper[OnRobot RG2]
    end
    
    subgraph Safety System
        Sensor[Robot State / Force] -->|Monitor| SafetyNode[Safety Monitor]
        SafetyNode -- "Emergency Stop" --> Controller
    end

```

---

## 3. 핵심 기술 (Key Technologies)

### 🧠 1. YOLOv11 OBB 기반 객체 인식

일반적인 Bounding Box(수평 박스)가 아닌 **OBB(Oriented Bounding Box, 회전된 박스)** 모델을 사용하여 물체의 회전 각도()까지 정밀하게 추론합니다.

* **객체 분류:** 햄버거(Burger), 음료(Coke/Soda), 사이드(Fries/Nugget) 등 메뉴 식별.
* **그리핑 최적화:** 물체가 놓인 각도에 맞춰 그리퍼를 회전시켜 파지 성공률을 극대화.

### 📐 2. 좌표 변환 시스템 (Coordinate Transformation)

카메라(2D 픽셀)에서 보는 세상과 로봇(3D 공간)이 있는 세상의 좌표를 일치시키는 기술입니다.

* **Depth Align:** RealSense의 RGB 센서와 Depth 센서 간의 시차를 보정하여 정확한 3D 거리 측정.
* **TF2 변환:** `Camera Link`  `End-effector`  `Robot Base` 좌표계 변환 행렬(Matrix) 연산 수행.

### 🛡️ 3. 이중 안전 정지 시스템 (Dual Safety Stop)

협동 로봇 운용 시 발생할 수 있는 사고를 방지하기 위해 두 가지 안전 로직을 구현했습니다.

1. **Safety Monitor Node:** 로봇의 상태 토픽을 실시간으로 구독(Subscribe)하며, 이상 징후(과도한 힘, 속도 초과 등) 감지 시 즉시 정지 명령 발행.
2. **Virtual Wall (작업 영역 제한):** MoveIt 설정 상에서 로봇이 트레이 작업 공간을 벗어나거나, 사람 쪽으로 향하지 못하도록 소프트웨어적인 가상 벽 설정.

---

## 4. 하드웨어 및 환경 (Environment)

| 항목 | 상세 내용 |
| --- | --- |
| **OS** | Ubuntu 22.04 LTS (Jammy Jellyfish) |
| **MiddleWare** | ROS2 Humble Hawksbill |
| **Manipulator** | Doosan Robotics M0609 |
| **Gripper** | OnRobot RG2 |
| **Camera** | Intel RealSense D435 |
| **PC Spec** | Intel i7, RTX 3060, 32GB RAM |

---

## 5. 설치 및 실행 (Installation & Usage)

### 📦 1. 사전 설정 (Prerequisites)

* 이 프로젝트는 `dsr_bringup2` 패키지가 필요합니다. (두산 로보틱스 공식 패키지 설치 필요)
* RealSense SDK 및 ROS2 Wrapper가 설치되어 있어야 합니다.

### 🚀 2. 실행 명령어 (Run Commands)

각 터미널 창을 분할하여 순서대로 실행해주세요.

#### Step 1: 로봇 연결 (Bringup)

실제 로봇(Real)과 시뮬레이션(Virtual) 모드 중 선택하여 실행합니다.

**A. 실제 로봇 연결 (Real Mode)**

> *IP 주소(host)는 로봇 설정에 맞게 변경하세요.*

```bash
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=192.168.1.100 port:=12345 model:=m0609

```

**B. 가상 시뮬레이션 (Virtual Mode)**

```bash
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=virtual host:=127.0.0.1 port:=12345 model:=m0609

```

#### Step 2: Vision 카메라 실행

RealSense 카메라 노드를 실행합니다. Depth 정렬 및 Pointcloud 기능을 활성화하여 좌표 오차를 최소화했습니다.

```bash
ros2 launch realsense2_camera rs_align_depth_launch.py depth_module.depth_profile:=848x480x30 rgb_camera.color_profile:=640x480x30 initial_reset:=true align_depth.enable:=true enable_rgbd:=true pointcloud.enable:=true

```

#### Step 3: 메인 시스템 실행 (Main System)

키오스크 UI, YOLO 노드, 모션 플래닝 노드를 포함한 전체 시스템을 구동합니다.

```bash
ros2 launch ff_robot robot_system.launch.py

```

---

Copyright 2025. Team B-2 All rights reserved.