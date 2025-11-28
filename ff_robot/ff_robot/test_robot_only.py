#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import DR_init
import time

# --- 로봇 설정 ---
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

# --- 이동할 위치들 ---
# 1. 홈 위치 (수직으로 서 있는 자세)
J_HOME_POS = [0, 0, 90, 0, 90, 0]

# 2. 관측 위치 (사물을 내려다보는 자세)
J_LOOK_POS = [-42.94, -54.85, 50.51, -3.11, 130.60, -266.60]

def main():
    rclpy.init()
    node = rclpy.create_node("simple_test_node", namespace=ROBOT_ID)
    
    # DSR 라이브러리 연결 설정
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    DR_init.__dsr__node = node
    
    try:
        from DSR_ROBOT2 import movej, get_current_posj
        
        print("\n========== [1단계: 로봇 제어 단독 테스트] ==========")
        
        # 1. 연결 확인
        print("✅ 1. 로봇 연결 성공.")
        current_j = get_current_posj()
        print(f"   📍 현재 관절 각도: {current_j}")
        
        # 2. 홈 위치 이동
        print(f"\n🚀 2. 홈 위치로 이동합니다... {J_HOME_POS}")
        # vel, acc는 반드시 float(30.0)으로 입력
        movej(J_HOME_POS, vel=30.0, acc=30.0)
        print("   ✅ 홈 위치 도착 완료.")
        
        # 잠시 대기 (육안 확인용)
        print("   ⏳ 2초 대기...")
        time.sleep(2.0)
        
        # 3. 관측 위치 이동
        print(f"\n🔭 3. 관측 위치로 이동합니다... {J_LOOK_POS}")
        movej(J_LOOK_POS, vel=30.0, acc=30.0)
        print("   ✅ 관측 위치 도착 완료.")
        
        print("\n🎉 [성공] 로봇 제어 기능에 문제가 없습니다.")
        print("==================================================\n")
        
    except ImportError:
        print("❌ [오류] DSR_ROBOT2 라이브러리를 찾을 수 없습니다. (환경 설정 확인)")
    except Exception as e:
        print(f"\n❌ [오류 발생] : {e}")
        print("   -> 힌트: 로봇이 'Manual' 모드이거나 'Emergency Stop' 상태인지 확인하세요.")
        print("   -> 힌트: 이미 목표 위치와 너무 가까우면 움직임이 없을 수 있습니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()