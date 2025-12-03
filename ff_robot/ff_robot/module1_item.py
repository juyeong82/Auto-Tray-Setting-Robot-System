#!/usr/bin/env python3
# item_placement_controller.py (Updated from robot_controller_node.py)
# 아이템 검출 → 집기 → 트레이 배치 전담 노드
# [Updated]
# 1. ITEM_PLACE_Z_OFFSET → ITEM_HEIGHTS로 변경
# 2. TRAY_FLOOR_Z 업데이트 (25.0 → -25.0)
# 3. DROP_SAFETY_MARGIN 업데이트 (30.0 → 10.0)

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import time
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R

import DR_init
from ff_robot_interfaces.srv import DetectObject, PlaceItem
from ff_robot.gripper import GripperManager
from ff_robot.order_logic import SlotManager

sys.stdout.reconfigure(line_buffering=True)

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================
GLOBAL_OFFSET_X = 0.0
GLOBAL_OFFSET_Y = 0.0
GLOBAL_OFFSET_Z = -60.0

TRAY_CENTER_OFFSET_X = 80.0
TRAY_0_EDGE_POS = [205.0, 20.0, 25.0, 43.35, -180.0, -134.82]
TRAY_1_EDGE_POS = [435.0, 20.0, 25.0, 43.35, -180.0, -134.82]

TRAY_FLOOR_Z = -25.0  # ⬆️ 변경: 25.0 → -25.0
SAFE_Z_FLOOR_LIMIT = 10.0


EXTRA_LIFT_HEIGHT = 150.0
DROP_SAFETY_MARGIN = 7.0  # ⬆️ 변경: 30.0 → 10.0

# J_ITEM_OBSERVE = [-34.0, 33.0, 15.0, 0.0, 132.0, 144.0]
J_ITEM_OBSERVE = [-33.197, 21.512, 35.707, -0.118, 122.787, 144.296]

J_ITEM_CHECKPOINT = [-45.4, 27.69, 55.06, 0.00, 97.67, 132.68] 

# 12/02/18:29 수정됨: 파지시 rz가 None으로 되어 있어 과도하게 회전하거나 특이점 발생하는 것 수정. 0, 180, None-> 60, 180, 60 으로 수정
ITEM_OFFSETS = {
    "burger1": {"z": 0.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "burger2": {"z": 0.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "burger3": {"z": 0.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "coke":    {"z": 10.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "cider":   {"z": 10.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "fries":   {"z": 0.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
    "nugget":  {"z": 0.0, "rx": 56.0, "ry": 179.87, "rz": -126.01},
}

# ⬆️ 변경: ITEM_PLACE_Z_OFFSET → ITEM_HEIGHTS로 이름 변경
ITEM_HEIGHTS = {
    "burger1": 30.0, "burger2": 30.0, "burger3": 30.0,
    "fries":   65.0, "nugget":  65.0,
    "coke":    125.0, "cider":   125.0
}

#햄버거에도 보정값을 추가해야 z축으로 이동한 후 xy로 이동함 -> 2 yolo에서는 쓰지 않기
FIXED_PICK_Z = {}
# {
#     "coke": 92.0, "cider": 92.0, "burger1": 50.0, "burger2": 50.0, "burger3": 50.0, 
# }

SCALE_X_TARGETS = ['cider', 'coke', 'fries', 'nugget']
SCALE_Y_TARGETS = ['burger1', 'burger2', 'burger3']

SCALE_FACTOR_X = 1.05
SCALE_FACTOR_Y = 1.05
TILT_FACTOR_X = 0.10

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

node_ = None
dsr_control_node_ = None
vision_cli = None
gripper = None
gripper_manager = None
T_GRIPPER_TO_CAM = None
slot_manager = None

def load_calibration():
    global T_GRIPPER_TO_CAM
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(current_dir, "T_gripper2camera.npy")
    if os.path.exists(npy_path):
        T_GRIPPER_TO_CAM = np.load(npy_path)
        node_.get_logger().info(f"✅ Calibration loaded: {npy_path}")
    else:
        node_.get_logger().error(f"❌ Calibration file not found: {npy_path}")
        sys.exit(1)

def get_tray_center_pose(slot_id):
    edge_pos = TRAY_0_EDGE_POS if slot_id == 0 else TRAY_1_EDGE_POS
    center_pos = list(edge_pos)
    center_pos[0] += TRAY_CENTER_OFFSET_X
    center_pos[3] = 56.0
    center_pos[4] = 179.87
    center_pos[5] = -126.01
    return center_pos

try:
    from ff_robot.onrobot import RG
except ImportError:
    class RG:
        def __init__(self, *args): pass
        def open_gripper(self): print("   👐 [Virtual] Open")
        def close_gripper(self, force=None): print("   ✊ [Virtual] Close")

def get_robot_pose_matrix(posx_list):
    x, y, z, rx, ry, rz = posx_list
    rot = R.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]
    return T

def transform_camera_to_base(cam_xyz):
    from DSR_ROBOT2 import get_current_posx
    try:
        curr_posx = get_current_posx()
        if curr_posx is None: return None
        if isinstance(curr_posx, tuple): curr_posx = curr_posx[0]
        T_base_gripper = get_robot_pose_matrix(curr_posx)
        T_base_cam = T_base_gripper @ T_GRIPPER_TO_CAM
        p_cam = np.array([cam_xyz[0]*1000, cam_xyz[1]*1000, cam_xyz[2]*1000, 1.0])
        p_base = T_base_cam @ p_cam
        return p_base[:3]
    except Exception as e:
        return None

def wait_for_motion():
    from DSR_ROBOT2 import check_motion
    time.sleep(0.1)
    while check_motion() != 0:
        time.sleep(0.05)
        if not rclpy.ok(): return False
    return True

def safe_movej(joints):
    from DSR_ROBOT2 import movej
    try:
        time.sleep(0.05)
        movej(joints, vel=50.0, acc=50.0)
        wait_for_motion()
        time.sleep(0.05)
        return True
    except: return False

def safe_movel(pos, desc="이동"):
    from DSR_ROBOT2 import movel, DR_BASE, DR_MV_MOD_ABS
    try:
        time.sleep(0.05)
        movel(pos, vel=[100.0, 100.0], acc=[100.0, 100.0], ref=DR_BASE, mod=DR_MV_MOD_ABS)
        wait_for_motion()
        time.sleep(0.05)
        return True
    except Exception as e:
        node_.get_logger().error(f"   ❌ [{desc}] 오류: {e}")
        return False

def safe_move_and_pick_item(base_pos, item_name, ref_x_pos, ref_y_pos, rot_rz):
    offset_info = ITEM_OFFSETS.get(item_name, {"z": 0.0, "rx": 0.0, "ry": 180.0, "rz": None})
    raw_x, raw_y, raw_z = base_pos[0], base_pos[1], base_pos[2]

    if item_name in SCALE_X_TARGETS:
        delta_x = raw_x - ref_x_pos
        target_x = ref_x_pos + (delta_x * SCALE_FACTOR_X) + GLOBAL_OFFSET_X
        tilt_z = (delta_x * TILT_FACTOR_X) if delta_x > 0 else 0.0
        target_y = raw_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"] - tilt_z
    elif item_name in SCALE_Y_TARGETS:
        delta_y = raw_y - ref_y_pos
        target_x = raw_x + GLOBAL_OFFSET_X
        target_y = ref_y_pos + (delta_y * SCALE_FACTOR_Y) + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"]
    else:
        target_x = raw_x + GLOBAL_OFFSET_X
        target_y = raw_y + GLOBAL_OFFSET_Y
        target_z = raw_z + GLOBAL_OFFSET_Z + offset_info["z"]

    if item_name in FIXED_PICK_Z:
        target_z = FIXED_PICK_Z[item_name]

    target_rx = offset_info.get("rx", 0.0)
    target_ry = offset_info.get("ry", 180.0)
    target_rz = rot_rz if offset_info.get("rz") is None else offset_info["rz"]

    if target_z < SAFE_Z_FLOOR_LIMIT: target_z = SAFE_Z_FLOOR_LIMIT

    # =========================================================================
    # [12/03 수정] EXTRA_LIFT_HEIGHT를 활용하여 진입/진출 높이를 동일하게 높임
    # =========================================================================
    
    # 1. 안전한 상공 높이 변수 정의 (Approach + Extra Lift)
    # 이 높이는 무조건 물체보다 150mm(기본100+추가50) 이상 높으므로 충돌 회피 가능
    high_z_pos = target_z + EXTRA_LIFT_HEIGHT

    # 2. 좌표 정의
    pick_pose = [target_x, target_y, target_z, target_rx, target_ry, target_rz]
    
    # [수정] 접근(approach)할 때도 high_z_pos를 사용하여 높게 진입
    approach_pose = [target_x, target_y, high_z_pos, target_rx, target_ry, target_rz]
    
    # [수정] 복귀(lift)할 때도 동일한 high_z_pos 사용
    lift_pose = [target_x, target_y, high_z_pos, target_rx, target_ry, target_rz]
    
    node_.get_logger().info(f"   🛡️ 고공 접근 설정: Z={high_z_pos:.1f} (Target Z={target_z:.1f})")
    
    # 3. 이동 시퀀스 수행
    
    # (1) 고공 접근 (대각선으로 오더라도 목적지 Z가 높아서 안전)
    if not safe_movel(approach_pose, "아이템 고공 접근"): return False, None
    
    if gripper_manager: gripper_manager.prepare_grip(item_name)
    
    # (2) 수직 하강
    if not safe_movel(pick_pose, "아이템 하강"): return False, None
    
    # (3) 그리퍼 동작
    if gripper_manager:
        try:
            gripper_manager.execute_grip(item_name)
        except Exception as e:
            node_.get_logger().warn(f"⚠️ Gripper execute failed (continuing): {e}")
    else:
        if gripper:
            try:
                gripper.close_gripper()
                time.sleep(0.5)
            except Exception as e:
                node_.get_logger().warn(f"⚠️ Gripper close failed (continuing): {e}")
    
    # (4) 수직 상승 (아까 진입했던 높은 높이로 복귀)
    if not safe_movel(lift_pose, "아이템 상승(High)"): return False, None
    
    # 배치(Place) 단계로 넘어갈 때도 이 높은 Z값을 유지하며 이동하도록 리턴값 전달
    return True, high_z_pos

def call_vision_service(target_name):
    if vision_cli is None or not vision_cli.service_is_ready():
        return None
    req = DetectObject.Request()
    req.target_object_id = target_name
    future = vision_cli.call_async(req)
    
    start = time.time()
    while not future.done():
        if time.time() - start > 5.0: return None
        time.sleep(0.1)
    try:
        res = future.result()
        if res.found:
            return {
                "position": [res.position.x, res.position.y, res.position.z],
                "rotation": [res.rx, res.ry, res.rz],
                "confidence": res.confidence
            }
    except: pass
    return None

# ==============================================================================
# SERVICE HANDLER: /place_item (2-Stage Vision + Top-Down Check)
# ==============================================================================
def handle_place_item(request, response):
    from DSR_ROBOT2 import get_current_posx
    
    item_name = request.item_name
    target_slot_id = request.target_slot_id
    item_index = request.item_index
    total_items = request.total_items
    
    node_.get_logger().info(f"🔧 [PlaceItem] 요청: '{item_name}' (2-Stage Vision)")
    
    # ---------------------------------------------------------
    # [Step 1] 1차 관측 (멀리서 대략적 위치 파악)
    # ---------------------------------------------------------
    if not safe_movej(J_ITEM_OBSERVE):
        response.success = False; response.message = "Failed to move to observation pose"
        return response
    
    time.sleep(1.0) # 이미지 안정화
    
    # 비전 호출 (1차)
    item_data_1 = call_vision_service(item_name)
    if not item_data_1:
        response.success = False; response.message = f"1st Vision failed: '{item_name}'"
        return response
    
    base_xyz_1 = transform_camera_to_base(item_data_1["position"])
    if base_xyz_1 is None:
        response.success = False; response.message = "1st Coord transform failed"
        return response

    # ---------------------------------------------------------
    # [Step 2] 정밀 관측을 위한 상공 이동 (Top-Down 접근)
    # ---------------------------------------------------------
    # 대각선 뷰에서 얻은 X, Y 좌표의 수직 상공으로 이동
    RE_CHECK_Z = 225.0  # 재확인 높이 (mm) - 충분히 높은 안전 고도
    
    # 수직 아래를 보기 위해 Rx=0, Ry=180, Rz=0 설정
    # (1차 인식된 X, Y 좌표 사용)
    pre_approach_pose = [base_xyz_1[0], base_xyz_1[1], RE_CHECK_Z, 0.0, 180.0, 0.0]
    
    # [중요] X, Y 먼저 이동하고 높이 맞춤 (직선 이동)
    if not safe_movel(pre_approach_pose, "정밀 관측 위치 이동"):
        response.success = False; response.message = "Failed to move to re-check pose"
        return response
        
    time.sleep(0.5) # 이동 후 진동 안정화

    # ---------------------------------------------------------
    # [Step 3] 2차 관측 (정밀 보정)
    # ---------------------------------------------------------
    item_data_2 = call_vision_service(item_name)
    
    final_x, final_y, final_z, final_rz = 0.0, 0.0, 0.0, 0.0
    
    if item_data_2:
        # 2차 인식 성공 시: 보정된 좌표 사용
        base_xyz_2 = transform_camera_to_base(item_data_2["position"])
        if base_xyz_2 is not None:
            final_x = base_xyz_2[0]
            final_y = base_xyz_2[1]
            final_z = base_xyz_2[2] # 비전 뎁스값 사용
            final_rz = item_data_2["rotation"][2] # Degree 단위
            node_.get_logger().info(f"   🎯 2차 보정 완료: Z값 {base_xyz_1[2]:.1f} -> {final_z:.1f}")
        else:
            final_x, final_y, final_z = base_xyz_1
            final_rz = item_data_1["rotation"][2]
    else:
        node_.get_logger().warn("   ⚠️ 2차 인식 실패 -> 1차 좌표 사용")
        final_x, final_y, final_z = base_xyz_1
        final_rz = item_data_1["rotation"][2]

    # [Z값 안전장치] 바닥 충돌 방지
    if final_z < SAFE_Z_FLOOR_LIMIT: 
        final_z = SAFE_Z_FLOOR_LIMIT

    # ---------------------------------------------------------
    # [Step 4] 아이템 집기 (보정된 좌표 사용)
    # ---------------------------------------------------------
    if gripper_manager: gripper_manager.prepare_grip(item_name)
    
    # 집을 때(Pick) 각도: 물체 각도(final_rz) 그대로 사용
    pick_rz = final_rz 
    full_pose = [final_x, final_y, final_z, 0, 0, pick_rz]
    
    # 현재 위치(상공)를 레퍼런스로 사용
    curr_pos = pre_approach_pose
    ref_x_pos, ref_y_pos = curr_pos[0], curr_pos[1]
    
    # safe_move_and_pick_item 호출
    # (주의: FIXED_PICK_Z가 켜져 있으면 final_z가 무시됩니다.)
    success, lifted_z = safe_move_and_pick_item(full_pose, item_name, ref_x_pos, ref_y_pos, pick_rz)
    
    if not success:
        response.success = False; response.message = "Failed to pick item"
        return response

    # ---------------------------------------------------------
    # [Step 5] 트레이 배치 (여기가 핵심)
    # ---------------------------------------------------------
    center_tray_pos = get_tray_center_pose(target_slot_id)
    center_tray_pos[1] += 10 # 약간의 Y 오프셋 (기존 코드 유지)
    grid_pos = slot_manager.get_tray_place_pose(center_tray_pos, item_index, total_items)
    
    # =========================================================
    # [수정] 트레이 배치 시 그리퍼 90도 회전 (Degree 단위)
    # =========================================================
    # ㅡ자(가로) -> ㅣ자(세로)로 회전하여 배치하여 충돌 방지
    grid_pos[5] += 90.0 
    # =========================================================
    
    # 상공 이동 (배치하러 가기) - 회전하면서 이동함
    approach_place = list(grid_pos)
    approach_place[2] = lifted_z # 아까 집고 올라온 높은 Z 유지
    
    if not safe_movel(approach_place, "배치 상공 접근"):
        response.success = False; response.message = "Failed approach"
        return response
    
    # 하강 및 놓기 (이미 회전된 상태)
    item_h = ITEM_HEIGHTS.get(item_name, 30.0) 
    final_place_z = TRAY_FLOOR_Z + item_h + DROP_SAFETY_MARGIN
    drop_pose = list(approach_place)
    drop_pose[2] = final_place_z
    
    if not safe_movel(drop_pose, "배치 하강"):
        response.success = False; response.message = "Failed descend"
        return response
    
    # 그리퍼 열기
    if gripper_manager: gripper_manager.release(item_name)
    else: 
        if gripper: gripper.open_gripper(); time.sleep(0.5)
        
    # 복귀
    depart_pose = list(drop_pose)
    depart_pose[2] = lifted_z
    safe_movel(depart_pose, "배치 후 상승")
    
    response.success = True
    response.message = "Success (2-Stage Vision)"
    response.actual_x = drop_pose[0]
    response.actual_y = drop_pose[1]
    response.actual_z = drop_pose[2]
    
    node_.get_logger().info(f"   ✅ '{item_name}' 배치 완료 (Top-Down 방식)")
    
    return response
    
    

def main(args=None):
    global node_, dsr_control_node_, vision_cli, gripper, gripper_manager, slot_manager
    
    rclpy.init(args=args)
    
    node_ = rclpy.create_node("item_placement_controller", namespace=ROBOT_ID)
    dsr_control_node_ = rclpy.create_node("dsr_item_internal_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_control_node_
    
    load_calibration()
    
    slot_manager = SlotManager()
    
    node_.get_logger().info("=========================================")
    node_.get_logger().info("🔧 Item Placement Controller Node (Updated)")
    node_.get_logger().info("=========================================")
    
    # 그리퍼 초기화
    try:
        gripper = RG("rg2", "192.168.1.1", "502")
        node_.get_logger().info("✅ Real Gripper Initialized")
    except:
        gripper = RG()
        node_.get_logger().info("⚠️ Virtual Gripper Initialized")
    
    gripper_manager = GripperManager(gripper)
    
    # 서비스 설정
    cb_group = ReentrantCallbackGroup()
    node_.create_service(PlaceItem, '/place_item', handle_place_item, callback_group=cb_group)
    vision_cli = node_.create_client(DetectObject, '/dsr01/detect_object', callback_group=cb_group)
    
    # 초기 자세 이동
    safe_movej(J_ITEM_OBSERVE)
    node_.get_logger().info("📍 Ready at J_ITEM_OBSERVE position")
    node_.get_logger().info("🎯 Service Ready: /place_item")
    
    executor = MultiThreadedExecutor()
    executor.add_node(node_)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()