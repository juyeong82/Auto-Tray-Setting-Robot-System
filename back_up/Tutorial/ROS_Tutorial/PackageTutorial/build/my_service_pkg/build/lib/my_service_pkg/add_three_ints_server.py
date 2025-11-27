import rclpy  # ROS2의 Python 클라이언트 라이브러리 (노드 생성, 통신 기능 등 제공)
from rclpy.node import Node  # ROS2 노드의 기반 클래스
from my_interfaces.srv import AddThreeInts

# 서비스 서버 클래스를 정의 (Node를 상속받아 ROS 노드 역할을 수행)
class AddThreeIntsServer(Node):
    def __init__(self):
        # 부모 클래스(Node)의 생성자를 호출하고 노드 이름을 'add_three_ints_server'로 지정
        super().__init__('add_three_ints_server')

        # 서비스 생성: AddThreeInts 타입을 사용하고, 서비스 이름은 'add_three_ints'
        # 요청이 들어오면 self.callback 함수가 호출됨
        self.srv = self.create_service(
            AddThreeInts,      # 사용할 서비스 타입 (요청/응답 정의 포함)
            'add_three_ints',  # 서비스 이름 (클라이언트가 이 이름으로 요청을 보냄)
            self.callback      # 요청이 들어왔을 때 호출할 함수
        )

    # 서비스 요청이 들어오면 자동으로 호출되는 콜백 함수
    def callback(self, request, response):
        # 수신한 요청 데이터 로그 출력
        self.get_logger().info(
            f"Received: {request.a} + {request.b} + {request.c}"
        )

        # 요청(request)에 담긴 세 정수를 더하여 응답(response)에 저장
        response.sum = request.a + request.b + request.c

        # 응답 객체를 반환하면 클라이언트에게 자동으로 전송됨
        return response

# 메인 함수: 노드를 실행하고 서비스 요청을 기다리는 루프를 시작함
def main(args=None):
    rclpy.init(args=args)           # rclpy 초기화 (필수)
    node = AddThreeIntsServer()     # 서비스 서버 노드 객체 생성
    rclpy.spin(node)                # 노드가 종료될 때까지 콜백 큐를 처리하며 실행 유지
    node.destroy_node()             # 노드가 종료되면 메모리 및 리소스 해제
    rclpy.shutdown()                # rclpy 종료 (ROS 통신 정리)
