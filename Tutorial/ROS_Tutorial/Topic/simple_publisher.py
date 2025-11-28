import rclpy  # ROS2 Python 클라이언트 라이브러리
from rclpy.node import Node  # Node 클래스를 가져옴 (ROS 노드 정의에 필요)
from std_msgs.msg import String  # 표준 메시지 타입 중 문자열 메시지를 사용

# 퍼블리셔 노드 정의
class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')  # 노드 이름을 'simple_publisher'로 설정하고 Node 초기화
        self.publisher_ = self.create_publisher(String, 'my_topic', 10)  # 'my_topic'라는 이름의 토픽을 생성하고 String 타입 메시지를 퍼블리시함 (큐 크기: 10)
        timer_period = 1.0  # 타이머 콜백 주기 (초) — 1초마다 메시지 전송
        self.timer = self.create_timer(timer_period, self.timer_callback)  # 주기적으로 호출될 타이머 콜백 함수 등록
        self.count = 0  # 메시지에 포함될 카운터 초기값

    # 주기적으로 호출되는 콜백 함수
    def timer_callback(self):
        msg = String()  # 메시지 객체 생성
        msg.data = f'Hello ROS2: {self.count}'  # 메시지 내용 설정 (카운터 포함 문자열)
        self.publisher_.publish(msg)  # 메시지를 퍼블리시
        self.get_logger().info(f'Published: "{msg.data}"')  # 로그에 퍼블리시한 메시지 출력
        self.count += 1  # 카운터 증가

# 메인 함수
def main(args=None):
    rclpy.init(args=args)  # rclpy 초기화
    node = SimplePublisher()  # 노드 객체 생성
    rclpy.spin(node)  # 노드를 실행하여 콜백 함수가 동작하도록 함 (무한 루프)

    node.destroy_node()  # 노드 종료 시 리소스 해제
    rclpy.shutdown()  # rclpy 종료


if __name__ == "__main__":
    main()