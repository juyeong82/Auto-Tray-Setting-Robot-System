import rclpy  # ROS2 Python 클라이언트 라이브러리
from rclpy.node import Node  # Node 클래스를 가져옴
from std_msgs.msg import String  # 퍼블리셔와 동일한 메시지 타입 (String)을 사용

# 서브스크라이버 노드 정의


class SimpleSubscriber(Node):
    def __init__(self):
        super().__init__('simple_subscriber')  # 노드 이름을 'simple_subscriber'로 설정하고 Node 초기화
        self.subscription = self.create_subscription(
            String,      # 수신할 메시지 타입
            'my_topic',  # 구독할 토픽 이름
            self.listener_callback,  # 메시지 수신 시 호출할 콜백 함수
            10)  # 큐 크기
        self.subscription  # 경고 방지를 위해 변수 사용 (사실상 의미는 없음)

    # 메시지 수신 시 호출되는 콜백 함수
    def listener_callback(self, my_msg):
        self.get_logger().info(f'I heard: "{my_msg.data}"')  # 수신한 메시지를 로그로 출력

# 메인 함수
def main(args=None):
    rclpy.init(args=args)  # rclpy 초기화
    node = SimpleSubscriber()  # 노드 객체 생성
    rclpy.spin(node)  # 콜백 함수가 실행되도록 노드를 실행
    node.destroy_node()  # 노드 종료
    rclpy.shutdown()  # rclpy 종료


if __name__ == "__main__":
    main()