import rclpy  # ROS2 Python 클라이언트 라이브러리
from rclpy.node import Node  # Node 클래스
from example_interfaces.srv import AddTwoInts  # 서버와 동일한 서비스 타입 사용

# 서비스 클라이언트 노드 정의
class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')  # 노드 이름 설정
        # 'add_two_ints' 서비스에 대한 클라이언트 생성
        self.client = self.create_client(AddTwoInts, 'add_two_ints')
        # 서비스가 준비될 때까지 반복해서 대기
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        # 요청 객체 생성
        self.request = AddTwoInts.Request()
    
    # 두 정수를 인자로 받아 서비스 요청을 보내고 결과를 반환
    def send_request(self, a, b):
        self.request.a = a  # 첫 번째 정수 설정
        self.request.b = b  # 두 번째 정수 설정
        future = self.client.call_async(self.request)  # 비동기 방식으로 요청 전송
        rclpy.spin_until_future_complete(self, future)  # 응답이 올 때까지 대기
        return future.result()  # 결과 반환

# 메인 함수 — 요청 보내기 및 결과 출력
def main(args=None):
    rclpy.init(args=args)  # rclpy 초기화
    node = AddTwoIntsClient()  # 클라이언트 노드 객체 생성
    result = node.send_request(17, 3)  # 7과 3을 더하는 요청 전송
    node.get_logger().info(f'Result: {result.sum}')  # 응답받은 결과 출력
    node.destroy_node()  # 노드 제거
    rclpy.shutdown()  # rclpy 종료


if __name__ == "__main__":
    main()