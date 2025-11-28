import rclpy  # ROS2 Python 클라이언트 라이브러리
from rclpy.node import Node  # ROS2 노드 클래스
from my_interfaces.srv import AddThreeInts

# 서비스 클라이언트 노드 클래스 정의
class AddThreeIntsClient(Node):
    def __init__(self):
        # 노드 초기화 및 이름 지정
        super().__init__('add_three_ints_client')
        

        # 클라이언트 생성: 'add_three_ints'라는 이름의 AddThreeInts 타입 서비스에 연결 시도
        self.client = self.create_client(
            AddThreeInts,        # 사용할 서비스 타입
            'add_three_ints'     # 연결할 서비스 이름 (서버와 동일해야 함)
        )

        # 서비스가 준비될 때까지 1초 간격으로 대기 (비동기 환경에서 중요)
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service...")  # 서비스가 준비될 때까지 반복 출력

        # 서비스 요청 객체 생성 (빈 요청)
        self.req = AddThreeInts.Request()

    # 숫자 3개를 인자로 받아 요청을 구성하고 서버로 전송하는 함수
    def send_request(self, a, b, c):
        self.req.a = a  # 요청의 첫 번째 숫자 설정
        self.req.b = b  # 두 번째 숫자 설정
        self.req.c = c  # 세 번째 숫자 설정

        # 비동기 방식으로 서비스 요청 전송 (콜백 없이 future 객체로 응답을 처리)
        future = self.client.call_async(self.req)

        # 응답이 도착할 때까지 블로킹 대기 (스레드-safe)
        rclpy.spin_until_future_complete(self, future)

        # 응답 데이터(future.result())를 반환
        return future.result()

# 메인 함수: 클라이언트 노드를 실행하고 요청을 전송
def main(args=None):
    rclpy.init(args=args)                     # rclpy 초기화
    node = AddThreeIntsClient()              # 클라이언트 노드 객체 생성

    # 1, 2, 3이라는 세 정수를 요청으로 전송하고 응답 결과를 저장
    result = node.send_request(1, 2, 3)

    # 받은 응답의 sum 값을 로그로 출력
    node.get_logger().info(f"Result: {result.sum}")

    node.destroy_node()                      # 노드 제거
    rclpy.shutdown()                         # rclpy 종료
