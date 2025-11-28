# interface만들기(메시지 타입 선언을 위한 파일 패키지 만들기)

```bash
ros2 pkg create my_interfaces --build-type ament_cmake  # package create
cd my_interfaces/                                       
rm -rf include src                            # remove unnecessary folder
srv/AddThreeInts.srv 파일 만들기                # message type define file 
CMakeLists.txt 파일과 package.xml파일 수정하기    # modify dependency
```

# 실제 코드 (서비스 서버와 서비스 클라이언트가 있는 소스코드)
```
ros2 pkg create my_service_pkg --build-type ament_python --dependencies rclpy my_interfaces    # package create
코드작성               #  my_interfaces 폴더 안에 만들기
setup.py수정          #  만든 파일을 setup.py에 추가해주기
```


# 실행
ros2 run my_service_pkg add_three_ints_server 
ros2 run my_service_pkg add_three_ints_client   # 새로운 터미널을 열어서
