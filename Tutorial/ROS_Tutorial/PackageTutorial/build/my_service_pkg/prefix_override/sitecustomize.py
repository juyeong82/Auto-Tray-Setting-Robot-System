import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rokey/Desktop/Tutorial/ROS_Tutorial/PackageTutorial/install/my_service_pkg'
