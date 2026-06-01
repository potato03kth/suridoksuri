from setuptools import find_packages, setup

package_name = "fc_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch",
         ["launch/phase1.launch.py", "launch/phase2.launch.py"]),
        (f"share/{package_name}/params",
         ["fc_ros/params/fc_ros_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="suridoksuri",
    maintainer_email="noreply@suridoksuri",
    description="ROS2/MAVROS 기반 FC 브릿지 노드 패키지",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "telemetry_node = fc_ros.nodes.telemetry_node:main",
            "offboard_node  = fc_ros.nodes.offboard_node:main",
            "mission_node   = fc_ros.nodes.mission_node:main",
        ],
    },
)
