cmake_minimum_required(VERSION 3.5)
project(transport_mpc)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# install(
#   DIRECTORY config description launch scripts 
#   DESTINATION share/${PROJECT_NAME}
# )

# Install Python modules
ament_python_install_package(transporter)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS
  scripts/start_controller.py
  scripts/start_controller_decen.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
