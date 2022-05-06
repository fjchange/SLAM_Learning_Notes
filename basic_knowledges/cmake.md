## CMake 入门笔记

引自[小白移动机器人](https://mp.weixin.qq.com/s/fHnTpTwmds4qBy481B8MRw)

Cmake是啥？跨平台编译工具，长于管理中大型项目工程。

能做啥？自动化生成makefile。

学啥？用啥学啥。

### 0、常用预定义变量

```
Variable	                Info
CMAKE_SOURCE_DIR	        根源代码目录，工程顶层目录。暂认为就是PROJECT_SOURCE_DIR
CMAKE_CURRENT_SOURCE_DIR	当前处理的 CMakeLists.txt 所在的路径
PROJECT_SOURCE_DIR	        工程顶层目录
CMAKE_BINARY_DIR	        运行cmake的目录。外部构建时就是build目录
CMAKE_CURRENT_BINARY_DIR	The build directory you are currently in.当前所在build目录
PROJECT_BINARY_DIR	        暂认为就是CMAKE_BINARY_DIR
```

### 1、第一个CMakeLists.txt文件

文件树如下：

```
├── CMakeLists.txt
├── main.cpp
```

CMakeLists.txt文件如下：

```
cmake_minimum_required(VERSION 3.5) #设置CMake最小版本
project (hello_cmake) #设置工程名，自动生成一些变量，比如PROJECT_NAME
add_executable(hello_cmake main.cpp) #生成可执行文件
```

注意事项：

* 每一个需要进行 CMake 操作的目录下面，都必须存在文件 **CMakeLists.txt** 。
* CMake 命令 **不区分大小写** 。习惯上，CMake 命令全小写，预定义变量全大写。
* 变量使用 `${}` 方式取值，但是在 `if` 控制语句中是直接使用变量名。
* command(parameter1 parameter2 …)，参数使用括号括起，参数之间使用空格或分号分开。

**外部构建和内部构建**

当然是推荐外部构建啦，内部构建留着落灰吧！

```
mkdir build
cd build/
cmake ..
```

### 2、如果工程里有其他的头文件呢？

文件树如下：

```
├── CMakeLists.txt
├── include
│   └── Hello.h
└── src
    ├── Hello.cpp
    └── main.cpp
```

CMakeLists.txt文件如下：

```
cmake_minimum_required(VERSION 3.5)#最低CMake版本
project (hello_headers)# 工程名
add_executable(hello_headers src/Hello.cpp #用所有的源文件生成一个可执行文件
						   src/main.cpp )#不建议对源文件使用变量
target_include_directories(hello_headers#设置这个可执行文件hello_headers需要包含的库的路径
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)
#PROJECT_SOURCE_DIR指工程顶层目录
#PRIVATE指定了库的范围
```

### 3、如果工程里包含静态库呢？

文件树如下：

```
├── CMakeLists.txt
├── include
│   └── static
│       └── Hello.h
└── src
    ├── Hello.cpp
    └── main.cpp
```

main.cpp

```
#include "static/Hello.h" // 这里需要注意

int main(int argc, char *argv[])
{
    Hello hi;
    hi.print();
    return 0;
}
```

CMakeLists.txt文件如下：

```
cmake_minimum_required(VERSION 3.5)
project(hello_library)
############################################################
# Create a library
############################################################
#库的源文件Hello.cpp生成静态库libhello_library.a
add_library(hello_library STATIC 
    src/Hello.cpp
)
target_include_directories(hello_library
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
)
# target_include_directories为一个目标（可能是一个库library也可能是可执行文件）添加头文件路径。
############################################################
# Create an executable
############################################################
#指定用哪个源文件生成可执行文件
add_executable(hello_binary 
    src/main.cpp
)
#链接可执行文件和静态库
target_link_libraries(hello_binary
    PRIVATE 
        hello_library
)
#链接库和包含头文件都有关于scope这三个关键字的用法。
```

### 4、如果工程里包含共享库呢？

文件树如下：

```
├── CMakeLists.txt
├── include
│   └── shared
│       └── Hello.h
└── src
    ├── Hello.cpp
    └── main.cpp
```

CMakeLists.txt文件如下：

```
cmake_minimum_required(VERSION 3.5)
project(hello_library)
############################################################
# Create a library
############################################################
#根据Hello.cpp生成动态库libhello_library.so
add_library(hello_library SHARED 
    src/Hello.cpp
)
#给动态库hello_library起一个别的名字hello::library
add_library(hello::library ALIAS hello_library)
#为这个库目标，添加头文件路径，PUBLIC表示包含了这个库的目标也会包含这个路径
target_include_directories(hello_library
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
)
############################################################
# Create an executable
############################################################
#根据main.cpp生成可执行文件
add_executable(hello_binary
    src/main.cpp
)
#链接库和可执行文件，使用的是这个库的别名。PRIVATE 表示
target_link_libraries( hello_binary
    PRIVATE 
        hello::library
)
```

### 5、如果工程里包含第三方库呢？

main.c如下所示：

```
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
/*Boost库是为C++语言标准库提供扩展的一些C++程序库的总称，由Boost社区组织开发、
维护。Boost库可以与C++标准库完美共同工作，并且为其提供扩展功能。*/
int main(int argc, char *argv[])
{
 ...
 return 0;
}
```

CMakeLists.txt文件如下：

```
cmake_minimum_required(VERSION 3.5)
project (third_party_include)
# find a boost install with the libraries filesystem and system
find_package(Boost 1.46.1 REQUIRED COMPONENTS filesystem system)
# check if boost was found
if(Boost_FOUND)
    message ("boost found")
else()
    message (FATAL_ERROR "Cannot find Boost")
endif()
# Add an executable
add_executable(third_party_include main.cpp)
# link against the boost libraries
target_link_libraries( third_party_include
    PRIVATE
        Boost::filesystem
)
```

如上所述，find_package（）函数将从CMAKE_MODULE_PATH中的文件夹列表中搜索“ FindXXX.cmake”中的CMake模块。find_package参数的确切格式取决于要查找的模块。这通常记录在FindXXX.cmake文件的顶部。

```
find_package(Boost 1.46.1 REQUIRED COMPONENTS filesystem system)
```

参数：

Boost-库名称。这是用于查找模块文件FindBoost.cmake的一部分

1.46.1 - 需要的boost库最低版本

REQUIRED - 告诉模块这是必需的，如果找不到会报错

COMPONENTS - 要查找的库列表。从后面的参数代表的库里找boost

可以使用更多参数，也可以使用其他变量。在后面的示例中提供了更复杂的设置。

 **尽管大多数现代库都使用导入的目标，但并非所有模块都已更新。如果未更新库，则通常会发现以下可用变量** ：

* xxx_INCLUDE_DIRS - 指向库包含目录的变量。
* xxx_LIBRARY - 指向库路径的变量。

  然后可以将它们添加到您的target_include_directories和target_link_libraries中，如下所示：

  ```
    # Include the boost headers
    target_include_directories( third_party_include
        PRIVATE ${Boost_INCLUDE_DIRS}
    )

    # link against the boost libraries
    target_link_libraries( third_party_include
        PRIVATE
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_FILESYSTEM_LIBRARY}
    )
  ```

### 6、如何设置构建类型呢？

文件树如下：

```
├── CMakeLists.txt
├── main.cpp
```

CMakeLists.txt文件如下：

```
cmake_minimum_required(VERSION 3.5)
#如果没有指定则设置默认编译方式
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  #在命令行中输出message里的信息
  message("Setting build type to 'RelWithDebInfo' as none was specified.")
  #不管CACHE里有没有设置过CMAKE_BUILD_TYPE这个变量，都强制赋值这个值为RelWithDebInfo
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  # 当使用cmake-gui的时候，设置构建级别的四个可选项
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

project (build_type)
add_executable(cmake_examples_build_type main.cpp)
```

CMake命令行中设置构建类型：

```
cmake .. -DCMAKE_BUILD_TYPE=Release
```

构建级别

* Release —— 不可以打断点调试，程序开发完成后发行使用的版本，占的体积小。它对代码做了优化，因此速度会非常快，

  在编译器中使用命令：`-O3 -DNDEBUG` 可选择此版本。
* Debug ——调试的版本，体积大。

  在编译器中使用命令：`-g` 可选择此版本。
* MinSizeRel—— 最小体积版本

  在编译器中使用命令：`-Os -DNDEBUG`可选择此版本。
* RelWithDebInfo—— 既优化又能调试。

  在编译器中使用命令：`-O2 -g -DNDEBUG`可选择此版本。

```
set(<variable> <value>... [PARENT_SCOPE])
```

### 7、如何设置编译方式呢？

CMakeLists.txt文件如下：

```
cmake_minimum_required(VERSION 3.5)
project (compile_flags)
add_executable(cmake_examples_compile_flags main.cpp)
#为可执行文件添加私有编译定义
target_compile_definitions(cmake_examples_compile_flags #建议使用
    PRIVATE EX3
)
```

对于编译器选项，还可以使用target_compile_options（）函数。

```
target_compile_definitions(<target>
   <INTERFACE|PUBLIC|PRIVATE> [items1...] )
```

`target` 指的是由 `add_executable()`产生的可执行文件或 `add_library()`添加进来的库。

`<INTERFACE|PUBLIC|PRIVATE>`指的是 `[items...]` 选项可以传播的范围；

`PUBLIC and INTERFACE`会传播 `<target>`的INTERFACE_COMPILE_DEFINITIONS属性，

`PRIVATE and PUBLIC`   会传播 `<target>` 的COMPILE_DEFINITIONS属性。

### 8、最终例子

CMakeLists.txt文件如下：

```
cmake_minimum_required(VERSION 3.0.2)
project(slam2d)
## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS  "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  rosbag
  tf
)

## System dependencies are found with CMake's conventions
find_package(Eigen3)
find_package(Ceres)
find_package(PCL)
find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES slam2d
 CATKIN_DEPENDS nav_msgs roscpp rospy sensor_msgs std_msg tf
 DEPENDS EIGEN3 PCL
)

###########
## Build ##
###########

include_directories(
# include
  src
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(slam2d_node  src/slam2d_node.cpp)
target_link_libraries(slam2d_node  ${catkin_LIBRARIES} 
							    ${CERES_LIBRARIES} 
							    ${PCL_LIBRARIES} 	   
							    ${OpenCV_LIBS})
							  
```
