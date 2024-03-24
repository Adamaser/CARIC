#include "baseline_planner.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "baseline_planner");
    ros::NodeHandle nh_init;
    
    ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh_init);

    Agent a(nh_ptr);  // 传入句柄指针初始化Agent类

    ros::MultiThreadedSpinner spinner(0);  // 在多线程环境中处理 ROS 节点的回调函数和消息循环，参数为0自动根据 CPU 核心数量来决定要创建多少个线程
    spinner.spin();

    return 0;
}