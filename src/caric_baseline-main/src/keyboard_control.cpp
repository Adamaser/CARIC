#include <iostream>
#include <ros/ros.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <geometry_msgs/Point.h>

double x = 0, y = 0, z = 10;

int main(int argc, char ** argv) {
    ros::init(argc, argv, "my_control");
    ros::NodeHandle n;
    ros::Rate rate(20);

    // 发布话题，/<unit_id>/command/trajectory
    ros::Publisher point_pub = n.advertise<geometry_msgs::Point>("/setpoint", 10);
    std::string inputLine;
    double x, y, z;
    
    // 需要循环发送控制指令
    while (ros::ok())
    {
        std::cout << "Enter point: ";
        std::getline(std::cin, inputLine); // 获取整行输入

        std::istringstream iss(inputLine); // 创建字符串流对象
        iss >> x >> y >> z; // 从字符串流中提取数字

        geometry_msgs::Point point_taget;
        point_taget.x = x;
        point_taget.y = y;
        point_taget.z = z;

        // 发布控制话题
        point_pub.publish(point_taget); 

        rate.sleep();
        std::cout << "running" << std::endl;
    }

    return 0;
}
