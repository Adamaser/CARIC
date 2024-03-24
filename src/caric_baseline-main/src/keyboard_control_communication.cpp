#include <iostream>
#include <ros/ros.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <geometry_msgs/Point.h>
#include <caric_mission/CreatePPComTopic.h>
#include <std_msgs/String.h>

double x = 0, y = 0, z = 5;

trajectory_msgs::MultiDOFJointTrajectory construct_pose(double x, double y, double z, double q_x=0, double q_y=0, double q_z=0.707, double q_w=0.707) {
    trajectory_msgs::MultiDOFJointTrajectory trajset_msg;
    trajectory_msgs::MultiDOFJointTrajectoryPoint trajpt_msg;

    geometry_msgs::Transform transform_msg;
    geometry_msgs::Twist accel_msg, vel_msg;

    transform_msg.translation.x = x;
    transform_msg.translation.y = y;
    transform_msg.translation.z = z;
    transform_msg.rotation.x = q_x;
    transform_msg.rotation.y = q_y;
    transform_msg.rotation.z = q_z;
    transform_msg.rotation.w = q_w;


    trajpt_msg.transforms.push_back(transform_msg);

    vel_msg.linear.x = 0;
    vel_msg.linear.y = 0;
    vel_msg.linear.z = 0;

    accel_msg.linear.x = 0;
    accel_msg.linear.y = 0;
    accel_msg.linear.z = 0;

    trajpt_msg.velocities.push_back(vel_msg);
    trajpt_msg.accelerations.push_back(accel_msg);
    trajset_msg.points.push_back(trajpt_msg);
    return trajset_msg;

}

void setpoint_recall(const geometry_msgs::Point msg) {
    x = msg.x;
    y = msg.y;
    z = msg.z;
    std::cout << "get point" << std::endl;
    return;
}

void gcs_task_assign(auto msg) {

}

int main(int argc, char ** argv) {
    ros::init(argc, argv, "my_control_communication");
    ros::NodeHandle n;
    ros::Rate rate(20);
    std::string uav_name;
    std::cout << "enter name of uav: ";
    std::cin >> uav_name;

    // 发布话题，/<unit_id>/command/trajectory
    ros::Publisher trajectory_pub = n.advertise<trajectory_msgs::MultiDOFJointTrajectory>("/" + uav_name + "/command/trajectory", 50);
    
    ros::Subscriber pos_sub = n.subscribe<geometry_msgs::Point>("/setpoint", 10, setpoint_recall);
    caric_mission::CreatePPComTopic srv;
    // 注册每个无人机
    auto client = n.serviceClient<caric_mission::CreatePPComTopic>("create_ppcom_topic");  
    auto cmd_pub_ = n.advertise<std_msgs::String>("/task_assign", 10);
    srv.request.source = "gcs";
    srv.request.targets.push_back("all");
    srv.request.topic_name = "/task_assign";
    srv.request.package_name = "std_msgs";
    srv.request.message_type = "String";
    client.call(srv);

    auto agent_position_sub_=n.subscribe<std_msgs::String>("/broadcast/gcs", 10, gcs_task_assign);
    
    
    // 需要循环发送控制指令
    while (ros::ok())
    {
        // 构造位姿、速度、加速度的数据
        trajectory_msgs::MultiDOFJointTrajectory trajset_msg = construct_pose(x, y, z);
        trajset_msg.header.stamp = ros::Time::now();
        // 发布控制话题
        trajectory_pub.publish(trajset_msg); 

        ros::spinOnce();

        rate.sleep();
        // std::cout << "running" << std::endl;
    }

    return 0;
}
