#include <iostream>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <visualization_msgs/MarkerArray.h>

#include "Eigen/Dense"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "Astar.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl/geometry/distance.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

#include <pcl/kdtree/kdtree_flann.h>
#include "utility.h"
#include <mutex>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <caric_mission/CreatePPComTopic.h>
#include <std_msgs/String.h>

#include "general_task_init.h"
#include "Astar.h"
#include <map>

#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>

#include "planner.h"
#include <nav_msgs/Path.h>

// for pcd
mutex mtx_cloud;

struct Point3D {
    double x, y, z;

    Point3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

    // 自定义比较函数，用于在std::map中排序
    bool operator<(const Point3D& other) const {
        if (x < other.x) return true;
        if (x > other.x) return false;
        if (y < other.y) return true;
        if (y > other.y) return false;
        return z < other.z;
    }
};

struct agent_local
{
    bool in_bounding_box = false;//判断在世界边界内的标志位
    bool planning_in_bounding_box = false;//在世界边界内规划标志位
    Eigen::Vector3i position_index;//位置索引
    Eigen::Vector3i planning_index;//规划位置索引
    double time = 0;//时间戳
    double state = 0;//状态标志
    double priority = 0;//优先级标志位
};

struct info
{
    bool get_info = false;//信息获取状态标志位
    double message_time = 0;//信息更新时间戳
    Eigen::Vector3d global_point;//全局坐标信息
    list<Eigen::Vector3d> global_path = {};//路径信息列表
    int state = 0;//无人机当前状态
    int priority = 0;//优先级
};

//名为info_agent代理信息管理器
class info_agent                          
{
public:
    
    info_agent()
    {
        namelist = {"/jurong", "/raffles", "/changi", "/sentosa", "/nanyang"};
        Agent_dict["/jurong"] = {false, 0, Eigen::Vector3d(0, 0, 1), {}, 0, 5};
        Agent_dict["/raffles"] = {false, 0, Eigen::Vector3d(0, 0, 2), {}, 0, 4};
        Agent_dict["/changi"] = {false, 0, Eigen::Vector3d(0, 0, 3), {}, 0, 3};
        Agent_dict["/sentosa"] = {false, 0, Eigen::Vector3d(0, 0, 4), {}, 0, 2};
        Agent_dict["/nanyang"] = {false, 0, Eigen::Vector3d(0, 0, 5), {}, 0, 1};
    }
    
    info_agent(vector<string> teammate)
    {
        namelist = {"/jurong", "/raffles", "/changi", "/sentosa", "/nanyang"};
        Agent_dict["/jurong"] = {false, 0, Eigen::Vector3d(0, 0, 1), {}, 0, 5};
        Agent_dict["/raffles"] = {false, 0, Eigen::Vector3d(0, 0, 2), {}, 0, 4};
        Agent_dict["/changi"] = {false, 0, Eigen::Vector3d(0, 0, 3), {}, 0, 3};
        Agent_dict["/sentosa"] = {false, 0, Eigen::Vector3d(0, 0, 4), {}, 0, 2};
        Agent_dict["/nanyang"] = {false, 0, Eigen::Vector3d(0, 0, 5), {}, 0, 1};
        for (int i = 0; i < teammate.size(); i++)
        {
            if (teammate[i] == "jurong")
            {
                leader = "/jurong";
            }
            else if (teammate[i] == "raffles")
            {
                leader = "/raffles";
            }
        }
    }
    
    int get_leader_state()//获取当前leader无人机的状态
    {
        return Agent_dict[leader].state;
    }
    
    string get_leader()//获取当前leader无人机的名称
    {
        return leader;
    }
    
    void get_leader_position(Eigen::Vector3d &target)//获取当前leader无人机的位置信息，存储在target中
    {
        // cout<<"leader:"<<leader<<endl;
        target = Agent_dict[leader].global_point;
    }
    
    void update_state(string name, int state_in)//更新传入name无人机的状态值
    {

        Agent_dict[name].state = state_in;
    }
    
    void reset_position_path(istringstream &str)//重置指定无人机的位置与路径信息
    {
        string name;
        getline(str, name, ';');
        if (name != "/jurong" && name != "/raffles" && name != "/changi" && name != "sentosa" && name != "/nanyang")
        {
            return;
        }
        else
        {
            string position_str;
            getline(str, position_str, ';');
            info info_temp;
            info_temp.get_info = true;
            info_temp.global_point = str2point(position_str);
            string path_point;
            while (getline(str, path_point, ';'))
            {
                info_temp.global_path.push_back(str2point(path_point));
            }
            info_temp.message_time = ros::Time::now().toSec();
            info_temp.state = Agent_dict[name].state;
            info_temp.priority = Agent_dict[name].priority;
            Agent_dict[name] = info_temp;
        }
    }

private:
    
    list<string> namelist;
    // list<Eigen::Vector3d> path_list;
    
    string leader;
    
    map<string, info> Agent_dict;
    
    void cout_name(string name)//输出指定无人机的所有信息
    {
        cout << name << endl;
        info info_in = Agent_dict[name];
        cout << "Priority:" << info_in.priority << endl;
        cout << "State:" << info_in.state << endl;
        cout << "Get info:" << info_in.get_info << endl;
        cout << "Time:" << info_in.message_time << endl;
        cout << "Global position:" << info_in.global_point.transpose() << endl;
        cout << "Path point:" << endl;
        for (auto &point : info_in.global_path)
        {
            cout << "node:" << point.transpose() << endl;
        }
        cout << endl;
    }
    
    Eigen::Vector3d str2point(string input)//将无人机的位置信息从字符串格式转为向量
    {
        Eigen::Vector3d result;
        std::vector<string> value;
        boost::split(value, input, boost::is_any_of(","));
        // cout<<input<<endl;
        if (value.size() == 3)
        {
            result = Eigen::Vector3d(stod(value[0]), stod(value[1]), stod(value[2]));
        }
        else
        {
            cout << input << endl;
            cout << "error use str2point 2" << endl;
        }
        return result;
    }
};

//初始化珊格地图
class grid_map
{
public:
    /********************建立珊格地图类，有三个不同的构造函数*************************/
    
    /********************构造函数1*****************/
    /*传入参数为空，系统默认构造*/
    ros::NodeHandlePtr nh_ptr_;
    ros::Publisher _path_minimum_pub;//发布minimum-snap轨迹

    grid_map() {}

    // Function use boundingbox message to build map
    /********************构造函数2*****************/
    /*参数：
    *@珊格边界：Boundingbox box
    *@珊格大小：grid_size_in
    *@无人机数：Teamsize_in
    *@无人机列表：team_list
    *函数功能：初始化珊格地图*/
    grid_map(Boundingbox box, Eigen::Vector3d grid_size_in, int Teamsize_in, vector<string> team_list, ros::NodeHandlePtr& nh_ptr)
    {   
        nh_ptr_ = nh_ptr;
        _path_minimum_pub = nh_ptr_->advertise<nav_msgs::Path>("/tra_generation", 10); 
        //将字符串映射到结构体agent_local
        local_dict["/jurong"] = {false, false, Eigen::Vector3i(0, 0, -1), Eigen::Vector3i(0, 0, -1), 0, 0, 5};
        local_dict["/raffles"] = {false, false, Eigen::Vector3i(0, 0, -1), Eigen::Vector3i(0, 0, -1), 0, 0, 4};
        local_dict["/sentosa"] = {false, false, Eigen::Vector3i(0, 0, -1), Eigen::Vector3i(0, 0, -1), 0, 0, 3};
        local_dict["/changi"] = {false, false, Eigen::Vector3i(0, 0, -1), Eigen::Vector3i(0, 0, -1), 0, 0, 2};
        local_dict["/nanyang"] = {false, false, Eigen::Vector3i(0, 0, -1), Eigen::Vector3i(0, 0, -1), 0, 0, 1};
        namelist = {"/jurong", "/raffles", "/changi", "/sentosa", "/nanyang"};
        for (auto &name : team_list)
        {
            if (name == "jurong" || name == "raffles")
            {
                continue;
            }
            follower.push_back("/" + name);
        }//分配leader和follower的命名空间
        team_size = Teamsize_in;  // 记录无人机数量
        fly_in_index = Eigen::Vector3i(0, 0, 0);//初始化珊格坐标系下的飞行索引
        rotation_matrix = box.getSearchRotation();  // 边界框的坐标变换旋转矩阵，旋转之后，局部坐标系下，z轴为立方体最长边的方向Rbw
        rotation_matrix_inv = rotation_matrix.inverse();  //反变换矩阵Rwb
        rotation_quat = Eigen::Quaterniond(rotation_matrix_inv);//将反变换矩阵表示为四元素
        map_global_center = box.getCenter();  // 边界框中心点的坐标
        map_quat_size = box.getRotExtents();  // 边界框坐标系下，各个边长距离中心点的距离
        grid_size = grid_size_in;  // 读取珊格大小
        initial_the_convert();  //计算地图的形状、中心索引以及初始化相关地图数据结构
        interval = floor(map_shape.z() / team_size);  //floor向下取整，局部坐标下，z轴方向就是最长边所在的方向，除于队伍成员数，对空间进行分块
        cout << "Teamsize:"
             << "team_size" << endl; // test
        for (int i = 1; i < Teamsize_in; i++)
        {
            region_slice_layer.push_back(i * interval);  //存储与切分区域有关的信息，每个团队成员都有一个切分区域
            finish_flag.push_back(0);  //往容器中添加一个0元素，可能用于跟踪任务或区域的完成状态？？
            finish_exp_flag.push_back(0);  //同上？？
        }
        set_under_ground_occupied();
    }

    // Function use grid size to build map used in construct the global map
    /********************构造函数3*****************/
    /*参数：
    *@珊格大小：grid_size_in
    *函数功能：初始化珊格地图*/
    grid_map(Eigen::Vector3d grid_size_in, ros::NodeHandlePtr& nh_ptr)
    {
        nh_ptr_ = nh_ptr;
        _path_minimum_pub = nh_ptr_->advertise<nav_msgs::Path>("/tra_generation", 10); 
        local_dict["/jurong"] = {false, false, Eigen::Vector3i(0, 0, -1), Eigen::Vector3i(0, 0, -1), 0, 0, 5};
        local_dict["/raffles"] = {false, false, Eigen::Vector3i(0, 0, -1), Eigen::Vector3i(0, 0, -1), 0, 0, 4};
        local_dict["/sentosa"] = {false, false, Eigen::Vector3i(0, 0, -1), Eigen::Vector3i(0, 0, -1), 0, 0, 3};
        local_dict["/changi"] = {false, false, Eigen::Vector3i(0, 0, -1), Eigen::Vector3i(0, 0, -1), 0, 0, 2};
        local_dict["/nanyang"] = {false, false, Eigen::Vector3i(0, 0, -1), Eigen::Vector3i(0, 0, -1), 0, 0, 1};
        namelist = {"/jurong", "/raffles", "/changi", "/sentosa", "/nanyang"};
        fly_in_index = Eigen::Vector3i(0, 0, 0);
        map_global_center = Eigen::Vector3d(0, 0, 0);//默认为0
        map_quat_size = Eigen::Vector3d(200, 200, 100);//默认地图尺寸
        grid_size = grid_size_in;
        rotation_matrix = Eigen::Matrix3d::Identity();  //转换初始化为单位矩阵
        rotation_matrix_inv = rotation_matrix.inverse();  
        rotation_quat = Eigen::Quaterniond(rotation_matrix_inv);
        initial_the_convert();
        set_under_ground_occupied();
    }

    // Function for update the map and interest point
    /*
    *函数作用：往地图中插入点，将其转化为珊格坐标系下的坐标，并更新地图
    *参数：
    *@插入点坐标信息：point_in
    */
    void insert_point(Eigen::Vector3d point_in)
    {
        Eigen::Vector3d point_in_local = rotation_matrix * (point_in - map_global_center);  //将世界坐标系下的点转化为边界框坐标系下的点（以原点为中心）
        if (out_of_range(point_in_local, false))  // 根据map_shape判断是否超过map范围，是的话直接退出函数
        {
            return;
        }
        Eigen::Vector3i bias_index(0, 0, 0);//记录珊格坐标系下的索引
        //在x维度上的坐标绝对值是否小于格子大小 grid_size.x() 的一半。如果是，说明点在格子的中心附近，偏移为0，yz轴类似
        if (fabs(point_in_local.x()) < 0.5 * grid_size.x())
        {
            bias_index.x() = 0;
        }
        else
        {
            if (point_in_local.x() > 0)  // 计算这个点云所处在的网格位置
            {
                bias_index.x() = floor((point_in_local.x() - 0.5 * grid_size.x()) / grid_size.x()) + 1;
            }
            else
            {
                bias_index.x() = -floor((-point_in_local.x() - 0.5 * grid_size.x()) / grid_size.x()) - 1;
            }
        }

        if (fabs(point_in_local.y()) < 0.5 * grid_size.y())
        {
            bias_index.y() = 0;
        }
        else
        {
            if (point_in_local.y() > 0)
            {
                bias_index.y() = floor((point_in_local.y() - 0.5 * grid_size.y()) / grid_size.y()) + 1;
            }
            else
            {
                bias_index.y() = -floor((-point_in_local.y() - 0.5 * grid_size.y()) / grid_size.y()) - 1;
            }
        }
        if (fabs(point_in_local.z()) < 0.5 * grid_size.z())
        {
            bias_index.z() = 0;
        }
        else
        {
            if (point_in_local.z() > 0)
            {
                bias_index.z() = floor((point_in_local.z() - 0.5 * grid_size.z()) / grid_size.z()) + 1;
            }
            else
            {
                bias_index.z() = -floor((-point_in_local.z() - 0.5 * grid_size.z()) / grid_size.z()) - 1;
            }
        }
        Eigen::Vector3i true_index = bias_index + map_index_center;  // 偏差加上地图中心坐标得到该地图坐标系下的珊格索引
        if (map[true_index.x()][true_index.y()][true_index.z()] == 1)  // 判断珊格是否被占用，如果标志为1表示已经被占用，退出函数
        {
            return; 
        }
        else//珊格未被占用
        {
            map[true_index.x()][true_index.y()][true_index.z()] = 1;  // 将此珊格标记为已被占用
            occupied_num++;  // 记录占用珊格数量加
            map_cloud_massage = point3i2str(true_index) + ";" + map_cloud_massage;  // 更新点云地图消息，将占用的网格索引添加到消息前部，并采用分号相隔
            for (int x = true_index.x() - 1; x < true_index.x() + 2; x++)  // 遍历周围立方体珊格
            {
                for (int y = true_index.y() - 1; y < true_index.y() + 2; y++)
                {
                    for (int z = true_index.z() - 1; z < true_index.z() + 2; z++)
                    {
                        if (out_of_range_index(Eigen::Vector3i(x, y, z)))  //检测是否超过了地图的范围
                        {
                            continue;
                        }
                        if (abs(x - true_index.x()) + abs(y - true_index.y()) + abs(z - true_index.z()) == 1)  // 如果珊格距离为1，即和该珊格接连，标记可能与新占用网格有关的感兴趣网格
                        {
                            if (map[x][y][z] == 0 && visited_map[x][y][z] == 0)  // 未被占用且未被访问过则标志为感兴趣
                            {
                                interest_map[x][y][z] = 1;
                            }
                            else
                            {
                                interest_map[x][y][z] = 0;
                            }
                        }
                    }
                }
            }
            return;
        }
    }

    /*
    *函数作用：绘制珊格地图
    *参数：
    *@点云信息队列：MarkerArray
    */
    visualization_msgs::MarkerArray Draw_map()
    {
        visualization_msgs::MarkerArray markers;
        for (int x = 0; x < map_shape.x(); x++)
        {
            for (int y = 0; y < map_shape.y(); y++)
            {
                for (int z = 0; z < map_shape.z(); z++)
                {
                    if (map[x][y][z] == 1)
                    {
                        markers.markers.push_back(generate_marker(Eigen::Vector3i(x, y, z), 0, markers.markers.size()));
                    }
                    else if (interest_map[x][y][z] == 1)
                    {
                        markers.markers.push_back(generate_marker(Eigen::Vector3i(x, y, z), 1, markers.markers.size()));
                    }
                }
            }
        }

        return markers;
    }

    /*
    *函数作用：更新当前全局坐标系下point的点、索引、局部坐标位置
    *参数：
    *@全局坐标系下的点：point
    */
    void update_position(Eigen::Vector3d point)
    {
        Eigen::Vector3d point_local = rotation_matrix * (point - map_global_center);  // 获得这个点在地图里面的局部坐标
        if (out_of_range(point_local, false))  // 判断是否超过范围
        {
            in_my_range = false;
            return;
        }
        Eigen::Vector3i index = get_index(point);  // 获取该无人机位置在珊格地图中的索引，即占了哪个珊格块
        // 首先获取搜索目标位置
        if (now_position_index != index && visited_map[index.x()][index.y()][index.z()] == 0 && search_direction.empty())  
        //检查新的位置索引是否与当前位置索引不同、是否未被访问过、当前搜索方向为空
        {
            search_direction = get_search_target(index);  // 获取云台搜索方向，可能返回的是空的
            time_start=ros::Time::now().toSec();//记录当前时间
        }
        // 每次到达指定位置会弹出一个位置（后文），当全部弹出时
        if (search_direction.empty())//当获取到的搜索位置也为空时，将此位置标记为访问过
        {
            visited_map[index.x()][index.y()][index.z()] = 1;
        }
        // 执行时间大于3秒后，将其标记为已访问
        if(fabs(ros::Time::now().toSec()-time_start)>3)
        {
            visited_map[index.x()][index.y()][index.z()] = 1;
        }
        now_position_global = point;
        now_position_index = index;
        now_position_local = point_local;
        in_my_range = true;
    }

    Eigen::Vector3i get_index(Eigen::Vector3d point_in)  // 获取某个点在珊格地图中的索引，即占了哪个珊格块
    {
        Eigen::Vector3d point_in_local = rotation_matrix * (point_in - map_global_center);
        Eigen::Vector3i bias_index(0, 0, 0);
        if (fabs(point_in_local.x()) < 0.5 * grid_size.x())
        {
            bias_index.x() = 0;
        }
        else
        {
            if (point_in_local.x() > 0)
            {
                bias_index.x() = floor((point_in_local.x() - 0.5 * grid_size.x()) / grid_size.x()) + 1;
            }
            else
            {
                bias_index.x() = -floor((-point_in_local.x() - 0.5 * grid_size.x()) / grid_size.x()) - 1;
            }
        }

        if (fabs(point_in_local.y()) < 0.5 * grid_size.y())
        {
            bias_index.y() = 0;
        }
        else
        {
            if (point_in_local.y() > 0)
            {
                bias_index.y() = floor((point_in_local.y() - 0.5 * grid_size.y()) / grid_size.y()) + 1;
            }
            else
            {
                bias_index.y() = -floor((-point_in_local.y() - 0.5 * grid_size.y()) / grid_size.y()) - 1;
            }
        }

        if (fabs(point_in_local.z()) < 0.5 * grid_size.z())
        {
            bias_index.z() = 0;
        }
        else
        {
            if (point_in_local.z() > 0)
            {
                bias_index.z() = floor((point_in_local.z() - 0.5 * grid_size.z()) / grid_size.z()) + 1;
            }
            else
            {
                bias_index.z() = -floor((-point_in_local.z() - 0.5 * grid_size.z()) / grid_size.z()) - 1;
            }
        }
        Eigen::Vector3i result = bias_index + map_index_center;
        return result;
    }
    
    /*
    @函数作用：分为两部分，第一部分为探索无人机，检查分配好的搜索区域是否被其他无人机搜索过，已经搜索时值为1，在更新的地图上再次调用Astar算法得到全局路径。第二部分为拍照无人机，功能相同
    @输入参数：
    1. target：目标点位置信息（全局坐标系下）
    2. myname：当前无人机的名称
    3. leader_name：leader无人机的名称
    4. flag：引用标志位
    5. islong：
    */
    void Astar_local(Eigen::Vector3d target, string myname, string leader_name, bool &flag, bool islong)
    {
        vector<vector<vector<int>>> map_temp = map;  // 创建三维地图初始化副本
        if (myname == "/jurong" || myname == "/raffles")  // 当前无人机为探索无人机时
        {
            if (true)
            { // Here condition should be whether need waiting;
                for (auto &name : namelist)//遍历所有在namelist中的无人机，检查当前无人机所规划的探索区域其他无人机是否已经探索过
                {
                    if (myname == name)//当前无人机是传入函数的无人机时退出当前循环
                    {
                        continue;
                    }
                    else
                    {
                        if (fabs(ros::Time::now().toSec() - local_dict[name].time) < 1 || true)  //检测时间间隔在1秒内, 这里跳过了
                        {
                            if (local_dict[name].in_bounding_box)//当前检查的无人机在bounding_box中
                            {
                                Eigen::Vector3i tar = local_dict[name].position_index;//标记当前无人机的位置已经占用
                                map_temp[tar.x()][tar.y()][tar.z()] = 1;
                            }
                            if (local_dict[name].planning_in_bounding_box)//检查当前无人机的路径存在于bounding box中
                            {
                                Eigen::Vector3i tar = local_dict[name].planning_index;//标记无人机的位置路径位置已经占用
                                map_temp[tar.x()][tar.y()][tar.z()] = 1;
                            }
                        }
                    }
                }
                Eigen::Vector3i tar_index = get_index(target);//获取当前无人机终点索引
                list<Eigen::Vector3i> path_tamp;//存储路径列表
                if (!islong)//判断是否超过目标长度
                {
                    path_tamp = astar_planner.get_path(map_temp, now_position_index, tar_index);
                }//采用Astar算法搜索路径
                else
                {
                    path_tamp = astar_planner.get_path_long(map_temp, now_position_index, tar_index);
                }

                if (path_tamp.empty())  //搜索到的路径为空时，标志为值为1，并将当前位置标记为最终位置，搜索不到后续怎么处理？？
                {
                    path_final_global = now_position_global;
                    path_index = path_tamp;
                    flag = true;
                }
                else  //搜索到路径时，将flag设置0表示搜索到了路径，将目标点作为最终位置，路径索引更新为搜索到的路径
                {
                    flag = false;
                    path_final_global = target;  
                    path_index = path_tamp;
                }
                generate_the_global_path();  // 更新全局路径
                return;
            }
            else
            {
                path_index = {};
                generate_the_global_path();
                return;
            }
        }
        else//无人机为拍照无人机时
        {
            for (auto &name : namelist)
            {
                if (myname == name || name == leader_name)//无人机为当前无人机户或者为leader无人机时，推出此次循环
                {
                    continue;
                }
                else
                {
                    if (fabs(ros::Time::now().toSec() - local_dict[name].time) < 1 || true)
                    {
                        if (local_dict[name].in_bounding_box)//其他无人机已经到过或者当前位置在此无人机的bounding box中时，标记为已经占用
                        {
                            Eigen::Vector3i tar = local_dict[name].position_index;
                            map_temp[tar.x()][tar.y()][tar.z()] = 1;
                        }
                        if (local_dict[name].planning_in_bounding_box)
                        {
                            Eigen::Vector3i tar = local_dict[name].planning_index;
                            map_temp[tar.x()][tar.y()][tar.z()] = 1;
                        }
                    }
                }
            }
            Eigen::Vector3i tar_index = get_index(target);
            list<Eigen::Vector3i> path_tamp;
            if (!islong)
            {
                path_tamp = astar_planner.get_path(map_temp, now_position_index, tar_index);
            }
            else
            {
                path_tamp = astar_planner.get_path_long(map_temp, now_position_index, tar_index);
            }
            if (path_tamp.empty())
            {
                path_final_global = now_position_global;
                path_index = path_tamp;
                flag = true;
            }
            else
            {
                if (fabs(ros::Time::now().toSec() - local_dict[leader_name].time) < 1)  // 如果当前时间和领导者的什么时间相差小于1？？，就从路径里面弹出最前面一个？？
                {
                    path_tamp.pop_front();
                }
                flag = false;
                path_final_global = target;
                path_index = path_tamp;
            }
            generate_the_global_path();
            return;
        }
    }
    
    void Astar_photo_my(Eigen::Vector3d target, string myname, bool &flag, bool islong)
    {
        vector<vector<vector<int>>> map_temp = map;
        for (auto &name : namelist)
        {
            if (myname == name)
            {
                continue;
            }
            else
            {
                if (fabs(ros::Time::now().toSec() - local_dict[name].time) < 1||true)
                {
                    if (local_dict[name].in_bounding_box)
                    {
                        Eigen::Vector3i tar = local_dict[name].position_index;
                        map_temp[tar.x()][tar.y()][tar.z()] = 1;
                    }
                    if (local_dict[name].planning_in_bounding_box)
                    {
                        Eigen::Vector3i tar = local_dict[name].planning_index;
                        map_temp[tar.x()][tar.y()][tar.z()] = 1;
                    }
                }
            }
        }
        Eigen::Vector3i tar_index = get_index(target);
        list<Eigen::Vector3i> path_tamp;
        if (!islong)
        {
            path_tamp = astar_planner.get_path(map_temp, now_position_index, tar_index);
        }
        else
        {
            path_tamp = astar_planner.get_path_long(map_temp, now_position_index, tar_index);
        }
        if (path_tamp.empty())
        {
            path_final_global = now_position_global;
            path_index = path_tamp;
            flag = true;
        }
        else
        {
            flag = false;
            path_final_global = target;
            path_index = path_tamp;
        }
        generate_the_global_path();
    }

    void Astar_photo(Eigen::Vector3d target, string myname, bool &flag)
    {
        vector<vector<vector<int>>> map_temp = map;
        for (auto &name : namelist)
        {
            if (myname == name)
            {
                continue;
            }
            else
            {
                if (fabs(ros::Time::now().toSec() - local_dict[name].time) < 1||true)
                {
                    if (local_dict[name].in_bounding_box)
                    {
                        Eigen::Vector3i tar = local_dict[name].position_index;
                        map_temp[tar.x()][tar.y()][tar.z()] = 1;
                    }
                    if (local_dict[name].planning_in_bounding_box)
                    {
                        Eigen::Vector3i tar = local_dict[name].planning_index;
                        map_temp[tar.x()][tar.y()][tar.z()] = 1;
                    }
                }
            }
        }
        Eigen::Vector3i tar_index = get_index(target);
        list<Eigen::Vector3i> path_tamp;
        path_tamp = astar_planner.get_path(map_temp, now_position_index, tar_index);
        if (path_tamp.empty())
        {
            path_final_global = now_position_global;
            path_index = path_tamp;
            flag = true;
        }
        else
        {
            flag = false;
            path_final_global = target;
            path_index = path_tamp;
        }
        generate_the_global_path();
    }
    
    Eigen::Vector3d get_fly_in_point_global()  //返回珊格地图中的全局坐标
    {
        // cout<<"fly in output"<<fly_in_index.transpose()<<endl;//test debug
        return get_grid_center_global(fly_in_index);
        // return get_grid_center_global(Eigen::Vector3i(0,0,1));
    }
    
    bool check_whether_fly_in(bool print)  // 判断无人机是否到达了飞入点
    {
        if (print)
        {
            cout << "now position" << now_position_index.transpose() << endl;
            cout << "fly in" << fly_in_index.transpose() << endl;
        }
        if ((now_position_index - fly_in_index).norm() < 2 && in_my_range)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    
    void update_fly_in_index(bool replan)  //通过在不同方向上搜索新位置，并检查一系列条件，来选择新的 fly_in_index 值
    {
        if (map[fly_in_index.x()][fly_in_index.y()][fly_in_index.z()] == 1 || replan)  // 如果飞入点是障碍物或者需要重新规划
        {
            int x = fly_in_index.x();
            int y = fly_in_index.y();
            int z = fly_in_index.z();
            int distance = min({abs(x), abs(y), abs(map_shape.x() - x), abs(map_shape.y() - y)});  // 当前点到地图边界的最短距离，考虑了 x 和 y 方向上的距离
            // 当前搜索的边界
            int top;
            int bottom;
            int left;
            int right;
            int i = x;
            int j = y;

            for (int k = z; k < map_shape.z(); k++)  // 从不同的高度搜索
            {
                for (distance; distance <= 3; distance++)
                {
                    top = map_shape.y() - distance;
                    bottom = distance;
                    left = distance;
                    right = map_shape.x() - distance;

                    while (i < right && j == bottom)
                    {
                        if (x == i && y == j && z == k)
                        {
                            i++;
                            continue;
                        }
                        if (i < 0 || j < 0 || k < 0 || i >= map_shape.x() || j >= map_shape.y() || k >= map_shape.z())  // 超出边界
                        {
                            i++;
                            continue;
                        }
                        if (map[i][j][k] == 0)  // 如果ijk这个珊格没有被占据，并且和原飞入点不是同一个，则找到了新的飞入点
                        {
                            if (fly_in_index == Eigen::Vector3i(i, j, k))
                            {
                                i++;
                                continue;
                            }
                            else
                            {
                                fly_in_index = Eigen::Vector3i(i, j, k);
                                return;
                            }
                        }
                        i++;
                    }
                    while (j < top && i == right)
                    {
                        if (x == i && y == j && z == k)
                        {
                            j++;
                            continue;
                        }
                        if (i < 0 || j < 0 || k < 0 || i >= map_shape.x() || j >= map_shape.y() || k >= map_shape.z())
                        {
                            j++;
                            continue;
                        }
                        if (map[i][j][k] == 0)
                        {
                            if (fly_in_index == Eigen::Vector3i(i, j, k))
                            {
                                j++;
                                continue;
                            }
                            else
                            {
                                fly_in_index = Eigen::Vector3i(i, j, k);
                                return;
                            }
                        }
                        j++;
                    }
                    while (i > left && j == top)
                    {
                        if (x == i && y == j && z == k)
                        {
                            i--;
                            continue;
                        }
                        if (i < 0 || j < 0 || k < 0 || i >= map_shape.x() || j >= map_shape.y() || k >= map_shape.z())
                        {
                            i--;
                            continue;
                        }
                        if (map[i][j][k] == 0)
                        {
                            if (fly_in_index == Eigen::Vector3i(i, j, k))
                            {
                                i--;
                                continue;
                            }
                            else
                            {
                                fly_in_index = Eigen::Vector3i(i, j, k);
                                return;
                            }
                        }
                        i--;
                    }
                    while (j > bottom && i == left)
                    {
                        if (x == i && y == j && z == k)
                        {
                            j--;
                            continue;
                        }
                        if (i < 0 || j < 0 || k < 0 || i >= map_shape.x() || j >= map_shape.y() || k >= map_shape.z())
                        {
                            j--;
                            continue;
                        }
                        if (map[i][j][k] == 0)
                        {
                            if (fly_in_index == Eigen::Vector3i(i, j, k))
                            {
                                j--;
                                continue;
                            }
                            else
                            {
                                fly_in_index = Eigen::Vector3i(i, j, k);
                                return;
                            }
                        }
                        j--;
                    }
                    i = distance + 1;
                    j = distance + 1;
                }
                cout << "Not find point in layer:" << k << endl;
                distance = 0;
            }
        }
    }
    
    Eigen::Vector3d target_point_mn_last={0,0,0};
    Eigen::Vector3d get_next_point_mn(bool& astar_finish, bool& flag)//得到下一路径点的全局坐标
    {
        Eigen::Vector3d target_point = now_position_global;

        // std::cout << "path_global size: " << path_global.size() << std::endl;
        if (!path_global.empty())  // 路径点不为空
        {
            // std::cout << "minimum_tra.poses.size: " << minimum_tra.poses.size() << ".  astar_finish: " << astar_finish << std::endl;

            if (minimum_tra.poses.empty() && astar_finish) {
                std::vector<Eigen::Vector3d> vectorNodes(path_global.begin(), path_global.end());
                vectorNodes.insert(vectorNodes.begin(), now_position_global);
                target_point_mn_last = path_global.back();
                planner planner(vectorNodes);
                minimum_tra = planner.tra;
                std::cout << "minimum finish." << std::endl;
            }
            
            
            if (!minimum_tra.poses.empty()) {
                target_point[0] = minimum_tra.poses[0].pose.position.x;
                target_point[1] = minimum_tra.poses[0].pose.position.y;
                target_point[2] = minimum_tra.poses[0].pose.position.z;
                minimum_tra.poses.erase(minimum_tra.poses.begin());

                minimum_tra.header.frame_id = "world";
                minimum_tra.header.stamp = ros::Time::now();
                _path_minimum_pub.publish(minimum_tra);
            }

            // std::cout << "path_global back: " << path_global.back() << std::endl;
            // std::cout << "minimum_tra.poses.size " << minimum_tra.poses.size() << std::endl;
            // std::cout << "***********target_point: " << target_point[0] << " " << target_point[1]  << " " << target_point[2]  << " " << std::endl;

        }
        else {
            astar_finish = false;
            flag = true;
            return target_point;
        }

        if (minimum_tra.poses.empty()) {
            astar_finish = false;
            flag = true;
            return target_point;
            // target_point = now_position_global;
        }
        double dis = std::sqrt((now_position_global[0]-target_point_mn_last[0])*(now_position_global[0]-target_point_mn_last[0]) \
                    + (now_position_global[1]-target_point_mn_last[1])*(now_position_global[1]-target_point_mn_last[1])\
                    + (now_position_global[2]-target_point_mn_last[2])*(now_position_global[2]-target_point_mn_last[2]));

        if(dis < 1) {
            flag = true;
        }

        std::cout << "当前位置距离目标点为：" << dis << std::endl;

        return target_point;
 
    }

    Eigen::Vector3d get_next_point(bool global)//得到下一路径点的全局坐标
    {
        if (!path_global.empty())  // 路径点不为空
        {
            Eigen::Vector3i index = get_index(path_global.front());  // 获取全局坐标下路径列表的下一个值
            if (map[index.x()][index.y()][index.z()] == 0)  // 当下一个坐标没有探索过或者里面没有被占据
            {
                return path_global.front();  // 返回为下一探索位置，由于path_global一直在更新，所以不用弹出已使用的点
            }
            else
            {
                return now_position_global;  // 下一个点已经探索过或者被占据，保持当前位置点
            }
        }
        else  // 全局路径path_global为空
        {
            if (map[now_position_index.x()][now_position_index.y()][now_position_index.z()] == 1 && global)  //当前位置已经探索过且为全局坐标
            {
                return now_position_global;
            }
            else
            {
                if (get_index(path_final_global) == now_position_index)
                {
                    return path_final_global;
                }
                else
                {
                    return get_grid_center_global(now_position_index);
                }
            }
        }
    }

    Eigen::Vector3d get_next_point_my(bool global, bool& flag, string myname = "self")//得到下一路径点的全局坐标
    {
        std::cout << myname << " path_global size: " << path_global.size() << std::endl;
        if (!path_global.empty())  // 路径点不为空
        {
            Eigen::Vector3i index = get_index(path_global.front());  // 获取全局坐标下路径列表的下一个值
            if (map[index.x()][index.y()][index.z()] == 0)  // 当下一个坐标没有探索过或者里面没有被占据
            {
                return path_global.front();  // 返回为下一探索位置，由于path_global一直在更新，所以不用弹出已使用的点
            }
            else
            {
                std::cout << myname << " 不可到达下一个点，返回当前位置 " << std::endl;
                // 如果下一个格子不可达，则
                flag = true;
                
                return now_position_global;  // 下一个点已经探索过或者被占据，保持当前位置点
            }
        }
        else  // 全局路径path_global为空
        {
            if (map[now_position_index.x()][now_position_index.y()][now_position_index.z()] == 1 && global)  //当前位置已经探索过且为全局坐标
            {
                return now_position_global;
            }
            else
            {
                if (get_index(path_final_global) == now_position_index)
                {
                    return path_final_global;
                }
                else
                {
                    return get_grid_center_global(now_position_index);
                }
            }
        }
    }

    // 判断是否到达目标点，输入目标点的索引
    bool reach_target (Eigen::Vector3d target_point_index) {
        double dis = std::sqrt((now_position_global[0]-target_point_index[0])*(now_position_global[0]-target_point_index[0]) \
                    + (now_position_global[1]-target_point_index[1])*(now_position_global[1]-target_point_index[1])\
                    + (now_position_global[2]-target_point_index[2])*(now_position_global[2]-target_point_index[2]));
        if (dis < 1) {
            return true;
        }
        return false;
    }
     
    bool go_fly_in_xy = false;

    Eigen::Vector3d get_next_point(bool global, Eigen::Vector3d now_global_position, int& flag_count, bool reach_flag = false, string myname = "name")  //得到下一路径点的全局坐标
    {
        if (!init_task_id)  // 查找该无人机对应的编号，用于分配空间块
        {
            int i = 0;
            while (i < follower.size())
            {
                if (follower[i] == myname)
                {
                    break;
                }
                i++;
            }
            task_id = i;
            init_task_id = true;
        }
        static int count_view_i = 1;
        if (box_viewpoint_list.empty() && viewpoint_save)
        {
            list<Eigen::Vector3i> path_index_temp;
            path_index_temp = Dijkstra_search_fly_in_xy(0, map_shape.z()-1, myname);
            go_fly_in_xy = true;
            cout<<"Path size:"<<path_index_temp.size()<<endl;
            if ((path_index_temp.empty() || path_index_temp.size() == 1) && viewpoint_save)
            {
                if (mystate != 3)
                {
                    mystate = 2;
                }
                return now_position_global;
            }
            path_index_temp.reverse();
            path_index = path_index_temp;
            generate_the_global_path();
            return path_global.front();  
        }
        if (!box_viewpoint_list.empty())  // 路径点不为空
        {
            Eigen::Vector3i last_viewpoint_index = get_index(last_viewpoint);
            if (!get_first_viewpoint) {
                get_first_viewpoint = true;
                last_viewpoint = box_viewpoint_list.front();
                box_viewpoint_list.pop_front();  // 改：删除头部元素
                std::cout << myname << " 得到第一个视点" << std::endl;
            }
            else if(flag_count > 1 || reach_flag) {  // || std::sqrt((now_position_global[0]-last_viewpoint_index[0])*(now_position_global[0]-last_viewpoint_index[0]) \
                    + (now_position_global[1]-last_viewpoint_index[1])*(now_position_global[1]-last_viewpoint_index[1])\
                    + (now_position_global[2]-last_viewpoint_index[2])*(now_position_global[2]-last_viewpoint_index[2])) < 2) {
                flag_count = 0;
                last_viewpoint = box_viewpoint_list.front();
                box_viewpoint_list.pop_front();
                std::cout << myname <<" =====================>>> 得到下一个视点: " << count_view_i << "还剩下视点数目：" << box_viewpoint_list.size() << std::endl;
                count_view_i++;
                double dis = std::sqrt((now_position_global[0]-last_viewpoint[0])*(now_position_global[0]-last_viewpoint[0]) + (now_position_global[1]-last_viewpoint[1])*(now_position_global[1]-last_viewpoint[1]) + (now_position_global[2]-last_viewpoint[2])*(now_position_global[2]-last_viewpoint[2]));
                std::cout << myname << " 当前位置为：" << now_position_global[0] << ", " << now_position_global[1] << ", "<< now_position_global[2] << ", "<< std::endl;
                std::cout << myname << " 目标视点位置为：" << last_viewpoint[0] << ", " << last_viewpoint[1] << ", "<< last_viewpoint[2] << ", "<< std::endl;
                std::cout << myname << " 当前位置距离目标点为：" << dis << std::endl;
            }
            return last_viewpoint;
        }
        else  // 全局路径path_global为空
        {
            if (map[now_position_index.x()][now_position_index.y()][now_position_index.z()] == 1 && global)  //当前位置已经探索过且为全局坐标
            {
                return now_position_global;
            }
            else
            {
                if (get_index(path_final_global) == now_position_index)
                {
                    return path_final_global;
                }
                else
                {
                    return get_grid_center_global(now_position_index);
                }
            }
        }
    }

    nav_msgs::Path get_path_show()//返回当前路径展示信息
    {
        return path_global_show_message;
    }
    
    void set_state(int a)  //将当前状态置为a，当a＝１时，代表飞入完成
    {
        mystate = a;
    }
    
    int get_state()//获取当前的状态信息
    {
        return mystate;
    }
    
    int get_state_leader()//获取当前领导者的状态：如果领导者本身的状态已经是 3，它会直接返回 3，否则，它会检查追随者的状态，如果所有追随者的状态都是 2，那么领导者的状态将被设置为 3，否则保持不变为 2
    {
        bool flag_state = true;
        if (mystate == 3)
        {
            return mystate;
        }
        for (auto &name : follower)
        {
            if (local_dict[name].state != 2)
            {
                flag_state = false;
                return mystate;
            }
        }
        if (flag_state && mystate == 2)
        {
            mystate = 3;
        }
        return mystate;
    }
    
    bool get_whether_pop()//返回为false为不弹出状态，返回为ture表示为弹出状态，
    {
        if (mystate == 3)
        {
            // cout<<"mystate:"<<mystate<<endl;
            // return true;
        }
        if (mystate != 3)  //状态不为3，不弹出状态
        {
            return false;
        }
        else  //状态为3，检查所有follower的local_dic状态，当有任何一个不为0，不弹出状态
        { 
            for (auto &name : follower)
            {
                if (local_dict[name].state != 0)
                {
                    return false;
                }
            }
            if(follower.size()==0&&mystate!=3){//当没有追随者且当前状态不为3时，不弹出
                return false;
            }
        }
        return true;
    }
    
    void exploration_layer_source(string myname, int region_index)//探索无人机更新path_index
    {
        list<Eigen::Vector3i> path_index_temp;
        if (is_not_empty())  // 当height为非空时，所在区域内搜索路径
        {
            path_index_temp = Dijkstra_search_2D_with_3D(height, region_slice_layer[region_index], myname);  // 比如说第一层是0到50米
        }
        else//当搜索区域为空时，搜索边界
        {
            path_index_temp = Dijkstra_search_edge(height, region_slice_layer[region_index], myname);
        }
        if (path_index_temp.empty())//当没有搜索到路径或者边界时
        {
            finish_exp_flag[region_index] = 1;//标记完成搜索标志位
            if (height < region_slice_layer[region_index])  //更新高度，最低处绕一圈，最高处绕一圈
            {
                height = region_slice_layer[region_index];
            }
            else
            {
                // 探索完成后，需要和拍照无人机进行通信，告知可以去拍照了，所以在当前的这个层上飞到和飞入点同样的xy坐标上，
                if (local_dict[follower[region_index]].state != 1)  // 如果这个区域的拍照无人机的状态不是1，就是完成了拍照或者处于等待状态
                {
                    finish_exp_flag[region_index] = 1;
                    path_index_temp = Dijkstra_search_fly_in_xy(interval * (region_index - 1), height, myname);
                    if (path_index_temp.empty() || path_index_temp.size() == 1)
                    {
                        path_index_temp = Dijkstra_search_edge(height, region_slice_layer[region_index], myname);
                        if (path_index_temp.empty())
                        {
                            height++;
                        }
                    }
                }
                else
                {
                    finish_flag[region_index] = 1;
                }
            }
        }
        path_index_temp.reverse();
        path_index = path_index_temp;
        generate_the_global_path();
    }

    // 改，用于探索
    void exploration_layer(string myname, int region_index, int high)  //探索无人机更新path_index
    {
        list<Eigen::Vector3i> path_index_temp;
        if (is_not_empty())  // 当height为非空时，所在区域内搜索路径
        {
            path_index_temp = Dijkstra_search_2D_with_3D(height, high, myname);  // 比如说第一层是0到50米
        }
        else//当搜索区域为空时，搜索边界
        {
            path_index_temp = Dijkstra_search_edge(height, high, myname);
        }
        std::cout << myname << " path_index_temp.size()---: " << path_index_temp.size() << std::endl;
        if (path_index_temp.empty())//当没有搜索到路径或者边界时
        {
            // finish_exp_flag[region_index] = 1;//标记完成搜索标志位
            // finish_id_bool = true;
            // if (height < high)  //更新高度，最低处绕一圈，最高处绕一圈
            // {
            //     height = high;
            //     // return;
            // }
            // else
            // {
            for (int i = 0; i < finish_exp_flag.size(); i++) {
                finish_exp_flag[i] = 1;
            }
            std::cout <<  myname << " path_index_temp.empty()======================height:" << height << std::endl;
            finish_id_bool = true;
            bool communication_success = true;
            for (int i = 0; i < follower.size(); i++) {
                if (local_dict[follower[i]].state != 1) {
                    communication_success = false;
                }
            }

            // 探索完成后，需要和拍照无人机进行通信，告知可以去拍照了，所以在当前的这个层上飞到和飞入点同样的xy坐标上，
            if (!communication_success)  // && local_dict[follower[region_index]].state != 2)  // 如果这个区域的拍照无人机的状态不是1，就是完成了拍照或者处于等待状态
            {
                // std::cout << "state != 1 ----------------------------------------------" << std::endl;
                // std::cout << "chanqi state: " << local_dict["/changi"].state << std::endl;
                // finish_exp_flag[region_index] = 1;
                // finish_id_bool = true;

                path_index_temp = Dijkstra_search_fly_in_xy(0, map_shape.z() - 1, myname);  // interval * (region_index - 1)
                if (path_index_temp.empty() || path_index_temp.size() == 1)
                {
                    path_index_temp = Dijkstra_search_edge(height, map_shape.z() - 1, myname);
                    if (path_index_temp.empty())
                    {
                        height++;
                    }
                }
            }
            else
            {
                box_all_finish_exp = true;
                std::cout <<  myname << " 排遣成功==========" << std::endl;
                finish_id_bool = true;
                for (int i = 0; i < finish_flag.size(); i++) {
                    finish_flag[i] = 1;
                }
                
                finish_flag_i++;
            }
        }
        path_index_temp.reverse();
        path_index = path_index_temp;
        generate_the_global_path();
    }

    void exploration_layer_last(string myname, int region_index, int high)  //探索无人机更新path_index
    {
        list<Eigen::Vector3i> path_index_temp;
        if (is_not_empty())  // 当height为非空时，所在区域内搜索路径
        {
            path_index_temp = Dijkstra_search_2D_with_3D(height, high, myname);  // 比如说第一层是0到50米
        }
        else//当搜索区域为空时，搜索边界
        {
            path_index_temp = Dijkstra_search_edge(height, high, myname);
        }
        if (path_index_temp.empty())//当没有搜索到路径或者边界时
        {
            // finish_exp_flag[region_index] = 1;//标记完成搜索标志位
            
            if (height < high)  //更新高度，最低处绕一圈，最高处绕一圈
            {
                height = high;
                // return;
            }
        }

        path_index_temp.reverse();
        path_index = path_index_temp;
        generate_the_global_path(); 
        
        if (path_index_temp.empty() && height >= map_shape.z()-1)  // 拍完了自己的任务
        {
            // std::cout << "-=-=-=-==-=-=-=-=-=-==-=" << local_dict["/chanqi"].state << std::endl;
            path_index_temp = Dijkstra_search_fly_in_xy(height / 2, high, myname);
            // cout<<"-=-=-= Path size:"<<path_index_temp.size()<<endl;
            if ((path_index_temp.empty() || path_index_temp.size() == 1))  // && finish_flag_leader)
            {
                box_all_finish_exp = true;
                for (int i = 0; i < finish_exp_flag.size(); i++) {
                    finish_exp_flag[i] = 1;
                }
            }
            path_index_temp.reverse();
            path_index = path_index_temp;
            generate_the_global_path();
            return;
        }
    }

    void take_photo_explorer(string myname) {
        if (myname == "/raffles" || myname == "/jurong")
        {
            if (finish_flag_leader)
            {

            }
            else
            {
                bool tamp = true;
                if(follower.size()==0){
                    tamp=false;
                }
                for (auto &name : follower)
                {
                    if (local_dict[name].state != 2)
                    {
                        tamp = false;
                        break;
                    }
                }
                finish_flag_leader = tamp;
            }
        }
        else
        {
            finish_flag_leader = false;
        }
        // list<Eigen::Vector3i> path_index_temp;
        // if (box_viewpoint_list.empty()) {
        //     path_index_temp = Dijkstra_search_fly_in_xy(height / 2, high, myname);

        //     path_index_temp.reverse();
        //     path_index = path_index_temp;
        //     generate_the_global_path();
        // }

        if (finish_flag_leader && box_viewpoint_list.empty() && viewpoint_save)
        {
            if (mystate != 3)
            {
                mystate = 2;
            }
        }
    }
    
    bool box_all_finish_exp = false;

    bool get_box_all_finish_exp() {
        return box_all_finish_exp;
    }

    void exploration(string myname)  // 检查是否探索完成，探索完开始调用拍照函数
    {
        // 这里作为判断当前边界框已经探索完
        // explor_finish = true;
        if (!box_all_finish_exp) {
            if (finish_flag_i == 0) {
                height = map_shape.z() / 2;
                exploration_layer(myname, 0, map_shape.z() / 2 + 5);
                return;
            }
            // exploration_layer_last(myname, 0, map_shape.z() - 1);
        }
        else {
            take_photo_explorer(myname);
        }
        // take_photo(myname);  //前n-1个区域已经探索完，则该探索无人机也作为一个普通拍照无人机去拍照
    }

    bool update_finish_id() {
        return finish_id_bool;
    }
    
    void update_viewpoint_box(vector<Eigen::Vector3d>& box_vp_tem) {  // 从视点节点中获取视点信息（探索者调用）
        // test_int = 1000000;
        if (viewpoint_save || box_vp_tem.empty())
            return;
        box_viewpoint_list_exp = list<Eigen::Vector3d>(box_vp_tem.begin(), box_vp_tem.end());  // box_vp_tem.end());  // box_vp_tem.begin() + std::min(10, static_cast<int>(box_vp_tem.size())));
        int nums = box_viewpoint_list_exp.size() / team_size;
        auto lastElements = std::prev(box_viewpoint_list_exp.end(), nums);
        box_viewpoint_list.insert(box_viewpoint_list.end(), lastElements, box_viewpoint_list_exp.end());
        viewpoint_save = true;
        std::cout << "update_viewpoint()成功：" << box_viewpoint_list_exp.size() << std::endl;
    }

    void update_viewpoint_box_from_topic(vector<Eigen::Vector3d>& box_vp_tem, std::string myname) {  // 从探索者无人机中接受视点信息
        if (!init_task_id)  // 查找该无人机对应的编号，用于分配空间块
        {
            int i = 0;
            while (i < follower.size())
            {
                if (follower[i] == myname)
                {
                    break;
                }
                i++;
            }
            task_id = i;
            init_task_id = true;
        }
        if (viewpoint_save || box_vp_tem.empty() || !init_task_id)
            return;
        int viewnums = box_vp_tem.size() / team_size;
        int begin = box_vp_tem.size() / team_size * task_id;
        auto startIter = box_vp_tem.begin() + begin;
        auto endIter = box_vp_tem.begin() + begin + viewnums;
        // box_viewpoint_list = list<Eigen::Vector3d>(box_vp_tem.begin(), box_vp_tem.end());  // box_vp_tem.end());  // box_vp_tem.begin() + std::min(10, static_cast<int>(box_vp_tem.size())));
        box_viewpoint_list.insert(box_viewpoint_list.end(), startIter, endIter);
        viewpoint_save = true;
        std::cout << "photo uav update_viewpoint()成功：" << box_viewpoint_list.size() << std::endl;
    }

    list<Eigen::Vector3d> get_viewpoint_vec() {
        return box_viewpoint_list_exp;
    }

    void take_photo_viewpoint(string myname) {
        int low = 0;
        int high = map_shape.z()-1;
        list<Eigen::Vector3i> path_index_temp;
        if (height < low + 1)
        {
            height = low + 1;  // 根据分块的最低点设定当前最低高度
        }
        finish_flag_leader = false;
        std::cout << "box_viewpoint_list size: " << box_viewpoint_list.size() << std::endl;
        if (box_viewpoint_list.empty() || finish_flag_leader)  // 拍完了自己的任务
        {
            path_index_temp = Dijkstra_search_fly_in_xy(low + 1, high - 1, myname);
            // cout<<"Path size:"<<path_index_temp.size()<<endl;
            if (path_index_temp.empty() || path_index_temp.size() == 1 || finish_flag_leader)
            {
                if (mystate != 3)
                {
                    mystate = 2;
                }
            }
            path_index_temp.reverse();
            path_index = path_index_temp;
            generate_the_global_path();
            return;
        }
        
        list<Eigen::Vector3d> point_global_list_tamp;//创建临时全局坐标系下的列表
        if (box_viewpoint_list.empty())//列表为空时，清空全局路径
        {
            path_global.clear();
            return;
        }
        nav_msgs::Path global_path_tamp;//定义nav_msgs类型的PATH，用于可视化

        // path_tamp.pop_back();//在列表中移除最后一个元素

        // path_tamp.pop_back();
        global_path_tamp.header.frame_id = "world";//定义global_path_tamp消息基准坐标系
        for (auto point_current : box_viewpoint_list)
        {
            // Eigen::Vector3d point_current = box_viewpoint_list.front();  // 从珊格索引获取其中心的实际坐标
            if (Developing)//将路径信息传给global_path_tamp
            {
                geometry_msgs::PoseStamped pose;
                pose.header.frame_id = "world";
                pose.header.stamp = ros::Time::now();
                pose.pose.position.x = point_current.x();
                pose.pose.position.y = point_current.y();
                pose.pose.position.z = point_current.z();
                pose.pose.orientation.w = 1.0;
                global_path_tamp.poses.push_back(pose);
            }
            point_global_list_tamp.push_back(point_current);
            // path_tamp.pop_back();
        }
        // point_global_list_tamp.push_back(path_final_global);
        // path_global_show_message = global_path_tamp;
        path_global = point_global_list_tamp;  // 最终结果，即将珊格索引转为坐标
        return;
    }

    void take_photo(string myname)//调用拍照函数更新拍照无人机的路径
    {
        if (!init_task_id)  // 查找该无人机对应的编号，用于分配空间块
        {
            int i = 0;
            while (i < follower.size())
            {
                if (follower[i] == myname)
                {
                    break;
                }
                i++;
            }
            task_id = i;
        }
        if(follower.size()==0){  // 没有跟随者，该架无人机探索整个边界框
            //yolo();
            take_photo_layer(0, map_shape.z()-1, myname);
            //yolo();
            return;
        }


        if (task_id == 0)  // 该无人机上跟随者，且编号为0
        {
            take_photo_layer(0, region_slice_layer[0], myname);
            return;
        }
        else if (task_id == follower.size())  // task_id等于跟随者的数量时，代表该组的探索者
        {
            take_photo_layer(region_slice_layer[task_id - 1], map_shape.z() - 1, myname);
        }
        else
        {
            take_photo_layer(region_slice_layer[task_id - 1], region_slice_layer[task_id], myname);
        }
    }
    
    void take_photo_layer(int low, int high, string myname)//更新拍照无人机的路径
    {
        list<Eigen::Vector3i> path_index_temp;
        if (height < low + 1)
        {
            height = low + 1;  // 根据分块的最低点设定当前最低高度
        }
        if (myname == "/raffles" || myname == "/jurong")
        {
            if (finish_flag_leader)
            {

            }
            else
            {
                bool tamp = true;
                if(follower.size()==0){
                    tamp=false;
                }
                for (auto &name : follower)
                {
                    if (local_dict[name].state != 2)
                    {
                        tamp = false;
                        break;
                    }
                }
                finish_flag_leader = tamp;
            }
        }
        else
        {
            finish_flag_leader = false;
        }

        if (height >= high || finish_flag_leader)  // 拍完了自己的任务
        {
            path_index_temp = Dijkstra_search_fly_in_xy(low + 1, high - 1, myname);
            // cout<<"Path size:"<<path_index_temp.size()<<endl;
            if (path_index_temp.empty() || path_index_temp.size() == 1 || finish_flag_leader)
            {
                if (mystate != 3)
                {
                    mystate = 2;
                }
            }
            path_index_temp.reverse();
            path_index = path_index_temp;
            generate_the_global_path();
            return;
        }

        path_index_temp = Dijkstra_search_2D_with_3D(height, high - 1, myname);
        if (path_index_temp.empty())
        {
            if (myname == "/raffles" || myname == "/jurong")
            {
                height += 2;  // 从4改为2，减小拍照时一次移动的距离
            }
            else
            {
                height += 2;  // 从4改为2，减小拍照时一次移动的距离
            }
        }
        path_index_temp.reverse();
        path_index = path_index_temp;
        generate_the_global_path();
        return;
    }

    bool is_not_empty()//在height高度检查所有平面位置是否占用，当有一个是非空返回ture
    {
        for (int x = 0; x < map_shape.x(); x++)
        {
            for (int y = 0; y < map_shape.y(); y++)
            {
                if (map[x][y][height] == 1)
                {
                    return true;
                }
            }
        }
        return false;
    }

    void update_gimbal(Eigen::Vector3d direction_global, bool print)  // 更新云台方向
    {
        if (print)  // 是否打印信息
        {
            cout << "direction_global:" << direction_global.transpose() << endl;
            // cout<<"matrix:"<<endl;
            // cout<<Rpy2Rot(direction_global)<<endl;
            if (!search_direction.empty())
            {
                cout << "target:" << search_direction.front().transpose() << endl;
                // cout<<"matrix:"<<endl;
                // cout<<Rpy2Rot(search_direction.front())<<endl;
            }
        }
        if (search_direction.empty())  // 路径搜索方向列表为空直接返回
        {
            return;
        }
        else if ((direction_global - search_direction.front()).norm() < 0.30)//全局方向向量和列表第一个元素之间的欧式距离小于0.3，即到达了指定方向，从列表中移除该目标方向
        {
            search_direction.pop_front();
            // cout<<"pop front search"<<endl;// test
        }
    }

    void insert_cloud_from_str(istringstream &msg)//读取点云并插入点云信息
    {
        string number_of_map;
        getline(msg, number_of_map, ';');
        int number_map = stoi(number_of_map);

        while (occupied_num < number_map)
        {
            string index_occupied;
            getline(msg, index_occupied, ';');

            Eigen::Vector3i index_occ_tamp;
            if (str2point3i(index_occupied, index_occ_tamp))
            {
                insert_map_index(index_occ_tamp);
            }
        }
    }

    void get_gimbal_rpy(Eigen::Vector3d &result)//获取云台姿态信息，当search_direction列表为空，它不执行任何操作，否则，它将获取 search_direction 列表的第一个元素作为云台的姿态信息
    {
        list<Eigen::Vector3d> search_temp = search_direction;  // 云台探索方向

        if (search_temp.empty())
        {
        }
        else
        {
            result = search_temp.front();
        }
    }
    bool get_mission_finished()//返回任务完成情况
    {
        return is_finished;
    }
    string get_num_str()//将整数值 occupied_num 转换为字符串，并在字符串末尾添加一个分号并返回
    {
        string result;
        result = to_string(occupied_num) + ";";
        return result;
    }
    string get_map_str()//返回点云地图信息
    {
        return map_cloud_massage;
    }
    void set_fly_in_index(string tar)//从传入的tar中提取坐标信息，并赋值给fly_in_index
    {
        try
        {
            Eigen::Vector3i vec_fly;
            if (str2point3i(tar, vec_fly))
            {
                fly_in_index = vec_fly;
            }
        }
        catch (const std::invalid_argument &e)
        {
            cout << "Invalid argument" << e.what() << endl;
            return;
        }
        catch (const std::out_of_range &e)
        {
            cout << "Out of range" << e.what() << endl;
            return;
        }
    }
    string get_fly_in_str()//将坐标信息转为字符串并返回
    {
        return (point3i2str(fly_in_index) + ";");
    }
    void update_local_dict(istringstream &str)//通过输入的字符串str提取无人机信息，更新无人机本地信息
    {
        string name;
        getline(str, name, ';');
        if (name != "/jurong" && name != "/raffles" && name != "/changi" && name != "/sentosa" && name != "/nanyang")
        {
            return;
        }
        else
        {
            string position_str;
            getline(str, position_str, ';');
            agent_local info_temp;
            // cout<<"position_str:"<<position_str<<endl;
            Eigen::Vector3d global_nbr_position_point = str2point(position_str);
            if (!out_of_range_global(global_nbr_position_point, false))
            {
                info_temp.in_bounding_box = true;
                info_temp.position_index = get_index(global_nbr_position_point);
            }
            string path_point;
            getline(str, path_point, ';');
            // cout<<"path_point:"<<path_point<<endl;
            Eigen::Vector3d next_nbr_path_po = str2point(path_point);
            if (!out_of_range_global(next_nbr_path_po, false))
            {
                info_temp.planning_in_bounding_box = true;
                info_temp.planning_index = get_index(next_nbr_path_po);
            }

            info_temp.time = ros::Time::now().toSec();
            info_temp.state = local_dict[name].state;
            info_temp.priority = local_dict[name].priority;
            local_dict[name] = info_temp;
            
        }
    }
    void update_state(string name, int state_in)//将无人机状态更新为state_in
    {
        local_dict[name].state = state_in;
    }
    list<string> get_state_string_list()//获取状态字符串列表信息
    {
        list<string> result;
        for (int i = 0; i < finish_exp_flag.size(); i++)
        {
            if (finish_exp_flag[i] == 1 && finish_flag[i] == 0)
            {
                string str_state_set = follower[i] + ";1;";
                // cout<<"str_state_set:"<<str_state_set<<"-=--=-=--==-=-=-=-=-=-=-" << std::endl;
                result.push_back(str_state_set);
            }
        }
        return result;
    }

    // Function whether a local point is out of range. for pcd
    bool out_of_range_box(Eigen::Vector3d point_in, bool out_put)
    {
        Eigen::Vector3d point=rotation_matrix*(point_in-map_global_center);
        if (fabs(point.x()) > (fabs(map_shape.x() / 2) + 0.5) * grid_size.x() || fabs(point.y()) > (fabs(map_shape.y() / 2) + 0.5) * grid_size.y() || fabs(point.z()) > (fabs(map_shape.z() / 2) + 0.5) * grid_size.z())
        {
            if (out_put)
            {
                cout << "xbool:" << (fabs(point.x()) > fabs(map_shape.x() / 2) * grid_size.x() ? "yes" : "no") << endl;
                cout << "xlim:" << fabs(map_shape.x() / 2) * grid_size.x() << endl;
                cout << "ylim:" << fabs(map_shape.y() / 2) * grid_size.y() << endl;
                cout << "grid size:" << grid_size.transpose() << endl;
                cout << "map_shape:" << map_shape.transpose() << endl;
                cout << "center:" << map_index_center.transpose() << endl;
                cout << "out range point" << point.transpose() << endl;
                cout << "Desired point" << (rotation_matrix * (get_grid_center_global(Eigen::Vector3i(0, 0, 0)) - map_global_center)).transpose() << endl;
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    // 判断视点范围，将边界框扩大
    bool out_of_range_box_viewpoint(Eigen::Vector3d point_in, bool out_put)
    {
        Eigen::Vector3d point=rotation_matrix*(point_in-map_global_center);
        if (fabs(point.x()) > (fabs(map_shape.x() / 2) + 0.5) * grid_size.x() || fabs(point.y()) > (fabs(map_shape.y() / 2) + 0.5) * grid_size.y() || fabs(point.z()) > (fabs(map_shape.z() / 2)) * grid_size.z())
        {
            if (out_put)
            {
                cout << "xbool:" << (fabs(point.x()) > fabs(map_shape.x() / 2) * grid_size.x() ? "yes" : "no") << endl;
                cout << "xlim:" << fabs(map_shape.x() / 2) * grid_size.x() << endl;
                cout << "ylim:" << fabs(map_shape.y() / 2) * grid_size.y() << endl;
                cout << "grid size:" << grid_size.transpose() << endl;
                cout << "map_shape:" << map_shape.transpose() << endl;
                cout << "center:" << map_index_center.transpose() << endl;
                cout << "out range point" << point.transpose() << endl;
                cout << "Desired point" << (rotation_matrix * (get_grid_center_global(Eigen::Vector3i(0, 0, 0)) - map_global_center)).transpose() << endl;
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    // Function to get the grid center point in global 将局部坐标系中的网格中心点转换为全局坐标系中的坐标
    Eigen::Vector3d get_grid_center_global(Eigen::Vector3i grid_index)
    {
        Eigen::Vector3d bias = (grid_index - map_index_center).cast<double>();
        Eigen::Vector3d local_result = bias.cwiseProduct(grid_size);
        Eigen::Vector3d global_result = rotation_matrix_inv * local_result + map_global_center;
        return global_result;
    }

private:
    nav_msgs::Path minimum_tra;

    list<Eigen::Vector3d> box_viewpoint_list;
    list<Eigen::Vector3d> box_viewpoint_list_exp;
    bool get_first_viewpoint = false;
    Eigen::Vector3d last_viewpoint;
    bool viewpoint_save = false;
    int test_int = 0;
    bool finish_id_bool = false;

    // communication part
    list<string> namelist;
    std::map<string, agent_local> local_dict;
    int mystate = 0;
    //
    int team_size;
    bool is_finished = false;
    AStar astar_planner;
    double time_start=0;

    bool init_task_id = false;
    int task_id = 0;
    string map_cloud_massage;
    int occupied_num = 0;
    int exploration_state = 0;
    vector<vector<vector<int>>> map;
    vector<vector<vector<int>>> interest_map;  // 如果某个点的四周有被占用的网格，并且该点没有被访问过，则设置为感兴趣的点
    vector<vector<vector<int>>> visited_map;
    Eigen::Vector3d grid_size;  // 每个网格在三个坐标轴上的尺寸
    Eigen::Matrix3d rotation_matrix;
    Eigen::Matrix3d rotation_matrix_inv;  // 旋转矩阵的逆，如果是边界框子地图，即Rbw，b为边界框，w为世界坐标系
    Eigen::Quaterniond rotation_quat;
    Eigen::Vector3d map_global_center;  // 地图中心点在世界坐标系下的坐标
    Eigen::Vector3i map_shape;  // 地图形状，三个坐标轴上的网格数
    Eigen::Vector3i map_index_center;  // 三维珊格地图的中心索引坐标（网格单位）
    Eigen::Vector3d map_quat_size;
    Eigen::Vector3d now_position_global;  // 无人机当前全局位置
    Eigen::Vector3d now_position_local;  // 无人机当前局部位置
    Eigen::Vector3i now_position_index;  // 无人机在地图中的索引
    Eigen::Vector3i fly_in_index;
    Eigen::Vector3d path_final_global;
    list<Eigen::Vector3d> path_global;  // 全局路径的坐标集合
    list<Eigen::Vector3i> path_index;
    list<Eigen::Vector3d> search_direction;  // 用于控制云台转向
    vector<int> region_slice_layer;  //存储与切分区域有关的信息，每个团队成员都有一个切分区域
    vector<int> finish_flag;
    int finish_flag_i = 0;
    vector<int> finish_exp_flag;
    vector<string> follower;
    bool Developing = true;
    bool in_my_range = false;
    nav_msgs::Path path_global_show_message;
    int height = 0;
    int interval = 0;
    bool finish_flag_leader = false;

    void initial_the_convert()
    {
        int x_lim;  // 一半长度占据的珊格数
        int y_lim;
        int z_lim;
        // 猜测：从边界框中心点出发，先减掉半个珊格，再计算剩下的长度有几个完整珊格，最后乘于2倍加上中间的1格得到一个边的完整珊格数目
        if (map_quat_size.x() < 0.5 * grid_size.x())
        {
            x_lim = 0;
        }
        else
        {
            x_lim = floor((map_quat_size.x() - 0.5 * grid_size.x()) / grid_size.x()) + 1;
        }
        if (map_quat_size.y() < 0.5 * grid_size.y())
        {
            y_lim = 0;
        }
        else
        {
            y_lim = floor((map_quat_size.y() - 0.5 * grid_size.y()) / grid_size.y()) + 1;
        }
        if (map_quat_size.z() < 0.5 * grid_size.z())
        {
            z_lim = 0;
        }
        else
        {
            z_lim = floor((map_quat_size.z() - 0.5 * grid_size.z()) / grid_size.z()) + 1;
        }
        x_lim = x_lim + 1;
        y_lim = y_lim + 1;
        z_lim = z_lim + 1;
        cout << "x lim:" << x_lim << endl;
        cout << "y lim:" << y_lim << endl;
        cout << "z lim:" << z_lim << endl;
        map_shape = Eigen::Vector3i(2 * x_lim + 1, 2 * y_lim + 1, 2 * z_lim + 1);  // 地图珊格数，这里+1是因为前面计算的时候减去了中间的一块
        cout << "Map shape:" << map_shape.transpose() << endl;
        map_index_center = Eigen::Vector3i(x_lim, y_lim, z_lim);  // 地图的中心
        cout << "Map Center Index:" << map_index_center.transpose() << endl;
        map = vector<vector<vector<int>>>(map_shape.x(), vector<vector<int>>(map_shape.y(), vector<int>(map_shape.z(), 0)));  // 三维珊格的整体尺寸
        interest_map = vector<vector<vector<int>>>(map_shape.x(), vector<vector<int>>(map_shape.y(), vector<int>(map_shape.z(), 0)));  // 感兴趣的地图？？
        visited_map = vector<vector<vector<int>>>(map_shape.x(), vector<vector<int>>(map_shape.y(), vector<int>(map_shape.z(), 0)));  // 已访问过的地图
        astar_planner = AStar(map, map_shape);  // A*初始化
    }
    void set_under_ground_occupied()  // 遍历地图中的每个网格，根据其中心点的 z 坐标来判断是否位于地下，并将相应的地图位置标记为被占用，可能用于后续的轨迹规划与避障
    {
        for (int x = 0; x < map_shape.x(); x++)
        {
            for (int y = 0; y < map_shape.y(); y++)
            {
                for (int z = 0; z < map_shape.z(); z++)
                {
                    Eigen::Vector3d grid_center_global = get_grid_center_global(Eigen::Vector3i(x, y, z));
                    if (grid_center_global.z() < 0.5 * grid_size.z())  // 判断是否除于最下方
                    {
                        map[x][y][z] = 1;
                    }
                }
            }
        }
    }
    // Function whether a local point is out of range
    bool out_of_range(Eigen::Vector3d point, bool out_put)  //用于检查是否超过世界坐标系的范围，超过世界坐标范围返回ture
    {
        if (fabs(point.x()) > (fabs(map_shape.x() / 2) + 0.5) * grid_size.x() || fabs(point.y()) > (fabs(map_shape.y() / 2) + 0.5) * grid_size.y() || fabs(point.z()) > (fabs(map_shape.z() / 2) + 0.5) * grid_size.z())
        {
            if (out_put)//是否要打印信息
            {
                cout << "xbool:" << (fabs(point.x()) > fabs(map_shape.x() / 2) * grid_size.x() ? "yes" : "no") << endl;
                cout << "xlim:" << fabs(map_shape.x() / 2) * grid_size.x() << endl;
                cout << "ylim:" << fabs(map_shape.y() / 2) * grid_size.y() << endl;
                cout << "grid size:" << grid_size.transpose() << endl;
                cout << "map_shape:" << map_shape.transpose() << endl;
                cout << "center:" << map_index_center.transpose() << endl;
                cout << "out range point" << point.transpose() << endl;
                cout << "Desired point" << (rotation_matrix * (get_grid_center_global(Eigen::Vector3i(0, 0, 0)) - map_global_center)).transpose() << endl;
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    // Function whether a local point is out of range
    bool out_of_range_global(Eigen::Vector3d point_in, bool out_put)
    {
        Eigen::Vector3d point=rotation_matrix*(point_in-map_global_center);
        if (fabs(point.x()) > (fabs(map_shape.x() / 2) + 0.5) * grid_size.x() || fabs(point.y()) > (fabs(map_shape.y() / 2) + 0.5) * grid_size.y() || fabs(point.z()) > (fabs(map_shape.z() / 2) + 0.5) * grid_size.z())
        {
            if (out_put)
            {
                cout << "xbool:" << (fabs(point.x()) > fabs(map_shape.x() / 2) * grid_size.x() ? "yes" : "no") << endl;
                cout << "xlim:" << fabs(map_shape.x() / 2) * grid_size.x() << endl;
                cout << "ylim:" << fabs(map_shape.y() / 2) * grid_size.y() << endl;
                cout << "grid size:" << grid_size.transpose() << endl;
                cout << "map_shape:" << map_shape.transpose() << endl;
                cout << "center:" << map_index_center.transpose() << endl;
                cout << "out range point" << point.transpose() << endl;
                cout << "Desired point" << (rotation_matrix * (get_grid_center_global(Eigen::Vector3i(0, 0, 0)) - map_global_center)).transpose() << endl;
            }
            return true;
        }
        else
        {
            return false;
        }
    }

    // Function whether a local point_index is out of range
    bool out_of_range_index(Eigen::Vector3i point)
    {
        if (point.x() >= 0 && point.x() < map_shape.x() && point.y() >= 0 && point.y() < map_shape.y() && point.z() >= 0 && point.z() < map_shape.z())
        {
            return false;
        }
        else
        {
            return true;
        }
    }
    // Function whether a local point_index is out of range and whether the z in limitation
    bool out_of_range_index(Eigen::Vector3i point, int top_z, int bottom_z)
    {
        if (top_z <= bottom_z)
        {
            cout << "Error use function" << endl;
        }
        if (point.x() >= 0 && point.x() < map_shape.x() && point.y() >= 0 && point.y() < map_shape.y() && point.z() > bottom_z && point.z() < top_z && point.z() >= 0 && point.z() < map_shape.z())
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    // Function to generate map marker
    visualization_msgs::Marker generate_marker(Eigen::Vector3i index, int type, int id)
    {
        // type- 0:occupied 1:interest
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.ns = "cube_marker_array";
        marker.id = id;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        Eigen::Vector3d grid_center = get_grid_center_global(index);
        marker.pose.position.x = grid_center.x();
        marker.pose.position.y = grid_center.y();
        marker.pose.position.z = grid_center.z();
        marker.pose.orientation.x = rotation_quat.x();
        marker.pose.orientation.y = rotation_quat.y();
        marker.pose.orientation.z = rotation_quat.z();
        marker.pose.orientation.w = rotation_quat.w();
        marker.scale.x = grid_size.x();
        marker.scale.y = grid_size.y();
        marker.scale.z = grid_size.z();
        if (type == 0 && grid_center.z() > 0)
        {
            marker.color.a = 0.5; // Don't forget to set the alpha!
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
        }
        else if (type == 1 && grid_center.z() > 0)
        {
            marker.color.a = 0.5; // Don't forget to set the alpha!
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
        }
        else
        {
            marker.color.a = 0.0; // Don't forget to set the alpha!
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
        }
        return marker;
    }
    // Function to generate 2D layer search path with 3D Dijkstra，从三维空间里面遍历layer平面上的所有感兴趣点
    list<Eigen::Vector3i> Dijkstra_search_2D_with_3D(int layer, int upper, string myname)
    {
        vector<vector<vector<int>>> grid = map;  // 复制一份网格地图
        for (auto &name : namelist)  // 将其它无人机所在的位置和将要去的位置置1
        {
            if (myname == name)
            {
                continue;
            }
            else
            {
                if (fabs(ros::Time::now().toSec() - local_dict[name].time) < 1||true)
                {
                    if (local_dict[name].in_bounding_box)
                    {
                        Eigen::Vector3i tar = local_dict[name].position_index;
                        // cout<<"position"<<tar.transpose()<<endl;
                        grid[tar.x()][tar.y()][tar.z()] = 1;
                    }
                    if (local_dict[name].planning_in_bounding_box)
                    {
                        Eigen::Vector3i tar = local_dict[name].planning_index;
                        // cout<<"motion"<<tar.transpose()<<endl;
                        grid[tar.x()][tar.y()][tar.z()] = 1;
                    }
                }
            }
        }
        Eigen::Vector3i start = now_position_index;  // 当前所在位置为起始点
        // 探索方向
        vector<Vector3i> directions = {Vector3i(0, 1, 0), Vector3i(0, -1, 0), Vector3i(1, 0, 0), Vector3i(-1, 0, 0), Vector3i(0, 0, 1), Vector3i(0, 0, -1)};

        // Initialize the queue and visited flag.初始化队列和访问标记向量
        queue<list<Eigen::Vector3i>> q;
        q.push({start});
        vector<vector<vector<bool>>> visited(map_shape.x(), vector<vector<bool>>(map_shape.y(), vector<bool>(map_shape.z(), false)));
        visited[start.x()][start.y()][start.z()] = true;

        while (!q.empty())
        {
            list<Eigen::Vector3i> path = q.front();
            q.pop();

            Eigen::Vector3i curr = path.back();
            // 循环退出条件是：路径的最后一个网格的索引值是感兴趣的，并且没有被访问过，高度在当前layer(最低)
            if (interest_map[curr.x()][curr.y()][curr.z()] == 1 && visited_map[curr.x()][curr.y()][curr.z()] == 0 && curr.z() == layer)
            {
                return path; // Found the path, return the complete path.
            }

            for (const auto &dir : directions)  // 宽度优先搜索
            {
                int nextRow = curr.x() + dir.x();
                int nextCol = curr.y() + dir.y();
                int nextHeight = curr.z() + dir.z();
                if (isValidMove(nextRow, nextCol, nextHeight,grid) && !visited[nextRow][nextCol][nextHeight] && nextHeight <= upper)
                {
                    list<Vector3i> newPath = path;
                    newPath.push_back(Vector3i(nextRow, nextCol, nextHeight));
                    q.push(newPath);
                    visited[nextRow][nextCol][nextHeight] = true;
                }
            }
        }
        // If the path is not found, return an empty list.
        return {};
    }
    // Function to generate 3D layer search/path planning with 3D Dijkstra 绕layer平面的四个边一圈
    list<Eigen::Vector3i> Dijkstra_search_edge(int layer, int upper, string myname)
    {
        vector<vector<vector<int>>> grid = map;
        for (auto &name : namelist)
        {
            if (myname == name)
            {
                continue;
            }
            else
            {
                if (fabs(ros::Time::now().toSec() - local_dict[name].time) < 1||true)
                {
                    // cout<<"name:"<<name<<endl;
                    if (local_dict[name].in_bounding_box)
                    {
                        // cout<<"pos insert"<<endl;
                        Eigen::Vector3i tar = local_dict[name].position_index;
                        grid[tar.x()][tar.y()][tar.z()] = 1;
                    }
                    if (local_dict[name].planning_in_bounding_box)
                    {
                        // cout<<"motion insert"<<endl;
                        Eigen::Vector3i tar = local_dict[name].planning_index;
                        grid[tar.x()][tar.y()][tar.z()] = 1;
                    }
                }
            }
        }
        Eigen::Vector3i start = now_position_index;
        vector<Vector3i> directions = {Vector3i(0, 1, 0), Vector3i(0, -1, 0), Vector3i(1, 0, 0), Vector3i(-1, 0, 0), Vector3i(0, 0, 1), Vector3i(0, 0, -1)};

        // Initialize the queue and visited flag.
        queue<list<Eigen::Vector3i>> q;
        q.push({start});
        vector<vector<vector<bool>>> visited(map_shape.x(), vector<vector<bool>>(map_shape.y(), vector<bool>(map_shape.z(), false)));
        visited[start.x()][start.y()][start.z()] = true;

        while (!q.empty())
        {
            list<Eigen::Vector3i> path = q.front();
            q.pop();

            Eigen::Vector3i curr = path.back();
            // 和Dijkstra_search_2d_with_3d的区别是这个返回条件，这里要求了目标点要是layer这个平面上的边界点
            if ((curr.x() == 0 || curr.y() == 0 || curr.x() == map_shape.x() - 1 || curr.y() == map_shape.y() - 1) && visited_map[curr.x()][curr.y()][curr.z()] == 0 && curr.z() == layer)
            {
                return path; // Found the path, return the complete path.
            }

            for (const auto &dir : directions)
            {
                int nextRow = curr.x() + dir.x();
                int nextCol = curr.y() + dir.y();
                int nextHeight = curr.z() + dir.z();
                if (isValidMove(nextRow, nextCol, nextHeight,grid) && !visited[nextRow][nextCol][nextHeight] && nextHeight <= upper)
                {
                    list<Vector3i> newPath = path;
                    newPath.push_back(Vector3i(nextRow, nextCol, nextHeight));
                    q.push(newPath);
                    visited[nextRow][nextCol][nextHeight] = true;
                }
            }
        }
        // If the path is not found, return an empty list.
        return {};
    }
    list<Eigen::Vector3i> Dijkstra_search_fly_in_xy(int lower, int upper, string myname)  // 输入是最低层和最高层、无人机名字
    {
        vector<vector<vector<int>>> grid = map;
        for (auto &name : namelist)
        {
            if (myname == name)
            {
                continue;
            }
            else
            {
                if (fabs(ros::Time::now().toSec() - local_dict[name].time) < 1||true)
                {   
                    if (local_dict[name].in_bounding_box)
                    {
                        Eigen::Vector3i tar = local_dict[name].position_index;
                        grid[tar.x()][tar.y()][tar.z()] = 1;
                    }
                    if (local_dict[name].planning_in_bounding_box)
                    {
                        Eigen::Vector3i tar = local_dict[name].planning_index;
                        grid[tar.x()][tar.y()][tar.z()] = 1;
                    }
                }
            }
        }
        Eigen::Vector3i start = now_position_index;  // 起始点是无人机当前的位置
        // 搜索方向
        vector<Vector3i> directions = {Vector3i(0, 1, 0), Vector3i(0, -1, 0), Vector3i(1, 0, 0), Vector3i(-1, 0, 0), Vector3i(0, 0, 1), Vector3i(0, 0, -1)};

        // Initialize the queue and visited flag.
        queue<list<Eigen::Vector3i>> q;  // 队列
        q.push({start});  // 压入起始点
        vector<vector<vector<bool>>> visited(map_shape.x(), vector<vector<bool>>(map_shape.y(), vector<bool>(map_shape.z(), false)));  // 用于判断是否访问过的三维vector
        visited[start.x()][start.y()][start.z()] = true;  // 起始点访问过了

        while (!q.empty())  // q不为空
        {
            list<Eigen::Vector3i> path = q.front();  // 弹出头部
            q.pop();

            Eigen::Vector3i curr = path.back();
            if ((curr.x() == fly_in_index.x() && curr.y() == fly_in_index.y()) && curr.z() >= lower && curr.z() <= upper)
            {
                return path; // Found the path, return the complete path.
            }

            for (const auto &dir : directions)
            {
                int nextRow = curr.x() + dir.x();
                int nextCol = curr.y() + dir.y();
                int nextHeight = curr.z() + dir.z();
                if (isValidMove(nextRow, nextCol, nextHeight,grid) && !visited[nextRow][nextCol][nextHeight] && nextHeight <= upper && nextHeight >= lower)
                {
                    list<Vector3i> newPath = path;
                    newPath.push_back(Vector3i(nextRow, nextCol, nextHeight));
                    q.push(newPath);
                    visited[nextRow][nextCol][nextHeight] = true;
                }
            }
        }
        // If the path is not found, return an empty list.
        return {};
    }
    //两个函数分别在全局地图中与珊格地图中检测指定位置是否能到达，是否被占用。
    bool isValidMove(int x, int y, int z)
    {
        if (x >= 0 && x < map_shape.x() && y >= 0 && y < map_shape.y() && z < map_shape.z() && z >= 0)
        {
            if (map[x][y][z] == 1)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else
        {
            return false;
        }
    }

    // 判断该点是否可以到达
    bool isValidMove(int x,int y,int z,vector<vector<vector<int>>> grid)
    {
        if (x >= 0 && x < map_shape.x() && y >= 0 && y < map_shape.y() && z < map_shape.z() && z >= 0)
        {
            if (grid[x][y][z] == 1)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else
        {
            return false;
        }
    }    

    void generate_the_global_path()//生成全局路径
    {   
        list<Eigen::Vector3i> path_tamp(path_index);//创建path_index列表副本
        list<Eigen::Vector3d> point_global_list_tamp;//创建临时全局坐标系下的列表
        if (path_index.empty())//列表为空时，清空全局路径
        {
            path_global.clear();
            return;
        }
        nav_msgs::Path global_path_tamp;//定义nav_msgs类型的PATH，用于可视化

        // if (path_tamp.size() == 1)
        // {
        //     path_tamp.pop_back();
        // }
        // else if (path_tamp.size() >= 2)
        // {
        //     Eigen::Vector3i my_index = path_tamp.back();
        //     path_tamp.pop_back();
        //     if ((now_position_global - get_grid_center_global(path_tamp.back())).norm() > (get_grid_center_global(path_tamp.back()) - get_grid_center_global(my_index)).norm())
        //     {
        //         path_tamp.push_back(my_index);
        //     }
        // }

        path_tamp.pop_back();//在列表中移除最后一个元素

        // path_tamp.pop_back();
        global_path_tamp.header.frame_id = "world";//定义global_path_tamp消息基准坐标系
        while (!path_tamp.empty())
        {
            Eigen::Vector3i index_current = path_tamp.back();//获取当前索引
            Eigen::Vector3d point_current = get_grid_center_global(index_current);  // 从珊格索引获取其中心的实际坐标
            if (Developing)//将路径信息传给global_path_tamp
            {
                geometry_msgs::PoseStamped pose;
                pose.header.frame_id = "world";
                pose.header.stamp = ros::Time::now();
                pose.pose.position.x = point_current.x();
                pose.pose.position.y = point_current.y();
                pose.pose.position.z = point_current.z();
                pose.pose.orientation.w = 1.0;
                global_path_tamp.poses.push_back(pose);
            }
            point_global_list_tamp.push_back(point_current);
            path_tamp.pop_back();
        }
        // point_global_list_tamp.push_back(path_final_global);
        path_global_show_message = global_path_tamp;
        path_global = point_global_list_tamp;  // 最终结果，即将珊格索引转为坐标
    }
    list<Eigen::Vector3d> get_search_target(Eigen::Vector3i true_index)//获取与输入坐标 true_index 相邻且可用的坐标点的全局坐标，并将它们存储在 point_list 中
    {
        list<Eigen::Vector3d> point_list;
        // cout<<"begin:"<<endl;//test  
        for (int x = true_index.x() - 1; x < true_index.x() + 2; x++)
        {
            for (int y = true_index.y() - 1; y < true_index.y() + 2; y++)
            {
                for (int z = true_index.z() - 1; z < true_index.z() + 2; z++)
                {
                    if (out_of_range_index(Eigen::Vector3i(x, y, z)))
                    {
                        continue;
                    }
                    if (abs(x - true_index.x()) + abs(y - true_index.y()) + abs(z - true_index.z()) == 1)  // 和输入点直接相邻的点
                    {
                        if (map[x][y][z] == 1)  // 目标点需被占用
                        {
                            point_list.push_back(get_rpy_limited_global(Eigen::Vector3d(x - true_index.x(), y - true_index.y(), z - true_index.z())));
                            // cout<<(Eigen::Vector3i(x,y,z)-true_index).transpose()<<endl;//test
                            // cout<<get_rpy_limited_global(Eigen::Vector3d(x-true_index.x(),y-true_index.y(),z-true_index.z())).transpose()<<endl;//test
                        }
                    }
                }
            }
        }
        // cout<<"end"<<endl;//test

        return point_list;
    }
    double get_rad(Eigen::Vector3d v1, Eigen::Vector3d v2)//求弧度制
    {
        return atan2(v1.cross(v2).norm(), v1.transpose() * v2);
    }
    Eigen::Vector3d get_rpy_limited_global(Eigen::Vector3d target_direction)//得到全局坐标系下目标方向的欧拉角度
    {
        Eigen::Vector3d global_target_direction = rotation_matrix_inv * target_direction;//通过旋转矩阵获得全局坐标系下的目标方向
        // Eigen::Vector3d global_target_direction=target_direction;
        Eigen::Quaterniond quaternion;//定义四元素变量
        quaternion.setFromTwoVectors(Eigen::Vector3d(1, 0, 0), global_target_direction);//将两者之间的旋转转换为四元素
        Eigen::Matrix3d rotation_matrix_here = quaternion.toRotationMatrix();//从四元素转换为旋转矩阵
        Eigen::Vector3d rpy = Rot2rpy(rotation_matrix_here);//旋转矩阵转为欧拉角
        if (abs(rpy.x() + rpy.z()) < 1e-6 || abs(rpy.x() - rpy.z()) < 1e-6)//欧拉角x 和 z 分量接近于零（小于 1e-6），则将它们设置为零
        {
            rpy.x() = 0;
            rpy.z() = 0;
        }

        if (rpy.y() > M_PI * 4 / 9)//限制欧拉角的范围
        {
            rpy.y() = M_PI * 4 / 9;
        }
        else if (rpy.y() < -M_PI * 4 / 9)
        {
            rpy.y() = -M_PI * 4 / 9;
        }
        return rpy;
    }
    Eigen::Vector3d Rot2rpy(Eigen::Matrix3d R)//旋转矩阵转欧拉角
    {
        // Eigen::Vector3d euler_angles=R.eulerAngles(2,1,0);
        // Eigen:;Vector3d result(euler_angles.z(),euler_angles.y(),euler_angles.x());
        // return result;
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d rpy(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        rpy(0) = r;
        rpy(1) = p;
        rpy(2) = y;

        return rpy;
    }
    Eigen::Matrix3d Rpy2Rot(Eigen::Vector3d rpy)//欧拉角转旋转矩阵
    {
        Eigen::Matrix3d result = Eigen::Matrix3d::Identity();
        result = Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix() * Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()).toRotationMatrix() * Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()).toRotationMatrix();
        return result;
    }
    string point3i2str(Eigen::Vector3i point)//点云转字符串
    {
        string result;
        result = to_string(point.x()) + "," + to_string(point.y()) + "," + to_string(point.z());
        return result;
    }
    bool str2point3i(string str, Eigen::Vector3i &result)//字符串转珊格
    {
        std::vector<string> value;
        Eigen::Vector3i result_tamp;
        boost::split(value, str, boost::is_any_of(","));
        if (value.size() == 3)
        {
            // cout<<"str:"<<str<<endl;
            try
            {
                result_tamp = Eigen::Vector3i(stoi(value[0]), stoi(value[1]), stoi(value[2]));
            }
            catch (const std::invalid_argument &e)
            {
                return false;
                cout << "Invalid argument" << e.what() << endl;
            }
            catch (const std::out_of_range &e)
            {
                cout << "Out of range" << e.what() << endl;
                return false;
            }
            result = result_tamp;
            return true;
        }
        else
        {
            return false;
            cout << "error use str2point 3" << endl;
            // cout<<"str:"<<str<<endl;
        }
    }
    void insert_map_index(Eigen::Vector3i true_index)//珊格坐标中插入占用点
    {
        map[true_index.x()][true_index.y()][true_index.z()] = 1;
        occupied_num++;
        for (int x = true_index.x() - 1; x < true_index.x() + 2; x++)
        {
            for (int y = true_index.y() - 1; y < true_index.y() + 2; y++)
            {
                for (int z = true_index.z() - 1; z < true_index.z() + 2; z++)
                {
                    if (out_of_range_index(Eigen::Vector3i(x, y, z)))
                    {
                        continue;
                    }
                    if (abs(x - true_index.x()) + abs(y - true_index.y()) + abs(z - true_index.z()) == 1)
                    {
                        if (map[x][y][z] == 0 && visited_map[x][y][z] == 0)
                        {
                            interest_map[x][y][z] = 1;
                        }
                        else
                        {
                            interest_map[x][y][z] = 0;
                        }
                    }
                }
            }
        }
    }
    Eigen::Vector3d str2point(string input)//字符串转点云
    {
        Eigen::Vector3d result;
        std::vector<string> value;
        boost::split(value, input, boost::is_any_of(","));
        if (value.size() == 3)
        {
            result = Eigen::Vector3d(stod(value[0]), stod(value[1]), stod(value[2]));
        }
        else
        {
            // cout<<"error use str2point 4"<<endl;
            result = Eigen::Vector3d(1000, 1000, 1000);
        }
        return result;
    }
};

// 用于地图信息转换
class mainbrain
{
public:
    ros::NodeHandlePtr nh_ptr_;
    mainbrain() {}
    mainbrain(string str, string name, ros::NodeHandlePtr& nh_ptr)//构造函数，根据传入参数判断是否为探索无人机，并且将路径信息分割并生成全局地图、 初始化部分成员变量
    {
        nh_ptr_ = nh_ptr;
        drone_rotation_matrix = Eigen::Matrix3d::Identity();//定义无人机旋转矩阵
        grid_size = Eigen::Vector3d(safe_distance, safe_distance, safe_distance);//定义珊格地图三维尺寸
        global_map = grid_map(grid_size, nh_ptr_);//根据定义的珊格地图尺寸初始化
        namespace_ = name;
        if (namespace_ == "/jurong" || namespace_ == "/raffles")//当前无人机为探索无人机时，is_leader标志位为ture
        {
            is_leader = true;
        }
        vector<string> spilited_str;  //用于存储将 str 字符串按分号(;)分割后的子字符串
        std::istringstream iss(str);
        std::string substring;
        while (std::getline(iss, substring, ';'))
        {
            spilited_str.push_back(substring);
        }
        generate_global_map(spilited_str[0]);  // ;的第一个子字符串记录这边界框覆盖范围、队伍分组以及负责路径的信息，进行记录
        if (spilited_str.size() > 1)  // 后面的子字符串记录的是边界框（路径）的具体信息，遍历所有path_index，并为每一个索引创建grid_map对象，并将其添加到map_set中
        {
            for (int j = 0; j < path_index.size(); j++)
            {
                map_set.push_back(grid_map(Boundingbox(spilited_str[path_index[j]]), grid_size, teammates_name.size(), teammates_name, nh_ptr_));
            }
        }
        else
        {
            cout << "Path Assigned Error!!!" << endl;
        }

        cout << "size of path assigned:  " << map_set.size() << endl;
        finish_init = true; //初始化完成
    }

    //更新地图信息，将点云信息添加到全局地图中
    void update_map(const sensor_msgs::PointCloud2ConstPtr &cloud, const sensor_msgs::PointCloud2ConstPtr &Nbr, const nav_msgs::OdometryConstPtr &msg)
    {
        if (!is_leader)  // 只有leadr有点云信息
        {
            return;
        }
        Eigen::Vector3d sync_my_position = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        vector<Eigen::Vector3d> Nbr_point;  // 用于存储邻居机器人的位置信息
        Nbr_point.push_back(sync_my_position);  // 将无人机自身位置信息压入
        CloudOdomPtr Nbr_cloud(new CloudOdom());  // 创建了一个指向 CloudOdom 类型的智能指针 Nbr_cloud，用于存储邻居机器人的里程计信息
        pcl::fromROSMsg(*Nbr, *Nbr_cloud);  // 邻居的信息存为了PointCloud2类型，转为pcl的类型
        for (const auto &point : Nbr_cloud->points)
        {
            Eigen::Vector3d cloud_point(point.x, point.y, point.z);
            if (std::fabs(point.t - Nbr->header.stamp.toSec()) > 0.2)  // 检查每个点的时间戳，如果时间戳与消息时间戳相差大于0.2秒
            {
                cout << "Missing Nbr" << endl;
            }
            else
            {
                Nbr_point.push_back(cloud_point);  // 压入邻居的位置
            }
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cloud(new pcl::PointCloud<pcl::PointXYZ>);  // 用于存储当前机器人的点云数据
        pcl::fromROSMsg(*cloud, *cloud_cloud);  // 将 ROS 消息 cloud 转换为点云数据，存储在 cloud_cloud 中
        for (const auto &point : cloud_cloud->points)  // 一个一个点云遍历
        {
            Eigen::Vector3d cloud_cloud_point(point.x, point.y, point.z);
            if (is_Nbr(cloud_cloud_point, Nbr_point))  // 检查这个点云是不是邻居无人机机体扫到的，是的话不存储这个点
            {
                continue;
            }
            else
            {
                insert_point(cloud_cloud_point);  // 在全局地图中插入这个点
            }

            // for pcd
            // if (map_set.size() > now_id && now_id == pcd_id) {
            //     mtx_cloud.lock();
            //     auto &elem = map_set[now_id];
            //     if (elem.out_of_range_box(cloud_cloud_point, false))  // 根据map_shape判断是否超过map范围，是的话直接退出函数
            //     {
            //         mtx_cloud.unlock();
            //         continue;
            //     }
                
            //     pointSet.insert(Point3D(point.x, point.y, point.z));
            //     mtx_cloud.unlock();
            // }
        }
    }
    
    void update_gimbal(Eigen::Vector3d gimbal_position)//更新云台姿态信息，计算云台旋转矩阵，更新相应的地图信息
    {
        if (!finish_init)
        {
            return;
        }
        Eigen::Matrix3d gimbal_rotation_matrix = Rpy2Rot(gimbal_position);
        Eigen::Matrix3d now_rot = gimbal_rotation_matrix * drone_rotation_matrix;
        Eigen::Vector3d rpy = Rot2rpy(now_rot);
        rpy.x() = 0;

        if (map_set.size() > now_id && !is_transfer)
        {
            map_set[now_id].update_gimbal(rpy, false);
        }
        else
        {
            global_map.update_gimbal(rpy, false);
        }
    }

    int get_exploration_finish() {
        return finish_id;
    }

    // 根据视点的位置，存到一个vector里面(每个box一个)-探索者
    void update_viewpoint(const geometry_msgs::PoseArray &msg, int box_id) {
        vector<Eigen::Vector3d> box_vp_tem;
        // std::cout << "box_id: " << box_id <<"now_id: " <<now_id << std::endl;
        for(auto vp : msg.poses) {
            Eigen::Vector3d view_point(vp.position.x, vp.position.y, vp.position.z);
            if(!map_set[box_id].out_of_range_box_viewpoint(view_point, false)) {  // 此处判断视点是否在当前边界框的大概范围内
                box_vp_tem.push_back(view_point);
            }
        }
        map_set[box_id].update_viewpoint_box(box_vp_tem);
        box_viewpoint_vec.push_back(box_vp_tem);
        std::cout << "完成了第" << box_id << "个边界框的视点存储，视点个数：" <<box_viewpoint_vec[box_id].size() << std::endl;
    }
    
    void update_position(Eigen::Vector3d point, Eigen::Matrix3d rotation)  // 更新无人机位置与姿态信息，更新无人机的旋转矩阵并更新对应地图
    {
        if (!odom_get)
        {
            initial_position = point;
            initial_position = initial_position + Eigen::Vector3d(0, 0, 3);  // 初始位置的高度加3，最后返回似乎只是悬停在初始位置的正上方，没有降落处理
        }
        drone_rotation_matrix = rotation;
        now_global_position = point;
        if (!finish_init)
        {
            return;
        }

        if (map_set.size() > now_id)  // 如果当前还在遍历边界框，则同时更新全局地图和子地图
        {
            global_map.update_position(point);
            map_set[now_id].update_position(point);
        }
        else
        {
            global_map.update_position(point);
        }
        odom_get = true;
    }
    
    void replan()//根据无人机当前状态与无人机类型，重新进行路径规划更新二次规划后的信息
    {
        if (!finish_init)
        {
            return;
        }

        if (map_set.size() == now_id && is_transfer)  // 猜测：完成了对边界框的探索，无人机回到原始位置？？
        {
            // fly home
            if (is_leader)  // 是带领者
            {   
                state = 0;  // state为０代表着回到初始位置状态？？
                bool flag = false;  // 探索的路径是否为空
                global_map.Astar_local(initial_position, namespace_, info_mannager.get_leader(), flag, true);  // Ａ*规划到初始位置上方的路径
                get_way_point = update_target_waypoint();  // 成功得到waypoint则返回true，如果global_map为空则让无人机停留在原地附近
                path_show = global_map.get_path_show();  // 用于可视化的路径
                if (flag)  // 搜索的路径为空时，高度加２，重新进入这个函数时，可以搜索新的路径，直到成功。
                {
                    initial_position.z()= initial_position.z()+2;
                }
                return;
            }
            else  // 如果是拍照无人机
            {
                state = 0;
                if (state == 0)
                {
                    is_transfer = true;  // 这个似乎多余，本来这个变量为真才能进入
                    if (info_mannager.get_leader_state() == 0 && !not_delete)  // 如果该队伍的带领者状态为０，并且not_delete为false？？
                    {
                        not_delete = true;
                    }
                }
                if (now_global_position.z() < 8)  // 当前位置高度低于８米时，回到初始位置
                {
                    bool flag = false;  // 探索的路径是否为空

                    global_map.Astar_local(initial_position, namespace_, info_mannager.get_leader(), flag, false);
                    if (flag)  // 探索的路径为空，改用长距离Ａ*再搜索一遍
                    {
                        global_map.Astar_local(initial_position, namespace_, info_mannager.get_leader(), flag, true);
                    }
                    get_way_point = update_target_waypoint();
                    path_show = global_map.get_path_show();
                    if (flag)  // 搜索的路径为空时，高度加１，重新进入这个函数时，可以搜索新的路径，直到成功。
                    {
                        initial_position.z() += 1;
                    }

                    return;
                }
                else  // 当前位置高度高于８米时，去到带领者的位置？？应该时跟着带领者飞行，低于八米回到其初始位置，改：高度比目标高度高1m，防止碰撞
                {
                    Eigen::Vector3d target;
                    info_mannager.get_leader_position(target);
                    target[2] += 2;  // 改：返回的时候高度加1，防止碰撞

                    bool flag = false;
                    global_map.Astar_local(target, namespace_, info_mannager.get_leader(), flag, false);
                    // if (flag)
                    // {
                    //     global_map.Astar_local(initial_position, namespace_, info_mannager.get_leader(), flag, true);
                    // }
                    get_way_point = update_target_waypoint();
                    path_show = global_map.get_path_show();

                    return;
                }
            }
        }
        else if (finish_init && is_transfer)  // 无人机在边界框间转换的状态
        {
            if (namespace_ == "/jurong" || namespace_ == "/raffles")
            {
               ////yolo();
                Eigen::Vector3d target = map_set[now_id].get_fly_in_point_global();  // 得到飞去这个边界框的目标点，第一次进入这个函数时，默认fly in point为局部坐标系下的０００
                bool flag = false;  // 探索的路径是否为空
                global_map.Astar_local(target, namespace_, info_mannager.get_leader(), flag, false);
                get_way_point = update_target_waypoint();
                path_show = global_map.get_path_show();
                map_set[now_id].update_fly_in_index(flag);  // 更新当前边界框的飞入点，默认是000点，搜索的原则是在最外围绕圈（重复时往里面靠一层），找到没有被占据的珊格
                is_transfer = !map_set[now_id].check_whether_fly_in(false);  // 判断无人机是否到达了飞入点，如果是，则is_transfer状态变为false
                if (!is_transfer)  // 无人机飞到飞入点后，状态改变，更新变量，state=1
                {
                    map_set[now_id].set_state(1);
                    state = map_set[now_id].get_state();
                }
               ////yolo();
            }                    
            else
            {
                if (state == 0)  // 状态为０是一开始的状态，下面的作用是？？
                {
                    is_transfer = true;
                    if (info_mannager.get_leader_state() == 0 && !not_delete)
                    {
                        not_delete = true;
                    }
                }
                if (info_mannager.get_leader_state() == 0)  // 如果领导者的状态是０，代表正在飞往下一个边界框的飞入点，拍照无人机跟随飞行
                {
                    Eigen::Vector3d target;
                    info_mannager.get_leader_position(target);
                    bool flag = false;
                    global_map.Astar_local(target, namespace_, info_mannager.get_leader(), flag, false);
                    get_way_point = update_target_waypoint();
                    path_show = global_map.get_path_show();
                    return;
                }
                else if (state == 1)  // 如果在转换中，并且state=1，则飞去飞入点
                {
                    Eigen::Vector3d target = map_set[now_id].get_fly_in_point_global();
                    bool flag = false;
                    global_map.Astar_photo(target, namespace_, flag);
                    get_way_point = update_target_waypoint();
                    path_show = global_map.get_path_show();
                    map_set[now_id].update_fly_in_index(flag); 
                    is_transfer = !map_set[now_id].check_whether_fly_in(false);
                    if (!is_transfer)
                    {
                        map_set[now_id].set_state(1);
                        state = map_set[now_id].get_state();
                    }
                    return;
                }
                else
                {
                    Eigen::Vector3d target = map_set[now_id].get_fly_in_point_global();
                    bool flag = false;
                    global_map.Astar_photo(target, namespace_, flag);
                    get_way_point = update_target_waypoint();
                    path_show = global_map.get_path_show();
                    map_set[now_id].update_fly_in_index(flag); 
                    return;
                }
            }
        }
        else
        {
            if (namespace_ == "/jurong" || namespace_ == "/raffles")  // 探索边界框
            {
               ////yolo();
                state = map_set[now_id].get_state_leader();  // 返回领导者状态
                map_set[now_id].exploration(namespace_);  // 探索区域
                if(map_set[now_id].update_finish_id()) {
                    finish_id = now_id;
                }

                path_show = map_set[now_id].get_path_show();
                get_way_point = update_target_waypoint();
               ////yolo();
                if (map_set[now_id].get_whether_pop())  // 猜测：当前边界框地图探索完毕，到下一个地图，所以now_id++
                {
                    now_id++;
                    state = 0;
                    is_transfer = true;
                    
                    return;
                }
            }
            else
            {
                if (map_set.size() > now_id)
                {
                    state = map_set[now_id].get_state();
                }
                else
                {
                    state = 0;
                }
                if (state == 0)
                {
                    is_transfer = true;
                    return;
                }
                if (map_set.size() > now_id)
                {
                    // 拍照无人机拍照策略
                    static bool reach_flag = false;;  // 用于判断何时更新点
                    static int flag_count = 0;
                    bool islong = false;
                    // 如果到达目标点, 得到下一个点
                    Eigen::Vector3d target = map_set[now_id].get_next_point(false, now_global_position, flag_count, reach_flag, namespace_);  
                    // get_way_point = update_target_waypoint(now_global_position);
                    // double dis = std::sqrt((now_global_position[0]-target[0])*(now_global_position[0]-target[0]) \
                    //                  + (now_global_position[1]-target[1])*(now_global_position[1]-target[1]) \
                    //                  + (now_global_position[2]-target[2])*(now_global_position[2]-target[2]));
                    // std::cout << namespace_ << " now_global_position : " << now_global_position << std::endl;
                    // std::cout << namespace_ << " flying to finnal target: " << target << std::endl;
                    Eigen::Vector3d target_index = global_map.get_grid_center_global(global_map.get_index(target));
                    double dis = std::sqrt((now_global_position[0]-target_index[0])*(now_global_position[0]-target_index[0]) \
                                    + (now_global_position[1]-target_index[1])*(now_global_position[1]-target_index[1]) \
                                    + (now_global_position[2]-target_index[2])*(now_global_position[2]-target_index[2]));
                    std::cout << namespace_ << " now_global_position : " << now_global_position << std::endl;
                    std::cout << namespace_ << " flying to finnal target: " << target << std::endl;
                    std::cout << namespace_ << " dis: " << dis << std::endl;
                    if (dis > 40)  // 如果大于四十米，用长A*
                        islong = true;
                    
                    if (dis < 0.5) reach_flag = true;  // 距离小于一定阈值判断到达，可以获取下一个点
                    else reach_flag = false;

                    bool flag = false;
                    global_map.Astar_photo_my(target, namespace_, flag, islong);  // 如果规划失败则flag为true
                    if (flag) flag_count++;  // 记录规划失败次数，超出3次抛弃这个视点
                    else flag_count = 0;

                    std::cout << namespace_ << " reach_flag : " << reach_flag << " flag: " << flag << " islong" << islong << std::endl;
                    if (!map_set[now_id].go_fly_in_xy) {
                        target_position = global_map.get_next_point_my(true, reach_flag, namespace_);  // 获取A*规划好的点
                        std::cout << namespace_ << " flying to now target: " << target_position << std::endl;
                        // map_set[now_id].get_next_point(false, now_global_position, flag, namespace_); 
                        // target_position = global_map.get_next_point_mn(Astar_finish, flag);  // A*规划的点发送出去
                        path_show = map_set[now_id].get_path_show();
                    }
                    else {
                        target_position = target;
                    }

                }
                else
                {
                    get_way_point = update_target_waypoint();
                }
            }
        }

    }
    bool update_target_waypoint()//更新目标路径点
    {
        if (odom_get && finish_init)
        {
            if (is_transfer || map_set.size() == now_id)  // 在边界框直接跳转
            {
                target_position = global_map.get_next_point(true);
                finish_first_planning = true;
                return true;
            } 
            else  // 局部，正在拍照或者探索
            {
                static int flag_count = 0;
                if (is_leader) {
                    if (!map_set[now_id].get_box_all_finish_exp()) {
                        target_position = map_set[now_id].get_next_point(false);
                    }
                    else {
                        static bool reach_flag = false;
                        bool islong = false;
                        Eigen::Vector3d target = map_set[now_id].get_next_point(false, now_global_position, flag_count, reach_flag, namespace_);  // 如果到达目标点, 得到下一个点
                        // double dis = std::sqrt((now_global_position[0]-target[0])*(now_global_position[0]-target[0]) \
                        //                 + (now_global_position[1]-target[1])*(now_global_position[1]-target[1]) \
                        //                 + (now_global_position[2]-target[2])*(now_global_position[2]-target[2]));
                        // std::cout << namespace_ << " now_global_position : " << now_global_position << std::endl;
                        // std::cout << namespace_ << " flying to finnal target: " << target << std::endl;
                        Eigen::Vector3d target_index = global_map.get_grid_center_global(global_map.get_index(target));
                        double dis = std::sqrt((now_global_position[0]-target_index[0])*(now_global_position[0]-target_index[0]) \
                                        + (now_global_position[1]-target_index[1])*(now_global_position[1]-target_index[1]) \
                                        + (now_global_position[2]-target_index[2])*(now_global_position[2]-target_index[2]));
                        std::cout << namespace_ << " now_global_position : " << now_global_position << std::endl;
                        std::cout << namespace_ << " flying to finnal target: " << target << std::endl;
                        std::cout << namespace_ << " dis: " << dis << std::endl;
                        if (dis > 50) 
                            islong = true;
                        
                        if (dis < 0.5) reach_flag = true;
                        else reach_flag = false;

                        bool flag = false;
                        global_map.Astar_photo_my(target, namespace_, flag, islong);
                        if (flag) flag_count++;
                        else flag_count = 0;
                        std::cout << namespace_ << " reach_flag : " << reach_flag << " flag: " << flag << std::endl;

                        // reach_flag = map_set[now_id].reach_target(target);
                        // reach_flag = map_set[now_id].reach_target(global_map.get_grid_center_global(global_map.get_index(target)));
                        // if ((!Astar_finish || flag) && !map_set[now_id].go_fly_in_xy) {  // 规划没有完成或者规划失败，并且不是去飞入点
                        //     global_map.Astar_photo_my(target, namespace_, flag, islong);  // A*规划,下一个点不可达则flag为true
                        //     if (flag) Astar_finish = false;
                        //     else Astar_finish = true;
                        // }
                        // std::cout << "target: " << target << std::endl;
                        // 如果到达目标点或者如果flag为true,则得到下一个点, 如果点列表为空则改变状态并飞到飞入点的xy上(通信)
                        if (!map_set[now_id].go_fly_in_xy) { 
                            target_position = global_map.get_next_point_my(true, reach_flag, namespace_);
                            std::cout << namespace_ << " flying to target: " << target_position << std::endl;
                            
                            
                            // map_set[now_id].get_next_point(false, now_global_position, flag, namespace_); 
                            // target_position = global_map.get_next_point_mn(Astar_finish, flag);  // A*规划的点发送出去
                        }
                        else {
                            target_position = target;
                        }
                    }
                    
                }
                else {
                    target_position = map_set[now_id].get_next_point(false, now_global_position, flag_count);
                }
                finish_first_planning = true;
                return true;
            }
        }
        else
        {
            return false;
        }
    }
    
    bool get_cmd(trajectory_msgs::MultiDOFJointTrajectory &cmd, geometry_msgs::Twist &gimbal)//获取无人机控制指令（包括云台信息与无人机轨迹信息）
    {
        if (!finish_first_planning)
        {
            return false;
        }
        if (get_way_point)
        {
            if (is_transfer || map_set.size() == now_id)
            {
                global_map.get_gimbal_rpy(target_angle_rpy);  // 获取云台转向控制信息
                // 构建控制消息，策略为：当前位置和目标位置相隔较近时，发送位置控制指令，较远时，先速度控制，接近后位置控制。偏航角由搜索方向确定
                cmd = position_msg_build(now_global_position, target_position, target_angle_rpy.z());
            }
            else
            {
                try
                {
                    map_set[now_id].get_gimbal_rpy(target_angle_rpy);
                }
                catch (...)
                {
                }

                cmd = position_msg_build(now_global_position, target_position, target_angle_rpy.z());
            }
            gimbal = gimbal_msg_build(target_angle_rpy);  // 构建云台控制数据
            return true;
        }
        else
        {
            return false;
        }
    }
    
    visualization_msgs::MarkerArray Draw_map()//可视化地图与路径
    {
        if (!finish_init)
        {
            visualization_msgs::MarkerArray nullobj;
            return nullobj;
        }
        if (map_set.size() == now_id)
        {
            return global_map.Draw_map();
        }
        else
        {
            return map_set[now_id].Draw_map();
        }
    }
    nav_msgs::Path Draw_Path()
    {
        path_show.header.frame_id = "world";
        return path_show;
    }
    // 管理无人机、无人机路径规划、无人机之间的通信
    bool get_position_plan_msg(string &msg)//返回 位置+无人机+坐标+路径 字符串信息
    {
        if (!odom_get || !finish_init)
        {
            return false;
        }
        else if (!is_planned)
        {
            msg = "position;" + namespace_ + ";" + point2str(now_global_position) + ";";
            return true;
        }
        else
        {
            msg = "position;" + namespace_ + ";" + point2str(now_global_position) + ";" + point2str(planning_point);
            return true;
        }
    }

    bool get_global_massage(string &msg)//获取全局地图信息的字符串消息，若无人机是领队并且初始化已完成，它将返回一个包含全局地图的消息字符串
    {
        if (!is_leader || !finish_init)  // 如果不是带领者或者没有初始化完成
        {
            return false;
        }
        // "mapglobal;Teamid;occupied_num;map_cloud_massage"
        msg = "mapglobal;" + to_string(Teamid) + ";" + global_map.get_num_str() + global_map.get_map_str();
        return true;
    }
    bool get_local_massage(string &msg)//获取本地地图的字符串信息
    {
        if (!is_leader || !finish_init || state >= 2 || map_set.size() == now_id)  // 需要满足这一系列条件，后面两个不知道是什么？？
        {
            return false;
        }
        // "map;Teamid;occupied_num;map_cloud_massage"
        msg = "map;" + to_string(Teamid) + ';' + map_set[now_id].get_num_str() + map_set[now_id].get_map_str();
        return true;
    }
    bool get_fly_in_massage(string &msg)//生成飞入点信息字符串，包括团队ID和当前无人机的飞入点信息
    {
        if (!is_leader || !finish_init || map_set.size() == now_id)
        {
            return false;
        }
        // "flyin;Teamid;fly_in_index(三维坐标);"
        msg = "flyin;" + to_string(Teamid) + ';' + map_set[now_id].get_fly_in_str();
        return true;
    }
    bool get_state_set_msg_list(list<string> &string_list)//生成状态设置信息字符串列表，包括团队ID、无人机名称和状态信息
    {
        if (!is_leader || !finish_init || map_set.size() == now_id)
        {
            return false;
        }
        list<string> string_list_tamp = map_set[now_id].get_state_string_list();
        if (string_list_tamp.empty())
        {
            return false;
        }
        else
        {
            for (auto &str : string_list_tamp)
            {
                string_list.push_back("state_set;" + to_string(Teamid) + ";" + str);
            }
            return true;
        }
    }
    bool get_state_massage(string &msg)// 生成状态信息字符串，包括无人机命名空间和状态
    {
        if (!finish_init) // || !is_leader)  // !!!!!md什么时候改了这里,牛逼,真煞笔
        {
            return false;
        }
        // "state;无人机名(namespace_);state;"
        msg = "state;" + namespace_ + ";" + to_string(state) + ";";
        return true;
    }

    bool get_viewpoint_massage(string &msg)  // 探索者生成view point信息
    {
        if (!finish_init || !is_leader || finish_id == -1)
        {
            return false;
        }
        // 构造的信息："viewpoint;teamid;box总数；视点个数；所有视点；视点个数；所有视点；"
        msg = "viewpoint;" + to_string(Teamid) + ";" + to_string(map_set.size()) + ";";
        for (int i = 0; i < map_set.size(); i++) {
            // msg += to_string(i) + ";";
            list<Eigen::Vector3d> view_point = map_set[i].get_viewpoint_vec();
            if (!view_point.empty()) {
                msg += to_string(view_point.size()) + ";";
                for (auto vp : view_point) {
                    msg += to_string(vp[0]) + "," + to_string(vp[1]) + "," + to_string(vp[2]) + ";";
                }
            }
            else {
                msg += "0;";
            }
        }
        
        return true;
    }

    void communicate(string str)//无人机之间的通信函数
    {
        if (!finish_init)  // 通信的前提是主控制大脑完成了初始化
        {
            return;
        }

        istringstream msg_stream(str);
        string topic;
        getline(msg_stream, topic, ';');

        if (topic == "position")
        {
            if (not_delete == false || (!is_leader && state == 2))
            {
                return;
            }
            istringstream global_po_str(str);
            istringstream local_po_str(str);
            getline(global_po_str, topic, ';');
            getline(local_po_str, topic, ';');
            info_mannager.reset_position_path(msg_stream);
            if (map_set.size() > now_id)
            {
                global_map.update_local_dict(global_po_str);
                map_set[now_id].update_local_dict(local_po_str);
            }
            else
            {
                global_map.update_local_dict(global_po_str);
            }

            return;
        }
        else if (topic == "state")
        {
            string orin;
            getline(msg_stream, orin, ';');
            string state_str;
            getline(msg_stream, state_str, ';');
            if (map_set.size() > now_id)
            {
                try
                {
                    if (orin == info_mannager.get_leader() && stoi(state_str) == 3 && state == 2 && not_delete)
                    {
                        not_delete = false;
                        now_id++;
                        return;
                    }
                    // std::cout << "00000000000000000000000000000000 update state: " << orin << state_str << std::endl;
                    map_set[now_id].update_state(orin, stoi(state_str));
                    info_mannager.update_state(orin, stoi(state_str));
                }
                catch (const std::invalid_argument &e)
                {

                    cout << "Invalid argument" << e.what() << endl;
                }
                catch (const std::out_of_range &e)
                {
                    cout << "Out of range" << e.what() << endl;
                }
            }
            else
            {
                try
                {
                    info_mannager.update_state(orin, stoi(state_str));
                }
                catch (const std::invalid_argument &e)
                {
                    cout << "Invalid argument" << e.what() << endl;
                }
                catch (const std::out_of_range &e)
                {
                    cout << "Out of range" << e.what() << endl;
                }
                return;
            }
        }
        else if (topic == "state_set")
        {
            string target_team;
            if (not_delete == false || (!is_leader && state == 2 || map_set.size() == now_id))  
            {
                return;
            }
            getline(msg_stream, target_team, ';');
            if (stoi(target_team) == Teamid)
            {
                string target_name;
                getline(msg_stream, target_name, ';');
                if (namespace_ == target_name)
                {
                    string state_str;
                    getline(msg_stream, state_str, ';');
                    state = stoi(state_str);
                    // std::cout << "00000000000000000000000000000000 update state: " << namespace_ << state_str << std::endl;
                    return;
                }
                else
                {
                    return;
                }
            }
            else
            {
                return;
            }
        }
        else if (topic == "map")
        {
            if (not_delete == false || (!is_leader && state == 2) || map_set.size() == now_id)
            {
                return;
            }
            string target_team;
            getline(msg_stream, target_team, ';');
            // cout<<"target team:"<<target_team<<endl;
            if (stoi(target_team) == Teamid)
            {
                // insert map_front from string
                if (map_set.size() > now_id)
                {
                    map_set[now_id].insert_cloud_from_str(msg_stream);
                }
                else
                {
                    return;
                }
            }
            else
            {
                return;
            }
        }
        else if (topic == "mapglobal")
        {
            string target_team;
            if (not_delete == false || (!is_leader && state == 2))
            {
                return;
            }

            getline(msg_stream, target_team, ';');

            if (stoi(target_team) == Teamid)
            {
                global_map.insert_cloud_from_str(msg_stream);
            }
            else
            {
                return;
            }
        }
        else if (topic == "visit")
        {
            string target_team;
            getline(msg_stream, target_team, ';');
            if (stoi(target_team) == Teamid)
            {
                // insert map_front from string
                // not develop this function now
            }
            else
            {
                return;
            }
        }
        else if (topic == "flyin")
        {
            if (not_delete == false || (!is_leader && state == 2) || map_set.size() == now_id)
            {

                return;
            }
            string target_team;
            getline(msg_stream, target_team, ';');
            if (stoi(target_team) == Teamid)
            {
                if (map_set.size() >= now_id)
                {
                    // insert fly_in_index
                    string fly_in_index;
                    getline(msg_stream, fly_in_index, ';');
                    map_set[now_id].set_fly_in_index(fly_in_index);
                }
                else
                {
                    return;
                }
            }
            else
            {

                return;
            }
        }
        else if (topic == "viewpoint") {
            if (is_leader) {
                return;
            }
            vector<Eigen::Vector3d> tem_vec;
            Eigen::Vector3d vpoint;
            string target_team;
            getline(msg_stream, target_team, ';');
            if (stoi(target_team) == Teamid) {
                string box_num;
                getline(msg_stream, box_num, ';');
                for (int i = 0; i < stoi(box_num); i++) {
                    tem_vec.clear();
                    string viewpoint_num;
                    getline(msg_stream, viewpoint_num, ';');
                    for (int j = 0; j < stoi(viewpoint_num); j++) {
                        string viewpoint;
                        getline(msg_stream, viewpoint, ';');
                        vpoint = str2point(viewpoint);
                        // std::istringstream iss(viewpoint);
                        // std::string token;
                        // for (int k = 0; k < 3; k++) {
                        //     getline(iss, token, ',');
                        //     vpoint[k] = stod(token);
                        // }
                        tem_vec.push_back(vpoint);
                    }
                    map_set[i].update_viewpoint_box_from_topic(tem_vec, namespace_);
                }
            }

        }
    }

private:
    bool Astar_finish = false;

    std::set<Point3D> pointSet;  // for pcd
    // bool explor_finish = false;
    int pcd_id = 0;
    vector<vector<Eigen::Vector3d>> box_viewpoint_vec;
    bool can_not_find_path = false;

    int now_id = 0;  // 当前在遍历的边界框的id
    int finish_id = -1; // 遍历完成的边界框的id
    int map_set_use = 0;
    int state = 0;  // ？？　０代表飞回初始位置或者飞去下一个飞入点，１代表飞到飞入点成功

    int pre_state = -1;
    bool pre_it = false;
    bool not_delete = true;  // 代表无人机没有被删除？？也就是还在运行检测中，似乎只对拍照无人机起效果
    bool is_planned = false;
    Eigen::Vector3d planning_point;  // 规划的目标点
    Eigen::Vector3d initial_position;  // 无人机的初始位置
    bool is_leader = false;
    vector<grid_map> map_set;
    grid_map global_map;

    int lowest_bound = 0;
    int highest_bound = 0;

    Eigen::Vector3d grid_size;
    vector<int> path_index;
    double safe_distance = 2.5;
    bool finish_init = false;  // 根据任务分配信息初始化主控制大脑标志位
    bool odom_get = false;
    string namespace_;//当前命名空间
    int Teamid;  // 当前无人机所属哪个队伍（0或者1）
    info_agent info_mannager;
    Eigen::Vector3d now_global_position;  // 点云坐标系下的无人机当前位置
    list<Eigen::Vector3i> global_index_path;
    list<Eigen::Vector3d> global_path;
    Eigen::Vector3d now_gimbal_position;
    nav_msgs::Path path_show;
    Eigen::Matrix3d gimbal_rotation_matrix;
    Eigen::Matrix3d drone_rotation_matrix;
    bool is_transfer = true;  // 代表无人机正在转换状态：飞到初始位置、边界框直接转换等？？
    bool is_exploration = false;
    Eigen::Vector3d target_angle_rpy;
    Eigen::Vector3d target_position;  // 无人机目标位置
    bool get_way_point = false;
    bool finish_first_planning = false;
    vector<string> teammates_name;
    void generate_global_map(string s)//这个函数用于解析输入字符串 s，并生成队伍分组的初始化信息
    {
        vector<string> spilited_str;
        std::istringstream iss(s);
        std::string substring;
        while (std::getline(iss, substring, ','))
        {
            spilited_str.push_back(substring);
        }
        Eigen::Vector3d max_region(stod(spilited_str[3]), stod(spilited_str[4]), stod(spilited_str[5]));  // 覆盖范围的最大xyz
        Eigen::Vector3d min_region(stod(spilited_str[0]), stod(spilited_str[1]), stod(spilited_str[2]));  // 覆盖范围的最小xyz
        vector<vector<string>> teams(2);
        vector<int> size_of_path(2);
        for (int i = 6; i < spilited_str.size(); i++)  // 读取队伍分配情况
        {
            if (spilited_str[i] == "team")
            {
                for (int j = i + 3; j < i + 3 + stoi(spilited_str[i + 2]); j++)
                {
                    teams[stoi(spilited_str[i + 1])].push_back(spilited_str[j]);  // 队伍0或者1的成员组成
                }
                i = i + 2 + stoi(spilited_str[i + 2]);
                continue;
            }
            if (spilited_str[i] == "path_size")
            {
                size_of_path[stoi(spilited_str[i + 1])] = stoi(spilited_str[i + 2]);  // 队伍0或者1负责的路径的数量
                i = i + 2;
                continue;
            }
        }
        for (int i = 0; i < 2; i++)  // 判断当前无人机所属哪个队伍（0或者1）
        {
            for (int j = 0; j < teams[i].size(); j++)
            {
                if ("/" + teams[i][j] == namespace_)
                {
                    Teamid = i;
                    break;
                }
            }
        }
        cout << "TeamID:" << Teamid << endl;
        // 记录当前无人机所属队伍负责的路径的编号，队伍0直接取前n个，队伍1取n+1到最后一个
        if (Teamid == 0)
        {
            for (int i = 0; i < size_of_path[0]; i++)
            {
                path_index.push_back(i + 1);
                cout << i + 1 << endl;
            }
            teammates_name = teams[Teamid];
        }
        else
        {
            for (int i = 0; i < size_of_path[1]; i++)
            {
                path_index.push_back(i + 1 + size_of_path[0]);
                cout << i + 1 + size_of_path[0] << endl;
            }
            teammates_name = teams[Teamid];
        }
        info_mannager = info_agent(teammates_name);  // 队伍信息初始化，这里判断出队长leader信息
    }
    void insert_point(Eigen::Vector3d point_in)  //这个函数用于将一个三维点插入到地图中
    {
        if (map_set.size() > now_id)  // 满足条件则将点同时插入全局地图和边界框地图，猜测：now_id是当前正在遍历的边界框？？
        {
            global_map.insert_point(point_in);
            for (auto &element : map_set)
            {
                element.insert_point(point_in);
            }
        }
        else  // 否则只插入全局地图
        {
            global_map.insert_point(point_in);
        }
    }
    // 检查这个点云是不是邻居无人机上的点
    bool is_Nbr(Eigen::Vector3d test, vector<Eigen::Vector3d> Nbr_point)  //用于检查一个点是否与一组邻近点 (Nbr_point) 中的任何点相邻
    {
        if (Nbr_point.size() == 0)
        {
            return false;
        }
        else
        {
            Eigen::Vector3d collision_box_size = grid_size;
            for (int i = 0; i < Nbr_point.size(); i++)
            {
                Eigen::Vector3d Nbr = Nbr_point[i];
                Eigen::Vector3d diff = Nbr - test;
                if (fabs(diff[0]) <= collision_box_size[0] && fabs(diff[1]) <= collision_box_size[1] && fabs(diff[2]) <= collision_box_size[2])
                {
                    return true;
                }
            }
            return false;
        }
    }
    //这个函数用于构建一个多自由度关节轨迹消息，通常用于控制机器人的位置和姿态
    trajectory_msgs::MultiDOFJointTrajectory position_msg_build(Eigen::Vector3d position, Eigen::Vector3d target, double target_yaw)
    {
        is_planned = true;
        planning_point = target;
        if (fabs(target_yaw) < M_PI / 2)
        {
            target_yaw = 0;
        }
        trajectory_msgs::MultiDOFJointTrajectory trajset_msg;
        trajectory_msgs::MultiDOFJointTrajectoryPoint trajpt_msg;
        trajset_msg.header.frame_id = "world";
        geometry_msgs::Transform transform_msg;
        geometry_msgs::Twist accel_msg, vel_msg;

        Eigen::Vector3d difference = (target - position);
        if (difference.norm() < 2)
        {
            transform_msg.translation.x = target.x();
            transform_msg.translation.y = target.y();
            transform_msg.translation.z = target.z();
            vel_msg.linear.x = 0;
            vel_msg.linear.y = 0;
            vel_msg.linear.z = 0;
        }
        else
        {
            Eigen::Vector3d target_pos = 2 * difference / difference.norm();  // 向量归一化后乘以２，限制了最大速度为２m每秒
            transform_msg.translation.x = 0;
            transform_msg.translation.y = 0;
            transform_msg.translation.z = 0;
            vel_msg.linear.x = target_pos.x();
            vel_msg.linear.y = target_pos.y();
            vel_msg.linear.z = target_pos.z();
        }
        transform_msg.rotation.x = 0;
        transform_msg.rotation.y = 0;
        transform_msg.rotation.z = sinf(target_yaw * 0.5);
        transform_msg.rotation.w = cosf(target_yaw * 0.5);

        trajpt_msg.transforms.push_back(transform_msg);

        accel_msg.linear.x = 0;
        accel_msg.linear.y = 0;
        accel_msg.linear.z = 0;

        trajpt_msg.velocities.push_back(vel_msg);
        trajpt_msg.accelerations.push_back(accel_msg);
        trajset_msg.points.push_back(trajpt_msg);

        trajset_msg.header.frame_id = "world";
        return trajset_msg;
    }
    //这个函数用于构建一个云台控制消息，通常用于控制相机或云台的姿态，基于输入的目标欧拉角 (roll, pitch, yaw) 构建了一个云台消息，包括线速度和角速度信息。
    geometry_msgs::Twist gimbal_msg_build(Eigen::Vector3d target_euler_rpy)
    {
        geometry_msgs::Twist gimbal_msg;
        gimbal_msg.linear.x = 1.0; // setting linear.x to -1.0 enables velocity control mode.
        if (fabs(target_euler_rpy.z()) < M_PI / 2)
        {
            gimbal_msg.linear.y = target_euler_rpy.y(); // if linear.x set to 1.0, linear,y and linear.z are the
            gimbal_msg.linear.z = target_euler_rpy.z(); // target pitch and yaw angle, respectively.
        }
        else
        {
            gimbal_msg.linear.y = target_euler_rpy.y(); // if linear.x set to 1.0, linear,y and linear.z are the
            gimbal_msg.linear.z = 0;                    // target pitch and yaw angle, respectively.
        }
        gimbal_msg.angular.x = 0.0;
        gimbal_msg.angular.y = 0.0; // in velocity control mode, this is the target pitch velocity
        gimbal_msg.angular.z = 0.0; // in velocity control mode, this is the target yaw velocity
        return gimbal_msg;
    }
    Eigen::Matrix3d Rpy2Rot(Eigen::Vector3d rpy)
    {
        Eigen::Matrix3d result = Eigen::Matrix3d::Identity();
        result = Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix() * Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()).toRotationMatrix() * Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()).toRotationMatrix();
        return result;
    }
    Eigen::Vector3d Rot2rpy(Eigen::Matrix3d R)
    {

        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d rpy(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        rpy(0) = r;
        rpy(1) = p;
        rpy(2) = y;

        return rpy;
    }
    string point2str(Eigen::Vector3d point)
    {
        string result;
        result = to_string(point.x()) + "," + to_string(point.y()) + "," + to_string(point.z());
        return result;
    }
    Eigen::Vector3d str2point(string input)
    {
        Eigen::Vector3d result;
        std::vector<string> value;
        boost::split(value, input, boost::is_any_of(","));
        if (value.size() == 3)
        {
            result = Eigen::Vector3d(stod(value[0]), stod(value[1]), stod(value[2]));
        }
        else
        {
            cout << "error use str2point 1" << endl;
        }
        return result;
    }
};

class Agent
{
public:
    Agent(ros::NodeHandlePtr &nh_ptr_)
    : nh_ptr(nh_ptr_)//初始化成员变量的时候直接赋值给nh_ptr
    {
        //定时器触发回调函数初始化，定时调用CallBack函数
        TimerProbeNbr = nh_ptr->createTimer(ros::Duration(1.0 / 10.0), &Agent::TimerProbeNbrCB, this);  // 发布规划的消息
        TimerPlan     = nh_ptr->createTimer(ros::Duration(1.0 / 2.0),  &Agent::TimerPlanCB,     this);  // 定时执行路径规划mm.replan()
        TimerCmdOut   = nh_ptr->createTimer(ros::Duration(1.0 / 10.0), &Agent::TimerCmdOutCB,   this);  // 发布控制指令
        TimerViz      = nh_ptr->createTimer(ros::Duration(1.0 / 1.0),  &Agent::TimerVizCB,      this);  // 发布可视化话题

        //创建订阅任务分配、广播信息对象
        task_sub_ = nh_ptr->subscribe("/task_assign" + nh_ptr->getNamespace(), 10, &Agent::TaskCallback, this);
        com_sub_  = nh_ptr->subscribe("/broadcast" + nh_ptr->getNamespace(), 10, &Agent::ComCallback, this);

        //初始化名为client为一个可以调用 "/create_ppcom_topic" 服务的客户端
        client    = nh_ptr->serviceClient<caric_mission::CreatePPComTopic>("/create_ppcom_topic");
        //创建发布话题对象
        communication_pub_ = nh_ptr->advertise<std_msgs::String>("/broadcast", 10);

        string str = nh_ptr->getNamespace();//获取当前节点的命名空间
        str.erase(0, 1);//删除str中的第一个字符，一般命名空间第一个字符为斜杠
        srv.request.source = str;//指定服务请求的来源或发送者
        srv.request.targets.push_back("all");//指定请求发送的目标为所有目标
        srv.request.topic_name = "/broadcast";//指定服务请求操作的话题
        srv.request.package_name = "std_msgs";//指定消息类型包
        srv.request.message_type = "String";//指定消息类型

        while (!serviceAvailable)//服务不可用时
        {
            serviceAvailable = ros::service::waitForService("/create_ppcom_topic", ros::Duration(10.0));//等待服务回应，最长等待10s
        }
        string result = "Begin";
        while (result != "success lah!")
        {
            client.call(srv);//调用名为 "/create_ppcom_topic" 的ROS服务，并将服务请求对象 srv 传递给服务
            result = srv.response.result;//获取服务响应
            printf(KYEL "%s\n" RESET, result.c_str());
            std::this_thread::sleep_for(chrono::milliseconds(1000));//暂停1秒，再次等待服务
        }
        communication_initialise = true;  // 通信初始化成功标志

        // 两个功能，一是发布初始位置；二是更新自身位置
        odom_sub_        = nh_ptr->subscribe("/ground_truth/odometry", 10, &Agent::OdomCallback, this);//this表示在回调函数中使用当前对象的实例来处理信息，自动分配内存防止内存泄漏
        
        // 更新云台的线速度
        gimbal_sub_      = nh_ptr->subscribe("/firefly/gimbal", 10, &Agent::GimbalCallback, this);

        // 以下订阅的话题的功能是对地图进行更新,同步的回调
        cloud_sub_       = new message_filters::Subscriber<sensor_msgs::PointCloud2>(*nh_ptr, "/cloud_inW", 10); // 世界坐标系下的点云信息
        nbr_sub_         = new message_filters::Subscriber<sensor_msgs::PointCloud2>(*nh_ptr, "/nbr_odom_cloud", 10);  // 当前无人机LOS内其它无人机的里程计信息
        odom_filter_sub_ = new message_filters::Subscriber<nav_msgs::Odometry>(*nh_ptr, "/ground_truth/odometry", 10);  // 当前无人机的里程计信息
        sync_            = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), *cloud_sub_, *nbr_sub_, *odom_filter_sub_);

        // 注册回调函数，boost::bind库来创建一个回调函数对象，它指定了要调用的回调函数以及回调函数的参数，
        // _1, _2, _3：这些是占位符，表示在调用回调函数时，_1 对应于第一个同步的消息，_2 对应于第二个同步的消息，_3 对应于第三个同步的消息。
        sync_->registerCallback(boost::bind(&Agent::MapCallback, this, _1, _2, _3));  

        //发布话题
        motion_pub_     = nh_ptr->advertise<trajectory_msgs::MultiDOFJointTrajectory>("/firefly/command/trajectory", 1);
        gimbal_pub_     = nh_ptr->advertise<geometry_msgs::Twist>("/firefly/command/gimbal", 1);

        // 可视化地图和路径
        map_marker_pub_ = nh_ptr->advertise<visualization_msgs::MarkerArray>("/firefly/map", 1);
        path_pub_       = nh_ptr->advertise<nav_msgs::Path>("/firefly/path_show", 10);

        // 订阅视点话题
        viewpoint_sub_      = nh_ptr->subscribe("/ViewPoints", 10, &Agent::ViewpointCallback, this);
    }

    void MapCallback(const sensor_msgs::PointCloud2ConstPtr &cloud,
                     const sensor_msgs::PointCloud2ConstPtr &Nbr,
                     const nav_msgs::OdometryConstPtr &msg)
    {

        // ensure the map initialization finished
        if (!map_initialise)
        {
            return;
        }
        // ensure time of messages sync 确保时间同步
        if (std::fabs(cloud->header.stamp.toSec() - Nbr->header.stamp.toSec()) > 0.2)
        {
            return;
        }
        mm.update_map(cloud, Nbr, msg);
    }

private:
    int box_i = 0;

    ros::NodeHandlePtr nh_ptr;  // nodehandle for communication 指向ROS节点句柄的指针

    ros::Timer TimerProbeNbr;   // To request updates from neighbours    用于更新邻居节点信息的定时器
    ros::Timer TimerPlan;       // To design a trajectory  用于进行路径规划的定时器
    ros::Timer TimerCmdOut;     // To issue control setpoint to unicon  用于发布控制点setpoint的定时器
    ros::Timer TimerViz;        // To vizualize internal states  用于可视化内部状态的定时器

    // callback Q1
    caric_mission::CreatePPComTopic srv; // This PPcom create for communication between neibors;
    ros::ServiceClient client;           // The client to create ppcom //定义ppcom服务端
    ros::Publisher communication_pub_;   // PPcom publish com  //发布ppcom通信信息
    bool serviceAvailable = false;       // The flag whether the communication service is ready   //初始化服务标志位
    ros::Subscriber task_sub_;           //订阅任务分配信息
    ros::Subscriber com_sub_;            //订阅ppcom通信信息
    string pre_task;

    // callback Q2
    ros::Subscriber odom_sub_;   // Get neibor_info update  //订阅里程计信息
    ros::Subscriber gimbal_sub_; // Get gimbal info update; //订阅万向节云台信息
    ros::Subscriber viewpoint_sub_;  // 订阅视点信息
    // callback Q3
    message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub_; //订阅点云信息
    message_filters::Subscriber<sensor_msgs::PointCloud2> *nbr_sub_; //
    message_filters::Subscriber<nav_msgs::Odometry>       *odom_filter_sub_; //订阅里程计信息
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                                            sensor_msgs::PointCloud2,
                                                            nav_msgs::Odometry> MySyncPolicy;//定义消息同步，同步sensor_msgs::PointCloud2、nav_msgs::Odometry消息
    // // boost::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;
    message_filters::Synchronizer<MySyncPolicy> *sync_;//定义消息同步器

    // callback Q4
    ros::Publisher motion_pub_; // motion command pub //无人机控制指令
    ros::Publisher gimbal_pub_; // motion gimbal pub  //云台控制指令

    // callback Q5
    ros::Publisher map_marker_pub_; //发布地图标记的ROS消息发布器
    ros::Publisher path_pub_; //用于发布路径信息的ROS消息发布器

    mainbrain mm; //具体化示例

    // variable for static map
    vector<Eigen::Vector3d> Nbr_point; //静态全局地图存储向量容器

    bool map_initialise = false;//地图初始化成功标志位
    bool communication_initialise = false;//服务请求初始化成功标志位
    // Callback function

    void TaskCallback(const std_msgs::String msg)
    {

        // cout<<nh_ptr->getNamespace()<<"Task begin"<<endl;
        if (pre_task == msg.data && pre_task != "")
        {
            map_initialise = true;
            return;
        }

        mm = mainbrain(msg.data, nh_ptr->getNamespace(), nh_ptr);  // 初始化主控制大脑
        pre_task = msg.data;
        map_initialise = true;
    }

    void ComCallback(const std_msgs::String msg)
    {
        if(!map_initialise)  // 如果地图没有初始化完成，则直接返回
        {
            return;
        }
        mm.communicate(msg.data);  // 获取msg的内容
        if (!serviceAvailable || !communication_initialise)  // 通信服务不可用或者通信初始化未完成，则返回
        {
            return;
        }
        std_msgs::String msg_map;
        if (mm.get_global_massage(msg_map.data))  // 领队者才会发送这个消息
        {
            communication_pub_.publish(msg_map);  // "mapglobal;Teamid;occupied_num？？;map_cloud_massage？？"
        }
        if (mm.get_local_massage(msg_map.data))  // 领队者才会发送这个消息
        {
            communication_pub_.publish(msg_map);  // "map;Teamid;occupied_num;map_cloud_massage"
        }
        if (mm.get_fly_in_massage(msg_map.data))  // 领队者才会发送这个消息
        {
            communication_pub_.publish(msg_map);  // "flyin;Teamid;fly_in_index(三维坐标);"
        }
        if (mm.get_state_massage(msg_map.data))  // 所有无人机都发送
        {
            communication_pub_.publish(msg_map);  // "state;无人机名(namespace_);state;"
        }

        if (mm.get_viewpoint_massage(msg_map.data))  // 探索者发送生成的view_point信息
        {
            communication_pub_.publish(msg_map);  // "viewpoint;teamid;探索完的box总数；box_id;视点个数；所有视点；box_id;视点个数；所有视点；"
        }
        list<string> msg_list;
        if (mm.get_state_set_msg_list(msg_list))  // 领队者才会发送这个消息，具体内容含义暂时不明？？
        {
            for (auto &str : msg_list)
            {
                std_msgs::String msg_list_item;
                msg_list_item.data = str;
                communication_pub_.publish(msg_list_item);
            }
        }

    }

    // 两个功能，一是发布初始位置；二是更新自身位置
    void OdomCallback(const nav_msgs::OdometryConstPtr &msg)
    {
        if(!map_initialise)  // 如果地图没有完成初始化。task init需要无人机的初始定位信息，由这里发送
        {
            Eigen::Vector3d initial_position = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
            std_msgs::String init_position_msg;
            init_position_msg.data="init_pos;"+nh_ptr->getNamespace()+";"+to_string(initial_position.x())+","+to_string(initial_position.y())+","+to_string(initial_position.z());
            if(communication_initialise)
            {
                communication_pub_.publish(init_position_msg);
            }
            return;
        }

        Eigen::Vector3d my_position = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        Eigen::Matrix3d R = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z).toRotationMatrix();
        mm.update_position(my_position, R);  // 更新自身的位置
    }

    // 更新云台的线速度
    void GimbalCallback(const geometry_msgs::TwistStamped &msg)
    {
        if(!map_initialise)
        {
            return;
        }
        Eigen::Vector3d position = Eigen::Vector3d(msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z);
        mm.update_gimbal(position);
    }

    void ViewpointCallback(const geometry_msgs::PoseArray &msg)
    {
        if(!map_initialise)
        {
            return;
        }
        // std::cout << "mm.get_exploration_finish: " << mm.get_exploration_finish() << "box_i: " << box_i << std::endl;
        if(mm.get_exploration_finish() == box_i) {  // 这里只有探索者能更新
            // 获取视点            
            // 调用mm.update_viewpoint(view, box_i)，将视点存储到mm
            std::cout << "*********************************************************************************get_viewpoint" << std::endl;
            vector<Eigen::Vector3d> vp_vec;
            mm.update_viewpoint(msg, box_i);
            box_i++;
            // box_i++;
        }
    }


    // 发布位置规划的消息
    void TimerProbeNbrCB(const ros::TimerEvent &)
    {
        if (!serviceAvailable || !map_initialise)
        {
            return;
        }
        std_msgs::String msg;
        if (mm.get_position_plan_msg(msg.data))
        {
            communication_pub_.publish(msg);
        }
        else
        {
            return;
        }
        return;
    }

    // 定时执行路径规划
    void TimerPlanCB(const ros::TimerEvent &)
    {
        if (!map_initialise)
        {
            return;
        }
       ////yolo();
        mm.replan();
       ////yolo();
        return;
    }

    // 发布控制指令
    void TimerCmdOutCB(const ros::TimerEvent &)
    {
        if (!map_initialise)
        {
            return;
        }
        
        trajectory_msgs::MultiDOFJointTrajectory position_cmd;
        geometry_msgs::Twist gimbal_msg;
        
        if (mm.get_cmd(position_cmd, gimbal_msg))  // 获取控制指令，包括无人机位置控制和云台控制
        {
            position_cmd.header.stamp = ros::Time::now();
            motion_pub_.publish(position_cmd);
            gimbal_pub_.publish(gimbal_msg);
        }

        return;
    }

    // 发布用于可视化的话题，
    void TimerVizCB(const ros::TimerEvent &)
    {

        if (!map_initialise)
        {
            return;
        }

        map_marker_pub_.publish(mm.Draw_map());
        path_pub_.publish(mm.Draw_Path());

        return;
    }
};
