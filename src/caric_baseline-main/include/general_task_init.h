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
#include <pcl/kdtree/kdtree_flann.h>

#include "utility.h"
#include <mutex>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <caric_mission/CreatePPComTopic.h>
#include <std_msgs/String.h>

struct position_info{
    Eigen::Vector3d position;  // 位置，无人机的里程计位置是在一个统一的世界坐标系下的
    bool update=false;  // 初始位置是否更新了
};

class Boundingbox
{
public:

    Boundingbox()  // 构造函数1:无参
    {
        center = Eigen::Vector3d(0, 0, 0);  // 中心位置
        volume = 0;  // 体积
        id = -1;  // 对应的编号
    };

    Boundingbox(string str)  // 构造函数2:字符串
    {
        vector<string> spilited_str;
        std::istringstream iss(str);
        std::string substring;
        while (std::getline(iss, substring, ','))  // 分割字符串。在遇到逗号字符之前,一直读取输入流中的字符
        {
            spilited_str.push_back(substring);
        }
        
        int i = 0;
        while (i < 24)
        {
            vertice.push_back(Eigen::Vector3d(stod(spilited_str[i]), stod(spilited_str[i + 1]), stod(spilited_str[i + 2])));
            i = i + 3;
        }
        
        while (i < 33)
        {
            rotation_matrix(i - 24) = stod(spilited_str[i]);
            i++;
        }
        while (i < 36)
        {
            center = Eigen::Vector3d(stod(spilited_str[i]), stod(spilited_str[i + 1]), stod(spilited_str[i + 2]));
            i = i + 3;
        }
        while (i < 39)
        {
            size_vector = Eigen::Vector3d(stod(spilited_str[i]), stod(spilited_str[i + 1]), stod(spilited_str[i + 2]));
            i = i + 3;
        }
        while (i < 42)
        {
            global_in_out.push_back(Eigen::Vector3d(stod(spilited_str[i]), stod(spilited_str[i + 1]), stod(spilited_str[i + 2])));
            i = i + 3;
        }
        while (i < 45)
        {
            global_in_out.push_back(Eigen::Vector3d(stod(spilited_str[i]), stod(spilited_str[i + 1]), stod(spilited_str[i + 2])));
            i = i + 3;
        }
        
        xsize = stod(spilited_str[i]);
        i++;
        ysize = stod(spilited_str[i]);
        i++;
        zsize = stod(spilited_str[i]);
        i++;
        id = stod(spilited_str[i]);
        i++;
        state = stod(spilited_str[i]);
        i++;
        volume = stod(spilited_str[i]);
        i++;
        use_x = stod(spilited_str[i]);
        i++;
        use_y = stod(spilited_str[i]);
        i++;
        use_z = stod(spilited_str[i]);
        i++;
    }

    Boundingbox(const std::vector<Eigen::Vector3d> &vec, int id_in, Eigen::Vector3d &start, Eigen::Vector3d &end, int state_in, bool x, bool y, bool z)
    {
        vertice = vec;
        id = id_in;
        center = Eigen::Vector3d(0, 0, 0);
        for (int index = 0; index < vec.size(); index++)
        {
            center = center + vec[index];
        }
        center = center / 8;
        xsize = (vec[1] - vec[0]).norm();
        ysize = (vec[3] - vec[0]).norm();
        zsize = (vec[4] - vec[0]).norm();
        volume = xsize * ysize * zsize;
        size_vector = Eigen::Vector3d(xsize / 2, ysize / 2, zsize / 2);
        Eigen::Vector3d xaxis, yaxis, zaxis;
        xaxis = (vec[1] - vec[0]).normalized();
        yaxis = (vec[3] - vec[0]).normalized();
        zaxis = (vec[4] - vec[0]).normalized();
        rotation_matrix << xaxis, yaxis, zaxis;
        global_in_out.push_back(start);
        global_in_out.push_back(end);
        state = state_in;
        use_x = x;
        use_y = y;
        use_z = z;
    }
    
    Boundingbox(const std::vector<Eigen::Vector3d> &vec, int id_in)  // 构造函数4：std::vector<Eigen::Vector3d>，ID
    {
        vertice = vec;  
        id = id_in;
        center = Eigen::Vector3d(0, 0, 0);
        // 计算立方体的中心
        for (int index = 0; index < vec.size(); index++)
        {
            center = center + vec[index];
        }
        center = center / 8;
        // 计算立方体在x、y、z上的长度
        xsize = (vec[1] - vec[0]).norm();
        ysize = (vec[3] - vec[0]).norm();
        zsize = (vec[4] - vec[0]).norm();

        volume = xsize * ysize * zsize;  // 计算立方体的体积
        size_vector = Eigen::Vector3d(xsize / 2, ysize / 2, zsize / 2);

        Eigen::Vector3d xaxis, yaxis, zaxis;  // 立方体的机身坐标
        xaxis = (vec[1] - vec[0]).normalized();
        yaxis = (vec[3] - vec[0]).normalized();
        zaxis = (vec[4] - vec[0]).normalized();

        // 计算各个面的中心点
        Eigen::Vector3d xplus, xminus, yplus, yminus, zplus, zminus;
        yminus = (vec[0] + vec[1] + vec[4] + vec[5]) / 4;
        yplus = (vec[2] + vec[3] + vec[6] + vec[7]) / 4;

        xminus = (vec[0] + vec[3] + vec[4] + vec[7]) / 4;
        xplus = (vec[1] + vec[2] + vec[5] + vec[6]) / 4;

        zminus = (vec[0] + vec[1] + vec[2] + vec[3]) / 4;
        zplus = (vec[4] + vec[5] + vec[6] + vec[7]) / 4;

        // 旋转矩阵
        rotation_matrix << xaxis, yaxis, zaxis;

        // 将最长边所在轴上的两个面压入global_in_out，作为路径的进入和出去点
        if ((zsize >= ysize) && (zsize >= xsize))
        {
            use_z = true;
            global_in_out.push_back(zminus);
            global_in_out.push_back(zplus);
        }
        else if ((xsize >= ysize) && (xsize >= zsize))
        {
            if (global_in_out.size() == 0)
            {
                use_x = true;
                global_in_out.push_back(xminus);
                global_in_out.push_back(xplus);
            }
        }
        else if ((ysize >= xsize) && (ysize >= zsize))
        {
            if (global_in_out.size() == 0)
            {
                use_y = true;
                global_in_out.push_back(yminus);
                global_in_out.push_back(yplus);
            }
        }
        else
        {
            if (global_in_out.size() == 0)
            {
                use_z = true;
                global_in_out.push_back(zminus);
                global_in_out.push_back(zplus);
            }
        }
    };

    ~Boundingbox(){};
    
    const Matrix3d getSearchRotation() const
    {
        Eigen::Vector3d axis_rotation_along(0.0, 0.0, 1.0);
        Eigen::Matrix3d transfer_matrix;
        Eigen::Matrix3d result;
        double angle = 0;
        // 以x轴作为最长边的两个面，根据进出的关系，要么绕y轴逆时针(0)旋转90度，或者顺时针(1)旋转90度，其它两个轴同理
        if (use_x)  
        {
            if (state == 0)
            {
                cout << "x+" << endl;
                axis_rotation_along = Eigen::Vector3d(0.0, 1.0, 0.0);
                angle = -M_PI / 2;
            }
            else if (state == 1)
            {
                cout << "x-" << endl;
                axis_rotation_along = Eigen::Vector3d(0.0, 1.0, 0.0);
                angle = M_PI / 2;
            }
            else
            {
                cout << "Error State" << endl;
            }
        }
        else if (use_y)
        {
            if (state == 0)
            {
                cout << "y+" << endl;
                axis_rotation_along = Eigen::Vector3d(1.0, 0.0, 0.0);
                angle = M_PI / 2;
            }
            else if (state == 1)
            {
                cout << "y-" << endl;
                axis_rotation_along = Eigen::Vector3d(1.0, 0.0, 0.0);
                angle = -M_PI / 2;
            }
            else
            {
                cout << "Error State" << endl;
            }
        }
        else if (use_z)
        {
            if (state == 0)
            {
                cout << "z+" << endl;
            }
            else if (state == 1)
            {
                cout << "z-" << endl;
                axis_rotation_along = Eigen::Vector3d(1.0, 0.0, 0.0);
                angle = M_PI;
            }
            else
            {
                cout << "Error State" << endl;
            }
        }
        else
        {
            cout << "noting happen" << endl;
        }

        transfer_matrix = Eigen::AngleAxisd(angle, axis_rotation_along);
        result = transfer_matrix * rotation_matrix.inverse();
        return result;
        
    }

    int getState()
    {
        return state;
    }

    double getVolume() const
    {
        return volume;
    }
    const Vector3d getCenter() const
    {
        return center;
    }
    const Matrix3d getRotation() const
    {
        return rotation_matrix;
    }
    const Vector3d getExtents() const
    {
        return size_vector;
    }
    const Vector3d getRotExtents() const  // 返回局部坐标系下各个边长距离中心点的距离
    {
        Eigen::Vector3d result(0, 0, 0);
        if (use_x)
        {
            if (state == 0)
            {
                // cout<<"x+"<<endl;
                result.x() = size_vector.z();
                result.y() = size_vector.y();
                result.z() = size_vector.x();
            }
            else if (state == 1)
            {
                result.x() = size_vector.z();
                result.y() = size_vector.y();
                result.z() = size_vector.x();
                // cout<<"x-"<<endl;
                // axis_rotation_along=Eigen::Vector3d(0.0,1.0,0.0);
                // angle=M_PI/2;
            }
            else
            {
                cout << "Error State" << endl;
            }
        }
        else if (use_y)
        {
            if (state == 0)
            {
                // cout<<"y+"<<endl;
                result.x() = size_vector.x();
                result.y() = size_vector.z();
                result.z() = size_vector.y();
            }
            else if (state == 1)
            {
                // cout<<"y-"<<endl;
                result.x() = size_vector.x();
                result.y() = size_vector.z();
                result.z() = size_vector.y();
            }
            else
            {
                cout << "Error State" << endl;
            }
        }
        else if (use_z)
        {
            if (state == 0)
            {
                // cout<<"z+"<<endl;
                result = size_vector;
            }
            else if (state == 1)
            {
                // cout<<"z-"<<endl;
                result = size_vector;
            }
            else
            {
                cout << "Error State" << endl;
            }
        }
        else
        {
            cout << "noting happen" << endl;
        }
        return result;
    }
    vector<Eigen::Vector3d> getVertices() const
    {
        return vertice;
    }
    Eigen::Vector3d get_global_in_out(int state) const
    {
        Eigen::Vector3d result = global_in_out[state];
        return result;
    }
    void edit_state(int state_in)
    {
        state = state_in;
    }
    void edit_id()
    {
        id = id + 1;
    }
    int getId()
    {
        return id;
    }

    // 按照比例将边界框拆分成两个
    void generate_start(double scale, Boundingbox &start, Boundingbox &end)
    {
        if (use_z)  // 判断用哪个轴上的两个面
        {
            vector<Eigen::Vector3d> new_vertice_start(8);
            vector<Eigen::Vector3d> new_vertice_end(8);
            if (state == 0)
            {
                new_vertice_start[0] = vertice[0];
                new_vertice_start[1] = vertice[1];
                new_vertice_start[2] = vertice[2];
                new_vertice_start[3] = vertice[3];
                new_vertice_start[4] = vertice[0] + (vertice[4] - vertice[0]) * scale;
                new_vertice_start[5] = vertice[1] + (vertice[5] - vertice[1]) * scale;
                new_vertice_start[6] = vertice[2] + (vertice[6] - vertice[2]) * scale;
                new_vertice_start[7] = vertice[3] + (vertice[7] - vertice[3]) * scale;

                new_vertice_end[0] = vertice[0] + (vertice[4] - vertice[0]) * scale;
                new_vertice_end[1] = vertice[1] + (vertice[5] - vertice[1]) * scale;
                new_vertice_end[2] = vertice[2] + (vertice[6] - vertice[2]) * scale;
                new_vertice_end[3] = vertice[3] + (vertice[7] - vertice[3]) * scale;

                new_vertice_end[4] = vertice[4];
                new_vertice_end[5] = vertice[5];
                new_vertice_end[6] = vertice[6];
                new_vertice_end[7] = vertice[7];

                Eigen::Vector3d new_start = (new_vertice_start[0] + new_vertice_start[1] + new_vertice_start[2] + new_vertice_start[3]) / 4;
                Eigen::Vector3d new_end = (new_vertice_start[4] + new_vertice_start[5] + new_vertice_start[6] + new_vertice_start[7]) / 4;
                Eigen::Vector3d end_start = (new_vertice_end[0] + new_vertice_end[1] + new_vertice_end[2] + new_vertice_end[3]) / 4;
                Eigen::Vector3d end_end = (new_vertice_end[4] + new_vertice_end[5] + new_vertice_end[6] + new_vertice_end[7]) / 4;

                start = Boundingbox(new_vertice_start, id, end_start, end_end, state, use_x, use_y, use_z);
                end = Boundingbox(new_vertice_end, id, end_start, end_end, state, use_x, use_y, use_z);
                return;
            }
            else
            {
                scale = 1 - scale;
                new_vertice_start[0] = vertice[0];
                new_vertice_start[1] = vertice[1];
                new_vertice_start[2] = vertice[2];
                new_vertice_start[3] = vertice[3];
                new_vertice_start[4] = vertice[0] + (vertice[4] - vertice[0]) * scale;
                new_vertice_start[5] = vertice[1] + (vertice[5] - vertice[1]) * scale;
                new_vertice_start[6] = vertice[2] + (vertice[6] - vertice[2]) * scale;
                new_vertice_start[7] = vertice[3] + (vertice[7] - vertice[3]) * scale;

                new_vertice_end[0] = vertice[0] + (vertice[4] - vertice[0]) * scale;
                new_vertice_end[1] = vertice[1] + (vertice[5] - vertice[1]) * scale;
                new_vertice_end[2] = vertice[2] + (vertice[6] - vertice[2]) * scale;
                new_vertice_end[3] = vertice[3] + (vertice[7] - vertice[3]) * scale;

                new_vertice_end[4] = vertice[4];
                new_vertice_end[5] = vertice[5];
                new_vertice_end[6] = vertice[6];
                new_vertice_end[7] = vertice[7];

                Eigen::Vector3d new_start = (new_vertice_start[0] + new_vertice_start[1] + new_vertice_start[2] + new_vertice_start[3]) / 4;
                Eigen::Vector3d new_end = (new_vertice_start[4] + new_vertice_start[5] + new_vertice_start[6] + new_vertice_start[7]) / 4;
                Eigen::Vector3d end_start = (new_vertice_end[0] + new_vertice_end[1] + new_vertice_end[2] + new_vertice_end[3]) / 4;
                Eigen::Vector3d end_end = (new_vertice_end[4] + new_vertice_end[5] + new_vertice_end[6] + new_vertice_end[7]) / 4;

                start = Boundingbox(new_vertice_end, id, end_start, end_end, state, use_x, use_y, use_z);
                end = Boundingbox(new_vertice_start, id, end_start, end_end, state, use_x, use_y, use_z);
                return;
            }
        }
        else if (use_x)
        {
            vector<Eigen::Vector3d> new_vertice_start(8);
            vector<Eigen::Vector3d> new_vertice_end(8);
            if (state == 0)
            {
                new_vertice_start[0] = vertice[0];
                new_vertice_start[3] = vertice[3];
                new_vertice_start[4] = vertice[4];
                new_vertice_start[7] = vertice[7];
                new_vertice_start[1] = vertice[0] + (vertice[1] - vertice[0]) * scale;
                new_vertice_start[2] = vertice[3] + (vertice[2] - vertice[3]) * scale;
                new_vertice_start[5] = vertice[4] + (vertice[5] - vertice[4]) * scale;
                new_vertice_start[6] = vertice[7] + (vertice[6] - vertice[7]) * scale;

                new_vertice_end[0] = vertice[0] + (vertice[1] - vertice[0]) * scale;
                new_vertice_end[3] = vertice[3] + (vertice[2] - vertice[3]) * scale;
                new_vertice_end[4] = vertice[4] + (vertice[5] - vertice[4]) * scale;
                new_vertice_end[7] = vertice[7] + (vertice[6] - vertice[7]) * scale;

                new_vertice_end[1] = vertice[1];
                new_vertice_end[2] = vertice[2];
                new_vertice_end[5] = vertice[5];
                new_vertice_end[6] = vertice[6];

                Eigen::Vector3d new_start = (new_vertice_start[0] + new_vertice_start[3] + new_vertice_start[4] + new_vertice_start[7]) / 4;
                Eigen::Vector3d new_end = (new_vertice_start[1] + new_vertice_start[2] + new_vertice_start[5] + new_vertice_start[6]) / 4;
                Eigen::Vector3d end_start = (new_vertice_end[0] + new_vertice_end[3] + new_vertice_end[4] + new_vertice_end[7]) / 4;
                Eigen::Vector3d end_end = (new_vertice_end[1] + new_vertice_end[2] + new_vertice_end[5] + new_vertice_end[6]) / 4;

                start = Boundingbox(new_vertice_start, id, end_start, end_end, state, use_x, use_y, use_z);
                end = Boundingbox(new_vertice_end, id, end_start, end_end, state, use_x, use_y, use_z);
                return;
            }
            else
            {
                scale = 1 - scale;
                new_vertice_start[0] = vertice[0];
                new_vertice_start[3] = vertice[3];
                new_vertice_start[4] = vertice[4];
                new_vertice_start[7] = vertice[7];
                new_vertice_start[1] = vertice[0] + (vertice[1] - vertice[0]) * scale;
                new_vertice_start[2] = vertice[3] + (vertice[2] - vertice[3]) * scale;
                new_vertice_start[5] = vertice[4] + (vertice[5] - vertice[4]) * scale;
                new_vertice_start[6] = vertice[7] + (vertice[6] - vertice[7]) * scale;

                new_vertice_end[0] = vertice[0] + (vertice[1] - vertice[0]) * scale;
                new_vertice_end[3] = vertice[3] + (vertice[2] - vertice[3]) * scale;
                new_vertice_end[4] = vertice[4] + (vertice[5] - vertice[4]) * scale;
                new_vertice_end[7] = vertice[7] + (vertice[6] - vertice[7]) * scale;

                new_vertice_end[1] = vertice[1];
                new_vertice_end[2] = vertice[2];
                new_vertice_end[5] = vertice[5];
                new_vertice_end[6] = vertice[6];

                Eigen::Vector3d new_start = (new_vertice_start[0] + new_vertice_start[3] + new_vertice_start[4] + new_vertice_start[7]) / 4;
                Eigen::Vector3d new_end = (new_vertice_start[1] + new_vertice_start[2] + new_vertice_start[5] + new_vertice_start[6]) / 4;
                Eigen::Vector3d end_start = (new_vertice_end[0] + new_vertice_end[3] + new_vertice_end[4] + new_vertice_end[7]) / 4;
                Eigen::Vector3d end_end = (new_vertice_end[1] + new_vertice_end[2] + new_vertice_end[5] + new_vertice_end[6]) / 4;

                start = Boundingbox(new_vertice_end, id, end_start, end_end, state, use_x, use_y, use_z);
                end = Boundingbox(new_vertice_start, id, end_start, end_end, state, use_x, use_y, use_z);
                return;
            }
        }
        else
        {
            vector<Eigen::Vector3d> new_vertice_start(8);
            vector<Eigen::Vector3d> new_vertice_end(8);
            if (state == 0)
            {
                new_vertice_start[0] = vertice[0];
                new_vertice_start[1] = vertice[1];
                new_vertice_start[5] = vertice[5];
                new_vertice_start[4] = vertice[4];
                new_vertice_start[3] = vertice[0] + (vertice[3] - vertice[0]) * scale;
                new_vertice_start[2] = vertice[1] + (vertice[2] - vertice[1]) * scale;
                new_vertice_start[6] = vertice[5] + (vertice[6] - vertice[5]) * scale;
                new_vertice_start[7] = vertice[4] + (vertice[7] - vertice[4]) * scale;

                new_vertice_end[0] = vertice[0] + (vertice[3] - vertice[0]) * scale;
                new_vertice_end[1] = vertice[1] + (vertice[2] - vertice[1]) * scale;
                new_vertice_end[5] = vertice[5] + (vertice[6] - vertice[5]) * scale;
                new_vertice_end[4] = vertice[4] + (vertice[7] - vertice[4]) * scale;

                new_vertice_end[3] = vertice[3];
                new_vertice_end[2] = vertice[2];
                new_vertice_end[6] = vertice[6];
                new_vertice_end[7] = vertice[7];

                Eigen::Vector3d new_start = (new_vertice_start[0] + new_vertice_start[1] + new_vertice_start[5] + new_vertice_start[4]) / 4;
                Eigen::Vector3d new_end = (new_vertice_start[3] + new_vertice_start[2] + new_vertice_start[6] + new_vertice_start[7]) / 4;
                Eigen::Vector3d end_start = (new_vertice_end[0] + new_vertice_end[1] + new_vertice_end[5] + new_vertice_end[4]) / 4;
                Eigen::Vector3d end_end = (new_vertice_end[3] + new_vertice_end[2] + new_vertice_end[6] + new_vertice_end[7]) / 4;

                start = Boundingbox(new_vertice_start, id, end_start, end_end, state, use_x, use_y, use_z);
                end = Boundingbox(new_vertice_end, id, end_start, end_end, state, use_x, use_y, use_z);
                return;
            }
            else
            {
                scale = 1 - scale;
                new_vertice_start[0] = vertice[0];
                new_vertice_start[1] = vertice[1];
                new_vertice_start[5] = vertice[5];
                new_vertice_start[4] = vertice[4];
                new_vertice_start[3] = vertice[0] + (vertice[3] - vertice[0]) * scale;
                new_vertice_start[2] = vertice[1] + (vertice[2] - vertice[1]) * scale;
                new_vertice_start[6] = vertice[5] + (vertice[6] - vertice[5]) * scale;
                new_vertice_start[7] = vertice[4] + (vertice[7] - vertice[4]) * scale;

                new_vertice_end[0] = vertice[0] + (vertice[3] - vertice[0]) * scale;
                new_vertice_end[1] = vertice[1] + (vertice[2] - vertice[1]) * scale;
                new_vertice_end[5] = vertice[5] + (vertice[6] - vertice[5]) * scale;
                new_vertice_end[4] = vertice[4] + (vertice[7] - vertice[4]) * scale;

                new_vertice_end[3] = vertice[3];
                new_vertice_end[2] = vertice[2];
                new_vertice_end[6] = vertice[6];
                new_vertice_end[7] = vertice[7];

                Eigen::Vector3d new_start = (new_vertice_start[0] + new_vertice_start[1] + new_vertice_start[5] + new_vertice_start[4]) / 4;
                Eigen::Vector3d new_end = (new_vertice_start[3] + new_vertice_start[2] + new_vertice_start[6] + new_vertice_start[7]) / 4;
                Eigen::Vector3d end_start = (new_vertice_end[0] + new_vertice_end[1] + new_vertice_end[5] + new_vertice_end[4]) / 4;
                Eigen::Vector3d end_end = (new_vertice_end[3] + new_vertice_end[2] + new_vertice_end[6] + new_vertice_end[7]) / 4;

                start = Boundingbox(new_vertice_end, id, end_start, end_end, state, use_x, use_y, use_z);
                end = Boundingbox(new_vertice_start, id, end_start, end_end, state, use_x, use_y, use_z);
                return;
            }
        }
    }

    string generate_string_version() const
    {
        string result = "";
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                result = result + to_string(vertice[i][j]) + ",";
            }
        }
        for (int i = 0; i < 9; i++)
        {
            result = result + to_string(rotation_matrix(i)) + ",";
        }
        for (int j = 0; j < 3; j++)
        {
            result = result + to_string(center[j]) + ",";
        }
        for (int j = 0; j < 3; j++)
        {
            result = result + to_string(size_vector[j]) + ",";
        }

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                result = result + to_string(global_in_out[i][j]) + ",";
            }
        }
        result = result + to_string(xsize) + ",";
        result = result + to_string(ysize) + ",";
        result = result + to_string(zsize) + ",";
        result = result + to_string(id) + ",";
        result = result + to_string(state) + ",";
        result = result + to_string(volume) + ",";
        result = result + to_string(use_x) + ",";
        result = result + to_string(use_y) + ",";
        result = result + to_string(use_z) + ",";
        // cout<<result<<endl;
        return result;
    }
    
private:
    vector<Eigen::Vector3d> vertice;  // 顶点
    double volume = 0;  // 体积
    Eigen::Vector3d center;  // 中心
    Eigen::Matrix3d rotation_matrix;  // 旋转矩阵
    Eigen::Vector3d size_vector;  // 三维向量，其中包含 x 轴、y 轴和 z 轴上的一半距离，用处？？
    double xsize, ysize, zsize;  // 立方体在x、y、z上的长度
    int id = 0;  // 编号
    vector<Eigen::Vector3d> global_in_out;  // 最长边所在轴对应的两个面中心点，作为路径的进入和出去点
    int state = 0;  // 0对应靠近原点的面是进入面，1对应远离原点的面是进入面
    bool use_x = false;  // 判断使用哪个轴上的两个面
    bool use_y = false;  
    bool use_z = false;  
};
// --------------------------------------------------------------------

class gcs_task_assign{
    public:
        gcs_task_assign(){

        }
        gcs_task_assign(ros::NodeHandlePtr &nh_ptr_)
        : nh_ptr_(nh_ptr_)
        {
            namelist = {"/jurong", "/raffles", "/changi", "/sentosa", "/nanyang"};
            // 每架无人机对应的位置信息map
            position_pair["/jurong"] = {Eigen::Vector3d(0, 0, 1), false};
            position_pair["/raffles"] = {Eigen::Vector3d(0, 0, 1), false};
            position_pair["/changi"] = {Eigen::Vector3d(0, 0, 1), false};
            position_pair["/sentosa"] = {Eigen::Vector3d(0, 0, 1), false};
            position_pair["/nanyang"]={Eigen::Vector3d(0, 0, 1), false};

            // 给最大值赋double类型数据的最小值，这样确保能更新成功
            xmax = -std::numeric_limits<double>::max();
            ymax = -std::numeric_limits<double>::max();
            zmax = -std::numeric_limits<double>::max();
            xmin = std::numeric_limits<double>::max();
            ymin = std::numeric_limits<double>::max();
            zmin = std::numeric_limits<double>::max();

            // 注册通信，gcs发布“/task_assign”到ppcom_rounter，ppcom_rounter把消息广播到 “/task_assign/每个无人机id”
            client = nh_ptr_->serviceClient<caric_mission::CreatePPComTopic>("create_ppcom_topic");  
            cmd_pub_ = nh_ptr_->advertise<std_msgs::String>("/task_assign", 10);
            srv.request.source = "gcs";  // 数据的源头
            srv.request.targets.push_back("all");  // 目标对象，这里是全部
            srv.request.topic_name = "/task_assign";
            srv.request.package_name = "std_msgs";
            srv.request.message_type = "String";  // 类型是string类型
            while (!serviceAvailable)
            {
                serviceAvailable = ros::service::waitForService("create_ppcom_topic", ros::Duration(10.0));
            }
            client.call(srv);  // 调用服务
            
            // 订阅边界框话题，/gcs/bounding_box_vertices是根据box_description.yaml构建的话题
            bbox_sub_ = nh_ptr_->subscribe<sensor_msgs::PointCloud>("/gcs/bounding_box_vertices", 10, &gcs_task_assign::bboxCallback, this);
            // 用于更新无人机第一次的位置，每台无人机都发布消息到/broadcast上，ppcom_rounter接受后将其发布到/broadcast/gcs上
            agent_position_sub_=nh_ptr_->subscribe<std_msgs::String>("/broadcast/gcs", 10, &gcs_task_assign::positionCallback, this);

            // 定时器回调函数，判断无人机的初始位置是否更新完毕，是的话agent_info_get变量置true可以进入边界框的任务分配
            Agent_ensure_Timer=nh_ptr_->createTimer(ros::Duration(1.0 / 10.0),  &gcs_task_assign::TimerEnsureCB,     this);
            // 定时器回调函数，定时发布任务
            Massage_publish_Timer=nh_ptr_->createTimer(ros::Duration(1.0 / 10.0),  &gcs_task_assign::TimerMessageCB,     this);
        }
    private:
        ros::NodeHandlePtr nh_ptr_;  // 句柄指针
        
        //communication related param
        bool serviceAvailable = false;  // 用于判断服务是否可用
        caric_mission::CreatePPComTopic srv;  // 调用create_ppcom_topic的服务数据
        ros::ServiceClient client;

        ros::Publisher cmd_pub_;  // 发布任务的话题

        ros::Subscriber agent_position_sub_;  // 智能体的位置订阅
        ros::Subscriber bbox_sub_;  // 边界框订阅
        
        ros::Timer Agent_ensure_Timer;
        ros::Timer Massage_publish_Timer;


        bool get_bbox=false;
        bool finish_massage_generate=false;
        bool agent_info_get=false;

        list<string> namelist;  // 每个无人机的名字列表
        map<string,position_info> position_pair;  // 无人机和位置信息的map
        double update_time;  // 记录无人机最后的第一次更新位置的时间

        double volumn_total=0;
        bool finish_bbox_record = false;
        vector<Boundingbox> box_set; // 存储边界框点组和编号
        
        vector<int> box_index;
        vector<int> state_vec;

        // 统计边界框覆盖到的xyz最值
        double xmax;
        double ymax;
        double zmax;
        double xmin;
        double ymin;
        double zmin;

        vector<vector<string>> team_info;  // 存储无人机团队成员，每个探险无人机带一队。即代码介绍里面的基于探索者数量的分组策略
        vector<vector<Boundingbox>> output_path;

        string result;  // 存储任务分配的结果

        // 边界框回调函数
        void bboxCallback(const sensor_msgs::PointCloud::ConstPtr &msg)
        {
            if(finish_bbox_record||!agent_info_get)  // 如果完成了边界框的记录，或者无人机的初始位置没有更新，直接return
            {
                // cout<<"now agent get"<<endl;
                // for(auto& name:namelist)
                // {
                //     if(position_pair[name].update){
                //         cout<<name<<endl;
                //     }
                // }
                return;
            }
            if(finish_massage_generate)  // 如果完成任务消息的生成，直接返回，所以这个callback只需成功运行一次
            {
                return;
            }

            sensor_msgs::PointCloud cloud = *msg;
            int num_points = cloud.points.size();  // 点云的总数
            if (num_points % 8 == 0 && num_points > 8 * box_set.size()) // 点云必须是8的整数倍
            {
                volumn_total = 0;  // 边界框总体积
                int num_box = num_points / 8;  // 统计有多少个边界框，每个边界框有八个点（这样构成一个立方体）
                for (int i = 0; i < num_box; i++)
                {
                    vector<Eigen::Vector3d> point_vec;
                    for (int j = 0; j < 8; j++)  
                    {
                        // 统计所有边界框覆盖到的区域的最值
                        if (cloud.points[8 * i + j].x > xmax)
                        {
                            xmax = cloud.points[8 * i + j].x;
                        }
                        if (cloud.points[8 * i + j].x < xmin)
                        {
                            xmin = cloud.points[8 * i + j].x;
                        }
                        if (cloud.points[8 * i + j].y > ymax)
                        {
                            ymax = cloud.points[8 * i + j].y;
                        }
                        if (cloud.points[8 * i + j].y < ymin)
                        {
                            ymin = cloud.points[8 * i + j].y;
                        }
                        if (cloud.points[8 * i + j].z > zmax)
                        {
                            zmax = cloud.points[8 * i + j].z;
                        }
                        if (cloud.points[8 * i + j].z < zmin)
                        {
                            zmin = cloud.points[8 * i + j].z;
                        }

                        // 将该组的8个点都存起来
                        point_vec.push_back(Eigen::Vector3d(cloud.points[8 * i + j].x, cloud.points[8 * i + j].y, cloud.points[8 * i + j].z));
                    }
                    box_set.push_back(Boundingbox(point_vec, i)); // 利用8点向量和编号构建边界框并存储
                    volumn_total += box_set[i].getVolume();  // 更新总的体积，Boundingbox里面计算了每个立方体的体积
                    point_vec.clear();
                }
                finish_bbox_record = true;
                //Team allocate 队伍分配
                Team_allocate();  // 按照距离将摄影无人机归类给探索无人机

                //Use best first to generate the global trajactory 
                /*
                使用best first方法生成全局轨迹
                每一步会选择离目标点最近的结点。
                Best-First 算法是一种贪心算法，一般通过定义一个启发式函数来引导着向离目标更近的方向前进。
                */
                Best_first_search();
                //Clip the best path 剪枝路径，分给两组无人机
                Clip_the_task();
                //Generate the massage 生成消息
                generate_massage();


            }
            else
            {
                return;
            }
            return;
        }

        void positionCallback(const std_msgs::String msg)
        {
            istringstream str(msg.data);  // 创建一个字符串流对象
            string type;  // 存储消息中的类型信息
            getline(str,type,';');  // 读取类型数据，以";"分割的第一个字符串
            if(type=="init_pos")
            {
                string origin;  // 存储消息中的origin，指的应该是无人机的ID
                getline(str,origin,';');
                string position_str;  // 该无人机的位置
                getline(str,position_str,';');
                // 如果这个无人机的位置没有更新过，则更新
                if(!position_pair[origin].update){
                    update_time=ros::Time::now().toSec();  // 记录更新时间
                    position_pair[origin].position=str2point(position_str);
                    position_pair[origin].update=true;  // 该无人机已经更新过初始位置了
                }
            }
        }

        // 计时器，如果已经记录到一些无人机的初始位置，并且当前时间和上一次更新无人机的时间超出10s，则停止记录初始位置，进入边界框的任务分配
        void TimerEnsureCB(const ros::TimerEvent &)
        {   
            if(agent_info_get)
            {
                return;
            }
            bool finish_agent_record=false; // 确保至少有一辆无人机的信息更新成功
            for(auto& name:namelist)
            {
                if(position_pair[name].update){
                    finish_agent_record=true;
                    break;
                }
            }
            if(!finish_agent_record)
            {
                return;
            }
            double time_now=ros::Time::now().toSec();

            if(fabs(time_now-update_time)>10)  // 完成智能体的记录，并且超过十秒没有更新，则代表智能体信息更新完成，可以开始任务分配
            {
                agent_info_get=true;
            }
        }

        // 这个定时器用于发布任务
        void TimerMessageCB(const ros::TimerEvent &){
            if(finish_massage_generate) // 如果完成了路径的生成
            {
                std_msgs::String task;
                task.data=result;
                cmd_pub_.publish(task);

            }
        }

        // 把字符串转换为点
        Eigen::Vector3d str2point(string input)
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

        // 按照距离将摄影无人机归类给探索无人机
        void Team_allocate()
        {
            team_info=vector<vector<string>>(2);  // 初始化二维向量，包含两个子向量

            // 如果/jurong的初始数据更新了，将其名字压入第一个子向量中。jurong和raffles都是探索无人机
            if(position_pair["/jurong"].update){ 
                team_info[0].push_back("/jurong");
            }
            // 如果/raffles的初始数据更新了，将其名字压入第一个子向量中
            if(position_pair["/raffles"].update){
                team_info[1].push_back("/raffles");
            }

            // 遍历所有无人机
            for(auto name:namelist){
                if(name=="/jurong"||name=="/raffles"){ // 如果是探索无人机就跳过
                    continue;
                }
                if(!position_pair[name].update){  // 如果初始位置没有更新的也跳过
                    continue;
                }
                if(team_info[0].size()>0&&team_info[1].size()>0){  // 如果两个探索无人机的位置都更新了
                    double jurong_dis=(position_pair[name].position-position_pair[team_info[0][0]].position).norm();
                    double raffles_dis=(position_pair[name].position-position_pair[team_info[1][0]].position).norm();
                    if(jurong_dis<=raffles_dis){  // 比较每个摄影师距离和那个近，近的放一起
                        team_info[0].push_back(name);
                    }else{
                        team_info[1].push_back(name);
                    }
                    continue;
                }else if(team_info[0].size()>0){  // 只有一架探索无人机，直接压入
                    team_info[0].push_back(name);
                    continue;
                }else if(team_info[1].size()>0){
                    team_info[1].push_back(name);
                    continue;
                }else{
                    continue;
                }
            }
            // 输出归类的结果
            for(int i=0;i<2;i++){
                cout<<"team"<<i<<": "<<endl;
                for(int j=0;j<team_info[i].size();j++){
                    cout<<team_info[i][j]<<endl;
                }
                cout<<"team"<<i<<" finished"<<endl;
            }

            
        }
    
    // best first方法生成全局轨迹
    void Best_first_search(){
        Eigen::Vector3d start_point;
        // 判断几个探索无人机有效。将其中一个的位置赋给start_point，默认是jurong
        if(team_info[0].size()>0&&team_info[1].size()>0){
            start_point=position_pair[team_info[0][0]].position;
        }else if(team_info[0].size()>0&&team_info[1].size()==0){
            start_point=position_pair[team_info[0][0]].position;
        }else if(team_info[0].size()==0&&team_info[1].size()>0){
            start_point=position_pair[team_info[1][0]].position;
        }

        /*
        1、以start_point为起点，找到一个距离该点最近的边界框，用global_in_out两个面中心进行对比，找到最近的一个面作为进去的边界框。
        2、将边界框的另外一个面中心作为出口，将其更新为start_point。
        3、重新第1步，直到所有边界框都遍历完，存储排好序后的边界框，这样生成了一条路径,后面将路径剪成两段飞给两组无人机
        */
        while(box_index.size()<box_set.size()){
            int index;  // 记录是哪个边界框
            int state;  // 记录从哪个面进去，0为靠近原点的面，1为远离原点的面
            double mindis=std::numeric_limits<double>::max();  // 初始化最小距离

            // 将所有边界框进行排序
            for(int i=0;i<box_set.size();i++){
                // 如果box_index里面已经有了第i个边界框，则继续
                if(find(box_index.begin(),box_index.end(),i)!=box_index.end()&&box_index.size()>0){
                    continue;
                }

                // 对每个边界框，计算它的位置（两个面中心点）和start_point的距离
                for(int j=0;j<2;j++){  
                    double dis=(box_set[i].get_global_in_out(j)-start_point).norm();
                    if(dis<mindis){  // 找到更小的距离则更新state和index
                        mindis=dis;
                        state=j;
                        index=i;
                    }
                }
            }
            box_index.push_back(index);  
            state_vec.push_back(state);  
            start_point=box_set[index].get_global_in_out(1-state);  // 更新起始点
        }
        // 打印路径，这个案例一共四个边界框
        cout<<"Path:"<<endl;
        for(int i=0;i<box_set.size();i++)
        {
            cout<<"bbox index:"<<box_index[i];
            cout<<" state:"<<state_vec[i]<<endl;
        }

    }

    // 分配边界框
    void Clip_the_task()
    {
        double volum_path = 0;
        int clip_index = -1;
        bool clip_in_boundingbox = false;

        Boundingbox replaced_in;
        Boundingbox replaced_out;
        vector<Boundingbox> BFS_result;  // 存储生成的路径
        // 将排列后的结果（边界框加面中心点）存入BFS_result
        for(int i=0;i<box_index.size();i++)
        {
            BFS_result.push_back(box_set[box_index[i]]);
            BFS_result[i].edit_state(state_vec[i]);
        }

        output_path.resize(2);

        // 判断哪架探索无人机有效，如果只有一架有效，则直接将整个排序好的边界框分给它
        if(team_info[0].size()==0&&team_info[1].size()>0)
        {
            output_path[1]=BFS_result;
            return;

        }else if(team_info[0].size()>0&&team_info[1].size()==0)
        {
            output_path[0]=BFS_result;
            return;
        }else if(team_info[0].size()>0&&team_info[1].size()>0)  // 如果两个探索无人机都有效
        {
            double factor=double(team_info[0].size())/double(team_info[0].size()+team_info[1].size());  // 计算探险无人机0组的数量占总的多少
            cout<<"factor"<<to_string(factor)<<endl;

            // 根据无人机数量的比例和边界框体积的比例，寻找分界点索引clip_index
            for(int j=0;j<BFS_result.size();j++)
            {
                volum_path +=BFS_result[j].getVolume();
                if (abs(volum_path / volumn_total - factor) <= 0.05)  // 如果两组无人机数量的比例和边界框体积的比例比较接近，则不需要拆分分界的边界框
                {
                    clip_index = j;  
                    clip_in_boundingbox = false;
                    break;
                }
                if (volum_path / volumn_total - factor > 0.05)  // 如果两组无人机数量的比例和边界框体积的比例超出一个阈值，则需要拆分分界的边界框
                {
                
                    clip_index = j;  // 标记分界的边界框是第几个
                    clip_in_boundingbox = true;  // 分界完成
                    double volum_more = volum_path - volumn_total * factor;  // 计算多出来的体积
                    double scaled_param = 1 - volum_more / (BFS_result[j].getVolume());  // 计算一个缩放参数
                    BFS_result[j].generate_start(scaled_param, replaced_in, replaced_out);  // 拆分边界框
                }
            }

            // 分配路径，两种情况，分界边界框是否需要拆分
            if(clip_in_boundingbox){
                for (int i = 0; i < BFS_result.size(); i++)
                {
                    if (i < clip_index)
                    {
                        output_path[0].push_back(BFS_result[i]);
                    }
                    else if (i == clip_index)
                    {
                        output_path[0].push_back(replaced_in);
                        output_path[1].push_back(replaced_out);
                    }
                    else
                    {
                        output_path[1].push_back(BFS_result[i]);
                    }
                }
                return;
            }else{
                for (int i = 0; i < BFS_result.size(); i++)
                {
                    if (i <= clip_index)
                    {
                        output_path[0].push_back(BFS_result[i]);
                    }
                    else
                    {
                        output_path[1].push_back(BFS_result[i]);
                    }
                } 
                return;       

            }
                
        }

    }

    // 构建边界框分配的最终消息
    void generate_massage()
    {
        result="";
        // 统计所有边界框覆盖到的区域的最值再加上一个松弛量，将区域放大一部分，作用需要在得到消息的处理中解读？？
        double loose_length=6.0;
        result = result + to_string(xmin - loose_length) + ",";
        result = result + to_string(ymin - loose_length) + ",";
        result = result + to_string(zmin - loose_length) + ",";

        result = result + to_string(xmax + loose_length) + ",";
        result = result + to_string(ymax + loose_length) + ",";
        result = result + to_string(zmax + loose_length) + ",";

        // 构建不同组别的无人机成员信息和负责的区域
        result=result+"team"+",0,"+to_string(team_info[0].size())+",";

        for(int i=0;i<team_info[0].size();i++){
            string str=team_info[0][i];
            str.erase(0, 1);  // 去掉"/"前缀
            result=result+str+",";
        }
        result=result+"team"+",1,"+to_string(team_info[1].size())+",";
        for(int i=0;i<team_info[1].size();i++){
            string str=team_info[1][i];
            str.erase(0, 1);
            result=result+str+",";
        }
        result = result + "path_size" + "," + "0" + "," + to_string(output_path[0].size()) + ",";
        result = result + "path_size" + "," + "1" + "," + to_string(output_path[1].size()) + ",";
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < output_path[i].size(); j++)
            {
                result = result + ";";
                result = result + output_path[i][j].generate_string_version();
            }
        }
        finish_massage_generate=true;

    }


};