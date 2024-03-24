#include <ros/ros.h>
#include <thread>
#include <iostream>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <visualization_msgs/Marker.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>
#include <string>
#include <chrono>
#include <random>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include "Pcl_Pro.h"
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <Eigen/Dense>
#include <pcl/octree/octree_search.h>
#include "OctreeMap.h"
using namespace std;
void Show_Cube(pcl::visualization::PCLVisualizer::Ptr viewer, OctMap* pp)
{
    int i=0;
    // 添加立方体到可视化对象
    for (const pcl::PointXYZ& point : pp->voxelCentroids) 
    {
        i++;
        double cube_side_length = pp->solution; // 修改为您想要的立方体边长

        // 计算立方体的最小和最大顶点
        pcl::PointXYZ min_pt, max_pt;
        min_pt.x = point.x - 0.5 * cube_side_length;
        min_pt.y = point.y - 0.5 * cube_side_length;
        min_pt.z = point.z - 0.5 * cube_side_length;
        max_pt.x = point.x + 0.5 * cube_side_length;
        max_pt.y = point.y + 0.5 * cube_side_length;
        max_pt.z = point.z + 0.5 * cube_side_length;
        // 创建立方体并设置透明度
        viewer->addCube(min_pt.x, max_pt.x, min_pt.y, max_pt.y, min_pt.z, max_pt.z, 0, 1, 0, std::to_string(i));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, std::to_string(i));
    }
}
int main(int argc, char** argv) 
{
    const char* lkhExecutablePath = "/home/nuc/ws_caric/src/vision_create/3rdprty/LKH";

    // 指定TSP问题描述文件路径
    const char* inputFilePath = "/home/nuc/ws_caric/src/vision_create/3rdprty/prpath.par";
    
    // 指定LKH输出文件路径
    const char* outputFilePath = "output.sol";

    // 构建LKH命令
    std::string command = std::string(lkhExecutablePath) + " " + inputFilePath;

    // 调用LKH求解器来解决TSP问题
    int exitCode = std::system(command.c_str());
    //std::string command = std::string("./LKH prpath.par");
    
    //std::system("cd /home/nuc/ws_caric/src/vision_create/3rdprty/");
    //int exitCode = std::system(command.c_str());
    /*
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    viewer->setBackgroundColor(0, 0, 0);
    pcl::io::loadPCDFile ("/home/nuc/ws_caric/src/vision_create/model/real_pcd.pcd", *cloud); //修改自己pcd文件所在路径
    //滤波
    //pcl::VoxelGrid<pcl::PointXYZ> sor;
    //sor.setInputCloud(cloud);
    //sor.setLeafSize(1,1,1); 
    //sor.filter(*cloud);
    //计算法线
    //pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    //pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    //ne.setInputCloud(cloud);
    //消除重复点云////////////////
    pcl::search::KdTree <pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);
    pcl::Indices pointIdxR;  // 保存每个近邻点的索引
    vector<float> Distance;  // 保存每个近邻点与查找点之间的欧式距离平方
	float radius = 0.5; // 距离阈值，若两点之间的距离为0.000001则认为是重合点
	set<int> remove_index;
	//对cloud中的每个点与邻域内的点进行比较
	for (auto& poiont_i : *cloud)
	{
		if (tree.radiusSearch(poiont_i, radius, pointIdxR, Distance) > 0)
		{
			if (pointIdxR.size() != 1)
			{
				for (size_t i = 1; i < pointIdxR.size(); ++i)
				{
					remove_index.insert(pointIdxR[i]);
				}
			}
		}
	}
    pcl::PointIndices::Ptr outliners(new pcl::PointIndices());
    copy(remove_index.cbegin(), remove_index.cend(), back_inserter(outliners->indices));
    //----------------------提取重复点索引之外的点云-------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	extract.setInputCloud(cloud);
	extract.setIndices(outliners);
	extract.setNegative(true);// 设置为true则表示保存索引之外的点
	extract.filter(*cloud_filtered);
    cout << "原始点云中点的个数为：" << cloud->points.size() << endl;
	cout << "删除的重复点的个数为:" << remove_index.size() << endl;
	cout << "去重之后点的个数为:" << cloud_filtered->points.size() << endl;

    viewer->addPointCloud<pcl::PointXYZ> (cloud_filtered,"sample cloud");//显示点云
    //viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (cloud,normals, 1.5, 0.4, "normals");//显示法向量
    //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");//设置点的大小
    while (!viewer->wasStopped()) 
    {
        // 更新点云显示
        viewer->spinOnce(100);
    }*/
}