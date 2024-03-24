
if (out_of_range(point_local, false))  // 判断是否超过范围
{
    in_my_range = false;
    return;
}

if (is_Nbr(cloud_cloud_point, Nbr_point))  // 检查这个点云是不是邻居无人机机体扫到的，是的话不存储这个点
{
    continue;
}

for (auto &element : map_set)
{
    Eigen::Vector3d point_in_local = rotation_matrix * (point_in - map_global_center);  //将世界坐标系下的点转化为边界框坐标系下的点（以原点为中心）
    if (out_of_range(point_in_local, false))  // 根据map_shape判断是否超过map范围，是的话直接退出函数
    {
        continue;
    }
}