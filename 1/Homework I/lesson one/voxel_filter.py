# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import random
# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
class voxel(object):
    def __init__(self):
        self.points = np.array(np.array([],dtype=np.float),dtype=np.float)
        self.mean = np.array(np.array([],dtype=np.float),dtype=np.float)
        self.inited = False
    def pushpoint(self, point):
        if(self.inited==False):
            self.points = np.array(np.array([[point]]))
            self.mean = point
            self.inited = True
        else:
            self.mean = (self.mean * self.points.shape[0] + point)/(self.points.shape[0]+1)
            self.points = np.insert(self.points, 0, values=np.array(np.array(point)), axis=0)
        return self.mean
    def randselect(self):
        randnum = random.randrange(0,self.points.shape[0])
        print("rand: ",randnum)
        return self.points[randnum, :]
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    pre_filtered = np.unique(np.array(point_cloud), axis=0)
    print("size: ", pre_filtered.shape)
    xyz_max = np.max(pre_filtered, axis=0)
    print("xyz_max: ",xyz_max)
    xyz_min = np.min(pre_filtered, axis=0)
    print("xyz_min: ", xyz_min)
    D = np.ceil((xyz_max - xyz_min) / leaf_size)
    if(D[0]==0 or D[1]==0 or D[2]==0):#点云成平面切法向量与轴平行时voxel grid应该使用2维降采样
        print("need 2d down sample")
        return
    voxel_vector = []
    for i in range((D[0] * D[1] * D[2]).astype(np.int64)):
        voxel_vector.append(voxel())#用类对象构造容器
    print(voxel_vector)
    num_vector = [0]*(D[0] * D[1] * D[2]).astype(np.int64)
    for i in range(pre_filtered.shape[0]):
        h = np.floor((pre_filtered[i] - xyz_min) / leaf_size)
        index = np.dot(h.reshape(1,3),np.array([1,D[0],D[0]*D[1]]))
        voxel_vector[int(index.astype(np.int32)[0])].pushpoint(pre_filtered[i])

    for i in range(len(voxel_vector)):
        if(voxel_vector[i].mean.shape[0]!=0):
            print("================")
            print(i, " voxel point size: ", voxel_vector[i].points.shape[0])
            print(i, " voxel mean: ", voxel_vector[i].mean.shape)
            print(i, " voxel rand: ", voxel_vector[i].randselect())
            filtered_points.append(voxel_vector[i].randselect()[0])
            # filtered_points.append(voxel_vector[i].mean)
            print("+++++++++++++++++")
        # print("voxel mean: ",voxel_object.mean)
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = "/home/dzy/work/pointcloud/ModelNet40/airplane/train/airplane_0001.ply"
    point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 10.0)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
