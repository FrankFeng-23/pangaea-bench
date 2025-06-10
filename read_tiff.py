import rasterio

# 指定tiff文件路径
tiff_file = "data/Biomassters/test_features/0a2d4cec_S1_00.tif"

# 打开tiff文件
with rasterio.open(tiff_file) as src:
    # 获取坐标参考系信息
    crs = src.crs
    print("坐标参考系:", crs)
    
    # 获取边界坐标信息（左，下，右，上）
    bounds = src.bounds
    print("边界:", bounds)
    
    # 获取仿射变换参数
    transform = src.transform
    print("仿射变换参数:", transform)
    
    # 获取分辨率（像素大小）
    resolution = src.res
    print("分辨率:", resolution)
