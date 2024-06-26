import pandas as pd
import numpy as np

def json_to_csv():
    # 读取 JSON 文件
    json_file = '/root/catkin_ws/src/detic_ros/datasets/metadata/lvis_v1_train_cat_info.json'
    data = pd.read_json(json_file)

    # 将数据写入 CSV 文件
    csv_file = 'lvis_v1_train_cat_info.csv'
    data.to_csv(csv_file, index=False)

    print(f"JSON 数据已成功转换为 CSV 文件: {csv_file}")

def csv_delete_column():
    # 读取 CSV 文件
    csv_file = '/root/catkin_ws/src/detic_ros/node_script/configs/lvis_v1_train_cat_info.csv'
    data = pd.read_csv(csv_file)

    # 保留 "id" 和 "name" 列
    selected_columns = data[['id', 'name']]

    # 将结果写入新的 CSV 文件
    output_csv_file = '/root/catkin_ws/src/detic_ros/node_script/configs/lvis_id_name.csv'
    selected_columns.to_csv(output_csv_file, index=False)

    print(f"已成功保留 'id' 和 'name' 列，并将其写入新的 CSV 文件: {output_csv_file}")

def csv_add_id_rgb():
    # 读取 CSV 文件
    csv_file = '/root/catkin_ws/src/detic_ros/node_script/configs/lvis_detected_class.csv'
    data = pd.read_csv(csv_file)

    # 生成随机的 RGB 颜色值
    num_rows = data.shape[0]
    data['id'] = range(1, len(data)+1)
    data['red'] = np.random.randint(0, 256, size=num_rows)
    data['green'] = np.random.randint(0, 256, size=num_rows)
    data['blue'] = np.random.randint(0, 256, size=num_rows)
    

    # 将结果写入新的 CSV 文件
    output_csv_file = '/root/catkin_ws/src/detic_ros/node_script/configs/lvis_detected.csv'
    data.to_csv(output_csv_file, index=False)

    print(f"已成功添加id和RGB列，并将其写入新的 CSV 文件: {output_csv_file}")

def csv_add_rgb():
    # 读取 CSV 文件
    csv_file = '/root/catkin_ws/src/detic_ros/node_script/configs/lvis_id_name.csv'
    data = pd.read_csv(csv_file)

    # 生成随机的 RGB 颜色值
    num_rows = data.shape[0]
    data['red'] = np.random.randint(0, 256, size=num_rows)
    data['green'] = np.random.randint(0, 256, size=num_rows)
    data['blue'] = np.random.randint(0, 256, size=num_rows)

    # 将结果写入新的 CSV 文件
    output_csv_file = '/root/catkin_ws/src/detic_ros/node_script/configs/lvis_id_name_rgb.csv'
    data.to_csv(output_csv_file, index=False)

    print(f"已成功添加 RGB 列，并将其写入新的 CSV 文件: {output_csv_file}")

def analyze():
    id_to_rgb = np.genfromtxt('/root/catkin_ws/src/detic_ros/node_script/configs/lvis_id_name_rgb.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
    print(np.shape(id_to_rgb))
    print(id_to_rgb[0][2])

csv_add_id_rgb()