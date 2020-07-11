import os
import pandas as pd


def check(lt):
    """
    check to ensure all file's suffix is '.csv'
    """
    res = []
    for i in range(len(lt)):
        if (lt[i][-3:] != 'csv'): continue
        res.append(lt[i])
    return res


def main():
    # 输入 例如:"D:\python_data\DataSets"
    data_path = input()
    path_list = []
    # 递归获取所有文件
    for dirpath, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            path_list.append(os.path.join(dirpath, filename))

    # check一下suffix
    path_list = check(path_list)

    # 取一个文件作为原始文件进行合并, 设表头为空
    df = pd.read_csv(path_list[0], header=None)
    for i in range(1, len(path_list) - 1, 1):
        df = df.append(pd.read_csv(path_list[i], header=None))
    # 重构索引和表头
    df = df.reset_index(drop=True).rename(
        columns={0: "time", 1: "cross", 2: "direction", 3: "leftflow", 4: "straightflow"})
    print(df)


if __name__ == '__main__':
    main()
