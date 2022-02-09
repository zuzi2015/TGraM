import os


def replaceDirName(rootDir):  # 修改rootDir路径下的文件夹名
    dirs = os.listdir(rootDir)
    for dir in dirs:
        print('oldname is:', dir)  # 输出老的名字

        if dir.split('_')[0] != 'Aircraft':
            temp = 'Ship_' + dir
            oldname = os.path.join(rootDir, dir)  # 老文件夹的名字
            newname = os.path.join(rootDir, temp)  # 新文件夹的名字
            os.rename(oldname, newname)  # 替换


if __name__ == '__main__':
    root = '/workspace/fairmot/src/data/AIR_MOT/images/train/'
    replaceDirName(root)

