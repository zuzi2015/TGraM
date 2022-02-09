import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


# seq_root = '/data/yfzhang/MOT/JDE/MOT20/images/train'
# label_root = '/data/yfzhang/MOT/JDE/MOT20/labels_with_ids/train'
seq_root = '/workspace/fairmot/src/data/AIR_MOT/images/train/'
label_root = '/workspace/fairmot/src/data/AIR_MOT/labels_with_ids/train'

mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    # seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    # seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    # seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
    seq_width = 1920
    seq_height = 1080

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq, 'img')
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, label, _ in gt:
        # if mark == 0 or not label == 1:
        #     continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            int(label - 1), tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
        # print(label_fpath)
        with open(label_fpath, 'a') as f:
            f.write(label_str)

    # for file in os.listdir(osp.join(seq_root, seq, 'img')):
    #     txt = osp.join(seq_label_root, file.replace('jpg', 'txt'))
    #     if not os.path.exists(txt):
    #         file = open(txt, 'w')



