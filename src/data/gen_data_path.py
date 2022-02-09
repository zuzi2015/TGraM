import os


def write(seq_root, label_root, out_root):
    seqs = [s for s in os.listdir(seq_root)]
    for seq in seqs:
        seq_label_root = os.path.join(label_root, seq, 'img')
        for file in os.listdir(os.path.join(seq_root, seq, 'img')):
            txt = os.path.join(seq_label_root, file.replace('jpg', 'txt'))
            if os.path.exists(txt):
                filepath = os.path.join('AIR_MOT/images/train', seq, 'img', file)
                print(filepath)
                with open(out_root, 'a+') as f:
                    f.write(filepath + '\r\n')


if __name__ == '__main__':
    out_path = 'temp.txt'
    seq_root = '/workspace/fairmot/src/data/AIR_MOT/images/train/'
    label_root = '/workspace/fairmot/src/data/AIR_MOT/labels_with_ids/train'
    write(seq_root, label_root, out_path)
