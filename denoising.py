import numpy as np
import cv2
import os

from utils.data_processing_3d import DataProcessing3D


def img_reader(image_path, shape=None):
    """
    as_type: 'uint8', 'string' or 'byte'
    """
    fid = open(image_path, 'rb')
    data = np.fromfile(fid, np.uint8)
    if shape is not None:
        data = data.reshape(shape)
    return data


def processing(src_name, dst_name, shape, progress=None, suffix='de', techs=['denoise']):
    """
    techs can be 'flip_ver', 'flip_hor', 'flip_vh', 'denoise'
    """
    print(progress, end='\r')
    try:
        img = img_reader(src_name, shape)
    except Exception as e:
        print('Error', e, src_name)
        return
    # Reshape to shape
    dst_name = dst_name[:dst_name.rindex('.')] + '_' + suffix + dst_name[dst_name.rindex('.'):]
    if 'denoise' in techs:
        if os.path.isfile(dst_name):
            # print('Exists, skipping', dst_name)
            return
        else:
            img = DataProcessing3D(img).nl_denoise_3d().image
            res = {'src': src_name, "dst": dst_name, "img": img}
            if not os.path.isdir(dst_name[:dst_name.rindex(os.path.sep)]):
                os.makedirs(dst_name[:dst_name.rindex(os.path.sep)])
            np.array(res['img']).tofile(res['dst'])


if __name__ == '__main__':
    NEW_SUF = '_denoised'
    import pandas as pd

    csvs = [
        './sample_data/datasheets/train.csv',
        './sample_data/datasheets/test.csv'
    ]
    selected = []
    counter = 0
    for cp in csvs:
        csv = pd.read_csv(cp)
        for idx, row in csv.iterrows():
            if not row['filename'].endswith('.img'):
                raise ValueError("Invalid file extension at %d as `%s`" % (idx, row['filename'][row['filename'].rindex('.'):]))
            if 'Optic Disc' == row['area']:
                shape = (200, 1024, 200)
            elif 'Macular' == row['area']:
                shape = (128, 1024, 512)
            else:
                raise ValueError("Invalid area found at %d as `%s`" % (idx, row['area']))
            src = os.path.abspath(os.path.join(row['root'], row['filename']))
            dst = str(src).replace(os.path.abspath(row['root']), os.path.abspath(row['root']) + NEW_SUF)
            counter += 1
            selected.append((src, dst, shape, counter))
        csv['root'] = csv['root'] + NEW_SUF
        csv['filename'] = csv['filename'].str[:-4] + "_de.img"
        csv.to_csv(cp[:cp.rindex('.')] + '_de' + cp[cp.rindex('.'):], index=None)

    selected = list(map(lambda x: (x[0], x[1], x[2], "%d/%d" % (x[3], len(selected))), selected))

    from multiprocessing import Pool
    with Pool(processes=20) as pool:
        pool.starmap(processing, selected, chunksize=1)
