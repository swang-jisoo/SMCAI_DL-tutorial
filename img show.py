data_dir = 'D:/PycharmProjects/dataverse_files' # 변경 필요
tar = train_tar
mask_tar, img_tar, mask_fname, img_fname = get_tar_fname(data_dir, tar)

m = 0 # range between 0-640
tar_idx = tar.index(os.path.dirname(img_fname[m])) - 1
x = img_tar[tar_idx].extractfile(img_fname[m])
y = mask_tar.extractfile(mask_fname[m])

x = Image.open(x)
y = Image.open(y)

x_trial = np.asarray(x)
y_trial = np.asarray(y) * 50

# input original image
img_x = Image.fromarray(x_trial, 'RGB')
img_x.show()

# mask image in grey scale
img_y = Image.fromarray(y_trial, 'L')
img_y.show()


def get_tar_fname(data_dir, tar):
    """ Get the path to the tar file and the image files inside of it. """
    img_tar, img_fname = [], []
    for t in range(len(tar)):
        tar_fname = os.path.join(data_dir, tar[t] + '.tar.gz')
        if t == 0:
            mask_tar = tarfile.open(tar_fname)
            mask_fname = mask_tar.getnames()[1:]
        else:
            i_tar = tarfile.open(tar_fname)
            img_tar.append(i_tar)
            img_fname += i_tar.getnames()[1:]
    mask_fname, img_fname = match_fname(mask_fname, img_fname)
    return mask_tar, img_tar, mask_fname, img_fname


def match_fname(mask_fname, img_fname):
    """ Match the image file name and the mask file name. """
    mask_fname_match, img_fname_match = [], []
    for m in mask_fname:
        m_fname_base = os.path.splitext(os.path.basename(m))[0]
        m_fname_split = m_fname_base.split('_')
        if 'mask' not in m_fname_split[0]:
            # Skip mask file names starting with '.mask'
            continue
        matcher = '_'.join(m_fname_split[1:4]) + '/' + '_'.join(m_fname_split[1:]) + '.jpg'
        for i in img_fname:
            if matcher == i:
                mask_fname_match.append(m)
                img_fname_match.append(i)
    return mask_fname_match, img_fname_match
