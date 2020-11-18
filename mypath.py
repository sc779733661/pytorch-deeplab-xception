class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'meter_seg_voc':
            # return 'E:\sc\image_data\meter\meter_seg_voc'
            return '/home/y/sc_dev/dilun/image_data/circular_voc'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
