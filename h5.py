import h5py
import numpy as np


def readh5(filepath, mode='r'):
    h5_dict = {}
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            dataset = node[:]
            if dataset.dtype.kind == 'O':  # Handle object (e.g., strings)
                dataset = np.array(dataset, dtype='U')
            else:
                dataset = np.array(dataset)
            h5_dict[name] = dataset

    with h5py.File(filepath, mode) as h5file:
        h5file.visititems(visitor_func)
    return h5_dict


def create_h5(filepath, data_dict, chunks=64):
    
    def create_dataset(h5file, path, values, chunksize):
        if values.dtype.kind == 'U':
            values = values.astype('S')
        h5file.create_dataset(path, data=values, chunks=chunksize)

    with h5py.File(filepath, "w") as h5file:
        for key, values in data_dict.items():
            if key == 'X_preid':
                continue
            if key.startswith(('X_','x_' ,'data_', 'record_')):
                prefix = 'data'
                name_parts = key.split('_')[1:]
            elif key.startswith(('y_','Y_','info_')):
                prefix = 'info'
                name_parts = key.split('_')[1:]
            else:
                name_parts = key.split('_')
                prefix     = ''
                
            name = '_'.join(name_parts)
            values = [np.array(v) if not isinstance(v, np.ndarray) else v for v in values]
            if len(set([v.shape for v in values])) == 1:
                values = np.stack(values, axis=0)
                chunk_size = (min(chunks, len(values)), *values.shape[1:]) if values.ndim > 1 else (min(chunks, len(values)),)
                create_dataset(h5file, f"{prefix}/{name}", values, chunk_size)
            else:
                if 'X_preid' not in data_dict and 'preid' not in data_dict:
                    raise ValueError('X_preid is required to store variable length data item (movies)')
                preid_key = 'X_preid' if 'X_preid' in data_dict else 'preid'
                for preid, v in zip(data_dict[preid_key], values):
                    create_dataset(h5file, f"{prefix}/{name}/{preid}", v, v.shape)