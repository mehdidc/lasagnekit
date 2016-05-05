

def gram_matrix(x):
    g = (x[:, :, None, :, :] * x[:, None, :, :, :]).sum(axis=(3, 4))
    return g.reshape((x.shape[0], -1))

