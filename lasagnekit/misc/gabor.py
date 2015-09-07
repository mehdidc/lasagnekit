import numpy as np
from scipy.optimize import minimize

def gabor_filter_fit(img):
    x_step = 2. / img.shape[1]
    y_step = 2. / img.shape[0]
    x = np.arange(-1, 1, x_step)
    y = np.arange(-1, 1, y_step)
    X , Y = np.meshgrid(x, y)
    def evaluate(params):
        a, b, sigma = params
        return ((img - np.cos(a * X + b *  Y) * np.exp(-(X**2 + Y**2) / (2*sigma**2)))**2).sum()
    x0 = (1, 1, 1)
    result = minimize(evaluate, x0, method='nelder-mead')
    a, b, sigma = result.get("x")
    return dict(a=a, b=b, sigma=sigma)

def gabor_filter_draw(w, h, params):
    a, b, sigma = params.get("a"), params.get("b"), params.get("sigma")
    x_step = 2. / w
    y_step = 2. / h
    x = np.arange(-1, 1, x_step)
    y = np.arange(-1, 1, y_step)
    X , Y = np.meshgrid(x, y)
    Z = np.cos(a * X + b *  Y) * np.exp(-(X**2 + Y**2) / (2*sigma**2))
    return Z
