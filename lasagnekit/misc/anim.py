import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class ImageAnimation(animation.TimedAnimation):
    def __init__(self, imgs, interval=1000, **kwargs):
        W, H = imgs.shape[1:3]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        self.imgs = imgs
        self.img = ax.imshow(np.random.uniform(imgs.min(), imgs.max(), (W, H)),
                             vmin=imgs.min(), vmax=imgs.max(),
                             interpolation='none',
                             **kwargs)
        animation.TimedAnimation.__init__(self, fig,
                                          interval=interval, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        self.img.set_data(self.imgs[i])
        self._drawn_artists = [self.img]

    def new_frame_seq(self):
        return iter(range(self.imgs.shape[0]))

    def _init_draw(self):
        pass

if __name__ == "__main__":
    imgs = np.random.uniform(size=(20, 10, 10))
    ani = ImageAnimation(imgs, cmap='gray')
    ani.save('test_sub.mp4', fps=10)
    plt.show()
