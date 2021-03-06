import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

import represilator as r
from image_annotated_heatmap import annotate_heatmap, heatmap


def analyse_diffusion_rate_and_density(size, number_of_examples=150):
    """
    Analiza tocke 1

    Kakšen vpliv ima na oscilacije gostota populacije celic, hitrost difuzije in velikost prostora?

    D [10^-2, 10^2]
    velikost prostora: 10x10
    gostota populacije: [0.01, 1]

    """
    x = []
    y = []
    result = []
    example_num = 0

    """Vpliv hitrosti difuzije na oscilacije"""
    for density in np.linspace(start=.1, stop=1, num=number_of_examples):
        tmp = []
        for d in np.linspace(start=.01, stop=1, num=number_of_examples):  # .1 1
            rep = r.Repressilator_S_PDE()
            rep.set_params(D1=d, density=density, size=size)
            # check if model oscillates
            [osc, freq, period, amplitude, damped] = rep.run()
            print("#%d | density: %f, | oscillates: %d, frequency: %f, period: %f, amplitude: %f" % (
                example_num, density, osc, freq, period, amplitude))

            example_num += 1
            del rep

            y.append(d)
            x.append(density)
            tmp.append(osc)
        result.append(tmp)

    result = np.array(result)
    print(result)

    x = np.unique(x)
    x = list(np.around(x, decimals=2))
    print(x)

    y = np.unique(y)
    y = list(np.around(y, decimals=2))
    print(y)

    # draw heat map
    make_heatmap(x, y, result)


def analyse_diffusion_rate(size=10, number_of_examples=150):
    """
    Vpliv hitrosti difuzije na oscilacije pri podani velikosti prostora
    :return:
    """
    example_num = 0
    x_per = []
    y_per = []

    x_amp = []
    y_amp = []

    for d in np.linspace(start=.01, stop=.8, num=number_of_examples):
        rep = r.Repressilator_S_PDE()
        rep.set_params(D1=d, density=0.4, size=size)
        [osc, freq, period, amplitude, damped] = rep.run()

        print("#%d | d: %f, | oscillates: %d, frequency: %f, period: %f, amplitude: %f" % (
            example_num, d, osc, freq, period, amplitude))

        if osc:
            # period graph
            x_per.append(period)
            y_per.append(d)
            # amplitude graph
            x_amp.append(amplitude)
            y_amp.append(d)

        example_num += 1
        del rep

    # draw graph
    plt.subplot(2, 1, 1)
    plt.title('Vpliv hitrosti difuzije na amplitudo v prostoru %dx%d' % (size, size))
    plt.plot(y_amp, x_amp)
    plt.ylabel('Amplituda')
    plt.xlabel('Hitrost difuzije')

    plt.subplot(2, 1, 2)
    plt.title('Vpliv hitrosti difuzije na periodo v prostoru %dx%d' % (size, size))
    plt.plot(y_per, x_per)
    plt.ylabel('Perioda')
    plt.xlabel('Hitrost difuzije')
    plt.show()


def analyse_density(size=10, number_of_examples=150):
    """
    Vpliv gostote celic na oscilacije pri podani velikosti prostora
    :return:
    """
    example_num = 0
    x_per = []
    y_per = []

    x_amp = []
    y_amp = []

    x_freq = []
    y_freq = []

    for density in np.linspace(start=.1, stop=1, num=number_of_examples):
        rep = r.Repressilator_S_PDE()
        rep.set_params(density=density, size=size)
        [osc, freq, period, amplitude, damped] = rep.run()

        print("#%d | density: %f, | oscillates: %d, frequency: %f, period: %f, amplitude: %f" % (
            example_num, density, osc, freq, period, amplitude))

        if osc:
            # period graph
            x_per.append(period)
            y_per.append(density)
            # amplitude graph
            x_amp.append(amplitude)
            y_amp.append(density)

            x_freq.append(freq)
            y_freq.append(density)

        example_num += 1
        del rep

    plt.subplot(3, 1, 1)
    plt.title('Vpliv gostote celic na amplitudo v prostoru %dx%d' % (size, size))
    # x_smooth = np.linspace(np.array(x_amp).max(), np.array(x_amp).min(), number_of_examples)
    # y_smooth = spline(x_amp, y_amp, x_smooth)
    y_smooth = gaussian_filter1d(y_amp, sigma=2)
    plt.plot(y_smooth, x_amp)
    plt.ylabel('Amplituda')
    plt.xlabel('Gostota')

    plt.subplot(3, 1, 2)
    plt.title('Vpliv gostote celic na periodo v prostoru %dx%d' % (size, size))
    y_smooth = gaussian_filter1d(y_per, sigma=2)
    plt.plot(y_smooth, x_per)
    plt.ylabel('Perioda')
    plt.xlabel('Gostota')

    plt.subplot(3, 1, 3)
    plt.title('Vpliv gostote celic na frekvenco v prostoru %dx%d' % (size, size))
    y_smooth = gaussian_filter1d(y_freq, sigma=2)
    plt.plot(y_smooth, x_freq)
    plt.ylabel('Frekvenca')
    plt.xlabel('Gostota')
    plt.show()


def make_heatmap(x, y, data, y_label='Gostota', x_label='Hitrost difuzije'):
    """
    Draw heat map
    :param x_label: x label title
    :param y_label: y label title
    :param x: x values array
    :param y: y values array
    :param data: numpy data array
    :return:
    """
    fig, ax = plt.subplots()

    im, cbar = heatmap(data, x, y, ax=ax, cmap="YlGn", cbarlabel="Oscilira [DA/NE]")
    texts = annotate_heatmap(im, valfmt="{x:d}")

    fig.tight_layout()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()


def draw():
    """
    Saved results from analyse_diffusion_rate_and_density.
    Use this to redraw heat maps
    """
    # 10x 10
    d_10 = [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
    tmp_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tmp_y = [0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.0]
    make_heatmap(tmp_x, tmp_y, d_10)

    # 15 x 15
    d_15 = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    tmp_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tmp_y = [0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.0]
    make_heatmap(tmp_x, tmp_y, d_15)

    # 20 x 20
    d_20 = [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
    tmp_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tmp_y = [0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.0]
    make_heatmap(tmp_x, tmp_y, d_20)


if __name__ == "__main__":
    # Vpliv hitrosti difuzije in gostote na oscilacije
    analyse_diffusion_rate_and_density(number_of_examples=10, size=10)
    analyse_diffusion_rate_and_density(number_of_examples=10, size=15)
    analyse_diffusion_rate_and_density(number_of_examples=10, size=20)

    # Vpliv hitrosti difuzije na oscilacije pri podani velikosti prostora
    analyse_diffusion_rate(number_of_examples=10, size=10)
    analyse_diffusion_rate(number_of_examples=20, size=15)
    analyse_diffusion_rate(number_of_examples=20, size=20)

    # Vpliv gostote celic na oscilacije pri podani velikosti prostora
    analyse_density(number_of_examples=150, size=10)
    analyse_density(number_of_examples=150, size=15)
    analyse_density(number_of_examples=150, size=20)
