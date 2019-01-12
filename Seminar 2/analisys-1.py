import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy.ndimage.filters import gaussian_filter1d

import represilator as r


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
    example_num = 0

    """Vpliv hitrosti difuzije na oscilacije"""
    for d in np.linspace(start=.01, stop=3, num=number_of_examples):
        for density in np.linspace(start=.1, stop=1, num=5):
            rep = r.Repressilator_S_PDE()
            rep.set_params(D1=d, density=density, size=size)
            # check if model oscillates
            [osc, freq, period, amplitude, damped] = rep.run()
            print("#%d | density: %f, | oscillates: %d, frequency: %f, period: %f, amplitude: %f" % (
                example_num, density, osc, freq, period, amplitude))

            example_num += 1
            del rep
            if osc:
                x.append(d)
                y.append(density)
            else:
                break

    # draw graph
    print(x)
    print(y)
    plt.title('Vpliv hitrosti difuzije na oscilacije v prostoru %dx%d' % (size, size))
    plt.plot(x, y)
    plt.ylabel('Gostota')
    plt.xlabel('Hitrost difuzije')
    plt.show()


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

    # # draw graph
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(x_amp, y_amp, 'r', x_per, y_per, 'b')
    # ax.set_title('Vplih hitrosti difuzije na oscilacije pri podani velikosti prostora')
    #
    # # display the plot
    # # plt.show()
    #
    # print(x_amp)
    # print(y_amp)

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


if __name__ == "__main__":
    analyse_diffusion_rate_and_density(size=10, number_of_examples=5)

    # Vpliv hitrosti difuzije na oscilacije pri podani velikosti prostora
    # analyse_diffusion_rate(number_of_examples=10, size=10)
    # analyse_diffusion_rate(number_of_examples=20, size=15)
    # analyse_diffusion_rate(number_of_examples=20, size=20)

    # Vpliv gostote celic na oscilacije pri podani velikosti prostora
    # analyse_density(number_of_examples=150, size=10)
    # analyse_density(number_of_examples=150, size=15)
    # analyse_density(number_of_examples=150, size=20)
