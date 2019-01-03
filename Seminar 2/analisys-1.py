import numpy as np
import matplotlib.pyplot as plt

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
            rep.load_params()
            rep.set_params(D1=d, density=density, size=size)
            # check if model oscillates
            oscillates = rep.run()[0]
            print("#%d | d: %f, density: %f, oscillates: %d" % (example_num, d, density, oscillates))
            example_num += 1
            if oscillates:
                x.append(d)
                y.append(density)
            else:
                break

    # draw graph
    print(x)
    print(y)
    plt.title('Vpliv hitrosti difuzije na oscilacije v prostoru %dx%d' % (size, size))
    plt.plot(x, y, 'ro')
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
    Vpliv hitrosti difuzije na oscilacije pri podani velikosti prostora
    :return:
    """
    example_num = 0
    x_per = []
    y_per = []

    x_amp = []
    y_amp = []

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

        example_num += 1

    plt.subplot(2, 1, 1)
    plt.title('Vpliv gostote celic na amplitudo v prostoru %dx%d' % (size, size))
    plt.plot(y_amp, x_amp)
    plt.ylabel('Amplituda')
    plt.xlabel('Gostota')

    plt.subplot(2, 1, 2)
    plt.title('Vpliv gostote celic na periodo v prostoru %dx%d' % (size, size))
    plt.plot(y_per, x_per)
    plt.ylabel('Perioda')
    plt.xlabel('Gostota')
    plt.show()


if __name__ == "__main__":
    # analyse_diffusion_rate_and_density(15)
    # analyse_diffusion_rate(number_of_examples=20)
    analyse_density(number_of_examples=20)
