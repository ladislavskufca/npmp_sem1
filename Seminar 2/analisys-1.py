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


if __name__ == "__main__":
    analyse_diffusion_rate_and_density(15)
