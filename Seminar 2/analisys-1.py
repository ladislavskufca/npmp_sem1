import numpy as np
import matplotlib.pyplot as plt

import represilator as r

NUMBER_OF_EXAMPLES = 100

"""
Analiza tocke 1

Kakšen vpliv ima na oscilacije gostota populacije celic, hitrost difuzije in velikost prostora? 

D [10^-2, 10^2]
velikost prostora: 10x10
gostota populacije: [0.01, 1]

"""


if __name__ == "__main__":
    x = []
    y = []
    example_num = 0

    """Vpliv hitrosti difuzije na oscilacije"""
    for d in np.linspace(start=.01, stop=1, num=NUMBER_OF_EXAMPLES):
        for density in np.linspace(start=.1, stop=1, num=4):
            rep = r.Repressilator_S_PDE()
            rep.load_params()
            rep.set_params(D1=d, density=density)
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
    plt.title('Vpliv hitrosti difuzije na oscilacije v prostoru 10x10')
    plt.plot(x, y, 'ro')
    plt.ylabel('Gostota')
    plt.xlabel('Hitrost difuzije')
    plt.show()
