import represilator as rep
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    r = rep.Repressilator_S_PDE()
    results = []


    # GRAF

    for ALFA in np.logspace(start=0.001, stop=10, num=10):
        for BETA in np.logspace(start=0.001, stop=10, num=10):
            print("alfa: %f, beta: %f."% (ALFA, BETA))
            r.load_params_range(alpha=ALFA, beta=BETA)
            oscilacija = r.run()[0]
            if oscilacija:
                plt.plot(ALFA, BETA, 'ro')

    plt.show()



    # ------------------------------------------------------------------------------------

    # HEATMAP

    # for ALFA in np.logspace(start=0.001, stop=10, num=3):
    #     helperArray = []

        # for BETA in np.logspace(start=0.001, stop=10, num=3):
        #     print("alfa: %f, beta: %f."% (ALFA, BETA))
        #     r.load_params_range(alpha=ALFA, beta=BETA)
        #     helperArray.append(r.run()[0])

        # results.append(helperArray)

    # plt.imshow(results, cmap='hot', interpolation='nearest')
    # plt.show()

