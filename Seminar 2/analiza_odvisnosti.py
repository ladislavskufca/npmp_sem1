# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import time
import represilator as rep

def makePlotFromData(filename):

    #TODO: get this thing working!

    filename = filename + ".txt"

    file = open(filename, "r")

    results = float(int(file.read()))
    fig, ax = plt.subplots()
    im = ax.imshow(results)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_title('Alfa in alfa')
    # plt.xlabel('Alfa')
    # plt.ylabel('Alfa')
    plt.show()
    fig.savefig(filename + ".png")

def makeHeatmap(numberOfSamples=1):

    # merjenje casa
    timeMeasure = time.time()

    r = rep.Repressilator_S_PDE()
    results = []

    # numberOfSamples --> USED FOR LOGSPACE function

    # GRAF
    fig = plt.figure(1)
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'xtick.major.size': 6})
    plt.rcParams.update({'ytick.major.size': 6})


    params = ["", "alpha", "beta", "delta_m", "kappa"]
    paramsNumbers = [1, 2, 3, 4]

    counter = 1
    for i in paramsNumbers:
        for j in xrange(i, 5):

            r.load_params_range()
            results = []
            ax = plt.subplot(4, 4, counter)

            if i == 1:
                plt.xlabel('Alfa')
            if i == 2:
                plt.xlabel('Beta')
            if i == 3:
                plt.xlabel('Delta_m')
            if i == 4:
                plt.xlabel('Kappa')

            if j == 1:
                plt.ylabel('Alfa')
            if j == 2:
                plt.ylabel('Beta')
            if j == 3:
                plt.ylabel('Delta_m')
            if j == 4:
                plt.ylabel('Kappa')

            for FIRST_PARAM in np.logspace(start=-3, stop=1, num=numberOfSamples, base=10.0):

                helperArray = []
                for SECOND_PARAM in np.logspace(start=-3, stop=1, num=numberOfSamples, base=10.0):

                    print("%s: %f, %s: %f."% (params[i], FIRST_PARAM, params[j], SECOND_PARAM))

                    if i == 1:
                        r.load_params_range_single(alpha=FIRST_PARAM)
                        plt.xlabel('Alfa')
                    if i == 2:
                        r.load_params_range_single(beta=FIRST_PARAM)
                        plt.xlabel('Beta')
                    if i == 3:
                        r.load_params_range_single(delta_m=FIRST_PARAM)
                        plt.xlabel('Delta_m')
                    if i == 4:
                        r.load_params_range_single(kappa=FIRST_PARAM)
                        plt.xlabel('Kappa')

                    if j == 1:
                        r.load_params_range_single(alpha=SECOND_PARAM)
                        plt.ylabel('Alfa')
                    if j == 2:
                        r.load_params_range_single(beta=SECOND_PARAM)
                        plt.ylabel('Beta')
                    if j == 3:
                        r.load_params_range_single(delta_m=SECOND_PARAM)
                        plt.ylabel('Delta_m')
                    if j == 4:
                        r.load_params_range_single(kappa=SECOND_PARAM)
                        plt.ylabel('Kappa')


                    helperArray.append(r.run()[0])

                results.append(helperArray)

            file = open("results-k-params/44{}-{}.txt".format(counter + i, numberOfSamples), "w")
            file.write(str(results))
            file.close()

            im = ax.imshow(results)
            ax.set_xscale('log')
            ax.set_yscale('log')
            # ax.set_title('Alfa in alfa')
            # plt.xlabel('Alfa')
            # plt.ylabel('Alfa')

            counter = counter + 1

        counter = counter + i


    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    plt.show()
    fig.savefig('results-k-params/final-{}.png'.format(numberOfSamples))

    print("Porabljen cas: {} s.".format(time.time() - timeMeasure))



if __name__ == "__main__":
    r = rep.Repressilator_S_PDE()
    results = []

    # IMPORTANT: RUN ONLY 1 of following methods!

    # run makeHeatmap for plot with all subplots
    samples = 2 # USED FOR LOGSPACE function
    makeHeatmap(samples)

    # run makePlotFromData with filename to draw only 1 heatmap for specific run
    # filename = "results-k-params/443-3" #WITHOUT TXT!
    # makePlotFromData(filename)
