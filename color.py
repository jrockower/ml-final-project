import pandas as pd
import numpy as np

def process_color(start, end, bins):
    '''
    Calculate color gradient for a single color
    Input:
        start (int):
        end (int):
        bins (int):

    Return:
        l (list): list size bins + 2 divided into even increments
    '''
    l = []
    if end > start:
        inc = (end-start) // (bins +2)
        color = start

        while color < end:
            l.append(color)
            color += inc

    else:
        inc = (start-end) // (bins+2)
        color = start

        while color > end:
            l.append(color)
            color -= inc

    l.append(end)

    return l

def gen_gradient(start, end, bins):
    red = process_color(start[0], end[0], bins)
    green = process_color(start[1], end[1], bins)
    blue = process_color(start[2], end[2], bins)

    colorscale = []
    for i in range(bins):
        rgb = 'rgb({},{},{})'.format(red[i], green[i], blue[i])
        colorscale.append(rgb)

    return colorscale


def gen_bins_and_gradient(bins, scale=True):
    if scale:
        start = (255, 250, 250)
        end = (0, 0, 128)
        end = (0, 78,69)
    else:
        #https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
        # For positive and negative
        start = (220, 50, 92)
        end = (0,90,181)
    red = process_color(start[0], end[0], bins)
    green = process_color(start[1], end[1], bins)
    blue = process_color(start[2], end[2], bins)

    colorscale = []
    for i in range(bins):
        rgb = 'rgb({},{},{})'.format(red[i], green[i], blue[i])
        colorscale.append(rgb)

    if not scale:
        # ensure median value is white
        med_index = len(colorscale) // 2
        colorscale[med_index] = 'rgb({},{},{})'.format(180,181,178)

    return colorscale

df = pd.DataFrame(np.random.randint(-100,100,size=(15, 4)), columns=list('ABCD'))

def set_bins(att, df):
    desc = df[att].describe()
    inc = desc['std']
    lb =  desc['min']
    ub =  desc['max']

    #Round to nearest 100
    if inc > 10:
        inc = ((inc // 10) + 1) * 10

    bins = [0.0]
    b = 0.0

    while b > lb:
        b -= inc
        bins.append(b)
    b=0

    while b < ub:
        b += inc
        bins.append(b)

    bins.sort()

    while len(bins) > 5:
        bins = bins[::2]

    return bins
