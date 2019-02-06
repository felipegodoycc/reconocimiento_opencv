from scipy.spatial import distance

# Directorio donde se guardan fotografias
DIR_FACES = "files/caras"
# Tamaño para reducir a miniaturas las fotografias
SIZE = 4
# Directorio donde se guarda el modelo
MODEL_FACES = 'files/models/'


def threshold(actual, anterior, thres=10):
    if len(actual) == len(anterior):
        for i in range(len(actual)):
            if anterior[i] + thres > actual[i] > anterior[i] - thres:
                pass
            else:
                return False
        return True
    else:
        return False


def euc(coord_a, coord_b):
    return distance.euclidean(coord_a, coord_b)
